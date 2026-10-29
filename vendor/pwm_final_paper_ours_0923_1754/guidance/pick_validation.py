"""Independent geometric screening of complete generated picks.

Acceptance means these measured proxies pass, not that physics execution works.
No particular proposed contact pair or recorded future is required.
"""
from __future__ import annotations

import math

import torch

from guidance.grasp import rotation_chord
from guidance.pick_objective import PickObjective


class PickValidator:
    def __init__(self, batch, target, geometry, calibration, *, hold_frames=6,
                 contact_tolerance=.005, penetration_tolerance=.001,
                 target_tolerance=.020, relative_step_tolerance=.003):
        if hold_frames < 2 or min(contact_tolerance, penetration_tolerance, target_tolerance, relative_step_tolerance) <= 0:
            raise ValueError('Invalid geometric validator tolerances')
        self.diagnostic=PickObjective(batch,target,geometry)
        self.batch,self.target=batch,target
        self.calibration={k:v.to(batch.positions) if torch.is_tensor(v) else v for k,v in calibration.items()}
        self.hold_frames=hold_frames
        self.contact_tolerance=contact_tolerance
        self.penetration_tolerance=penetration_tolerance
        self.target_tolerance=target_tolerance
        self.relative_step_tolerance=relative_step_tolerance

    @torch.no_grad()
    def __call__(self,pred):
        if pred['t'].shape[1] < self.hold_frames:
            raise ValueError('Not enough frames for the requested hold')
        f=self.diagnostic.features(pred)
        if not all(bool(torch.isfinite(v).all()) for v in pred.values() if torch.is_floating_point(v)):
            raise FloatingPointError('Nonfinite predicted trajectory')
        columns={int(k):i for i,k in enumerate(pred['rigid_idx'].tolist())}
        h,j=columns[1],columns[0]; fingers=[columns[2],columns[3]]
        R,t=pred['R'],pred['t']; c=self.calibration
        rel=(R[:,:,h].transpose(-1,-2)[:,:,None] @ (t[:,:,fingers]-t[:,:,h,None])[...,None])[...,0]
        q=c['q_anchor']+((rel-c['finger_anchor'])*c['axis']).sum(-1)
        qfull=torch.cat([c['q_anchor'][None,None],q],dim=1)
        speed=(qfull[:,1:]-qfull[:,:-1]).abs()/float(c['dt'])
        relative_t=(R[:,:,h].transpose(-1,-2) @ (t[:,:,j]-t[:,:,h])[...,None])[...,0]
        relative_R=R[:,:,h].transpose(-1,-2) @ R[:,:,j]
        translation=(relative_t[:,1:]-relative_t[:,:-1]).norm(dim=-1)
        rotation=rotation_chord(relative_R[:,1:],relative_R[:,:-1])
        depth=torch.maximum(f['inside_object'].amax(-1),f['inside_gripper_hull'].amax(-1))[:,1:]
        limits=c['limits']; H=self.hold_frames
        residuals=dict(contact=f['gaps'][:,-H:].amax()-self.contact_tolerance,
            penetration=depth.amax()-self.penetration_tolerance,
            support=(self.diagnostic.support_z-f['bottom'][:,1:]).amax()-.003,
            gripper_support=(self.diagnostic.support_z-f['gripper_bottom'][:,1:]).amax()-.003,
            opening=torch.maximum(limits[:,0]-q,q-limits[:,1]).amax()-.0001,
            speed=(speed-limits[:,2]).amax()*float(c['dt']),
            relative_translation=translation[:,-(H-1):].amax()-self.relative_step_tolerance,
            relative_rotation=.04*(rotation[:,-(H-1):].amax()-2*math.sin(math.radians(5)/2)),
            goal=(t[:,-H:,j]-self.target[:,None]).norm(dim=-1).amax()-self.target_tolerance)
        values={k:float(v) for k,v in residuals.items()}
        physical=[v for k,v in values.items() if k!='goal']
        return dict(accepted=all(v<=0 for v in values.values()),
                    interaction_accepted=all(v<=0 for v in physical),
                    max_violation_m=max(0.,max(values.values())),
                    max_interaction_violation_m=max(0.,max(physical)),residuals_m=values,
                    label='sampled geometric screening; simulator execution unverified')


def selection_key(loss,validation=None):
    if validation is None: return (float(loss),)
    if validation['accepted']: return (0.,float(loss))
    # A physically credible interaction that misses the goal must outrank a
    # phantom goal hit. Goal residual must not dominate fallback feasibility.
    if validation.get('interaction_accepted',False): return (1.,float(loss))
    violation=validation.get('max_interaction_violation_m',validation['max_violation_m'])
    return (2.,violation,float(loss))
