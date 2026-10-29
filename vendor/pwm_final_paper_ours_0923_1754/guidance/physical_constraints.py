"""Differentiable counterparts to the independent pick/place geometry screens.

These constraints use observed geometry and joint calibration, with no grasp
proposal or prescribed closure time. Feasibility is a geometric proxy only.
"""
import math

import torch

from guidance.grasp import rotation_chord
from utils.pose_ops import rotation_geodesic_error


class PhysicalConstraints:
    def __init__(self, objective, *, task='pick', hold_frames=6):
        if task not in ('pick', 'place') or hold_frames < 2:
            raise ValueError('Expected pick/place constraints and at least two hold frames')
        self.objective = objective
        self.task = task
        self.hold_frames = hold_frames

    def tolerance_scales(self):
        """Fixed task tolerances for optional dimensionless AL violations.

        These do not change acceptance thresholds. A value of2 represents a
        signed violation twice that constraint's declared tolerance.
        """
        o=self.objective
        scales=dict(penetration=getattr(o,'mesh_penetration_tolerance',.001),
                    support=.003,gripper_support=.003,opening=.0001,
                    speed=float(o.calibration['limits'][:,2].min())*float(o.calibration['dt']),
                    goal=.020)
        if self.task=='pick':
            scales.update(contact=getattr(o,'mesh_contact_tolerance',.005),
                          relative_translation=.003,relative_rotation=.04*2*math.sin(math.radians(5)/2))
        else:
            scales.update(rest=.005,release=.005,settle=.002)
            if o.target_R is not None:scales['goal_rotation']=.04*math.radians(15)
        return scales

    def __call__(self, pred):
        o, H = self.objective, self.hold_frames
        if pred['t'].shape[1] < H:
            raise ValueError('Not enough frames for the requested hold')
        f = o.features(pred)
        c = o.calibration
        columns = {int(k): i for i, k in enumerate(pred['rigid_idx'].tolist())}
        j, h = columns[o.object_body], columns[o.hand_body]
        R, t = pred['R'], pred['t']
        q = f['joint_q']
        qfull = torch.cat([c['q_anchor'][None, None], q], dim=1)
        speed = (qfull[:, 1:]-qfull[:, :-1]).abs()/float(c['dt'])
        limits = c['limits']
        depth = torch.maximum(f['inside_object'].amax(-1), f['inside_gripper_hull'].amax(-1))[:, 1:]
        values = dict(
            penetration=depth.amax()-.001,
            support=(o.support_z-f['bottom'][:, 1:]).amax()-.003,
            gripper_support=(o.support_z-f['gripper_bottom'][:, 1:]).amax()-.003,
            opening=torch.maximum(limits[:, 0]-q, q-limits[:, 1]).amax()-.0001,
            speed=(speed-limits[:, 2]).amax()*float(c['dt']),
            goal=(t[:, -H:, j]-o.target[:, None]).norm(dim=-1).amax()-.020)
        if self.task == 'pick':
            relative_t = (R[:, :, h].transpose(-1, -2) @ (t[:, :, j]-t[:, :, h])[..., None])[..., 0]
            relative_R = R[:, :, h].transpose(-1, -2) @ R[:, :, j]
            translation = (relative_t[:, 1:]-relative_t[:, :-1]).norm(dim=-1)
            rotation = rotation_chord(relative_R[:, 1:], relative_R[:, :-1])
            values.update(contact=f['gaps'][:, -H:].amax()-.005,
                relative_translation=translation[:, -(H-1):].amax()-.003,
                relative_rotation=.04*(rotation[:, -(H-1):].amax()-2*math.sin(math.radians(5)/2)))
        else:
            values.update(rest=(f['bottom'][:, -H:]-o.support_z).abs().amax()-.005,
                release=.005-f['gaps'][:, -H:].amin(),
                settle=f['object_step'][:, -(H-1):].norm(dim=-1).amax()-.002)
            if o.target_R is not None:
                angles = rotation_geodesic_error(R[:, -H:, j], o.target_R.to(R))
                values['goal_rotation'] = .04*(angles.amax()-math.radians(15))
        if 'mesh_distance' in f:
            distance=f['mesh_distance']
            values['penetration']=-distance[:,1:].amin()-o.mesh_penetration_tolerance
            if self.task=='pick':
                values['contact']=distance[:,-H:,1:].amax()-o.mesh_contact_tolerance
            else:
                values['release']=.005-distance[:,-H:,1:].amin()
        if getattr(o,'goal_region',None) is not None:
            values['goal']=o.goal_region.signed_violation(t[:,-H:,j]).amax()
        return values
