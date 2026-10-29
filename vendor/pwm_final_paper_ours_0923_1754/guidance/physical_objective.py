"""Pickup interaction costs with peak geometry and articulated joint penalties.

No grasp proposal, prescribed contact pair, phase schedule or recorded future is
used. These remain geometric penalties; they do not establish executed dynamics.
"""
import math

import torch

from guidance.pick_objective import PickObjective
from utils.pose_ops import rotation_geodesic_error


class PhysicalTermSubset:
    """Select auxiliary costs without changing task goals or final evaluation.

    An empty subset evaluates the held pose goal directly, avoiding unused mesh
    queries during differentiation. Nonempty subsets retain the original feature
    definitions. Default callers do not use this wrapper.
    """
    def __init__(self, base, enabled):
        self.base = base
        self.enabled = frozenset(enabled)
        available = set(base.weights) - {'rotation'}
        if not self.enabled <= available:
            raise ValueError(f'Unknown physical terms: {sorted(self.enabled - available)}')

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, pred, batch):
        base = self.base
        if not self.enabled:
            j = int((pred['rigid_idx'] == base.object_body).nonzero()[0])
            track = (pred['t'][:, :, j] - base.target[:, None]).norm(dim=-1)
            frames = getattr(base, 'settle_frames', base.goal_frames)
            if track.shape[1] < frames:
                raise ValueError('Prediction does not cover the requested goal interval')
            if base.goal_region is not None:
                track = base.goal_region.distance(pred['t'][:, :, j])
            terms = {'goal': base.reduce_goal(track[:, -frames:])}
            if 'rotation' in base.weights:
                terms['rotation'] = (rotation_geodesic_error(pred['R'][:, -frames:, j],
                    base.target_R.to(pred['R'])).mean() if base.target_R is not None
                    else pred['t'].new_zeros(()))
        else:
            terms = base.terms(base.features(pred))
        loss = base.goal_weight * terms['goal'] + sum(
            (weight if key == 'rotation' else base.physical_weight * weight) * terms[key]
            for key, weight in base.weights.items() if key in self.enabled or key == 'rotation')
        diagnostics = {key + ('_cost_rad' if key == 'rotation' else '_cost_mm'):
            float(value.detach()) * (1 if key == 'rotation' else 1000)
            for key, value in terms.items()}
        return loss, diagnostics


class PickPhysicalObjective(PickObjective):
    def __init__(self,batch,target,geometry,calibration,*,joint_weight=10.,speed_weight=10.,
                 goal_frames=1,goal_region=None,goal_reduction='mean',**kwargs):
        super().__init__(batch,target,geometry,**kwargs)
        if isinstance(goal_frames,bool) or not isinstance(goal_frames,int) or goal_frames<1:
            raise ValueError('Goal frames must be a positive integer')
        self.goal_frames=goal_frames
        self.goal_region=goal_region
        if goal_reduction not in ('mean','max'):
            raise ValueError('Goal reduction must be mean or max')
        self.goal_reduction=goal_reduction
        if any(not math.isfinite(v) or v<0 for v in (joint_weight,speed_weight)):
            raise ValueError('Joint penalty weights must be finite and nonnegative')
        self.calibration={k:v.to(batch.positions) if torch.is_tensor(v) else v for k,v in calibration.items()}
        self.weights.update(joint=joint_weight,joint_speed=speed_weight)
        self.weights['gripper_support']=self.weights['support']

    def reduce_goal(self, distances):
        """Position-only reduction; max matches the held-position task screen."""
        return distances.amax() if self.goal_reduction=='max' else distances.mean()

    def features(self,pred):
        f=super().features(pred);c=self.calibration
        if pred['t'].shape[1]<self.goal_frames:
            raise ValueError('Prediction does not cover the requested goal interval')
        col={int(k):i for i,k in enumerate(pred['rigid_idx'].tolist())}
        if self.goal_region is not None:
            f['region_goal_distance']=self.goal_region.distance(pred['t'][:,:,col[self.object_body]])
        if self.goal_frames>1:
            f['physical_goal_track']=(pred['t'][:,:,col[self.object_body]]-self.target[:,None]).norm(dim=-1)
        h=col[self.hand_body];fingers=[col[k] for k in self.fingers]
        R,t=pred['R'],pred['t']
        local=(R[:,:,h].transpose(-1,-2)[:,:,None] @ (t[:,:,fingers]-t[:,:,h,None])[...,None])[...,0]
        q=c['q_anchor']+((local-c['finger_anchor'])*c['axis']).sum(-1)
        qfull=torch.cat([c['q_anchor'][None,None],q],dim=1)
        limits=c['limits']
        f.update(joint_q=q,joint_violation=torch.maximum(limits[:,0]-q,q-limits[:,1]).clamp_min(0),
            joint_speed_violation=((qfull[:,1:]-qfull[:,:-1]).abs()-limits[:,2]*float(c['dt'])).clamp_min(0))
        return f

    def terms(self,f):
        terms=super().terms(f)
        if self.goal_frames>1:
            terms['goal']=self.reduce_goal(f['physical_goal_track'][:,-self.goal_frames:])
        if self.goal_region is not None:
            terms['goal']=self.reduce_goal(f['region_goal_distance'][:,-self.goal_frames:])
        depth=torch.maximum(f['inside_object'].amax(-1),f['inside_gripper_hull'].amax(-1))[:,1:]
        excess=(depth-.001).clamp_min(0)
        support=(self.support_z-f['bottom'][:,1:]-self.support_tolerance).clamp_min(0)
        gripper_support=(self.support_z-f['gripper_bottom'][:,1:]-self.support_tolerance).clamp_min(0)
        terms.update(penetration=excess.mean()+excess.amax(),support=support.mean()+support.amax(),
            gripper_support=gripper_support.mean()+gripper_support.amax(),
            joint=f['joint_violation'].mean()+f['joint_violation'].amax(),
            joint_speed=f['joint_speed_violation'].mean()+f['joint_speed_violation'].amax())
        return terms
