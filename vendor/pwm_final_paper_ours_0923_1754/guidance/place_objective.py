"""Proposal-free placing costs and geometric audit; execution remains unverified."""
import math
import torch

from guidance.physical_objective import PickPhysicalObjective
from guidance.pick_validation import PickValidator
from utils.pose_ops import rotation_geodesic_error


class PlacePhysicalObjective(PickPhysicalObjective):
    def __init__(self,*args,target_R=None,rotation_weight=.04,settle_frames=6,**kwargs):
        if kwargs.get('goal_region') is not None:
            raise ValueError('Position-region objectives currently support picking only')
        super().__init__(*args,**kwargs)
        self.target_R=target_R;self.rotation_weight=rotation_weight;self.settle_frames=settle_frames
        self.weights.update(rest=5.,release=2.,settle=2.,rotation=rotation_weight)

    def features(self,pred):
        f=super().features(pred)
        j=(pred['rigid_idx']==self.object_body).nonzero(as_tuple=True)[0].item()
        f['goal_rotation']=rotation_geodesic_error(pred['R'][:,:,j],self.target_R.to(pred['R'])) if self.target_R is not None else pred['t'].new_zeros(pred['t'].shape[:2])
        f['goal_distance_track']=(pred['t'][:,:,j]-self.target[:,None]).norm(dim=-1)
        return f

    def terms(self,f):
        terms=super().terms(f);H=min(self.settle_frames,f['object_step'].shape[1])
        # Keep proximity during transport; allow and request release at the rest
        # target. The picking objective's final held-contact term is inapplicable.
        gaps=f['gaps'][:,1:];lifted=((f['bottom'][:,1:]-self.support_z-.003)/.02).clamp(0,1)
        terms.update(contact=(lifted*(gaps.amax(-1)-self.contact_tolerance).clamp_min(0)).mean(),
            goal=self.reduce_goal(f['goal_distance_track'][:,-H:]),
            rest=(f['bottom'][:,-H:]-self.support_z).abs().mean(),
            release=(.005-gaps[:,-H:].amin(-1)).clamp_min(0).mean(),
            settle=f['object_step'][:,-max(1,H-1):].norm(dim=-1).mean(),
            rotation=f['goal_rotation'][:,-H:].mean())
        return terms

    def __call__(self,pred,batch):
        loss,diag=super().__call__(pred,batch)
        angle=diag.pop('rotation_cost_mm')/1000
        diag.update(rotation_cost_rad=angle,rotation_weighted_cost_mm=angle*self.rotation_weight*1000)
        return loss,diag


class PlaceValidator(PickValidator):
    def __init__(self,*args,target_R=None,**kwargs):
        super().__init__(*args,**kwargs);self.target_R=target_R

    @torch.no_grad()
    def __call__(self,pred):
        base=super().__call__(pred);f=self.diagnostic.features(pred);H=self.hold_frames
        values={k:v for k,v in base['residuals_m'].items() if k not in ('contact','relative_translation','relative_rotation')}
        values.update(rest=float((f['bottom'][:,-H:]-self.diagnostic.support_z).abs().max())-.005,
            release=.005-float(f['gaps'][:,-H:].amin()),
            settle=float(f['object_step'][:,-(H-1):].norm(dim=-1).amax())-.002)
        if self.target_R is not None:
            j=(pred['rigid_idx']==0).nonzero(as_tuple=True)[0].item()
            angles=rotation_geodesic_error(pred['R'][:,-H:,j],self.target_R.to(pred['R']))
            values['goal_rotation']=.04*(float(angles.max())-math.radians(15))
        physical=[v for k,v in values.items() if k not in ('goal','goal_rotation')]
        return dict(accepted=all(v<=0 for v in values.values()),interaction_accepted=all(v<=0 for v in physical),
            max_violation_m=max(0.,max(values.values())),max_interaction_violation_m=max(0.,max(physical)),
            residuals_m=values,label='placing: sampled geometric rest/release screen; simulator execution unverified')
