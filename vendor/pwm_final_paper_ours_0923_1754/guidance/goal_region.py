"""Explicit position-region goals, separate from precise point-goal experiments."""
import torch


class PositionGoalRegion:
    """An axis-aligned closed box in physical metres; axes may be unbounded.

    This describes a different task from reaching the box's centre. For example,
    a lower height bound and bounded XY workspace describe a minimum-height lift.
    Bounds are fixed task inputs, never optimized or inferred from future states.
    """
    def __init__(self, lower, upper):
        self.lower=torch.as_tensor(lower,dtype=torch.float64).detach().clone()
        self.upper=torch.as_tensor(upper,dtype=torch.float64).detach().clone()
        if self.lower.shape!=(3,) or self.upper.shape!=(3,):
            raise ValueError('Position-region bounds must each contain three coordinates')
        if (torch.isnan(self.lower).any() or torch.isnan(self.upper).any()
                or torch.isposinf(self.lower).any() or torch.isneginf(self.upper).any()
                or not bool((self.lower<self.upper).all())):
            raise ValueError('Position-region bounds must define nonempty intervals without NaNs')
        if not bool(torch.isfinite(self.lower).any() or torch.isfinite(self.upper).any()):
            raise ValueError('A position region needs at least one finite bound')

    def signed_violation(self, positions):
        """Largest coordinate-bound violation; nonpositive means inside the box."""
        return torch.maximum(self.lower.to(positions)-positions,
                             positions-self.upper.to(positions)).amax(-1)

    def distance(self, positions):
        """Euclidean distance to the box, zero throughout its interior."""
        return torch.maximum(self.lower.to(positions)-positions,
                             positions-self.upper.to(positions)).clamp_min(0).norm(dim=-1)


class PositionRegionValidator:
    """Keep the supplied physical screen, replacing only its point-goal criterion."""
    def __init__(self, validator, region, *, object_body=0, hold_frames=6):
        if hold_frames<1:
            raise ValueError('Region validation needs a positive hold interval')
        self.validator,self.region=validator,region
        self.object_body,self.hold_frames=object_body,hold_frames

    @torch.no_grad()
    def __call__(self,pred):
        if pred['t'].shape[1]<self.hold_frames:
            raise ValueError('Prediction does not cover the region hold interval')
        base=self.validator(pred)
        column=(pred['rigid_idx']==self.object_body).nonzero(as_tuple=True)[0].item()
        values=dict(base['residuals_m'])
        values['goal']=float(self.region.signed_violation(pred['t'][:,-self.hold_frames:,column]).max())
        return dict(base,accepted=all(v<=0 for v in values.values()),
                    max_violation_m=max(0.,max(values.values())),residuals_m=values,
                    label='Position-region task; '+base['label'])
