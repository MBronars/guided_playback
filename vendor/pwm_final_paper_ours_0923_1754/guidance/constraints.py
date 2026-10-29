"""Interleaved augmented-Lagrangian updates for measured geometric constraints.

Finite gradient steps on nonconvex geometric proxies do not guarantee feasibility.
Final candidates must be screened independently; this is not a HardFlow port.
"""
import math

import torch


class ConstrainedGraspObjective:
    def __init__(self, objective, *, penalty=2000., dual_every=2, residual_fn=None,
                 residual_scales=None):
        if not math.isfinite(penalty) or penalty <= 0 or dual_every < 1:
            raise ValueError('Invalid augmented-Lagrangian settings')
        self.objective=objective
        self.penalty=float(penalty)
        self.dual_every=int(dual_every)
        self.multipliers={}
        self.residual_fn=residual_fn
        self.residual_scales=dict(residual_scales or {})
        if any(not math.isfinite(v) or v<=0 for v in self.residual_scales.values()):
            raise ValueError('Constraint scales must be finite and positive')

    def __call__(self,pred,batch):
        base,diag=self.objective(pred,batch)
        residuals=(self.residual_fn(pred) if self.residual_fn is not None else
                   self.objective.residuals(self.objective.features(pred)))
        if self.residual_scales and self.residual_scales.keys()!=residuals.keys():
            raise ValueError('Constraint scales must cover exactly the active residuals')
        augmented=base*0
        for key,value in residuals.items():
            raw=value
            if self.residual_scales:value=value/self.residual_scales[key]
            multiplier=self.multipliers.get(key,0.)
            augmented=augmented+((multiplier+self.penalty*value).clamp_min(0).square()-multiplier**2)/(2*self.penalty)
            diag['dual_'+key]=multiplier
            diag['constraint_'+key+'_m']=float(raw.detach())
            if self.residual_scales:
                diag['constraint_'+key+'_scaled']=float(value.detach())
        diag['selection_loss']=float(base.detach())
        diag['augmented_penalty']=float(augmented.detach())
        return base+augmented,diag

    def after_step(self,diagnostics,update):
        if (update+1)%self.dual_every==0:
            for key,value in diagnostics.items():
                if key.startswith('constraint_') and key.endswith('_m'):
                    name=key[len('constraint_'):-2]
                    scaled=value/self.residual_scales.get(name,1.)
                    self.multipliers[name]=max(0.,self.multipliers.get(name,0.)+self.penalty*scaled)
