"""Coherent rigid observation biases for inference robustness experiments."""
from dataclasses import replace
import math

import torch

from utils.pose_ops import so3_exp


def rigid_observation_bias(batch, *, translation_m=0., rotation_deg=0., seed=0, roots=(0,)):
    """Bias observed root bodies and their descendants by a common rigid transform.

    Magnitudes are fixed norms, not per-axis standard deviations. Body templates,
    identities and true future reference labels are retained. Intended for hidden-
    action inference; this does not construct noisy training targets.
    """
    if batch.batch_size!=1 or batch.body_poses is None:
        raise ValueError('Observation corruption currently requires one pose-native scene')
    if min(translation_m,rotation_deg)<0 or not all(math.isfinite(v) for v in (translation_m,rotation_deg)):
        raise ValueError('Observation-bias magnitudes must be finite and nonnegative')
    parent=batch.body_parent[0] if batch.body_parent is not None else torch.full_like(batch.body_type[0],-1)
    if len(set(roots))!=len(roots) or any(k<0 or k>=len(parent) or parent[k]>=0 for k in roots):
        raise ValueError('Choose distinct root bodies; articulated children move with their parent')
    generator=torch.Generator(device='cpu').manual_seed(int(seed))
    poses=batch.body_poses.clone(); positions=batch.positions.clone(); C=batch.context_len
    receipts=[]
    for root in roots:
        group={int(root)}
        while True:
            expanded=group|{i for i,p in enumerate(parent.tolist()) if p in group}
            if expanded==group: break
            group=expanded
        group=sorted(group)
        direction=torch.randn(3,generator=generator,dtype=torch.float64)
        direction/=direction.norm()
        axis=torch.randn(3,generator=generator,dtype=torch.float64); axis/=axis.norm()
        rotation=so3_exp((axis*math.radians(rotation_deg))[None])[0].to(positions)
        translation=(direction*translation_m).to(positions)
        pivot=batch.body_poses[0,C-1,root,:3,3]
        rows=torch.isin(batch.particle_body[0],torch.tensor(group,device=batch.device))
        if translation_m or rotation_deg:
            positions[0,:C,rows]=(batch.positions[0,:C,rows]-pivot) @ rotation.T+pivot+translation
            for body in group:
                poses[0,:C,body,:3,:3]=rotation @ batch.body_poses[0,:C,body,:3,:3]
                poses[0,:C,body,:3,3]=(batch.body_poses[0,:C,body,:3,3]-pivot) @ rotation.T+pivot+translation
        receipts.append(dict(root=root,bodies=group,translation_m=translation.cpu().tolist(),
                             rotation_axis=axis.tolist(),rotation_deg=rotation_deg))
    changed=replace(batch,positions=positions,body_poses=poses)
    return changed,dict(seed=seed,kind='constant coherent observed rigid-pose bias',transforms=receipts,
        reference_future='unchanged true future; do not use this batch for noisy-target training')


def density_estimation_bias(batch,scale=1.,bodies=(0,)):
    """Change only the planner's density input; retain a separate true batch."""
    if not math.isfinite(scale) or scale<=0:
        raise ValueError('Density scale must be finite and positive')
    if any(k<0 or k>=batch.body_type.shape[1] for k in bodies):
        raise ValueError('Invalid body for density bias')
    mask=torch.isin(batch.particle_body,torch.tensor(bodies,device=batch.device))
    rho=torch.where(mask,batch.rho*scale,batch.rho)
    return replace(batch,rho=rho),dict(kind='density-estimation multiplicative bias',scale=scale,bodies=list(bodies),
        physical_truth='not modified; this is an input-sensitivity probe until simulator execution is qualified')
