"""Explicit finger command policies for simulator qualification, in metres."""
import math
import numpy as np
import torch

from utils.pose_ops import quat_wxyz_to_matrix


def mesh_preload_commands(achieved, proximity, *, preload=.006, policy='instant', release_delta=.002):
    """Translate achieved finger coordinates into an explicit servo program.

    Instant preserves the historical stateless proximity gate exactly. Latched
    engages when both fingers are near and retains preload through proximity
    flicker. A deliberate mean opening of release_delta from the tightest held
    coordinates releases it; another such closing stroke rearms the latch.
    Increasing each calibrated coordinate means opening. This is an optional
    controller experiment, not an inferred contact force or a prescribed grasp.
    """
    q=np.asarray(achieved);near=np.asarray(proximity)
    if (q.ndim!=2 or len(q)<1 or q.shape[1]!=2 or near.shape!=(len(q),) or near.dtype!=np.bool_
            or not np.isfinite(q).all() or not math.isfinite(preload) or preload<0
            or not math.isfinite(release_delta) or release_delta<=0):
        raise ValueError('Expected finite two-finger coordinates, Boolean proximity and valid preload/release distances')
    if policy not in ('instant','latched'):raise ValueError('Unknown mesh preload policy')
    if policy=='instant':return q-preload*near[:,None],near.copy()
    active=np.zeros(len(q),dtype=bool);state='waiting';minimum=float(q[0].mean());maximum=minimum
    for i,value in enumerate(q):
        aperture=float(value.mean())
        if state=='holding':
            minimum=min(minimum,aperture)
            if aperture-minimum>=release_delta:
                state='opening';maximum=aperture
        elif state=='opening':
            maximum=max(maximum,aperture)
            if maximum-aperture>=release_delta:state='waiting'
        if state=='waiting' and near[i]:state='holding';minimum=aperture
        active[i]=state=='holding'
    return q-preload*active[:,None],active


def damping_compensation(desired_velocity, *, stiffness, damping):
    """Position-target offset cancelling a PD servo's desired-velocity damping.

    For force = kp*(target-q)-kv*qdot, offset=(kv/kp)*desired_velocity.
    This compensates neither acceleration nor contact load. Inputs are tensors;
    the caller retains its existing travel/force limits and contact-load policy.
    """
    velocity=torch.as_tensor(desired_velocity)
    kp=torch.as_tensor(stiffness,device=velocity.device,dtype=velocity.dtype)
    kv=torch.as_tensor(damping,device=velocity.device,dtype=velocity.dtype)
    if (not velocity.is_floating_point() or not bool(torch.isfinite(velocity).all())
            or not bool(torch.isfinite(kp).all()) or not bool(torch.isfinite(kv).all())
            or bool((kp<=0).any()) or bool((kv<0).any())):
        raise ValueError('Damping compensation needs finite velocity, positive stiffness and nonnegative damping')
    return velocity*(kv/kp)


def finger_command(achieved,*,mode='achieved',frame=1,recorded=None,
                   close_frame=None,preload_m=.006,lower=0.,upper=.04):
    if not bool(torch.isfinite(achieved).all()) or lower>=upper:
        raise ValueError('Invalid achieved finger positions or limits')
    if mode=='recorded':
        if recorded is None or recorded.shape!=achieved.shape or not bool(torch.isfinite(recorded).all()):
            raise ValueError('Recorded-command control requires finite source targets')
        return recorded.clone()  # Never alter the positive-control command.
    if mode=='achieved': return achieved.clamp(lower,upper)
    if mode!='preload' or close_frame is None or close_frame<1 or not math.isfinite(preload_m) or preload_m<0:
        raise ValueError('Preload mode requires an explicit closure frame and nonnegative finite preload')
    target=achieved-preload_m if frame>=close_frame else achieved
    return target.clamp(lower,upper)


def native_tracks_to_model(native_poses,native_to_template_R,native_to_template_t,canonical_shift):
    """Invert replay preparation's frame conversion for physical-unit metrics."""
    pose=torch.as_tensor(native_poses,dtype=torch.float64)
    mr=torch.as_tensor(native_to_template_R,dtype=torch.float64)
    mt=torch.as_tensor(native_to_template_t,dtype=torch.float64)
    shift=torch.as_tensor(canonical_shift,dtype=torch.float64)
    R=quat_wxyz_to_matrix(pose[...,3:]) @ mr.transpose(-1,-2)
    t=pose[...,:3]+shift-(R @ mt[...,None])[...,0]
    return R,t
