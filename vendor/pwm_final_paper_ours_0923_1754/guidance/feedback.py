"""Construct coherent, future-free rigid observations for replanning.

This refreshes model inputs only. The caller owns the observation source, task
goal, execution policy and simulator state; calling it is not a closed-loop test.
"""
from dataclasses import replace

import torch


@torch.no_grad()
def rigid_planning_batch(batch, *, horizon):
    """Resize an unconditional planning horizon using only the observed context.

    Unlike feedback from external poses, this preserves the input particle
    observations bitwise. Future slots are placeholders, never ground truth.
    All static metadata and articulation retain the original fixture convention.
    """
    if batch.body_poses is None:
        raise ValueError('Rigid planning requires observed body poses')
    C = batch.context_len
    rebuilt = rigid_feedback_batch(batch, batch.body_poses[:, :C, :, :3, :3],
                                   batch.body_poses[:, :C, :, :3, 3], horizon=horizon)
    context = batch.positions[:, :C].clone()
    if not bool(torch.isfinite(context).all()):
        raise ValueError('Planning context contains nonfinite particle observations')
    return replace(rebuilt, positions=torch.cat([
        context, context[:, -1:].expand(-1, horizon, -1, -1)], dim=1))


@torch.no_grad()
def rigid_feedback_batch(batch, rotations, translations, *, horizon):
    """Use the latest C observed poses, in body-index order, and static templates.

    Inputs are [1,L,K,3,3] rotations and [1,L,K,3] translations in model physical
    coordinates, L>=C. Future slots repeat the last observation only to satisfy
    the sampler's tensor contract; all future action/target deltas are zero.
    No recorded future position, pose, action or target is copied into the result.
    Supported scope matches unconditional noise optimization: one all-rigid scene.
    """
    if (batch.batch_size != 1 or batch.body_type is None
            or not bool((batch.body_type == 1).all()) or batch.action_active is not None):
        raise ValueError('Rigid feedback requires one entirely rigid scene without temporal particle actions')
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError('Feedback horizon must be a positive integer')
    C, K, N = batch.context_len, batch.body_type.shape[1], batch.positions.shape[2]
    R = torch.as_tensor(rotations, device=batch.device, dtype=torch.float32).detach()
    t = torch.as_tensor(translations, device=batch.device, dtype=torch.float32).detach()
    if (R.ndim != 5 or R.shape[0] != 1 or R.shape[1] < C
            or R.shape[2:] != (K, 3, 3) or t.shape != R.shape[:-1]):
        raise ValueError('Feedback poses must cover the context and every body in body-index order')
    R, t = R[:, -C:].clone(), t[:, -C:].clone()
    if not bool(torch.isfinite(R).all() and torch.isfinite(t).all()):
        raise ValueError('Feedback context contains nonfinite poses')
    with torch.autocast(device_type=batch.device.type, enabled=False):
        if (float((R.transpose(-1, -2) @ R - torch.eye(3, device=batch.device)).abs().max()) > 2e-4
                or float((torch.linalg.det(R) - 1).abs().max()) > 2e-4):
            raise ValueError('Feedback rotations must be proper rotation matrices')
        poses = torch.eye(4, device=batch.device, dtype=R.dtype).expand(1, C+horizon, K, 4, 4).clone()
        poses[:, :C, :, :3, :3], poses[:, :C, :, :3, 3] = R, t
        poses[:, C:] = poses[:, C-1:C]
        context = torch.zeros(1, C, N, 3, device=batch.device, dtype=R.dtype)
        valid = batch.get_valid_mask()[0]
        if bool(((batch.particle_body[0, valid] < 0) | (batch.particle_body[0, valid] >= K)).any()):
            raise ValueError('A valid feedback particle has no rigid body')
        for k in range(K):
            rows = valid & (batch.particle_body[0] == k)
            template = batch.body_templates[0, k, batch.body_template_mask[0, k]].float()
            if int(rows.sum()) != len(template):
                raise ValueError('Rigid feedback templates do not match their particle populations')
            context[:, :, rows] = torch.einsum('bcij,nj->bcni', R[:, :, k], template) + t[:, :, k, None]
        positions = torch.cat([context, context[:, -1:].expand(-1, horizon, -1, -1)], dim=1)
    pose_zeros = torch.zeros(1, horizon, K, 6, device=batch.device, dtype=R.dtype)
    return replace(batch, positions=positions, body_poses=poses,
                   actions=torch.zeros(1, horizon, N, 3, device=batch.device, dtype=R.dtype),
                   pose_actions=pose_zeros.clone(), pose_deltas_gt=pose_zeros)


@torch.no_grad()
def reanchor_finger_calibration(batch, calibration, *, hand_body=1, fingers=(2, 3)):
    """Keep the absolute joint coordinate convention, updating its observed anchor.

    Without rebasing q_anchor, the first predicted finger velocity would be
    measured from the initial episode's opening rather than the latest opening.
    This helper is for the calibrated prismatic fingers used by pickup objectives.
    """
    if batch.batch_size != 1:
        raise ValueError('Finger calibration reanchoring requires one scene')
    c = {k: v.to(batch.positions).clone() if torch.is_tensor(v) else v
         for k, v in calibration.items()}
    with torch.autocast(device_type=batch.device.type, enabled=False):
        pose = batch.body_poses[0, batch.context_len-1].float()
        local = (pose[hand_body, :3, :3].T @
                 (pose[list(fingers), :3, 3] - pose[hand_body, :3, 3]).T).T
        c['q_anchor'] = c['q_anchor'] + ((local-c['finger_anchor'])*c['axis']).sum(-1)
        c['finger_anchor'] = local
    return c
