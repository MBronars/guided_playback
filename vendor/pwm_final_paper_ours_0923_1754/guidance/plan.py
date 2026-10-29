"""Actuator plans: world-frame pose deltas of the actuated rigid bodies over the
window, applied to a batch as everything the backbones read about the actuator.

A plan is a tensor [B, F, Ka, 6] of [dt (m), rotation vector (rad)] per future
transition for the Ka actuated rigid bodies (ascending body index), the same
currency as `TrajectoryBatch.pose_actions`. `apply_plan` rewrites, differentiably:
`pose_actions` (the pose-token action skip), the actuated bodies' future
`body_poses` (integrated from the C-1 anchor), the actuated particles' future
`positions` and per-particle `actions` (template @ pose, transition differences),
and `pose_deltas_gt` rows of the actuated bodies. Passive bodies and deformable
particles are untouched.
"""
from __future__ import annotations

import dataclasses

import torch

from utils.pose_ops import so3_exp


def actuated_bodies(batch) -> torch.Tensor:
    """[Ka] body indices of actuated rigid bodies (element 0 is representative)."""
    rigid = batch.body_type[0] == 1
    act = batch.pose_action_mask[0] & rigid
    return act.nonzero(as_tuple=True)[0]


def plan_from_batch(batch) -> torch.Tensor:
    """The achieved actuator motion of the batch: the plan that reproduces it."""
    return batch.pose_actions[:, :, actuated_bodies(batch)].clone()


def to_normalized(batch, deltas: torch.Tensor) -> torch.Tensor:
    """Plan in the pose-delta stats' unit range (the space gradient planners step in)."""
    return batch.pose_delta_stats.normalize(deltas, clip=False)


def from_normalized(batch, z: torch.Tensor) -> torch.Tensor:
    return batch.pose_delta_stats.denormalize(z)


def body_rows(batch, body: int) -> torch.Tensor:
    """[B, P] particle rows of one body, per element (collated layouts differ)."""
    valid = batch.get_valid_mask()
    rows = [((batch.particle_body[e] == body) & valid[e]).nonzero(as_tuple=True)[0]
            for e in range(batch.batch_size)]
    n = rows[0].numel()
    assert all(r.numel() == n for r in rows), f"body {body}: row counts differ across the batch"
    return torch.stack(rows)


def integrate(anchor_R, anchor_t, deltas):
    """Compose world deltas from an anchor pose. anchor_R [B, K, 3, 3], anchor_t
    [B, K, 3], deltas [B, F, K, 6] -> (R [B, F, K, 3, 3], t [B, F, K, 3])."""
    B, F, K, _ = deltas.shape
    R, t = anchor_R, anchor_t
    Rs, ts = [], []
    for f in range(F):
        dR = so3_exp(deltas[:, f, :, 3:].reshape(-1, 3)).reshape(B, K, 3, 3)
        R = dR @ R
        t = t + deltas[:, f, :, :3]
        Rs.append(R)
        ts.append(t)
    return torch.stack(Rs, 1), torch.stack(ts, 1)


def apply_plan(batch, deltas: torch.Tensor):
    """Return a batch whose actuator inputs follow `deltas` [B, F, Ka, 6]."""
    if batch.joint_type is not None:
        raise NotImplementedError("Plans over articulated bodies (joint coordinates) are not implemented yet")
    aidx = actuated_bodies(batch)
    B, F, Ka, _ = deltas.shape
    C = batch.context_len
    assert Ka == aidx.numel(), f"plan has {Ka} bodies, batch has {aidx.numel()} actuated bodies"
    assert F == batch.num_future_frames, f"plan has {F} transitions, window has {batch.num_future_frames}"
    deltas = deltas.to(batch.positions.dtype)
    anchor = batch.body_poses[:, C - 1, aidx]
    R, t = integrate(anchor[:, :, :3, :3].to(deltas.dtype), anchor[:, :, :3, 3].to(deltas.dtype), deltas)

    pose_actions = batch.pose_actions.clone()
    pose_actions[:, :, aidx] = deltas
    pose_deltas_gt = batch.pose_deltas_gt.clone()
    pose_deltas_gt[:, :, aidx] = deltas
    body_poses = batch.body_poses.clone()
    body_poses[:, C:, aidx, :3, :3] = R.to(body_poses.dtype)
    body_poses[:, C:, aidx, :3, 3] = t.to(body_poses.dtype)
    positions = batch.positions.clone()
    actions = batch.actions.clone()
    b_idx = torch.arange(B, device=deltas.device)[:, None]
    for j, body in enumerate(aidx.tolist()):
        n_t = int(batch.body_template_mask[0, body].sum())
        tmpl = batch.body_templates[:, body, :n_t].to(deltas.dtype)                       # [B, P, 3]
        pts = torch.einsum("bfij,bpj->bfpi", R[:, :, j], tmpl) + t[:, :, j, None]        # [B, F, P, 3]
        prev0 = torch.einsum("bij,bpj->bpi", anchor[:, j, :3, :3].to(deltas.dtype), tmpl) + anchor[:, j, None, :3, 3].to(deltas.dtype)
        prev = torch.cat([prev0[:, None], pts[:, :-1]], dim=1)
        rows = body_rows(batch, body)                                                      # [B, P]
        positions[b_idx, C:, rows] = pts.transpose(1, 2).to(positions.dtype)
        actions[b_idx, :, rows] = (pts - prev).transpose(1, 2).to(actions.dtype)
    return dataclasses.replace(batch, positions=positions, actions=actions, body_poses=body_poses,
                               pose_actions=pose_actions, pose_deltas_gt=pose_deltas_gt)
