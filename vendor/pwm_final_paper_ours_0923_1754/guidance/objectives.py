"""Objectives on a sampled window, in physical units (metres, radians).

Every objective is `fn(pred, batch) -> (loss, diagnostics)` where `pred` is the
dict from `materialize` (positions [B, F, N, 3] in metres; for pose-native
models also R [B, F, Kr, 3, 3], t [B, F, Kr, 3] and rigid_idx). Losses are
batch means so planners can compare them across candidates; diagnostics are
plain floats in millimetres and degrees.
"""
from __future__ import annotations

import math
import torch

from utils.pose_native import materialize_pose_native
from utils.pose_ops import rotation_geodesic_error


def one_window(batch):
    """The batch as ONE window of C-1+F transitions from frame 0: what a context-generating
    checkpoint's sampler returns (guidance.sampler.sample_window), anchored at the recorded
    frame-0 poses. The per-particle action stream is padded with zeros over the context rows;
    the rigid pose stream (pose_actions / pose_action_mask) is untouched."""
    import dataclasses
    C1 = batch.context_len - 1
    if C1 == 0:
        return batch
    ctx_act = torch.zeros_like(batch.actions[:, :1]).expand(-1, C1, -1, -1)
    kw = dict(context_len=1, actions=torch.cat([ctx_act, batch.actions], 1))
    if batch.action_active is not None:
        kw["action_active"] = torch.cat([batch.context_action_active, batch.action_active], 1)
        kw["context_action_active"] = batch.context_action_active[:, :0]
    return dataclasses.replace(batch, **kw)


def materialize(batch, x0_particles, x0_pose):
    """Normalized sampler output -> physical positions (and rigid poses).

    An output that covers C-1+F rows (a context-generating checkpoint) is integrated from
    frame 0 through one_window(batch); `positions` / `t` then hold every generated frame."""
    rows = (x0_pose if x0_pose is not None else x0_particles).shape[1]
    if rows == batch.context_len - 1 + batch.num_future_frames and batch.context_len > 1:
        batch = one_window(batch)
    if x0_pose is not None:
        # Plans can become the context of another window, just as in production
        # rollout. Keep both paths on the same stable rotation representation.
        pos, R, t, ridx = materialize_pose_native(batch, x0_particles, x0_pose, stabilize_joints=True)
        return {"positions": pos, "R": R, "t": t, "rigid_idx": ridx}
    pos = batch.predictions_to_positions(batch.prediction_stats.denormalize(x0_particles))
    return {"positions": pos}


def _rigid_column(pred, body: int) -> int:
    where = (pred["rigid_idx"] == body).nonzero(as_tuple=True)[0]
    assert where.numel() == 1, f"body {body} is not a rigid body of this batch"
    return int(where[0])


def body_target(body: int, target_t: torch.Tensor, target_R: torch.Tensor | None = None,
                frame: int = -1, rotation_weight: float = 0.0, goal_weight: float = 1.0):
    """Translation (and optional rotation) of one rigid body at `frame` against a
    target. target_t [B, 3] metres; target_R [B, 3, 3]. Loss = mean distance in
    metres * goal_weight (+ rotation_weight * geodesic radians)."""
    if not math.isfinite(goal_weight) or goal_weight < 0:
        raise ValueError('Goal weight must be finite and nonnegative')
    def fn(pred, batch):
        j = _rigid_column(pred, body)
        dist = (pred["t"][:, frame, j] - target_t.to(pred["t"])).norm(dim=-1)        # [B]
        loss = goal_weight * dist.mean()
        diag = {"target_distance_mm": float(dist.detach().mean()) * 1000.0}
        if target_R is not None:
            ang = rotation_geodesic_error(pred["R"][:, frame, j], target_R.to(pred["R"]))  # [B] radians
            loss = loss + rotation_weight * ang.mean()
            diag["target_rotation_deg"] = float(ang.detach().mean()) * 180.0 / torch.pi
        return loss, diag
    return fn


def particle_target(target: torch.Tensor, mask: torch.Tensor | None = None, frame: int = -1):
    """Per-particle distance at `frame` to `target` [B, N, 3] over `mask` [B, N]
    (default: all valid particles). Loss = mean distance in metres."""
    def fn(pred, batch):
        sel = batch.get_valid_mask() if mask is None else (mask.to(batch.device) & batch.get_valid_mask())
        d = (pred["positions"][:, frame] - target.to(pred["positions"])).norm(dim=-1)   # [B, N]
        loss = d[sel].mean()
        return loss, {"particle_target_mm": float(loss.detach()) * 1000.0}
    return fn


def phantom_motion(passive_body: int, contact_radius: float):
    """Motion of a passive rigid body in transitions where no actuated particle is
    within `contact_radius` (metres) of it at the transition start. Loss = mean
    unexplained displacement in metres per transition."""
    def fn(pred, batch):
        j = _rigid_column(pred, passive_body)
        C = batch.context_len
        pos = pred["positions"]
        valid = batch.get_valid_mask()
        act = batch.action_mask & valid
        body = (batch.particle_body == passive_body) & valid
        starts = torch.cat([batch.positions[:, C - 1:C], pos[:, :-1]], dim=1)          # [B, F, N, 3]
        B, F = starts.shape[:2]
        gaps = []
        for e in range(B):
            a, b = starts[e][:, act[e]], starts[e][:, body[e]]                        # [F, Na, 3], [F, Nb, 3]
            gaps.append(torch.cdist(a, b).amin(dim=(1, 2)))                           # [F]
        gap = torch.stack(gaps)                                                         # [B, F]
        prev_t = torch.cat([batch.body_poses[:, C - 1, passive_body, None, :3, 3], pred["t"][:, :-1, j]], dim=1)
        moved = (pred["t"][:, :, j] - prev_t).norm(dim=-1)                              # [B, F]
        free = (gap > contact_radius).to(moved)
        loss = (moved * free).sum(dim=1).mean()
        return loss, {"phantom_mm": float(loss.detach()) * 1000.0, "free_transitions": float(free.mean())}
    return fn


def floor_penetration(floor_z: float = 0.0):
    """Mean depth below the floor over valid particles and frames, metres."""
    def fn(pred, batch):
        depth = (floor_z - pred["positions"][..., 2]).clamp(min=0.0)                    # [B, F, N]
        sel = batch.get_valid_mask()[:, None].expand_as(depth)
        loss = depth[sel].mean()
        return loss, {"penetration_mm": float(loss.detach()) * 1000.0}
    return fn


def box_clearance(boxes, clearance: float = 0.005, bodies=None, frames=slice(None)):
    """Keep every particle (of `bodies`, default all valid rows) at least `clearance` metres
    outside each axis-aligned box. boxes: list of (centre [3], size [3]) in world metres,
    size = full extent. Loss = mean over particles and frames of the shortfall below the
    clearance (metres) + the worst shortfall. Geometry-free: a sampled-point probe on the
    checkpoint's own particles, not a convex-surface distance (guidance/obstacles.py)."""
    spec = [(torch.as_tensor(c, dtype=torch.float32), torch.as_tensor(s, dtype=torch.float32) / 2) for c, s in boxes]
    def fn(pred, batch):
        pos = pred["positions"][:, frames]                                                  # [B, F', N, 3]
        sel = batch.get_valid_mask()
        if bodies is not None:
            sel = sel & torch.isin(batch.particle_body, torch.as_tensor(bodies, device=sel.device))
        sel = sel[:, None].expand(pos.shape[:3])
        worst = pos.new_zeros(())
        total = pos.new_zeros(())
        for centre, half in spec:
            q = (pos - centre.to(pos)).abs() - half.to(pos)                                  # [B, F', N, 3]
            outside = q.clamp(min=0).norm(dim=-1)
            inside = q.amax(dim=-1).clamp(max=0)                                             # <= 0 inside the box
            signed = outside + inside                                                        # exact box SDF
            short = (clearance - signed).clamp(min=0)[sel]
            total = total + short.mean()
            worst = torch.maximum(worst, short.max() if short.numel() else worst)
        loss = total + worst
        return loss, {"box_shortfall_mm": float(total.detach()) * 1000.0, "box_worst_mm": float(worst.detach()) * 1000.0}
    return fn


def weighted(*terms):
    """weighted((w1, fn1), (w2, fn2), ...) -> one objective; diagnostics merge."""
    def fn(pred, batch):
        total, diag = 0.0, {}
        for w, term in terms:
            loss, d = term(pred, batch)
            total = total + w * loss
            diag.update(d)
        diag["objective"] = float(total.detach()) if torch.is_tensor(total) else float(total)
        return total, diag
    return fn
