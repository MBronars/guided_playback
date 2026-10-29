"""Receding horizon: plan one window, commit `stride` frames, slide, re-plan.

Windows come from inference.window_bounds and inference.rollout_window, the
evaluator's contract (inference.windowed_rollout): the context of every window
after the first is the model's own committed output (particles and the carried
rigid-pose track, never ground truth), the actuator's committed poses are the
plan's, and the final window is shortened to the remaining horizon. No regenerated
overlap is spliced onto an older committed trajectory.
"""
from __future__ import annotations

import torch

from guidance.plan import actuated_bodies, apply_plan, integrate
from inference import rollout_window, window_bounds


def receding_horizon(model, diffusion, batch, planner, objective, *, horizon: int, stride: int,
                     future_len: int, steps: int, schedule: str = "pyramid", uncertainty_scale: float = 1.0,
                     generator=None, planner_kwargs: dict | None = None, verbose: bool = False):
    """Returns dict(positions [B, C+H, N, 3] committed, R/T tracks for rigid bodies,
    deltas [B, H, Ka, 6] the executed plan, windows: list of PlanResult, nfe)."""
    C = batch.context_len
    T = batch.positions.shape[1]
    H = min(horizon, T - C)
    if H < 1 or stride < 1 or future_len < 1:
        raise ValueError("horizon, stride and future_len must be positive")
    target_len = C + H
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    aidx = actuated_bodies(batch)
    a_cols = torch.tensor([int((ridx == b).nonzero()[0]) for b in aidx.tolist()], device=batch.device)
    committed = batch.positions[:, :C].clone()
    R_track = batch.body_poses[:, :C, ridx, :3, :3].float().clone()
    T_track = batch.body_poses[:, :C, ridx, :3, 3].float().clone()
    executed, windows, nfe = [], [], 0
    kwargs = dict(planner_kwargs or {})
    while committed.shape[1] < target_len:
        s, F_w, skip = window_bounds(committed.shape[1], target_len, future_len, C)
        win = rollout_window(batch, committed, s, F_w, rigid_idx=ridx, R_track=R_track, T_track=T_track)
        result = planner(model, diffusion, win, objective, steps=steps, schedule=schedule,
                         uncertainty_scale=uncertainty_scale, generator=generator, **kwargs)
        last = (s + F_w == target_len)
        keep = F_w - skip if last else min(stride, F_w - skip)
        pred_pos = result.pred["positions"].clone()
        R_new, T_new = result.pred["R"].clone(), result.pred["t"].clone()
        # actuated particle rows follow the plan exactly (template @ planned pose)
        planned = apply_plan(win, result.deltas).positions[:, C:]
        for e in range(batch.batch_size):
            rows = win.action_mask[e] & win.get_valid_mask()[e]
            pred_pos[e][:, rows] = planned[e][:, rows].to(pred_pos.dtype)
        # actuated bodies follow the plan exactly, not the model's own reading of it
        anchor = win.body_poses[:, C - 1, aidx]
        R_a, T_a = integrate(anchor[:, :, :3, :3].float(), anchor[:, :, :3, 3].float(), result.deltas.float())
        R_new[:, :, a_cols], T_new[:, :, a_cols] = R_a, T_a
        committed = torch.cat([committed, pred_pos[:, skip:skip + keep]], dim=1)
        R_track = torch.cat([R_track, R_new[:, skip:skip + keep]], dim=1)
        T_track = torch.cat([T_track, T_new[:, skip:skip + keep]], dim=1)
        executed.append(result.deltas[:, skip:skip + keep])
        windows.append(result)
        nfe += result.nfe
        if verbose:
            print(f"[horizon] window s={s} F={F_w} keep={keep} loss={result.loss:.5f} nfe={result.nfe}")
    return {"positions": committed, "R": R_track, "T": T_track, "rigid_idx": ridx,
            "deltas": torch.cat(executed, dim=1), "windows": windows, "nfe": nfe}
