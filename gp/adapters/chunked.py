"""Chunked (block-autoregressive) rollout for pwm_final_paper checkpoints: NOT the streaming staircase.

Window w covers rows [start, start + 19): its first C-1 = 3 rows are CONTEXT, already known and fed clean (at t = 1, the
convention for committed rows), and its 16 future rows are denoised TOGETHER from level 1 to 0 in `steps` calls (one shared
level per call). The next window starts 16 rows later, so its context is the last 3 rows the previous window generated.
The last window is shifted back to end at N; rows it covers that are already generated are fed clean too.

Context of the first window = the RECORDED first 3 transitions of the episode (pose deltas of the recorded body poses,
normalized with the checkpoint's pose_delta_stats), i.e. generation starts after recorded frame 3.
Churn (eta = 1): row j re-formed at the i-th call of its window as x = x0 + l_i (E[:, j, i] - x0); E[:, j, 0] = initial draw."""
import torch
from torch.utils.checkpoint import checkpoint
from .paper_core import pose_channel_mask, MEMLOG


def recorded_context_rows(streamer, batch):
    """[B, C-1, Kr, 6] normalized pose deltas of the recorded transitions 0 .. C-2 (rigid bodies)."""
    import sys
    from utils.pose_native import pose_sequence_deltas, joint_layout
    C = streamer.C
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    p = batch.body_poses[:, :C].float()
    layout = joint_layout(batch, torch.arange(p.shape[2], device=p.device))
    d = pose_sequence_deltas(p, layout)[:, :, ridx]                                   # [B, C-1, Kr, 6] raw
    return batch.pose_delta_stats.normalize(d, clip=False).float()


def window_starts(N, NW, C):
    starts, committed = [], C - 1
    while committed < N:
        s = min(committed - (C - 1), N - NW); starts.append(s); committed = s + NW
    return starts


def rollout_chunked(S, batch, X, XP, *, steps, ctx_rows, grad=False, fps_seeds=None, churn=None):
    C, NW, dev = S.C, S.NW, S.dev
    B, N = X.shape[:2]
    if churn is not None:
        XP = churn[:, :, 0]
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    pmask = pose_channel_mask(batch, ridx)
    pm = (lambda v: v * pmask.to(v)) if pmask is not None else (lambda v: v)
    rows, prows = list(X.unbind(1)), [pm(v) for v in XP.unbind(1)]
    gen0 = bool(getattr(S, "act_cmd", False))                                         # act_cmd: frame 0 only; the first window generates all its rows
    if not gen0:
        for j in range(C - 1):                                                        # recorded context, clean
            prows[j] = pm(ctx_rows[:, j].to(XP.dtype)); rows[j] = torch.zeros_like(rows[j])
    seeds = None if fps_seeds is None else tuple(int(v) for v in fps_seeds)
    done = [j < C - 1 and not gen0 for j in range(N)]
    cl = float(getattr(S, "ctx_level", 0.0) or 0.0)                                   # kept rows' honest level (0 = clean)
    pos_a, bp_a, a_idx, k = batch.positions[:, 0], batch.body_poses[:, 0], 0, 0
    for start in window_starts(N, NW, C):
        while a_idx < start:
            pos_a, bp_a = S._step_anchor(batch, pos_a, bp_a, rows[a_idx], prows[a_idx]); a_idx += 1
        pos_w = pos_a[:, None].expand(-1, NW + 1, -1, -1); bp_w = bp_a[:, None].expand(-1, NW + 1, -1, -1, -1)
        todo = [j for j in range(start, start + NW) if not done[j]]
        kept = [j - start for j in range(start, start + NW) if done[j]]
        if cl > 0 and kept:                                                           # one draw per window, shared by every member (candidates differ only in actions)
            gk = torch.Generator(device=dev).manual_seed(7919 * (start + 1))
            ek = torch.randn(1, len(kept), *rows[0].shape[1:], device=dev, generator=gk)
            ekp = torch.randn(1, len(kept), *prows[0].shape[1:], device=dev, generator=gk)
        for i in range(steps):
            l0, l1 = 1.0 - i / steps, 1.0 - (i + 1) / steps
            lv = torch.tensor([l0 if j in todo else cl for j in range(start, start + NW)], device=dev)
            tt = torch.clamp(torch.round(lv * S.T), 1, S.T - 1).long()[None].expand(B, -1)
            xs, xps = torch.stack(rows[start:start + NW], 1), torch.stack(prows[start:start + NW], 1)
            if cl > 0 and kept:                                                       # x_t = x0 + t (eps - x0) on the kept rows
                xs, xps = xs.clone(), xps.clone()
                xs[:, kept] = xs[:, kept] + cl * (ek.to(xs.dtype) - xs[:, kept])
                xps[:, kept] = pm(xps[:, kept] + cl * (ekp.to(xps.dtype) - xps[:, kept]))
            args = (batch, xs[:, :C - 1], xs[:, C - 1:], xps[:, :C - 1], xps[:, C - 1:], tt, pos_w, bp_w, start, k, seeds)
            x0, x0p = checkpoint(S._call, *args, use_reentrant=False) if grad else S._call(*args)
            k += 1
            for j in todo:
                jj = j - start
                if l1 <= 0:
                    rows[j], prows[j] = x0[:, jj], pm(x0p[:, jj])
                else:
                    rows[j] = x0[:, jj] + (l1 / l0) * (rows[j] - x0[:, jj])
                    prows[j] = pm(x0p[:, jj] + l1 * (pm(churn[:, j, i + 1]) - x0p[:, jj])) if churn is not None else pm(x0p[:, jj] + (l1 / l0) * (prows[j] - x0p[:, jj]))
        for j in todo:
            done[j] = True
        if MEMLOG:
            print(f"[mem] window {start} allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    return S._materialize(batch, rows, prows)[0], k
