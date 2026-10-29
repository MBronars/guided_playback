"""Receding-horizon variant of clean_core.Staircase for closed-loop block pushing (push0418 checkpoints).

The first c transitions are EXECUTED (from the simulator), not generated: they enter as clean rows at level 0 with their
real poses, and only transitions c..n_tr-1 are denoised. Frame 0 stays the episode start, so the plan always ends at the
episode's last frame (receding horizon: only the remaining frames are predicted). full window: when fewer than
Cm1 + window rows remain, the window start slides back so every model call still reads Cm1 + window rows; the extra
rows are executed transitions at level 0. Used only when a request carries a history; open-loop runs keep clean_core."""
import numpy as np
import torch
from .clean_core import (Staircase, _ckpt, _stream_window, _stream_frame_positions, hide_actions, _integrate_and_write)


def _qmat(q):
    w, x, y, z = [q[..., i] for i in range(4)]
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def history_prefix(batch, obj_mask, hist, keep=None):
    """Executed frames 0..c (simulator, world frame: block_pos [c+1,3], block_quat [c+1,4] wxyz, gripper_xy [c+1,2]) as model
    rows. Poses are carried relative to frame 0, so the model-frame convention of the scenario batch (gp/push_scenes.py) holds:
    cube R_t = dR_t R0, T_t = dR_t (T0 - c0) + c0 + (p_t - p_0) with dR_t = R(q_t) R(q_0)^T; rod R_t = R0, T_t = T0 + (g_t - g_0, 0).
    Returns dict(P [B, c, Kr, 6] normalized pose deltas, R/T/pos lists of c+1 frames)."""
    from utils.pose_ops import so3_log
    dev = batch.device; B = batch.batch_size
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]; act = batch.pose_action_mask[0, ridx]
    R0 = batch.body_poses[:, 0, ridx, :3, :3].float(); T0 = batch.body_poses[:, 0, ridx, :3, 3].float()
    w = obj_mask.float()[..., None]; c0 = (batch.positions[:, 0].float() * w).sum(1) / w.sum(1)          # [B, 3] cube centroid, frame 0
    bp = torch.as_tensor(hist["block_pos"], dtype=torch.float32, device=dev); bq = torch.as_tensor(hist["block_quat"], dtype=torch.float32, device=dev)
    g = torch.as_tensor(hist["gripper_xy"], dtype=torch.float32, device=dev); n = bp.shape[0]
    Rq = _qmat(bq); dR = Rq @ Rq[0].transpose(-1, -2)                                                  # [n, 3, 3]
    R, T = [], []
    for t in range(n):
        Rt, Tt = R0.clone(), T0.clone()
        for k in range(len(ridx)):
            if bool(act[k]):                                                                          # rod: translation only
                Tt[:, k] = T0[:, k] + torch.cat([g[t] - g[0], torch.zeros(1, device=dev)])
            else:                                                                                     # cube
                Rt[:, k] = dR[t] @ R0[:, k]
                Tt[:, k] = (dR[t] @ (T0[:, k] - c0).T).T + c0 + (bp[t] - bp[0])
        R.append(Rt); T.append(Tt)
    pos = [batch.positions[:, 0]]
    zero = torch.zeros_like(batch.positions[:, 0])
    for t in range(1, n):
        pos.append(_stream_frame_positions(batch, ridx, R[t], T[t], pos[-1], zero))
    rows = []
    for t in range(n - 1):
        dt = T[t + 1] - T[t]; dRr = R[t + 1] @ R[t].transpose(-1, -2)
        aa = so3_log(dRr.reshape(-1, 3, 3)).reshape(B, len(ridx), 3)
        rows.append(torch.cat([dt, aa], -1))
    P = batch.pose_delta_stats.normalize(torch.stack(rows, 1), clip=False).float() if rows else None
    c = n - 1; k = c if keep is None else min(int(keep), c)
    if k < c:                                            # keep only the last k executed transitions; the anchor becomes executed frame c - k
        R, T, pos = R[c - k:], T[c - k:], pos[c - k:]; P = P[:, c - k:] if k > 0 else None
    return dict(P=P, R=R, T=T, pos=pos, c=k, anchor_frame=c - k, executed=c)


class PrefixStaircase(Staircase):
    def __init__(self, model, diffusion, cfg, batch, steps, horizon, prefix, seed=0, amp_dtype=None, full_window=True, stop_at=None):
        """stop_at: commit only transitions [0, stop_at) of the `horizon` scheduled ones. With horizon >= stop_at + window the
        rows past stop_at keep entering at pure noise, so EVERY model call reads Cm1 + window rows (no shrinking tail),
        and the rollout returns frames 0..stop_at; the in-flight rows past stop_at are discarded."""
        self.prefix, self.c, self.full_window = prefix, int(prefix["c"]), bool(full_window)
        self.stop_at = None if stop_at is None else int(stop_at)
        super().__init__(model, diffusion, cfg, batch, steps, horizon=horizon, seed=seed, amp_dtype=amp_dtype)
        dev = batch.device; levels, call = [], 0
        j = torch.arange(self.n_tr, dtype=torch.float64)
        while True:                                                     # new rows start their staircase at transition c
            lv = (self.S + (j - self.c) * self.u - call).clamp(0.0, float(self.S)); lv[:self.c] = 0.0
            levels.append(lv)
            if bool((lv[:self.stop_at] <= 0).all()):
                break
            call += 1
        self.L = torch.stack(levels).to(dev); self.n_calls = self.L.shape[0] - 1
        t_next = self.L[1:] / self.S; stepped = (self.L[:-1] - self.L[1:]) > 0
        self.used = stepped & (t_next > 0); self.t_next = t_next

    def rollout(self, X0, P0_, E_, mode="churn", timers=None):
        torch.manual_seed(self.seed); torch.cuda.manual_seed(self.seed)
        batch, B, S, top, T_train, Cm1, c = self.batch, self.batch.batch_size, self.S, self.top, self.T_train, self.Cm1, self.c
        use_ckpt = torch.is_grad_enabled(); ridx, dev = self.ridx, batch.device; valid = batch.get_valid_mask()
        pre = self.prefix
        Ps = [pre["P"][:, j] for j in range(c)] + [P0_[:, j] for j in range(c, self.n_tr)]
        R, Tt, pos = list(pre["R"]), list(pre["T"]), list(pre["pos"])
        committed, k = c, 0; self.rows_seen = []; stop = self.n_tr if self.stop_at is None else self.stop_at
        last_rs = c; self.resample_log = []
        while committed < stop:
            lv, lvn = self.L[k], self.L[k + 1]
            s = max(0, committed - Cm1)
            if self.full_window and self.n_tr - s < Cm1 + self.window:
                s = max(0, self.n_tr - (Cm1 + self.window))
            nr = min(Cm1 + self.window, self.n_tr - s); self.rows_seen.append(int(nr))
            win = _stream_window(batch, pos[s], R[s], Tt[s], ridx, s, nr)
            if self.hide:
                win = hide_actions(win, 1.0)
            tb = (lv[s:s + nr] / S * top).float().unsqueeze(0).expand(B, -1)
            xc = torch.stack([X0[:, q] for q in range(s, s + Cm1)], 1)
            xf = torch.stack([X0[:, q] for q in range(s + Cm1, s + nr)], 1)

            def call_model(pc, pf, _win=win, _tb=tb, _xc=xc, _xf=xf):
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                    part_f, pose_f = self.model(_win, _xf, _tb[:, Cm1:], noisy_context=_xc, context_timesteps=_tb[:, :Cm1],
                                                noisy_pose=pf, noisy_ctx_pose=pc)
                    out = torch.cat([self.base._last_ctx_pose_pred, pose_f], 1)
                assert part_f is None
                return out.float()

            pc = torch.stack(Ps[s:s + Cm1], 1); pf = torch.stack(Ps[s + Cm1:s + nr], 1)
            pred = _ckpt(call_model, pc, pf, use_reentrant=False) if use_ckpt else call_model(pc, pf)
            p_rows = torch.cat([pc, pf], 1)
            dt = ((lv[s:s + nr] - lvn[s:s + nr]) / S * (top / T_train)).float()
            tn = (lvn[s:s + nr] / S * (top / T_train)).float()
            if mode == "euler":
                v = self.diffusion.as_velocity(pred, p_rows, tb)
            else:
                x0 = self.diffusion.get_x0_est(pred, tb, p_rows)
            for i in range(nr):
                if float(dt[i]) > 0:
                    j = s + i
                    Ps[j] = Ps[j] - dt[i] * v[:, i] if mode == "euler" else (1 - tn[i]) * x0[:, i] + tn[i] * E_[k, j]
            k += 1
            while committed < stop and float(lvn[committed]) <= 0.0:
                j = committed
                raw = batch.pose_delta_stats.denormalize(torch.stack(Ps[s:j + 1], 1).float()).float()
                dummy = torch.zeros((B, j + 1 - s, batch.positions.shape[2], 3), device=dev)
                _, R_all, T_all = _integrate_and_write(win, ridx, raw, R[s], Tt[s], dummy, valid)
                R.append(R_all[:, -1]); Tt.append(T_all[:, -1])
                pos.append(_stream_frame_positions(batch, ridx, R_all[:, -1], T_all[:, -1], pos[j], X0[:, j]))
                committed += 1
            rs = getattr(self, "resample", None)
            if rs and mode != "euler" and not torch.is_grad_enabled() and committed < stop and committed // rs["every"] > last_rs // rs["every"]:
                # population reseeding at a commit point: score every member by the model's clean lookahead (object centroid at the
                # furthest row of this window, from the x0 estimates of the in-flight rows), draw the next population with softmax
                # weights (member 0 = the current best, always kept), and copy the parents' committed frames and in-flight rows.
                # Rows that have not entered the window and every later injected draw stay the slot's own, so copies diverge.
                last_rs = committed; e = s + nr; el = min(e, stop)                       # score no further than the goal frame
                seq = [Ps[q] if q < committed else x0[:, q - s] for q in range(s, el)]
                raw = batch.pose_delta_stats.denormalize(torch.stack(seq, 1).float()).float()
                dummy = torch.zeros((B, el - s, batch.positions.shape[2], 3), device=dev)
                _, R_la, T_la = _integrate_and_write(win, ridx, raw, R[s], Tt[s], dummy, valid)
                p_la = _stream_frame_positions(batch, ridx, R_la[:, -1], T_la[:, -1], pos[committed], X0[:, el - 1])
                m = rs["mask"].float()
                cen = (p_la * m[..., None]).sum(1) / m.sum(1, keepdim=True)
                d = (cen - rs["target"]).norm(dim=-1)                                    # [B] lookahead distance to the target
                w = torch.softmax(-(d - d.min()) / (rs["temp"] * d.std().clamp_min(1e-6)), 0)
                g = torch.Generator(device=dev).manual_seed(int(self.seed) * 7919 + committed)
                idx = torch.multinomial(w, B, replacement=True, generator=g); idx[0] = d.argmin()
                Ps = [pq[idx] if q < e else pq for q, pq in enumerate(Ps)]
                R = [r[idx] for r in R]; Tt = [t[idx] for t in Tt]; pos = [p_[idx] for p_ in pos]
                self.resample_log.append(dict(committed=int(committed), parents=int(idx.unique().numel()), d_min=round(float(d.min()) * 1e3, 1), d_mean=round(float(d.mean()) * 1e3, 1)))
        if self.resample_log:
            print(f"[reseed] {len(self.resample_log)} reseeds; distinct parents kept per reseed {[x['parents'] for x in self.resample_log]} of {B}; lookahead best mm {[x['d_min'] for x in self.resample_log]}", flush=True)
        return torch.stack(pos, 1)
