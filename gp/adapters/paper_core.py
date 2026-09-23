"""Streaming rollout core for a context-predicting pose-native checkpoint, PICK/PLACE edition (shared by server.py).

Adapted 2026-09-22 from /data/user_data/mbronars/reports/mould_server/stream_core.py (moulding: dough particles +
paddle). Differences: (1) entirely RIGID scenes (pickup: hand + two prismatic fingers + one object, 1,183 points): the
UNet skips its particle decode there (particle output and context particle stash are None), so those rows are zeros --
materialization rebuilds every rigid row from its integrated pose, so they are never read; (2) episodes come from ONE
domain of a multi-domain checkpoint (`domain`), the whole clip from its first frame, padded by repeating the last
recorded frame so that N transitions fit the loader's window contract (the streamer reads only frame 0 of it).

One rollout = N transitions from frame 0. Schedule = diffusion/stream_states.window_states(g=0, c=keep, S, N, n=19),
the reference sampler the training 'stream' draws were enumerated from: every transition enters at pure noise, the
19-row window keeps `keep` committed rows at its front and slides (back at the end), a row is committed when its
level reaches 0 with the model's x0 as its content, and committed rows are fed at t = 1. Frame 0 is the only given
state; every action is hidden; the window's anchor frame and poses come from the rollout's own rows.

Precision/speed: every model call runs under bf16 autocast; the model can be regionally compiled (compile_model).
With grad=True (guidance) each model call is activation-checkpointed, so backprop through the whole chain stores
only the per-call inputs.
"""
import dataclasses, os, sys
import torch
from torch.utils.checkpoint import checkpoint

REPO = "/data/user_data/mbronars/packages/pwm_final_paper"
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from omegaconf import OmegaConf
from inference import load_model_and_diffusion, rollout_window
from utils.pose_native import materialize_pose_native, pose_channel_mask
from utils.conditioning import hide_actions
from data.dataset import create_dataloader
from diffusion.stream_states import window_states

MEMLOG = bool(int(os.environ.get("PWM_MEMLOG", "0")))            # per-call memory trace (debug)


class FixedFPS:
    """Opt-in (fix_fps): per-MEMBER farthest-point-sampling start points.

    The checkpoint's UNet resamples its FPS start point every forward (unet_fps_random_start) from the global CPU RNG
    via pytorch3d.sample_farthest_points. When active, that call is routed through `_fps`, which draws each batch
    element's start from its OWN generator seeded by (member seed, model-call index k): a member sees the same draws
    on every guidance iteration and in every window of the rollout, members differ from each other, and the backward
    recompute of a checkpointed call (same k) redraws them identically. Off: the model's own behaviour, unchanged."""
    def __init__(self):
        self.seeds, self.k, self.n = None, 0, 0

    def begin(self, k):                                   # called at the top of every model call
        self.k, self.n = k, 0

    def _fps(self, points, lengths=None, K=50, random_start_point=False):
        from pytorch3d import _C
        from pytorch3d.ops.utils import masked_gather
        if self.seeds is None or not random_start_point:
            return _ORIG_FPS(points, lengths=lengths, K=K, random_start_point=random_start_point)
        N, P, _ = points.shape
        dev = points.device
        lengths = torch.full((N,), P, dtype=torch.int64, device=dev) if lengths is None else lengths.long()
        K = (torch.full((N,), K, dtype=torch.int64, device=dev) if isinstance(K, int) else
             torch.as_tensor(K, dtype=torch.int64, device=dev))
        B = len(self.seeds); per = N // B                  # rows are (member, object) in member-major order
        starts = torch.zeros(N, dtype=torch.int64)
        for e in range(B):
            g = torch.Generator().manual_seed(int(self.seeds[e]) * 1_000_003 + self.k * 7919 + self.n)
            for o in range(per):
                starts[e * per + o] = torch.randint(int(lengths[e * per + o]), (1,), generator=g).item()
        self.n += 1
        idx = _C.sample_farthest_points(points, lengths, K, starts.to(dev))
        return masked_gather(points, idx), idx


FPS = FixedFPS()
import models.unet_std as _unet_std
_ORIG_FPS = _unet_std.sample_farthest_points
_unet_std.sample_farthest_points = FPS._fps                 # a pass-through unless FPS.seeds is set


def load(ckpt, device):
    cwd = os.getcwd(); os.chdir(REPO)                       # relative paths in the checkpoint config (stream table)
    try:
        model, diffusion, cfg, _ = load_model_and_diffusion(ckpt, device)
    finally:
        os.chdir(cwd)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, diffusion, cfg


def compile_model(model, regional=True, l0=True):
    """utils/compilation.prepare_compilation: regional, static shapes. l0=True also compiles the L0 units the training
    default leaves eager (ragged across mixed scenes; every mould scene is 256 + 1024 points). Measured per model call
    at B=4 (A100, bf16): forward 126.7 -> 100.6 ms, checkpointed fwd+bwd 455.4 -> 396.7 ms; compile ~75 s once."""
    from utils.compilation import prepare_compilation
    return prepare_compilation(model, {"compile": True, "compile_regional": regional, "compile_dynamic": False,
                                       "compile_l0": l0})


def pad_frames(b, transitions):
    """Extend a batch to N+1 frames by repeating its last recorded frame (positions, body poses; per-transition
    streams by zeros). The streamer reads only frame 0 and the shapes; `gt` saved with a result is the padded clip
    and `gt_frames` says how many are recorded."""
    T = b.positions.shape[1]
    if T >= transitions + 1:
        return b, T
    extra = transitions + 1 - T
    rep = lambda v: torch.cat([v, v[:, -1:].expand(-1, extra, *([-1] * (v.dim() - 2)))], 1)
    zero = lambda v: torch.cat([v, torch.zeros_like(v[:, -1:]).expand(-1, extra, *([-1] * (v.dim() - 2)))], 1)
    kw = dict(positions=rep(b.positions), body_poses=rep(b.body_poses), actions=zero(b.actions))
    for name in ("pose_deltas_gt", "pose_actions", "action_active"):
        v = getattr(b, name)
        if v is not None:
            kw[name] = zero(v)
    return dataclasses.replace(b, **kw), T


def episode_batch(cfg, indices, transitions, device, domain="pickup"):
    """Held-out test episodes of `domain` (whole clip from its first frame, padded to N+1 frames) as B=1 batches."""
    cwd = os.getcwd(); os.chdir(REPO)
    try:
        loader = create_dataloader(
            wds_location=cfg.data.wds_location, format=cfg.data.format, normalization=cfg.normalization,
            prediction_target=cfg.prediction_target, context_len=cfg.data.context_len,
            future_len=cfg.data.future_len, batch_size=1, num_workers=0,
            emit_sidecar=cfg.data.emit_sidecar, bucket_max_items=cfg.data.bucket_max_items,
            articulated=bool(cfg.model.get("articulated_bodies", False)), domains=[domain],
            eval_domains=None, domain_weights=None, seed=0, roots=OmegaConf.to_container(cfg.data.roots, resolve=True),
            window_motion=None, batch_across_episodes=None, training=False, shuffle=False, no_shard_split=False,
            first_window=True)
        # the loader slices the clip to context+future_len frames; ask it for the whole clip instead
        for d in loader.loaders.values():             # UnifiedValRoundRobin: one DataLoader per domain
            d.dataset.future_len = 10 ** 6            # T <= window: _window() returns the whole clip
        want, got = sorted(set(indices)), {}
        for i, b in enumerate(loader):
            if i in want:
                got[i], _ = pad_frames(b, transitions)
                got[i].gt_frames = b.positions.shape[1]
            if i >= want[-1]:
                break
    finally:
        os.chdir(cwd)
    return got


def stack_batches(batches):
    """Concatenate B=1 TrajectoryBatches along the batch dim (same layout: every pickup episode is 671 + 512)."""
    first = batches[0]
    kw = {}
    for f in dataclasses.fields(first):
        v = getattr(first, f.name)
        if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1:
            kw[f.name] = torch.cat([getattr(b, f.name) for b in batches], 0)
    out = dataclasses.replace(first, **kw)
    out.gt_frames = [getattr(b, "gt_frames", b.positions.shape[1]) for b in batches]
    return out


def object_particles(batch):
    """[B, P] mask of the passive rigid object's particles (rigid, not actuated), from the UNHIDDEN batch."""
    passive = (batch.body_type == 1) & ~batch.pose_action_mask                   # [B, K]
    return passive.gather(1, batch.particle_body) & batch.get_valid_mask()


def object_centroid(batch, pos, mask=None):
    w = (object_particles(batch) if mask is None else mask).to(pos.dtype)[..., None]
    return (pos * w).sum(-2) / w.sum(-2)


class Streamer:
    def __init__(self, model, diffusion, cfg, device):
        self.model, self.diffusion, self.cfg, self.dev = model, diffusion, cfg, device
        self.T = diffusion.num_train_timesteps
        self.C = cfg.data.context_len
        self.NW = cfg.data.context_len - 1 + cfg.data.future_len          # 19 rows

    def _whole(self, b, L=None):
        """The batch as ONE window of its first L transitions anchored at frame 0 (for materialization)."""
        L = b.positions.shape[1] - 1 if L is None else L
        pos = b.positions[:, :L + 1]
        kw = dict(context_len=1, positions=pos, body_poses=b.body_poses[:, :L + 1], actions=torch.zeros_like(pos[:, 1:]))
        if b.action_active is not None:
            kw["action_active"] = torch.zeros_like(pos[:, 1:, :, 0], dtype=torch.bool)
            kw["context_action_active"] = b.context_action_active[:, :0]
        return dataclasses.replace(b, **kw)

    joint_limits = None           # (q_min, q_max) metres of |q| for prismatic fingers; None = no limit (the model's raw output)

    def _limit_fingers(self, batch, pos, R, t, q_prev=None):
        """Saturating prismatic joint limit: per finger, q_f = clip(q_{f-1} + (q_f - q_{f-1})) frame by frame, where q is
        the finger centroid's offset from its parent along the joint axis (parent frame). Finger translations and point
        rows are rebuilt from the clipped q; rotations are untouched. Differentiable (clip). Returns pos, t, q_last."""
        if self.joint_limits is None:
            return pos, t, None
        qmin, qmax = self.joint_limits
        ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
        col = {int(k): j for j, k in enumerate(ridx.tolist())}
        act, jt, par = batch.pose_action_mask[0], batch.joint_type[0], batch.body_parent[0]
        valid = batch.get_valid_mask()
        B, F = t.shape[:2]
        t = t.clone(); pos = pos.clone(); q_last = {}
        for k in (act & (jt == 2)).nonzero()[:, 0].tolist():
            j, jp = col[k], col[int(par[k])]
            axis = batch.joint_axis[:, k] / batch.joint_axis[:, k].norm(dim=-1, keepdim=True)         # [B, 3] parent frame
            Rp, tp = R[:, :, jp], t[:, :, jp]                                                            # [B, F, 3, 3], [B, F, 3]
            local = torch.einsum("bfji,bfj->bfi", Rp, t[:, :, j] - tp)                                   # parent-frame offset
            q = torch.einsum("bfi,bi->bf", local, axis)                                                  # signed
            sgn = torch.sign(q[:, :1]).clamp(min=-1).where(q[:, :1] != 0, torch.ones_like(q[:, :1]))
            qa = (q * sgn)                                                                                # |q| along the outward side
            prev = qa[:, 0].clamp(qmin, qmax) if q_prev is None else q_prev[k]
            out = []
            for f in range(F):
                d = qa[:, f] - qa[:, f - 1] if f > 0 else qa[:, f] - (q_prev[k] if q_prev is not None else qa[:, f])   # increment
                prev = (prev + d).clamp(qmin, qmax); out.append(prev)
            qc = torch.stack(out, 1) * sgn                                                               # signed, clipped
            local_c = local + (qc - q)[..., None] * axis[:, None]
            t_new = tp + torch.einsum("bfij,bfj->bfi", Rp, local_c)
            t[:, :, j] = t_new
            q_last[k] = torch.stack(out, 1)[:, -1]
            n = int(batch.body_template_mask[0, k].sum()); tmpl = batch.body_templates[:, k, :n].float()
            for b in range(B):
                rows = (batch.particle_body[b] == k) & valid[b]
                pos[b, :, rows] = torch.einsum("fij,pj->fpi", R[b, :, j], tmpl[b]) + t_new[b][:, None]
        return pos, t, q_last

    def _materialize(self, batch, rows, prows):
        """Positions [B, L+1, P, 3] and rigid (R, t) over the first L = len(rows) transitions, from frame 0."""
        L = len(rows)
        pred, R, t, _ = materialize_pose_native(self._whole(batch, L), torch.stack(rows, 1).float(), torch.stack(prows, 1).float())
        pred, t, _ = self._limit_fingers(batch, pred, R, t)
        return torch.cat([batch.positions[:, :1], pred], 1), R, t

    def _step_anchor(self, batch, pos_a, bp_a, row, prow):
        """Advance the anchor frame by ONE committed transition: the same materialization (deformable rows = previous
        + delta, rigid rows = template @ integrated pose) run over a one-transition window anchored at pos_a / bp_a."""
        ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
        b = dataclasses.replace(self._whole(batch, 1), positions=torch.stack([pos_a, pos_a], 1),
                                body_poses=torch.stack([bp_a, bp_a], 1))
        pred, R, t, _ = materialize_pose_native(b, row[:, None].float(), prow[:, None].float())
        if self.joint_limits is not None:
            # the anchor's own finger q (already within limits) is the previous state of the saturating clip
            pred, t, _ = self._limit_fingers(b, pred, R, t, q_prev=self._anchor_q(b, bp_a))
        bp = bp_a.clone()
        bp[:, ridx, :3, :3] = R[:, 0]; bp[:, ridx, :3, 3] = t[:, 0]
        return pred[:, 0], bp

    def _anchor_q(self, batch, bp):
        """|q| of every prismatic finger in a [B, K, 4, 4] pose set (parent-frame offset along the joint axis)."""
        act, jt, par = batch.pose_action_mask[0], batch.joint_type[0], batch.body_parent[0]
        out = {}
        for k in (act & (jt == 2)).nonzero()[:, 0].tolist():
            p = int(par[k]); axis = batch.joint_axis[:, k] / batch.joint_axis[:, k].norm(dim=-1, keepdim=True)
            local = torch.einsum("bji,bj->bi", bp[:, p, :3, :3], bp[:, k, :3, 3] - bp[:, p, :3, 3])
            out[k] = (local * axis).sum(-1).abs()
        return out

    def _window_state(self, batch, start, *rows_and_prows):
        """The window's anchor frames and poses [start, start + NW], rebuilt from rows [0, start + NW)."""
        n = len(rows_and_prows) // 2
        pos, R, t = self._materialize(batch, list(rows_and_prows[:n]), list(rows_and_prows[n:]))
        ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
        others = ~torch.isin(torch.arange(batch.body_type.shape[1], device=pos.device), ridx)
        bp = batch.body_poses[:, :n + 1].clone()
        bp[:, 1:, ridx, :3, :3] = R; bp[:, 1:, ridx, :3, 3] = t
        bp[:, 1:, others] = bp[:, :1, others]
        return pos[:, start:], bp[:, start:]

    def _call(self, batch, xs_ctx, xs_fut, xps_ctx, xps_fut, tt, pos_w, bp_w, start, k=0, seeds=None):
        # seeds/k are ARGUMENTS so a checkpointed call's backward recompute (after the rollout returned) redraws the
        # same per-member FPS starts; the fixed-FPS state is set here and cleared on the way out.
        C, NW = self.C, self.NW
        FPS.seeds = seeds; FPS.begin(k)
        try:
            return self._call_model(batch, xs_ctx, xs_fut, xps_ctx, xps_fut, tt, pos_w, bp_w, start)
        finally:
            FPS.seeds = None

    def _call_model(self, batch, xs_ctx, xs_fut, xps_ctx, xps_fut, tt, pos_w, bp_w, start):
        C, NW = self.C, self.NW
        pos_full = torch.zeros_like(batch.positions)                   # shape only: every field is overridden below
        wb = rollout_window(batch, pos_full, start + C, NW - (C - 1))
        wb = dataclasses.replace(wb, positions=pos_w, body_poses=bp_w, pose_deltas_gt=torch.zeros_like(wb.pose_deltas_gt),
                                 actions=torch.zeros_like(wb.actions))
        wb = hide_actions(wb, 1.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fut, pose_fut = self.model(wb, xs_fut, tt[:, C - 1:], noisy_context=xs_ctx, context_timesteps=tt[:, :C - 1],
                                       noisy_pose=xps_fut, noisy_ctx_pose=xps_ctx)
        if fut is None:                                                # entirely rigid scene: no particle decode
            B, NWr = pose_fut.shape[0], tt.shape[1]
            part = torch.zeros(B, NWr, xs_fut.shape[2], 3, device=pose_fut.device)
        else:
            part = torch.cat([self.model._last_ctx_part_pred, fut], 1)
        return part.float(), torch.cat([self.model._last_ctx_pose_pred, pose_fut], 1).float()

    def churn_levels(self, steps, keep, N):
        """[N, n_max] level at which row j is re-formed by its i-th injection (0 = no such injection; the final
        decrement to level 0 injects nothing). Replays the schedule arithmetic of rollout() exactly."""
        NW = self.NW
        calls = window_states(0, keep, steps, N, NW)
        m, D = NW - keep, steps * (NW - keep)
        sched = lambda j, call: min(D, max(0, D + j * steps - call * m)) / D
        lev, committed, inj = [1.0] * N, 0, [[] for _ in range(N)]
        for k in range(len(calls)):
            start = min(max(0, committed - keep), N - NW)
            for j in range(start, start + NW):
                l0, l1 = lev[j], min(lev[j], sched(j, k + 1))
                if l0 <= 0 or l1 == l0:
                    continue
                if l1 > 0:
                    inj[j].append(l1)
                lev[j] = l1
            while committed < N and lev[committed] <= 0:
                committed += 1
        n_max = max(len(v) for v in inj)
        out = torch.zeros(N, n_max)
        for j, v in enumerate(inj):
            out[j, :len(v)] = torch.tensor(v)
        return out

    def rollout(self, batch, X, XP, *, steps, keep=3, grad=False, fps_seeds=None, churn=None, churn_part=None):
        """X [B, N, P, 3], XP [B, N, Kr, 6] initial noise -> positions [B, N+1, P, 3]. Differentiable when grad.

        churn (full churn, eta = 1): a pre-decided table of pose draws E [B, N, 1 + n_max, Kr, 6]; E[:, j, 0] is
        row j's initial noise (XP is ignored) and E[:, j, i] the fresh Gaussian used at row j's i-th injection, at the
        level churn_levels(steps, keep, N)[j, i-1]: x = x0 + l_new (eps - x0) instead of carrying the old noise
        x = x0 + (l_new/l_old)(x - x0). The rollout is a deterministic function of E.

        churn_part [B, N, 1 + n_max, P, 3] does the same for the PARTICLE rows, on the same levels (X is then
        ignored). Full churn means every noised quantity is re-formed, particles included; a deformable scene (dough)
        must pass it. Without it the particle rows keep the deterministic carry -- correct only for rigid scenes,
        where the UNet emits no particle output at all."""
        C, NW, dev = self.C, self.NW, self.dev
        B, N = X.shape[:2]
        if churn is not None:
            if churn.shape[:2] != (B, N):
                raise ValueError(f"churn draws must be [B, N, 1 + n_max, Kr, 6], got {tuple(churn.shape)}")
            XP = churn[:, :, 0]
            n_inj = [0] * N                                             # injections done per row
        if churn_part is not None:
            if churn is None:
                raise ValueError("churn_part needs the pose churn table (full churn re-forms both)")
            if churn_part.shape[:2] != (B, N) or churn_part.shape[2] != churn.shape[2]:
                raise ValueError(f"churn_part must be [B, N, {churn.shape[2]}, P, 3], got {tuple(churn_part.shape)}")
            X = churn_part[:, :, 0]
        if N < NW:
            raise ValueError(f"need at least {NW} transitions")
        ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
        others = ~torch.isin(torch.arange(batch.body_type.shape[1], device=dev), ridx)
        pmask = pose_channel_mask(batch, ridx)
        pm = (lambda v: v * pmask.to(v)) if pmask is not None else (lambda v: v)
        rows, prows = list(X.unbind(1)), [pm(v) for v in XP.unbind(1)]
        seeds = None if fps_seeds is None else tuple(int(v) for v in fps_seeds)
        calls = window_states(0, keep, steps, N, NW)
        m, D = NW - keep, steps * (NW - keep)
        sched = lambda j, call: min(D, max(0, D + j * steps - call * m)) / D
        lev, committed = [1.0] * N, 0
        # The model reads only the window's FIRST frame (positions + body poses; the rest are overwritten by the
        # noisy rows -- checked: corrupting frames 1: changes x0 by exactly 0). That anchor depends on committed
        # rows only, so it is advanced one committed row at a time instead of re-integrated from frame 0 per call.
        pos_a, bp_a, a_idx = batch.positions[:, 0], batch.body_poses[:, 0], 0
        for k, (_, _, wl) in enumerate(calls):
            start = min(max(0, committed - keep), N - NW)
            assert all(abs(a - b) < 1e-9 for a, b in zip(wl, lev[start:start + NW])), "schedule drifted from window_states"
            while a_idx < start:                                        # rows < start are committed (start <= committed)
                pos_a, bp_a = self._step_anchor(batch, pos_a, bp_a, rows[a_idx], prows[a_idx])
                a_idx += 1
            pos_w = pos_a[:, None].expand(-1, NW + 1, -1, -1)
            bp_w = bp_a[:, None].expand(-1, NW + 1, -1, -1, -1)
            tt = torch.clamp(torch.round(torch.tensor(wl, device=dev) * self.T), 1, self.T - 1).long()[None].expand(B, -1)
            xs, xps = torch.stack(rows[start:start + NW], 1), torch.stack(prows[start:start + NW], 1)
            args = (batch, xs[:, :C - 1], xs[:, C - 1:], xps[:, :C - 1], xps[:, C - 1:], tt, pos_w, bp_w, start, k, seeds)
            x0, x0p = (checkpoint(self._call, *args, use_reentrant=False) if grad else self._call(*args))
            if MEMLOG and (k % 10 == 0 or k == len(calls) - 1):
                print(f"[mem] call {k}/{len(calls)} grad={grad} allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
            for j in range(start, start + NW):
                l0, l1 = lev[j], min(lev[j], sched(j, k + 1))
                if l0 <= 0 or l1 == l0:
                    continue
                r, jj = l1 / l0, j - start
                if churn is not None and l1 > 0:
                    n_inj[j] += 1
                    prows[j] = pm(x0p[:, jj] + l1 * (pm(churn[:, j, n_inj[j]]) - x0p[:, jj]))
                    rows[j] = (x0[:, jj] + l1 * (churn_part[:, j, n_inj[j]] - x0[:, jj]) if churn_part is not None
                               else x0[:, jj] + r * (rows[j] - x0[:, jj]))
                else:
                    rows[j] = x0[:, jj] + r * (rows[j] - x0[:, jj])
                    prows[j] = pm(x0p[:, jj] + r * (prows[j] - x0p[:, jj]))
                lev[j] = l1
            while committed < N and lev[committed] <= 0:
                committed += 1
        assert committed == N
        return self._materialize(batch, rows, prows)[0], len(calls)
