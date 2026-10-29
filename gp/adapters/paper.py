"""pwm_final_paper checkpoints (context-predicting pose-native UNet, unified loader): pickup / rl_push / mould domains.

Wraps paper_core (= reports/pickplace_server/stream_core.py, verbatim copy): Streamer with the stream staircase,
full-churn draw table, fixed per-member FPS, joint limits. One State = (X particle draws [fixed, inert for rigid
scenes], XP pose draws or the churn table E)."""
import dataclasses
import numpy as np
import torch
from . import paper_core as sc
from .base import Adapter, State

FRAME_DT = {"pickup": 1 / 20, "rl_push": 1 / 20, "yikai_small": 1 / 20, "mould": 0.05, "pushing": 1 / 20}       # pushing = 04_18 (PushBoundary), 20 Hz
SIM = {"pickup": "isaac", "rl_push": "isaac", "mould": "genesis", "pushing": "sapien"}


class PaperAdapter(Adapter):
    name = "paper"
    select_path = None          # a select json (sweeps/assetsweep_select.json): `episodes` then index ITS entries

    def load(self, ckpt, device, compile=True):
        self.dev = device
        self.model, self.diffusion, self.cfg = sc.load(ckpt, device)
        self.step = int(torch.load(ckpt, map_location="cpu", weights_only=False)["step"])
        if compile:
            sc.compile_model(self.model, l0=True)
        self.S = sc.Streamer(self.model, self.diffusion, self.cfg, device)
        self.mode = "compiled regional + L0, bf16" if compile else "eager bf16"
        return self

    def episodes(self, indices, transitions, domain=None):
        domain = domain or list(self.cfg.data.domains)[0]
        self.domain = domain
        self.frame_dt = FRAME_DT.get(domain, 1 / 20); self.sim = SIM.get(domain)
        if self.select_path:                                   # benchmark scenes, matched by their frame-0 object centroid
            import json
            sel = json.load(open(self.select_path)); entries = [sel[i] for i in indices]
            got = sc.episode_batch_select(self.cfg, entries, transitions, self.dev, domain=domain)
            batch = sc.stack_batches(got); gf = batch.gt_frames
            batch = batch.to(self.dev); batch.gt_frames = gf; batch.episodes = [e["label"] for e in entries]
            return batch
        got = sc.episode_batch(self.cfg, indices, transitions, self.dev, domain=domain)
        batch = sc.stack_batches([got[i] for i in indices]); gf = batch.gt_frames
        batch = batch.to(self.dev); batch.gt_frames = gf; batch.episodes = list(indices)
        return batch

    def prepare(self, batch, transitions, steps, keep, seed, churn, t_freeze, fix_fps):
        hist = getattr(self, "history", None)
        if hist is not None and ("positions" in hist or hist.get("frame0")):   # closed loop: executed frames as clean rows (gp/adapters/paper_prefix.py)
            from .paper_prefix import prepare_prefix
            return prepare_prefix(self, batch, transitions, steps, keep, seed, churn, t_freeze, fix_fps, hist)
        B, N, P = batch.positions.shape[0], transitions, batch.positions.shape[2]
        self.S.joint_limits = self.joint_limits
        g = torch.Generator(device=self.dev).manual_seed(seed)
        X = torch.randn(B, N, P, 3, device=self.dev, generator=g)
        Kr = int((batch.body_type[0] == 1).sum())
        XP = torch.randn(B, N, Kr, 6, device=self.dev, generator=g)
        fixed = dict(X=X, steps=steps, keep=keep, fps_seeds=[seed * 1000 + i for i in range(B)] if fix_fps else None, churn=churn)
        from .paper_core import window_states
        chunked = getattr(self, "sampler_mode", "stream") == "chunked"
        calls = len(window_states(0, keep, steps, N, self.S.NW))
        if chunked:
            from .chunked import recorded_context_rows, window_starts
            fixed.update(chunked=True, ctx_rows=recorded_context_rows(self.S, batch))
            with torch.no_grad():                                                   # the recorded context must reproduce recorded frames 1..3
                cr = fixed["ctx_rows"]; C1 = cr.shape[1]
                rec, _, _ = self.S._materialize(batch, list(torch.zeros(B, C1, P, 3, device=self.dev).unbind(1)), list(cr.unbind(1)))
                v = batch.get_valid_mask()[:, None, :, None]
                self.ctx_check_mm = float(((rec[:, 1:] - batch.positions[:, 1:C1 + 1]).abs() * v).amax() * 1e3)
            print(f"[chunked] recorded context rows reproduce recorded frames 1..{C1} to {self.ctx_check_mm:.3f} mm (max abs)", flush=True)
            calls = steps * len(window_starts(N, self.S.NW, self.S.C))
            if churn:                                                               # draw i of row j = its i-th call; level after call i = 1 - i/steps
                E = torch.randn(B, N, steps + 1, Kr, 6, device=self.dev, generator=g)
                lev = torch.tensor([1.0] + [1.0 - i / steps for i in range(1, steps + 1)], device=self.dev)
                free = ((lev >= t_freeze - 1e-9) & (lev > 0))[None, None, :, None, None].expand_as(E).clone()
                shell = E.flatten(3).norm(dim=-1)[..., None, None]
                info = dict(model_calls=calls, draws_per_row=steps + 1, free_draws_per_row=int(free[0, 0, :, 0, 0].sum()),
                            sampler="chunked 3-context / 16-future windows, recorded first context, full churn (eta=1), joint draw optimization")
                return State(vars=[E.requires_grad_(True)], free=[free], shell=[shell], fixed=fixed, info=info)
            shell = XP.flatten(2).norm(dim=-1)[..., None, None]
            return State(vars=[XP.requires_grad_(True)], free=[None], shell=[shell], fixed=fixed,
                         info=dict(model_calls=calls, sampler="chunked 3-context / 16-future windows, recorded first context, deterministic"))
        deformable = bool((batch.body_type[0] == 0).any())                          # dough etc.: particle draws matter
        if churn:
            inj = self.S.churn_levels(steps, keep, N).to(self.dev)
            levels = torch.cat([torch.ones(N, 1, device=self.dev), inj], 1)
            E = torch.randn(B, N, levels.shape[1], Kr, 6, device=self.dev, generator=g)
            free = ((levels >= t_freeze - 1e-9) & (levels > 0))[None, :, :, None, None].expand_as(E).clone()
            shell = E.flatten(3).norm(dim=-1)[..., None, None]
            info = dict(draws_per_row=int(levels.shape[1]), free_per_row_min=int(free[0, :, :, 0, 0].sum(1).min()),
                        free_per_row_max=int(free[0, :, :, 0, 0].sum(1).max()), model_calls=calls, sampler="full churn (eta=1), joint draw optimization")
            st = State(vars=[E.requires_grad_(True)], free=[free], shell=[shell], fixed=fixed, info=info)
            if deformable:
                # full churn re-forms the PARTICLE rows too (same levels); without this the dough rows would carry
                # deterministically, which is only correct for rigid scenes (no particle output at all).
                EX = torch.randn(B, N, levels.shape[1], P, 3, device=self.dev, generator=g)
                free_x = ((levels >= t_freeze - 1e-9) & (levels > 0))[None, :, :, None, None].expand_as(EX).clone()
                st.vars.append(EX.requires_grad_(True)); st.free.append(free_x)
                st.shell.append(EX.flatten(3).norm(dim=-1)[..., None, None])
                info["particle_churn"] = True
        else:
            shell = XP.flatten(2).norm(dim=-1)[..., None, None]
            st = State(vars=[XP.requires_grad_(True)], free=[None], shell=[shell], fixed=fixed,
                       info=dict(model_calls=calls, sampler="deterministic (initial-noise optimization)"))
        if deformable and not churn and not getattr(self, "pose_only", False):        # no churn: optimize the initial particle draws (pose_only: keep them fixed, as for rigid scenes)
            st.vars.append(X.requires_grad_(True)); st.free.append(None); st.shell.append(X.flatten(2).norm(dim=-1)[..., None, None])
            st.info["particle_draws_optimized"] = True
            fm = int(getattr(self, "freeze_modes", 0) or 0)
            if fm > 0:                                                                # keep the low-order spatial modes of the dough noise at their sampled values
                om = self.object_mask(batch).float()                                  # [B, P]
                p0 = batch.positions[:, 0].float()                                    # [B, P, 3] frame-0 points
                cols = [om]
                if fm >= 2:
                    mu = (p0 * om[..., None]).sum(1, keepdim=True) / om.sum(1, keepdim=True)[..., None]
                    cols += [((p0 - mu)[..., k] * om) for k in range(3)]
                Q, _ = torch.linalg.qr(torch.stack(cols, -1))                         # [B, P, M] orthonormal over the dough points (zero elsewhere)
                st.fixed["mode_basis"] = [None] * (len(st.vars) - 1) + [Q]
                st.info["frozen_modes"] = Q.shape[-1]
        return st

    def rollout(self, batch, state, grad):
        if state.fixed.get("prefix") is not None:
            from .paper_prefix import rollout_prefix
            return rollout_prefix(self, batch, state, grad)
        f = state.fixed; X = state.vars[1] if len(state.vars) > 1 else f["X"]
        if f.get("chunked"):
            from .chunked import rollout_chunked
            ch = state.vars[0] if f["churn"] else None
            pos, _ = rollout_chunked(self.S, batch, X, None if ch is not None else state.vars[0], steps=f["steps"], ctx_rows=f["ctx_rows"], grad=grad, fps_seeds=f["fps_seeds"], churn=ch)
            return pos
        if f["churn"]:
            part = state.vars[1] if state.info.get("particle_churn") else None        # [B, N, 1+n_max, P, 3]
            pos, _ = self.S.rollout(batch, f["X"], None, steps=f["steps"], keep=f["keep"], grad=grad,
                                    fps_seeds=f["fps_seeds"], churn=state.vars[0], churn_part=part)
        else:
            pos, _ = self.S.rollout(batch, X, state.vars[0], steps=f["steps"], keep=f["keep"], grad=grad, fps_seeds=f["fps_seeds"])
        return pos

    def object_mask(self, batch):
        passive = ~batch.pose_action_mask                                             # [B, K]: rigid or deformable, not actuated
        return passive.gather(1, batch.particle_body) & batch.get_valid_mask()

    def roles(self, batch):
        jt_t = batch.joint_type if getattr(batch, "joint_type", None) is not None else torch.zeros_like(batch.body_type)
        pb, am, jt, valid = (batch.particle_body.cpu().numpy(), batch.pose_action_mask.cpu().numpy(), jt_t.cpu().numpy(), batch.get_valid_mask().cpu().numpy())
        out = np.full(pb.shape, 2, np.int8)
        for b in range(pb.shape[0]):
            for k in np.nonzero(am[b])[0]:
                out[b, pb[b] == k] = 1 if jt[b, k] == 2 else 0
            out[b, ~valid[b]] = -1
        return out

    def export(self, batch):
        cpu = lambda v: v.detach().cpu().numpy()
        act = batch.pose_action_mask
        object_ids = torch.where(act.gather(1, batch.particle_body), 1, 0)          # Genesis contract: 0 object, 1 actuator
        jt = batch.joint_type if getattr(batch, "joint_type", None) is not None else torch.zeros_like(batch.body_type)
        return dict(particle_body=cpu(batch.particle_body), pose_action_mask=cpu(act), joint_type=cpu(jt),
                    body_type=cpu(batch.body_type), object_ids=cpu(object_ids), body_poses0=cpu(batch.body_poses[:, 0]),
                    gt=cpu(batch.positions), gt_frames=np.array(batch.gt_frames))
