"""pwm_final_clean PICKUP checkpoints (e.g. pickup_uncond_x0_regpass_cont200k final.pt) through the package's fast path:
inference.streaming_chain (the same differentiable streaming staircase guidance/streaming_guidance.py optimizes), but under
bf16 autocast with the model regionally compiled, at any S. Episodes = entries of a select json ({label,
f0_obj_centroid_model}) matched in the pickup val set, like streaming_guidance.py --select; `episodes` indexes that list
(repeats = population). State = the initial pose noise P0 (deterministic Euler staircase)."""
import dataclasses, json, os, sys
import numpy as np
import torch
from .base import Adapter, State

REPO = os.environ.get("GP_CLEAN_REPO", "/data/user_data/mbronars/packages/pwm_final_clean")   # GP_CLEAN_REPO: opt-in other tree (mb/clean-flowmap)
ROOTS = os.environ.get("GP_PICKUP_ROOTS", "/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_pick_success").split(":")   # GP_PICKUP_ROOTS=a:b overrides (e.g. the place_success root for held starts)
SELECT = "/data/user_data/mbronars/packages/guided_playback/sweeps/pickonly_val5.json"


class CleanPickupAdapter(Adapter):
    name = "clean_pickup"
    frame_dt = 1 / 20
    sim = "isaac"
    window = None
    select_path = SELECT
    bf16 = True

    def load(self, ckpt, device, compile=True):
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        cwd = os.getcwd(); os.chdir(REPO)
        try:
            from inference import load_model_and_diffusion
            self.model, self.diffusion, self.cfg, self.ckpt_steps = load_model_and_diffusion(ckpt, device)
        finally:
            os.chdir(cwd)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval(); self.dev = device
        self.step = int(torch.load(ckpt, map_location="cpu", weights_only=False).get("step", -1))
        if compile:
            from utils.compilation import prepare_compilation
            import torch._dynamo as dynamo
            prepare_compilation(self.model, dict(compile=True, compile_regional=True, compile_l0=True, compile_dynamic=False))
            for n, v in (("cache_size_limit", 4096), ("recompile_limit", 4096), ("accumulated_recompile_limit", 16384)):
                if hasattr(dynamo.config, n):
                    setattr(dynamo.config, n, v)
        self.mode = ("compiled regional + L0, " if compile else "eager, ") + ("bf16" if self.bf16 else "fp32")
        return self

    def episodes(self, indices, transitions, domain=None):
        from scripts.unified.eval_unified import evaluation_dataset
        sys.path.insert(0, os.path.join(REPO, "guidance"))
        from streaming_guidance import select_and_group
        self.domain = "pickup"
        C, F = int(self.cfg.data.context_len), int(self.cfg.data.future_len)
        cwd = os.getcwd(); os.chdir(REPO)
        try:
            ds = evaluation_dataset(self.cfg, "pickup", C, F, roots_override={"pickup": ROOTS})
            ds.emit_sidecar = True; ds.articulated = bool(getattr(self.model.cfg, "articulated_bodies", False)); ds._pose_stats(); ds.future_len = 10_000
            sel = json.load(open(self.select_path)); pick = [dict(sel[i], label=f"{sel[i]['label']}_m{k}") for k, i in enumerate(indices)]
            groups = select_and_group(ds, pick, len(pick), self.dev)
        finally:
            os.chdir(cwd)
        labels, batch = groups[0]
        batch = batch.to(self.dev)
        T = batch.positions.shape[1]
        if int(transitions) + 1 > T:                        # longer than the recorded clip: repeat its last frame (only frame 0 is read)
            from .paper_core import pad_frames
            batch, _ = pad_frames(batch, int(transitions))
        batch.gt_frames = [T] * len(indices); batch.episodes = [sel[i]["label"] for i in indices]
        self.n_tr = int(transitions)
        return batch

    def prepare(self, batch, transitions, steps, keep, seed, churn, t_freeze, fix_fps):
        from inference import streaming_noise
        if churn and not (getattr(self, "stop_at", None) or getattr(self, "history_clouds", None) is not None):
            raise SystemExit("clean_pickup: churn needs the full-window samplers (stop_at or history_clouds)")
        g = torch.Generator(device=self.dev).manual_seed(seed)
        X0, P0 = streaming_noise(self.model, batch, self.n_tr, g)
        W = self.window or int(self.cfg.data.future_len)
        u = max(steps / W, 1e-9)
        info = dict(sampler=f"streaming_chain window {W}, S {steps}, deterministic Euler, initial pose-noise opt, {self.mode}", transitions=self.n_tr)
        shell = P0.flatten(2).norm(dim=-1)[..., None, None]
        ff = int(getattr(self, "freeze_frames", 0) or 0); free = None
        if ff > 0:                                                    # grasp-phase freeze: rows of frames < ff keep their unguided draw
            free = (torch.arange(P0.shape[1], device=P0.device) >= ff).to(P0.dtype)[None, :, None, None].expand_as(P0).clone()
            info["freeze_frames"] = ff
        fixed = dict(X0=X0, kw=dict(window=W, num_inference_steps=int(steps)))
        if os.environ.get("GP_ENTER_FULL_NOISE"):                   # opt-in: rows sliding into the window keep their true level
            fixed["kw"]["enter_full_noise"] = True; info["sampler"] += ", enter_full_noise"
        if os.environ.get("GP_STREAM_U"):                           # opt-in: calls between consecutive transitions (default steps / window)
            fixed["kw"]["uncertainty_scale"] = float(os.environ["GP_STREAM_U"]); info["sampler"] += f", u {fixed['kw']['uncertainty_scale']}"
        hc = getattr(self, "history_clouds", None)
        if hc is not None and len(hc) > 1:                          # closed loop: executed frames 0..c as clean prefix rows, noise only for the rest
            from .pick_chain import executed_prefix
            pre, err = executed_prefix(batch, hc); c = len(pre["P"])
            X0 = X0[:, c:].contiguous(); P0 = P0[:, c:].contiguous(); shell = P0.flatten(2).norm(dim=-1)[..., None, None]
            if free is not None: free = free[:, c:].contiguous()
            fixed = dict(X0=X0, pre=pre, kw=dict(window=W, S=int(steps)))
            info.update(executed=c, prefix_rebuild_err_mm=round(err * 1e3, 3), generated=int(P0.shape[1]))
        vars_, free_, shell_ = [P0], [free], [shell]
        if churn:                                                   # full churn bank (push recipe, DNO SDE variant): one draw per (model call, row)
            n_rows = int(P0.shape[1]); n_calls = int(steps) + int(np.ceil((max(n_rows, 1) - 1) * u)) + 3
            E = torch.randn((n_calls, n_rows, P0.shape[0], P0.shape[2], 6), device=P0.device, generator=g)
            vars_.append(E); free_.append(None); shell_.append(E.flatten(3).norm(dim=-1)[..., None, None])
            fixed["churn"] = True; info.update(churn=True, bank_calls=n_calls)
            fixed["t_freeze"] = float(t_freeze or 0.0)                 # > 0: only bank draws injected at level >= t_freeze are optimized (mask set on the first rollout)
        if fix_fps:                                                 # reseed the global RNG before every rollout: same FPS draws each time (push adapter behaviour)
            fixed["fps_seed"] = int(seed); info["fix_fps"] = True
        return State(vars=[v.clone().requires_grad_(True) for v in vars_], free=free_, shell=shell_, fixed=fixed, info=info)

    def rollout(self, batch, state, grad):
        from inference import streaming_chain
        if "fps_seed" in state.fixed:
            torch.manual_seed(state.fixed["fps_seed"]); torch.cuda.manual_seed(state.fixed["fps_seed"])
        lev = None
        if state.fixed.get("churn") and state.fixed.get("t_freeze", 0.0) > 0 and not state.fixed.get("free_set"):
            lev = torch.zeros(state.vars[1].shape[0], state.vars[1].shape[1])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            stop = getattr(self, "stop_at", None)
            if "pre" in state.fixed:                                   # closed loop, full windows (the window slides back over executed rows)
                from .pick_chain import chain_prefix
                out = chain_prefix(self.model, self.diffusion, batch, state.fixed["pre"], state.fixed["X0"], state.vars[0], full_window=True, grad_checkpoint=grad,
                                   E=state.vars[1] if state.fixed.get("churn") else None, lev_out=lev, **state.fixed["kw"])
                out["calls"] = out.get("calls", 0)
            elif stop:                                                   # full windows to the end: noise covers stop + window transitions, commit 0..stop
                from .pick_chain import streaming_chain_stop
                out = streaming_chain_stop(self.model, self.diffusion, batch, state.fixed["X0"], state.vars[0], grad_checkpoint=grad, stop_at=int(stop),
                                           E=state.vars[1] if state.fixed.get("churn") else None, lev_out=lev, **state.fixed["kw"])
            else:
                out = streaming_chain(self.model, self.diffusion, batch, state.fixed["X0"], state.vars[0], grad_checkpoint=grad, **state.fixed["kw"])
        if lev is not None:                                              # threshold mask from the levels the sampler actually used
            E = state.vars[1]; tf = state.fixed["t_freeze"]
            fm = ((lev >= tf - 1e-9) & (lev > 0)).to(E.device)[:, :, None, None, None].expand_as(E).clone()
            state.free[1] = fm; state.fixed["free_set"] = True
            state.info.update(bank_draws_used=int((lev > 0).sum()), bank_draws_free=int(fm[:, :, 0, 0, 0].sum()))
            print(f"[clean_pickup] churn threshold {tf}: {state.info['bank_draws_free']} of {state.info['bank_draws_used']} injected draws optimized", flush=True)
        state.info["model_calls"] = int(out["calls"])
        return out["positions"].float()

    def object_mask(self, batch):
        return ((batch.body_type == 1) & ~batch.pose_action_mask).gather(1, batch.particle_body) & batch.get_valid_mask()

    def roles(self, batch):
        pb, am = batch.particle_body.cpu().numpy(), batch.pose_action_mask.cpu().numpy()
        jt = batch.joint_type.cpu().numpy() if getattr(batch, "joint_type", None) is not None else np.zeros(am.shape, np.int64)
        valid = batch.get_valid_mask().cpu().numpy(); out = np.full(pb.shape, 2, np.int8)
        for b in range(pb.shape[0]):
            for k in np.nonzero(am[b])[0]:
                out[b, pb[b] == k] = 1 if jt[b, k] == 2 else 0
            out[b, ~valid[b]] = -1
        return out

    def export(self, batch):
        cpu = lambda v: v.detach().cpu().numpy()
        jt = batch.joint_type if getattr(batch, "joint_type", None) is not None else torch.zeros_like(batch.body_type)
        return dict(particle_body=cpu(batch.particle_body), pose_action_mask=cpu(batch.pose_action_mask), joint_type=cpu(jt), body_type=cpu(batch.body_type),
                    object_ids=cpu(batch.object_ids), body_poses0=cpu(batch.body_poses[:, 0]), gt=cpu(batch.positions), gt_frames=np.array(batch.gt_frames))
