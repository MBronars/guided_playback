"""Gradient-free receding-horizon planner in the world model (MPPI with argmin selection instead of the weighted
average of pwm_efficient/mppi_in_world_model.py):

  per replan (committed c transitions so far):
    draw K fresh noises for the remaining 60 - c transitions, sample K futures with the streaming chain continuing
    from the committed prefix (committed transitions are clean context rows at level 0, exactly as the streaming
    staircase feeds its own committed rows), score each by the objective at the final frame, keep the argmin,
    commit its next `commit` transitions (particles + all rigid poses: hand, fingers, object), repeat.

Candidates run in chunks of the batch size (the server's compiled B), so no recompiles. Reloaded by the server on
every /plan request (bin/serve.py), so edits here need no restart."""
import json, os, time
import numpy as np
import torch
from .session import Args, guided_run  # noqa: F401  (Args used by the server)
from .objectives import Objective, centroid


def _chain_from_prefix(model, diffusion, batch, pre, Xn, Pn, *, window, S, full_window=False, churn_gen=None):
    """streaming_chain (pwm_final_clean inference.py) with the first c transitions already committed.
    pre = dict(X=[c x [B,N,3]], P=[c x [B,Kr,6]], pos=[c+1 x [B,N,3]], R=[c+1], T=[c+1]); Xn/Pn [B, n_new, ...] fresh noise.
    Returns positions [B, 1+c+n_new, N, 3] and the clean normalized rows X, P (lists) for committing.
    full_window: never truncate -- when fewer than `window` unfinished rows remain, slide the window start back so every model
    call sees Cm1 + window rows; the extra rows are already-committed transitions at level 0 (clean, never stepped)."""
    from utils.conditioning import hide_actions, model_metadata
    from utils.pose_native import _integrate_and_write, pose_channel_mask
    from inference import _stream_window, _stream_frame_positions
    base = model_metadata(model)
    C = batch.context_len; Cm1 = C - 1
    u = float(S / window)
    hide = bool(getattr(base, "_hides_given_actions", False))
    T_train = diffusion.num_train_timesteps
    top = float(T_train) if bool(getattr(base, "_sampler_full_clock", False)) else float(T_train - 1)
    c = len(pre["X"]); n_new = int(Pn.shape[1]); n_tr = c + n_new
    B, dev = batch.batch_size, batch.device
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    pmask = pose_channel_mask(batch, ridx); pmask = None if pmask is None else pmask.to(dev)
    Xs = list(pre["X"]) + [Xn[:, j] for j in range(n_new)]
    Ps = list(pre["P"]) + [Pn[:, j] if pmask is None else Pn[:, j] * pmask for j in range(n_new)]
    R, T, pos = list(pre["R"]), list(pre["T"]), list(pre["pos"])
    valid = batch.get_valid_mask()
    j_all = torch.arange(n_tr, dtype=torch.float64)

    def levels(call):   # committed rows level 0; the new staircase starts at transition c
        lv = (S + (j_all - c) * u - call).clamp(0.0, float(S)); lv[:c] = 0.0
        return lv.to(dev)

    committed, call = c, 0; rows_seen = []
    max_calls = S + int(np.ceil((n_new - 1) * u)) + 2
    while committed < n_tr:
        if call > max_calls:
            raise RuntimeError("prefix streaming schedule did not converge (bug)")
        lv, lv_next = levels(call), levels(call + 1)
        s = max(0, committed - Cm1)
        if full_window and n_tr - s < Cm1 + window:
            s = max(0, n_tr - (Cm1 + window))
        n_rows = min(Cm1 + window, n_tr - s)
        rows_seen.append(int(n_rows))
        win = _stream_window(batch, pos[s], R[s], T[s], ridx, s, n_rows)
        if hide:
            win = hide_actions(win, 1.0)
        lv_in = lv[s:s + n_rows]
        tb = (lv_in / S * top).float().unsqueeze(0).expand(B, -1)
        xc = torch.stack(Xs[s:s + Cm1], 1); xf = torch.stack(Xs[s + Cm1:s + n_rows], 1)
        pc = torch.stack(Ps[s:s + Cm1], 1); pf = torch.stack(Ps[s + Cm1:s + n_rows], 1)
        part_f, pose_f = model(win, xf, tb[:, Cm1:], noisy_context=xc, context_timesteps=tb[:, :Cm1], noisy_pose=pf, noisy_ctx_pose=pc)
        pose_c = getattr(base, "_last_ctx_pose_pred", None); part_c = getattr(base, "_last_ctx_pred", None)
        pred_pose = torch.cat([pose_c, pose_f], dim=1)
        has_part = part_f is not None and part_c is not None
        p_rows = torch.cat([pc, pf], 1); x_rows = torch.cat([xc, xf], 1)
        dt = ((lv_in - lv_next[s:s + n_rows]).clamp(min=0.0) / S * (top / T_train)).float()
        vp = diffusion.as_velocity(pred_pose, p_rows, tb)
        vx = diffusion.as_velocity(torch.cat([part_c, part_f], 1), x_rows, tb) if has_part else None
        if churn_gen is not None:                                    # full churn (eta = 1), as the guided churn sampler: re-form from x0 + a fresh draw
            x0p = diffusion.get_x0_est(pred_pose, tb, p_rows); tnr = (lv_next[s:s + n_rows] / S * (top / T_train)).float()
        for i in range(n_rows):
            if float(dt[i]) > 0:
                j = s + i
                if churn_gen is None:
                    new_p = Ps[j] - dt[i] * vp[:, i]
                else:
                    new_p = (1 - tnr[i]) * x0p[:, i] + tnr[i] * torch.randn(Ps[j].shape, device=Ps[j].device, generator=churn_gen, dtype=Ps[j].dtype)
                Ps[j] = new_p if pmask is None else new_p * pmask
                if vx is not None:
                    Xs[j] = Xs[j] - dt[i] * vx[:, i]
        call += 1
        while committed < n_tr and float(lv_next[committed]) <= 0.0:
            j = committed
            raw = batch.pose_delta_stats.denormalize(torch.stack(Ps[s:j + 1], 1).float()).float()
            dummy = torch.zeros((B, j + 1 - s, batch.positions.shape[2], 3), device=dev)
            _, R_all, T_all = _integrate_and_write(win, ridx, raw, R[s], T[s], dummy, valid)
            R.append(R_all[:, -1]); T.append(T_all[:, -1])
            pos.append(_stream_frame_positions(batch, ridx, R_all[:, -1], T_all[:, -1], pos[j], Xs[j]))
            committed += 1
    return dict(positions=torch.stack(pos, 1), X=Xs, P=Ps, R=R, T=T, calls=call, rows=rows_seen)


def handle(ad, base, req, lock, log):
    """POST /plan body: the /guide knobs (domain, task, select, episodes = the asset repeated B times, transitions,
    steps, offset, seed, name, desc) + K (candidates per replan, multiple of B) + commit (transitions per replan)."""
    A = Args(base, **{k: v for k, v in req.items() if k not in ("K", "commit", "full_window")})
    K, commit = int(req.get("K", 16)), int(req.get("commit", 5)); full = bool(req.get("full_window", False))
    if not A.name: A["name"] = time.strftime("plan_%Y%m%d_%H%M%S")
    with lock:
        return _plan(ad, A, K, commit, log, full)


def _plan(ad, A, K, commit, log, full=False):
    from inference import streaming_noise
    dev = torch.device("cuda")
    eps = [int(v) for v in A.episodes.split(",")]; B = len(eps)
    if K % B: raise ValueError(f"K={K} must be a multiple of the batch {B}")
    ad.select_path = A.get("select") or type(ad).select_path
    batch = ad.episodes(eps, A.transitions, domain=A.domain)
    mask = ad.object_mask(batch); c0 = centroid(batch.positions[:, 0], mask)
    off = torch.tensor([float(v) for v in A.offset.split(",")], device=dev)[None].expand(B, -1).clone()
    if A.task in ("push", "move"): off[:, 2] = 0
    target = c0 + off
    obj = Objective(target, mask, batch.get_valid_mask())
    n, S, W = int(A.transitions), int(A.steps), int(ad.cfg.data.future_len)
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    pre = dict(X=[], P=[], pos=[batch.positions[:, 0]], R=[batch.body_poses[:, 0, ridx, :3, :3].float()], T=[batch.body_poses[:, 0, ridx, :3, 3].float()])
    log(f"[plan] {A.name}: K {K} (chunks of {B}), commit {commit}, {n} transitions, S {S}, target mm {[round(float(v) * 1e3) for v in target[0]]}")
    replans, first_sample, t_all = [], None, time.time()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=ad.bf16):
        r = 0
        while len(pre["X"]) < n:
            c = len(pre["X"]); torch.cuda.synchronize(); t0 = time.time()
            best, losses, seeds = None, [], []
            K0 = int(A.get("K0", 0) or 0)                              # first replan only: an initial pool of K0 candidates (rounded up to chunks of B)
            for ch in range(-(-max(K0, K) // B) if r == 0 else K // B):
                sd = int(A.seed) * 1_000_003 + r * 1009 + ch; seeds.append(sd)
                g = torch.Generator(device=dev).manual_seed(sd); torch.manual_seed(sd)   # global RNG = the UNet's random FPS
                Xn, Pn = streaming_noise(ad.model, batch, n - c, g)
                out = _chain_from_prefix(ad.model, ad.diffusion, batch, pre, Xn, Pn, window=W, S=S, full_window=full, churn_gen=g if bool(A.get("churn")) else None)
                d, _ = obj(out["positions"].float()); d = d.float()
                losses += [round(v * 1e3, 2) for v in d.tolist()]
                if first_sample is None: first_sample = out["positions"][:1].float().clone()
                k = int(d.argmin())
                if best is None or float(d[k]) < best[0]:
                    best = (float(d[k]), ch, k, {key: [t[k:k + 1] for t in out[key]] for key in ("X", "P", "R", "T")} | dict(pos=out["positions"][k:k + 1].float()))
            m = min(commit, n - c); bl = best[3]
            for j in range(c, c + m):   # commit the winner's next m transitions, broadcast back to all B rows
                pre["X"].append(bl["X"][j].expand(B, *bl["X"][j].shape[1:]).clone()); pre["P"].append(bl["P"][j].expand(B, *bl["P"][j].shape[1:]).clone())
                pre["R"].append(bl["R"][j + 1].expand(B, *bl["R"][j + 1].shape[1:]).clone()); pre["T"].append(bl["T"][j + 1].expand(B, *bl["T"][j + 1].shape[1:]).clone())
                pre["pos"].append(bl["pos"][:, j + 1].expand(B, -1, -1).clone())
            torch.cuda.synchronize(); sec = round(time.time() - t0, 2)
            replans.append(dict(replan=r, committed_before=c, remaining=n - c, seconds=sec, best_mm=round(best[0] * 1e3, 2), best_chunk=best[1], best_row=best[2],
                                cand_mm=losses, seeds=seeds))
            log(f"[plan] replan {r:2d}: committed {c:2d} -> {c + m:2d}, {K} candidates over {n - c} transitions, best {best[0] * 1e3:.1f} mm "
                f"(median {np.median(losses):.1f}), {sec} s")
            r += 1
    plan = torch.stack(pre["pos"], 1)[:1].float()                  # [1, 1+n, N, 3]
    d_plan, _ = obj(plan.expand(B, -1, -1, -1)); d0, _ = obj(first_sample.expand(B, -1, -1, -1))
    total = round(time.time() - t_all, 1)
    log(f"[plan] done: committed plan {float(d_plan[0]) * 1e3:.1f} mm to target, first plain sample {float(d0[0]) * 1e3:.1f} mm, {total} s")
    out = os.path.join(A.runs, ad.domain, A.name); os.makedirs(out, exist_ok=True)
    ex = {k: v[:1] for k, v in ad.export(batch).items()}
    np.savez(os.path.join(out, "rollout.npz"), predicted=plan.cpu().numpy(), unguided=first_sample.cpu().numpy(), roles=ad.roles(batch)[:1],
             object_mask=mask[:1].cpu().numpy(), episodes=np.array(batch.episodes[:1]), target=target[:1].cpu().numpy(), offsets=off[:1].cpu().numpy(),
             boxes=np.zeros((0, 6), np.float32), trace_mm=np.array([[r_["best_mm"]] for r_ in replans], np.float32), step=np.array(ad.step), **ex)
    rj = dict({k: v for k, v in dict(A).items()}, method="mppi_argmin", K=K, commit=commit, full_window=full, adapter=ad.name, domain=ad.domain, sim=ad.sim, step=ad.step,
              frame_dt=ad.frame_dt, mode=ad.mode, best_mm=[round(float(d_plan[0]) * 1e3, 2)], unguided_mm=[round(float(d0[0]) * 1e3, 2)],
              iteration_seconds=[r_["seconds"] for r_ in replans], plan_seconds=total, replans=replans, episodes=list(batch.episodes[:1]),
              offsets=off[:1].cpu().numpy().tolist(), joint_limits_mm=None, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(rj, open(os.path.join(out, "result.json"), "w"), indent=1)
    return dict(name=A.name, out=out, best_mm=rj["best_mm"], unguided_mm=rj["unguided_mm"], seconds=total, iteration_seconds=rj["iteration_seconds"])
