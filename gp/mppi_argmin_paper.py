"""MPPI-argmin (no gradients) on the PAPER adapter's stream sampler (pwm_final_paper checkpoints: mould / pickup / rl_push).
Same loop as gp/mppi_argmin.py: per replan draw K fresh candidates of the remaining transitions from the committed prefix,
score each at the last frame, keep the argmin, commit its next `commit` transitions, repeat.

Sampler = paper_core.Streamer.rollout continued from a committed prefix: committed rows are clean (level 0), the window is
always the full NW = 19 rows (start = min(max(0, committed - keep), N - NW): it slides back at the end instead of shrinking,
exactly as the guided runs), full churn (eta = 1, fresh Gaussian at every re-forming, pose AND particle rows) drawn from a
per-chunk torch.Generator, fixed per-member FPS seeds. With churn off and no prefix it reproduces Streamer.rollout bitwise
(scratch/test_mould_prefix.py). Reloaded by the server on every /plan request."""
import json, os, time
import numpy as np
import torch
from .session import Args
from .objectives import Objective, centroid


def rollout_prefix(St, batch, pre, X_new, XP_new, *, steps, keep, gen=None, fps_seeds=None, churn=True):
    """Continue a stream rollout from `pre` = dict(rows, prows [c committed clean rows], pos_a, bp_a, a_idx [anchor]).
    X_new [B, n_new, P, 3], XP_new [B, n_new, Kr, 6]: initial noise of the new rows. churn: every re-forming uses a fresh
    draw from `gen` (x = x0 + l (eps - x0)); off = deterministic carry. Returns positions [B, 1 + c + n_new, P, 3], the
    final clean rows/prows, the anchor as it stood when the loop began (unchanged), and the call count."""
    from .adapters.paper_core import pose_channel_mask
    C, NW, dev = St.C, St.NW, St.dev
    B, n_new = X_new.shape[:2]
    c = len(pre["rows"]); N = c + n_new
    if N < NW:
        raise ValueError(f"need at least {NW} transitions in total")
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    pmask = pose_channel_mask(batch, ridx)
    pm = (lambda v: v * pmask.to(v)) if pmask is not None else (lambda v: v)
    rows = list(pre["rows"]) + list(X_new.unbind(1))
    prows = list(pre["prows"]) + [pm(v) for v in XP_new.unbind(1)]
    seeds = None if fps_seeds is None else tuple(int(v) for v in fps_seeds)
    m, D = NW - keep, steps * (NW - keep)
    sched = lambda j, call: 0.0 if j < c else min(D, max(0, D + (j - c) * steps - call * m)) / D
    lev, committed = [0.0] * c + [1.0] * n_new, c
    pos_a, bp_a, a_idx = pre["pos_a"], pre["bp_a"], int(pre["a_idx"])
    k = 0
    max_calls = steps + int(np.ceil(n_new * steps / m)) + 3
    while committed < N:
        if k > max_calls:
            raise RuntimeError("prefix stream schedule did not converge (bug)")
        start = min(max(0, committed - keep), N - NW)
        while a_idx < start:                                                # anchor = the window's first frame (committed rows only)
            pos_a, bp_a = St._step_anchor(batch, pos_a, bp_a, rows[a_idx], prows[a_idx]); a_idx += 1
        pos_w = pos_a[:, None].expand(-1, NW + 1, -1, -1)
        bp_w = bp_a[:, None].expand(-1, NW + 1, -1, -1, -1)
        wl = lev[start:start + NW]
        tt = torch.clamp(torch.round(torch.tensor(wl, device=dev) * St.T), 1, St.T - 1).long()[None].expand(B, -1)
        xs, xps = torch.stack(rows[start:start + NW], 1), torch.stack(prows[start:start + NW], 1)
        x0, x0p = St._call(batch, xs[:, :C - 1], xs[:, C - 1:], xps[:, :C - 1], xps[:, C - 1:], tt, pos_w, bp_w, start, k, seeds)
        for j in range(start, start + NW):
            l0, l1 = lev[j], min(lev[j], sched(j, k + 1))
            if l0 <= 0 or l1 == l0:
                continue
            jj = j - start
            if churn and l1 > 0:
                ep = torch.randn(xs.shape[0], *rows[j].shape[1:], device=dev, generator=gen)
                epp = torch.randn(xps.shape[0], *prows[j].shape[1:], device=dev, generator=gen)
                rows[j] = x0[:, jj] + l1 * (ep - x0[:, jj])
                prows[j] = pm(x0p[:, jj] + l1 * (pm(epp) - x0p[:, jj]))
            else:
                r = l1 / l0
                rows[j] = x0[:, jj] + r * (rows[j] - x0[:, jj])
                prows[j] = pm(x0p[:, jj] + r * (prows[j] - x0p[:, jj]))
            lev[j] = l1
        k += 1
        while committed < N and lev[committed] <= 0:
            committed += 1
    positions = St._materialize(batch, rows, prows)[0]
    return dict(positions=positions, rows=rows, prows=prows, calls=k)


def handle(ad, base, req, lock, log):
    """POST /plan on a paper-adapter server. Body: domain, task, episodes (the episode repeated B times), transitions, steps,
    keep, offset, seed, fix_fps, name, desc + K (multiple of B), commit, churn (default true)."""
    A = Args(base, **{k: v for k, v in req.items() if k not in ("K", "commit")})
    K, commit = int(req.get("K", 6)), int(req.get("commit", 5))
    if not A.name: A["name"] = time.strftime("plan_%Y%m%d_%H%M%S")
    if A.churn is None: A["churn"] = True
    with lock:
        return _plan(ad, A, K, commit, log)


def _plan(ad, A, K, commit, log):
    dev = torch.device("cuda"); St = ad.S
    eps = [int(v) for v in A.episodes.split(",")]; B = len(eps)
    if K % B: raise ValueError(f"K={K} must be a multiple of the batch {B}")
    batch = ad.episodes(eps, A.transitions, domain=A.domain)
    St.joint_limits = ad.joint_limits
    mask = ad.object_mask(batch); c0 = centroid(batch.positions[:, 0], mask)
    off = torch.tensor([float(v) for v in A.offset.split(",")], device=dev)[None].expand(B, -1).clone()
    if A.task in ("push", "move"): off[:, 2] = 0
    target = c0 + off
    obj = Objective(target, mask, batch.get_valid_mask())
    n, S, keep, P = int(A.transitions), int(A.steps), int(A.keep), batch.positions.shape[2]
    Kr = int((batch.body_type[0] == 1).sum()); churn = bool(A.churn)
    pre = dict(rows=[], prows=[], pos_a=batch.positions[:, 0], bp_a=batch.body_poses[:, 0], a_idx=0)
    log(f"[plan] {A.name}: paper/{ad.domain} K {K} (chunks of {B}), commit {commit}, {n} transitions, S {S}, keep {keep}, churn {churn}, "
        f"target mm {[round(float(v) * 1e3) for v in target[0]]}")
    replans, first_sample, t_all = [], None, time.time()
    with torch.no_grad():
        r = 0
        while len(pre["rows"]) < n:
            c = len(pre["rows"]); torch.cuda.synchronize(); t0 = time.time()
            best, losses, seeds = None, [], []
            K0 = int(A.get("K0", 0) or 0)                              # first replan only: an initial pool of K0 candidates (rounded up to chunks of B)
            for ch in range(-(-max(K0, K) // B) if r == 0 else K // B):
                sd = int(A.seed) * 1_000_003 + r * 1009 + ch; seeds.append(sd)
                g = torch.Generator(device=dev).manual_seed(sd)
                Xn = torch.randn(B, n - c, P, 3, device=dev, generator=g); XPn = torch.randn(B, n - c, Kr, 6, device=dev, generator=g)
                fps = [sd * 10 + i for i in range(B)] if A.fix_fps else None
                out = rollout_prefix(St, batch, pre, Xn, XPn, steps=S, keep=keep, gen=g, fps_seeds=fps, churn=churn)
                d, _ = obj(out["positions"].float()); d = d.float()
                losses += [round(v * 1e3, 2) for v in d.tolist()]
                if first_sample is None: first_sample = out["positions"][:1].float().clone()
                k = int(d.argmin())
                if best is None or float(d[k]) < best[0]:
                    best = (float(d[k]), ch, k, dict(rows=[t[k:k + 1] for t in out["rows"]], prows=[t[k:k + 1] for t in out["prows"]]))
            mcommit = min(commit, n - c); bl = best[3]
            for j in range(c, c + mcommit):                                        # commit the winner's next rows, broadcast to all B members
                pre["rows"].append(bl["rows"][j].expand(B, -1, -1).clone()); pre["prows"].append(bl["prows"][j].expand(B, -1, -1).clone())
            new_c = len(pre["rows"])
            while pre["a_idx"] < max(0, new_c - keep):                              # advance the shared anchor through the committed rows
                pre["pos_a"], pre["bp_a"] = St._step_anchor(batch, pre["pos_a"], pre["bp_a"], pre["rows"][pre["a_idx"]], pre["prows"][pre["a_idx"]])
                pre["a_idx"] += 1
            torch.cuda.synchronize(); sec = round(time.time() - t0, 2)
            replans.append(dict(replan=r, committed_before=c, remaining=n - c, seconds=sec, best_mm=round(best[0] * 1e3, 2), best_chunk=best[1],
                                best_row=best[2], cand_mm=losses, seeds=seeds, calls=int(out["calls"])))
            log(f"[plan] replan {r:2d}: committed {c:3d} -> {new_c:3d}, {K} candidates over {n - c} transitions ({out['calls']} calls each), "
                f"best {best[0] * 1e3:.1f} mm (median {np.median(losses):.1f}), {sec} s")
            r += 1
        plan = St._materialize(batch, pre["rows"], pre["prows"])[0][:1].float()
    d_plan, _ = obj(plan.expand(B, -1, -1, -1)); d0, _ = obj(first_sample.expand(B, -1, -1, -1))
    total = round(time.time() - t_all, 1)
    log(f"[plan] done: committed plan {float(d_plan[0]) * 1e3:.1f} mm to target, first plain sample {float(d0[0]) * 1e3:.1f} mm, {total} s")
    out_dir = os.path.join(A.runs, ad.domain, A.name); os.makedirs(out_dir, exist_ok=True)
    ex = {k: v[:1] for k, v in ad.export(batch).items()}
    np.savez(os.path.join(out_dir, "rollout.npz"), predicted=plan.cpu().numpy(), unguided=first_sample.cpu().numpy(), roles=ad.roles(batch)[:1],
             object_mask=mask[:1].cpu().numpy(), episodes=np.array([str(e) for e in batch.episodes[:1]]), target=target[:1].cpu().numpy(),
             offsets=off[:1].cpu().numpy(), boxes=np.zeros((0, 6), np.float32), trace_mm=np.array([[x["best_mm"]] for x in replans], np.float32),
             step=np.array(ad.step), **ex)
    rj = dict({k: v for k, v in dict(A).items()}, method="mppi_argmin", K=K, commit=commit, full_window=True, adapter=ad.name, domain=ad.domain,
              sim=ad.sim, step=ad.step, frame_dt=ad.frame_dt, mode=ad.mode, best_mm=[round(float(d_plan[0]) * 1e3, 2)],
              unguided_mm=[round(float(d0[0]) * 1e3, 2)], iteration_seconds=[x["seconds"] for x in replans], plan_seconds=total, replans=replans,
              episodes=list(batch.episodes[:1]), offsets=off[:1].cpu().numpy().tolist(),
              joint_limits_mm=None if ad.joint_limits is None else [v * 1e3 for v in ad.joint_limits], created=time.strftime("%Y-%m-%d %H:%M:%S"),
              sampler=f"paper stream, full {St.NW}-row windows, keep {keep}, S {S}, " + ("full churn (eta=1) fresh draws" if churn else "deterministic"))
    json.dump(rj, open(os.path.join(out_dir, "result.json"), "w"), indent=1)
    return dict(name=A.name, out=out_dir, best_mm=rj["best_mm"], unguided_mm=rj["unguided_mm"], seconds=total, iteration_seconds=rj["iteration_seconds"])
