"""Generic guided rollout: iterate (rollout with grad -> objective -> update the free draws), best iterate per member,
iteration 0 = the unguided rollout of the same draws, a final no-grad pass scores the last update.

Optimizers: adam (torch Adam on the draws, lr cosine-annealed to lr_final when given) or unitgrad (x -= lr * sqrt(D)
* g / |g| per (member, row): the push0418 / mould recipe). Shell: after each update re-project every draw onto its
original Gaussian-shell radius (off by default: the user's decision for pickup)."""
import json, math, os, time
import numpy as np
import torch
from .objectives import centroid


def _lr_at(it, iters, lr, lr_final):
    if lr_final is None or iters <= 1:
        return lr
    return lr_final + (lr - lr_final) * 0.5 * (1 + math.cos(math.pi * it / (iters - 1)))


def run(adapter, batch, state, objective, iterations, lr, lr_final=None, optimizer="adam", shell=False, log=print):
    B = batch.positions.shape[0]
    opt = torch.optim.Adam(state.vars, lr=lr) if optimizer == "adam" else None
    best_d = torch.full((B,), float("inf"), device=batch.positions.device); best = unguided = None
    trace, terms, times, peaks = [], [], [], []
    for it in range(iterations + 1):
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
        if it < iterations:
            pos = adapter.rollout(batch, state, grad=True)
        else:
            with torch.no_grad():
                pos = adapter.rollout(batch, state, grad=False)
        d, tm = objective(pos)
        if it == 0:
            unguided = pos.detach()
        better = d.detach() < best_d; best_d = torch.where(better, d.detach(), best_d)
        best = pos.detach() if best is None else torch.where(better[:, None, None, None], pos.detach(), best)
        trace.append([round(v, 2) for v in (tm["goal"].detach() * 1e3).tolist()])
        terms.append({k: [round(v, 3) for v in (t.detach() * 1e3).tolist()] for k, t in tm.items()} | dict(objective=[round(v, 2) for v in (d.detach() * 1e3).tolist()]))
        if it < iterations:
            cur_lr = _lr_at(it, iterations, lr, lr_final)
            if optimizer == "adam":
                for g in opt.param_groups: g["lr"] = cur_lr
                opt.zero_grad(); d.sum().backward()
                for v, free in zip(state.vars, state.free):
                    if free is not None and v.grad is not None: v.grad.mul_(free)
                opt.step()
            else:
                grads = torch.autograd.grad(d.sum(), state.vars, allow_unused=True)
                with torch.no_grad():
                    for v, g, free in zip(state.vars, grads, state.free):
                        if g is None: continue
                        D = v.shape[-1] * v.shape[-2]
                        step = cur_lr * (D ** 0.5) * g / g.flatten(-2).norm(dim=-1).clamp_min(1e-12)[..., None, None]
                        if free is not None: step = step * free
                        v.sub_(step)
            if shell:
                with torch.no_grad():
                    for v, r in zip(state.vars, state.shell):
                        if r is None: continue
                        n = v.flatten(-2).norm(dim=-1).clamp_min(1e-8)[..., None, None]; v.mul_(r / n)
        torch.cuda.synchronize(); times.append(round(time.time() - t0, 1)); peaks.append(round(torch.cuda.max_memory_allocated() / 2**30, 1))
        extra = "".join(f"  {k} mm {terms[-1][k]}" for k in ("box", "floor") if any(x > 0 for x in terms[-1][k]))
        log(f"it {it:2d}: object-to-target mm {trace[-1]}{extra}  {times[-1]} s  peak {peaks[-1]} GiB")
    return dict(predicted=best, unguided=unguided, best_mm=[round(v, 2) for v in (best_d * 1e3).tolist()], trace_mm=trace, terms=terms,
                iteration_seconds=times, peak_gib=peaks)


def save(out, adapter, batch, res, objective, meta):
    """rollout.npz + result.json (the contract every replay backend and the viewer read)."""
    os.makedirs(out, exist_ok=True)
    ex = adapter.export(batch)
    target = objective.target; off = meta.pop("offsets")
    np.savez(os.path.join(out, "rollout.npz"), predicted=res["predicted"].cpu().numpy(), unguided=res["unguided"].cpu().numpy(),
             roles=adapter.roles(batch), object_mask=adapter.object_mask(batch).cpu().numpy(), episodes=np.array(batch.episodes),
             target=target.cpu().numpy(), offsets=off, boxes=np.array([torch.cat([c, 2 * h]).cpu().numpy() for c, h in objective.boxes], np.float32).reshape(-1, 6),
             trace_mm=np.array(res["trace_mm"], np.float32), step=np.array(adapter.step), **ex)
    rj = dict(meta, adapter=adapter.name, domain=adapter.domain, sim=adapter.sim, step=adapter.step, frame_dt=adapter.frame_dt, mode=adapter.mode,
              best_mm=res["best_mm"], unguided_mm=res["trace_mm"][0], trace_mm=res["trace_mm"], terms=res["terms"], iteration_seconds=res["iteration_seconds"],
              peak_gib=res["peak_gib"], episodes=list(batch.episodes), joint_limits_mm=None if adapter.joint_limits is None else [v * 1e3 for v in adapter.joint_limits])
    json.dump(rj, open(os.path.join(out, "result.json"), "w"), indent=1)
    return rj
