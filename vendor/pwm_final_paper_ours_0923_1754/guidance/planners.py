"""Planners: choose the actuator plan and/or the sampler's noise for one window.

Every planner has the signature
    planner(model, diffusion, batch, objective, *, steps, schedule, uncertainty_scale,
            generator, budget) -> PlanResult
and reports its cost as NFE (model forward passes). Gradient-free planners
(best_of_k, cem) never call backward and are the matched-compute controls for the
gradient planners (noise_opt, actuator_opt, joint_opt), which carry the settings
validated on the historical pushing experiments: Adam, lr 0.05 to 0.20, K=4
restarts, no terminal inpainting. Noise optimization changes all initial particle
and pose noise; actuator optimization changes supplied actuator inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from guidance import objectives as objlib
from guidance.plan import apply_plan, from_normalized, plan_from_batch, to_normalized
from guidance.sampler import draw_noise, sample_window, schedule_timesteps


@dataclass
class PlanResult:
    deltas: torch.Tensor                 # [B, F, Ka, 6] chosen actuator plan
    noise: tuple                         # (x_t, x_pose) that produced the chosen sample
    pred: dict                           # materialized sample of the chosen plan
    loss: float
    diagnostics: dict
    nfe: int
    trace: list = field(default_factory=list)   # per-iteration [{"iter", "loss", ...}]


def window_nfe(diffusion, batch, steps, schedule, uncertainty_scale, *, future_len: int | None = None) -> int:
    """Forward passes per window, optionally for a specified length instead of the batch's."""
    frames = batch.num_future_frames if future_len is None else future_len
    K = schedule_timesteps(diffusion, frames, steps, schedule, uncertainty_scale, "cpu")
    return int(sum(1 for m in range(len(K) - 1) if not bool((K[m] == K[m + 1]).all())))


def evaluate(model, diffusion, batch, deltas, noise, objective, *, steps, schedule, uncertainty_scale):
    guided = apply_plan(batch, deltas)
    x0p, x0pose = sample_window(model, diffusion, guided, noise, steps=steps, schedule=schedule,
                                uncertainty_scale=uncertainty_scale)
    pred = objlib.materialize(guided, x0p, x0pose)
    loss, diag = objective(pred, guided)
    return loss, diag, pred


def best_of_k(model, diffusion, batch, objective, *, steps, schedule="pyramid", uncertainty_scale=1.0,
              generator=None, budget=None, k: int = 4, deltas=None):
    """Sample K windows under the given (default: the batch's own) plan; keep the best."""
    deltas = plan_from_batch(batch) if deltas is None else deltas
    best, trace = None, []
    with torch.no_grad():
        for i in range(k):
            noise = draw_noise(model, batch, generator)
            loss, diag, pred = evaluate(model, diffusion, batch, deltas, noise, objective, steps=steps,
                                        schedule=schedule, uncertainty_scale=uncertainty_scale)
            trace.append({"iter": i, "loss": float(loss), **diag})
            if best is None or float(loss) < best.loss:
                best = PlanResult(deltas, noise, pred, float(loss), diag, 0)
    best.nfe = k * window_nfe(diffusion, batch, steps, schedule, uncertainty_scale)
    best.trace = trace
    return best


def _waypoints_to_deltas(waypoints: torch.Tensor, F: int) -> torch.Tensor:
    """waypoints [B, W, Ka, 6] cumulative pose offsets at W evenly spaced future
    frames (the last at frame F-1) -> per-transition deltas [B, F, Ka, 6] by linear
    interpolation of the cumulative track from zero."""
    B, W, Ka, _ = waypoints.shape
    knots = torch.linspace(0, F - 1, W + 1, device=waypoints.device)[1:]              # [W]
    frames = torch.arange(F, device=waypoints.device, dtype=waypoints.dtype)           # [F]
    track = torch.zeros(B, F, Ka, 6, device=waypoints.device, dtype=waypoints.dtype)
    prev_knot, prev_val = torch.zeros((), device=waypoints.device), torch.zeros(B, Ka, 6, device=waypoints.device, dtype=waypoints.dtype)
    for w in range(W):
        knot, val = knots[w], waypoints[:, w]
        span = (knot - prev_knot).clamp(min=1e-6)
        alpha = ((frames - prev_knot) / span).clamp(0, 1)                                 # [F]
        inside = (frames > prev_knot) & (frames <= knot)
        interp = prev_val[:, None] + alpha[None, :, None, None] * (val - prev_val)[:, None]
        track = torch.where(inside[None, :, None, None], interp, track)
        prev_knot, prev_val = knot, val
    zero = torch.zeros(B, 1, Ka, 6, device=waypoints.device, dtype=waypoints.dtype)
    return torch.diff(torch.cat([zero, track], dim=1), dim=1)


def cem(model, diffusion, batch, objective, *, steps, schedule="pyramid", uncertainty_scale=1.0,
        generator=None, budget=None, iterations: int = 4, population: int = 16, elites: int = 4,
        waypoints: int = 4, sigma_t: float = 0.02, sigma_r: float = 0.05, init_deltas=None):
    """Cross-entropy method over actuator waypoints. The plan is W cumulative SE(3)
    offsets per actuated body (metres, radians) at evenly spaced future frames,
    interpolated linearly to per-transition deltas. Population candidates are
    sampled around the elite mean; one sampler window per candidate with a fresh
    noise draw. Gradient-free: NFE = iterations * population windows."""
    B, F = batch.batch_size, batch.num_future_frames
    base = plan_from_batch(batch) if init_deltas is None else init_deltas
    Ka = base.shape[2]
    # initial mean: the base plan's cumulative track sampled at the waypoint frames
    knots = torch.linspace(0, F - 1, waypoints + 1, device=base.device)[1:].round().long()
    mean = base.cumsum(dim=1)[:, knots]                                                  # [B, W, Ka, 6]
    std = torch.cat([torch.full((3,), sigma_t), torch.full((3,), sigma_r)]).to(base).expand_as(mean).clone()
    best, trace = None, []
    with torch.no_grad():
        for it in range(iterations):
            cands = mean[None] + std[None] * torch.randn((population,) + mean.shape, device=base.device,
                                                          generator=generator, dtype=base.dtype)
            scores, results = [], []
            for c in range(population):
                deltas = _waypoints_to_deltas(cands[c], F)
                noise = draw_noise(model, batch, generator)
                loss, diag, pred = evaluate(model, diffusion, batch, deltas, noise, objective, steps=steps,
                                            schedule=schedule, uncertainty_scale=uncertainty_scale)
                scores.append(float(loss)); results.append((deltas, noise, pred, diag))
            order = sorted(range(population), key=lambda c: scores[c])
            elite = cands[order[:elites]]
            mean, std = elite.mean(dim=0), elite.std(dim=0, unbiased=False) + 1e-6
            top = order[0]
            trace.append({"iter": it, "loss": scores[top], "population_mean": sum(scores) / population, **results[top][3]})
            if best is None or scores[top] < best.loss:
                d, n, p, dg = results[top]
                best = PlanResult(d, n, p, scores[top], dg, 0)
    best.nfe = iterations * population * window_nfe(diffusion, batch, steps, schedule, uncertainty_scale)
    best.trace = trace
    return best


def _adam(model, diffusion, batch, objective, *, steps, schedule, uncertainty_scale, generator,
          optimize_noise: bool, optimize_plan: bool, restarts: int, iterations: int, lr: float,
          init_deltas=None):
    base = (plan_from_batch(batch) if init_deltas is None else init_deltas).detach()
    best, trace = None, []
    nfe_per = window_nfe(diffusion, batch, steps, schedule, uncertainty_scale)
    nfe = 0
    for r in range(restarts):
        x_t, x_pose = draw_noise(model, batch, generator)
        z = to_normalized(batch, base).clone()      # plan leaf lives in the stats' unit range
        leaves = []
        if optimize_noise:
            x_t.requires_grad_(); leaves.append(x_t)
            if x_pose is not None:
                x_pose.requires_grad_(); leaves.append(x_pose)
        if optimize_plan:
            z.requires_grad_(); leaves.append(z)
        opt = torch.optim.Adam(leaves, lr=lr)
        for it in range(iterations):
            opt.zero_grad(set_to_none=True)
            deltas = from_normalized(batch, z)
            loss, diag, pred = evaluate(model, diffusion, batch, deltas, (x_t, x_pose), objective, steps=steps,
                                        schedule=schedule, uncertainty_scale=uncertainty_scale)
            nfe += nfe_per                      # forward; the backward pass is extra compute, not NFE
            trace.append({"restart": r, "iter": it, "loss": float(loss.detach()), **diag})
            if best is None or float(loss.detach()) < best.loss:
                best = PlanResult(deltas.detach().clone(), (x_t.detach().clone(), None if x_pose is None else x_pose.detach().clone()),
                                  {k: (v.detach() if torch.is_tensor(v) else v) for k, v in pred.items()}, float(loss.detach()), diag, 0)
            loss.backward()
            opt.step()
    best.nfe = nfe
    best.trace = trace
    return best


def noise_opt(model, diffusion, batch, objective, *, steps, schedule="pyramid", uncertainty_scale=1.0,
              generator=None, budget=None, restarts: int = 4, iterations: int = 10, lr: float = 0.1, init_deltas=None):
    """Adam over the initial noise; the actuator plan stays as given."""
    return _adam(model, diffusion, batch, objective, steps=steps, schedule=schedule, uncertainty_scale=uncertainty_scale,
                 generator=generator, optimize_noise=True, optimize_plan=False, restarts=restarts,
                 iterations=iterations, lr=lr, init_deltas=init_deltas)


def actuator_opt(model, diffusion, batch, objective, *, steps, schedule="pyramid", uncertainty_scale=1.0,
                 generator=None, budget=None, restarts: int = 1, iterations: int = 10, lr: float = 0.05, init_deltas=None):
    """Adam over the actuator plan in normalized pose-delta units; noise fixed per restart."""
    return _adam(model, diffusion, batch, objective, steps=steps, schedule=schedule, uncertainty_scale=uncertainty_scale,
                 generator=generator, optimize_noise=False, optimize_plan=True, restarts=restarts,
                 iterations=iterations, lr=lr, init_deltas=init_deltas)


def joint_opt(model, diffusion, batch, objective, *, steps, schedule="pyramid", uncertainty_scale=1.0,
              generator=None, budget=None, restarts: int = 1, iterations: int = 10, lr: float = 0.05, init_deltas=None):
    """Adam over both the noise and the normalized actuator plan (the historical OOD winner)."""
    return _adam(model, diffusion, batch, objective, steps=steps, schedule=schedule, uncertainty_scale=uncertainty_scale,
                 generator=generator, optimize_noise=True, optimize_plan=True, restarts=restarts,
                 iterations=iterations, lr=lr, init_deltas=init_deltas)


PLANNERS = {"best_of_k": best_of_k, "cem": cem, "noise_opt": noise_opt,
            "actuator_opt": actuator_opt, "joint_opt": joint_opt}
