"""Noise optimization for a context-generating, unconditional checkpoint (one window).

Every action is hidden; the sampler generates the C-1 context rows and the F future rows from
their own noise (guidance.sampler.sample_window, context_predicted path). Adam moves the POSE
noise of both row sets so that the materialized window meets an objective on the frame-0-anchored
trajectory; the particle noise is inert for entirely rigid scenes (rigid rows are rebuilt from
poses) and is left as drawn. Restart r = 0 iteration 0 is the unguided sample of that seed.
"""
from __future__ import annotations

import time

import torch

from guidance import objectives as objlib
from guidance.sampler import draw_noise, sample_window
from utils.conditioning import hide_actions


def sample_and_score(model, diffusion, hidden, noise, objective, *, steps, schedule, stream_keep, grad):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        x0p, x0pose = sample_window(model, diffusion, hidden, noise, steps=steps, schedule=schedule,
                                    stream_keep=stream_keep, use_checkpoint=True)
        pred = objlib.materialize(hidden, x0p, x0pose)
        loss, diag = objective(pred, hidden)
    return loss, diag, pred


def optimize_noise(model, diffusion, batch, objective, *, steps=25, schedule="full_sequence", stream_keep=None,
                   restarts=2, iterations=16, lr=0.1, prior_weight=1e-3, seed=54321, log=print):
    """Returns dict(best=..., unguided=..., trace=[...]) with materialized windows on CPU."""
    if batch.batch_size != 1:
        raise ValueError("optimize one window at a time (per-element selection)")
    hidden = hide_actions(batch, 1.0)
    device = batch.device
    best = None
    unguided = None
    trace = []
    total_nfe = 0
    for r in range(restarts):
        g = torch.Generator(device=device).manual_seed(seed + r)
        drawn = draw_noise(model, hidden, g)
        leaves = [drawn[1].detach().clone().requires_grad_(True), drawn[3].detach().clone().requires_grad_(True)]
        fixed = (drawn[0].detach(), drawn[2].detach())
        opt = torch.optim.Adam(leaves, lr=lr)
        for it in range(iterations + 1):
            t0 = time.time()
            noise = (fixed[0], leaves[0], fixed[1], leaves[1])
            grad = it < iterations
            loss, diag, pred = sample_and_score(model, diffusion, hidden, noise, objective, steps=steps,
                                                schedule=schedule, stream_keep=stream_keep, grad=grad)
            # the Gaussian prior on the optimized noise: keep its second moment at one
            prior = sum((z.pow(2).mean() - 1.0).abs() for z in leaves)
            total = loss + prior_weight * prior
            rec = dict(restart=r, iteration=it, loss=float(loss), prior=float(prior), **diag,
                       noise_rms=[float(z.detach().pow(2).mean().sqrt()) for z in leaves])
            if grad:
                opt.zero_grad(set_to_none=True)
                total.backward()
                rec["grad_norm"] = [float(z.grad.norm()) for z in leaves]
                opt.step()
            rec["seconds"] = time.time() - t0
            trace.append(rec)
            out = dict(positions=pred["positions"].detach().cpu(), t=pred["t"].detach().cpu(), loss=float(loss),
                       diag=diag, restart=r, iteration=it, noise=tuple(n.detach().cpu() for n in noise))
            if r == 0 and it == 0:
                unguided = out
            if best is None or out["loss"] < best["loss"]:
                best = out
            log(f"restart {r} it {it:2d} loss {float(loss):.5f} " +
                " ".join(f"{k} {v:.2f}" for k, v in diag.items() if k != "objective") +
                f"  {rec['seconds']:.1f}s" + (f"  grad {rec['grad_norm'][0]:.2e}/{rec['grad_norm'][1]:.2e}" if grad else ""))
    return dict(best=best, unguided=unguided, trace=trace, hidden_context_len=hidden.context_len)
