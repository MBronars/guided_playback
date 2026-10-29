"""Deterministic loss measurement for overfit diagnostics.

The training loss is sampled at a random diffusion timestep, and for velocity-parameterized
flow matching the loss spans ~370x across t (0.0007 at t=0.95, 0.27 at t=0.02). One logged
step is therefore one draw from a distribution wider than any effect we compare across arms:
measured point-to-point scatter on the round-2 logs was 3.4-38x while the between-arm
difference was 1.2-3.2x. Reading a floor or a plateau off that series is not possible.

`fixed_t_loss` removes the sampling: fixed batch, fixed t grid, fixed noise seeds. Re-running
returns the same numbers, so a difference between two runs is a real difference. `tail`
summarises a training history with an estimator instead of its last point.
"""
import numpy as np
import torch

T_GRID = (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95)
T_LARGE = 0.5           # gate on t >= this: the region free of the 1/t tail
NOISE_SEED = 1000


@torch.no_grad()
def fixed_t_loss(model, batch, diffusion, grid=T_GRID, draws=6, *, loss_options=None):
    """The production objective on fixed noise seeds and fixed normalized times.

    All noised frames (including context when enabled) use the selected t. Masks,
    sample/velocity heads, rigid pose terms and auxiliary weights are exactly those
    of trajectory_objective. This is a controlled slice, not an estimate of the
    training timestep distribution. Caller mode and RNG are unchanged.
    """
    import copy
    from diffusion import TrajectoryFlowMatching
    from train import trajectory_objective

    if not isinstance(diffusion, TrajectoryFlowMatching):
        raise TypeError("fixed_t_loss requires TrajectoryFlowMatching")
    if draws < 1 or not grid or any(not 0 < t <= 1 for t in grid):
        raise ValueError("Need positive draws and a nonempty grid in (0, 1]")
    fixed = copy.copy(diffusion)
    options = dict(diffusion_forcing=True)
    options.update(loss_options or {})
    device = batch.device
    was_training = model.training
    model.eval()
    out = {}
    try:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            for t in grid:
                def times(batch_size, device, num_frames=None, num_objects=None, low=0):
                    shape = (batch_size,)
                    if num_frames is not None:
                        shape += (num_frames,)
                        if num_objects is not None:
                            shape += (num_objects,)
                    return torch.full(shape, t * diffusion.num_train_timesteps, device=device)

                fixed.sample_timesteps = times
                losses = []
                for draw in range(draws):
                    # Seed only the device whose RNG is forked, not every visible GPU.
                    torch.random.default_generator.manual_seed(NOISE_SEED + draw)
                    if device.type == "cuda":
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(NOISE_SEED + draw)
                    loss, _, _ = trajectory_objective(model, batch, fixed, **options)
                    losses.append(float(loss))
                if not np.isfinite(losses).all():
                    raise RuntimeError(f"Non-finite fixed-t probe at t={t}")
                out[float(t)] = (float(np.mean(losses)), float(np.std(losses)))
    finally:
        model.train(was_training)
    return out


def large_t_loss(sweep, t_min=T_LARGE):
    """Mean fixed-t loss over t >= t_min -- the headline overfit number."""
    v = [m for t, (m, _) in sweep.items() if t >= t_min]
    return float(np.mean(v))


def tail(hist, frac=3):
    """Estimator + noise floor + trend for the last 1/`frac` of a loss history."""
    h = np.asarray(hist, dtype=float)
    w = h[len(h) - len(h) // frac:]
    lo = max(float(w.min()), 1e-12)
    x = np.arange(len(w), dtype=float)
    slope = float(np.polyfit(x, np.log10(np.maximum(w, 1e-12)), 1)[0] * 1000)
    return {"n": int(len(w)), "mean": float(w.mean()), "median": float(np.median(w)),
            "min": lo, "max": float(w.max()), "scatter": float(w.max()) / lo,
            "slope_dex_per_1k": slope, "still_descending": slope < -0.02}


def guard(name, hist, min_points=201):
    """Reject a loss history that would silently produce a misleading chart.

    Catches the failure that put `0.0000` on a log axis: `print(f"{loss:.4f}")` truncated
    5e-6 to zero, a parser read 0.0, and the plot floored it to 1e-7 and drew a flat line
    at the bottom of the panel. Nothing along that path complained.
    """
    h = np.asarray(hist, dtype=float)
    if h.size < min_points:
        raise ValueError(f"{name}: {h.size} points, need >= {min_points} for the estimator")
    if not np.isfinite(h).all():
        raise ValueError(f"{name}: non-finite values")
    if (h <= 0).any():
        raise ValueError(f"{name}: {(h <= 0).sum()} non-positive values -- a printf format "
                         f"or a plot floor has replaced the data on the way to the chart")
    return h
