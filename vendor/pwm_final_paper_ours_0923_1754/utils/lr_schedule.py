"""Learning-rate schedules counted in logical, synchronized updates."""
import math

import torch


def make_scheduler(optimizer, train_cfg, num_processes=1):
    """Compensate Accelerate's per-process scheduler stepping, without resume ticks.

    Construct once, then restore state. Constructing with a loaded last_epoch
    would execute an extra initial step and change the resumed learning rate.
    """
    kind = str(train_cfg.get("lr_schedule", "cosine"))
    steps = int(train_cfg["steps"])
    warmup = int(train_cfg.get("warmup_steps", 0) or 0)
    warmup_frac = float(train_cfg.get("wsd_warmup_frac", 0.05))
    decay_frac = float(train_cfg.get("wsd_decay_frac", 0.1))
    if kind not in ("cosine", "constant", "wsd") or steps <= 0 or num_processes < 1:
        raise ValueError("Invalid learning-rate schedule, step budget or process count")
    if kind == "wsd":
        if not 0 <= warmup_frac < 1 or not 0 < decay_frac <= 1 - warmup_frac:
            raise ValueError("WSD warmup/decay fractions must fit within the step budget")
        warmup = int(round(steps * warmup_frac))
    if not 0 <= warmup < steps:
        raise ValueError("warmup_steps must be nonnegative and below train.steps")
    horizon, ramp = steps * num_processes, warmup * num_processes
    if kind == "cosine" and ramp == 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=horizon, eta_min=float(train_cfg["lr"]) * 0.01)
    else:
        def factor(index):
            if ramp and index < ramp:
                return (index if kind == "wsd" else index + 1) / ramp
            if kind == "constant":
                return 1.0
            if kind == "wsd":
                decay_start = horizon * (1 - decay_frac)
                return 1.0 if index < decay_start else max(0.0, (horizon - index) / max(1.0, horizon - decay_start))
            progress = min((index - ramp) / max(1, horizon - ramp), 1.0)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scheduler._pwm_schedule = dict(kind=kind, steps=steps, warmup=warmup,
                                   decay_frac=decay_frac if kind == "wsd" else None,
                                   processes=num_processes, lr=float(train_cfg["lr"]))
    return scheduler


def validate_scheduler_state(scheduler, state):
    expected = getattr(scheduler, "_pwm_schedule", None)
    if expected is None:
        return
    recorded = state.get("_pwm_schedule")
    if recorded is not None and recorded != expected:
        raise RuntimeError(f"Resume schedule differs from checkpoint: saved={recorded}, requested={expected}")
    if recorded is None and "T_max" in state and state["T_max"] != expected["steps"] * expected["processes"]:
        raise RuntimeError("Legacy checkpoint cosine horizon differs from the configured logical schedule")
