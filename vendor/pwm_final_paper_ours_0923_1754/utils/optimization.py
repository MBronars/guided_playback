"""A shared optimizer-step boundary for Accelerate/DDP training.

An accumulation group is accepted or rejected as a whole, on every rank.
Gradients are cleared AFTER that decision; clearing them before backward loses
earlier microbatches. Non-finite loss/gradients are rejected even with clipping or
spike thresholds disabled. Checkpoints should only be saved at sync boundaries.
"""
import math

import torch
import torch.distributed as dist


def backward_step(loss, model, optimizer, accelerator, *, grad_clip=None,
                  loss_ema=None, loss_spike_threshold=float("inf"),
                  grad_norm_spike_threshold=float("inf")):
    """Backward once; step only at a healthy, collectively agreed sync boundary.

    The total unified loss is checked for both finiteness and loss spikes.
    The optimizer owns only
    transient accumulation flags; its parameter/momentum state is not touched
    on rejected groups. Returns logging flags and the pre-clipping grad norm.
    """
    flags = list(getattr(optimizer, "_pwm_accumulation_flags", None) or (False,) * 3)
    value = float(loss.detach())
    flags[0] |= not math.isfinite(value)
    if loss_ema is not None:
        flags[0] |= not math.isfinite(float(loss_ema))
        flags[1] |= value > loss_spike_threshold * loss_ema

    accelerator.backward(loss)
    synchronized = bool(accelerator.sync_gradients)
    if not synchronized:
        optimizer._pwm_accumulation_flags = flags
        return {"skipped": any(flags), "optimizer_step": False,
                "sync_gradients": False, "nonfinite": flags[0], "grad_norm": None}

    # clip_grad_norm_ also unscales fp16 gradients through Accelerate. An infinite
    # bound measures the norm without clipping, so the no-clip path is still safe.
    bound = float("inf") if grad_clip is None else float(grad_clip)
    grad_norm = float(accelerator.clip_grad_norm_(model.parameters(), bound))
    flags[0] |= not math.isfinite(grad_norm)
    flags[2] |= grad_norm > grad_norm_spike_threshold
    agreed = torch.tensor(flags, dtype=torch.int32, device=loss.device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(agreed, op=dist.ReduceOp.MAX)
    flags = [bool(v) for v in agreed.tolist()]
    skipped = any(flags)
    if skipped:
        scaler = getattr(accelerator, "scaler", None)
        if scaler is not None:
            # Clear unscale bookkeeping without optimizer.step. Every rank makes
            # the SAME scale update, including ranks whose own gradients were
            # finite. A finite spike does not change the loss scale.
            scale = scaler.get_scale()
            if flags[0]:
                scale *= scaler.get_backoff_factor()
            scaler.update(new_scale=scale)
    else:
        optimizer.step()
        skipped = bool(getattr(optimizer, "step_was_skipped", False))

    optimizer.zero_grad(set_to_none=True)
    optimizer._pwm_accumulation_flags = None
    return {"skipped": skipped, "optimizer_step": not skipped,
            "sync_gradients": True, "nonfinite": flags[0], "grad_norm": grad_norm}
