"""Clean particle-context conditioning shared by every sampling entrypoint."""
import dataclasses

import torch


def model_metadata(model):
    """Read provenance through DDP/compiled wrappers without bypassing forward.

    Callers must still execute the original model: its wrapper may own autocast,
    output conversion, distributed synchronization or compilation.
    """
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        wrapped = getattr(model, "module", None)
        if wrapped is None:
            wrapped = getattr(model, "_orig_mod", None)
        if wrapped is None:
            break
        model = wrapped
    return model


def clean_context(model, batch, *, default_timesteps=True):
    """Return normalized history and its trained timestep representation.

    None requests the backbone default: a literal zero embedding in the
    Transformer, timestep zero in the UNet. 0 is clean DF history, and 1 is the
    explicit clean_ctx_timesteps recipe. Preserve the trained convention. Models
    built by models.factory carry this provenance; the fallback preserves the
    historical direct-construction API used by standalone tests/evaluators.
    """
    base = model_metadata(model)
    if not getattr(getattr(base, "cfg", None), "all_relative", False):
        return None, None
    context = batch.delta_stats.normalize(batch.context_deltas)
    fallback = 0 if getattr(base, "_ctx_timesteps_trained", default_timesteps) else None
    timestep = getattr(base, "_clean_context_timestep", fallback)
    times = (None if timestep is None else torch.full(
        (batch.batch_size, batch.context_len - 1), timestep, device=batch.device, dtype=torch.long))
    return context, times


def hide_actions(batch, probability):
    """Replace the actuated bodies' given future motion by the learned no-action input.

    Per batch element: probability 0 returns the batch itself, 1 hides every element
    (the model must generate the actuator), and anything in between draws one
    Bernoulli per element from the ambient generator. Only the pose-token action
    stream is hidden (`pose_actions`, `pose_action_mask`): the per-particle `actions`
    stream stays, so this is a complete hiding only for models that do not read it
    (the UNet with unet_actions=false, which utils/config.py enforces for
    train.action_dropout). Body identity (geometry, joint type, material) is not
    future information and is kept.
    """
    if probability <= 0 or batch.pose_action_mask is None:
        return batch
    mask = batch.pose_action_mask
    if probability >= 1:
        keep = torch.zeros(batch.batch_size, dtype=torch.bool, device=mask.device)
    else:
        keep = torch.rand(batch.batch_size, device=mask.device) >= probability
    return dataclasses.replace(
        batch, pose_action_mask=mask & keep[:, None],
        pose_actions=batch.pose_actions * keep[:, None, None, None].to(batch.pose_actions.dtype))
