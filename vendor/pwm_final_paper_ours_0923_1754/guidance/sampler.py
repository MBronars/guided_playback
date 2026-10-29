"""Differentiable single-window diffusion-forcing sampler.

Step for step the same chain as `inference.run_diffusion_inference_df` (same
scheduling matrix, timestep mapping, Euler update and clean-context contract),
without the no-grad guard: gradients reach the initial noise and every batch
field the backbone reads, in particular the actuator inputs. Each model call runs
under non-reentrant activation checkpointing when gradients are enabled, so the
memory of the chain is one forward pass plus the per-step states.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from diffusion.flow_matching import TrajectoryFlowMatching
from diffusion.scheduling import generate_scheduling_matrix
from utils.conditioning import clean_context, model_metadata


def pose_native(model, batch) -> bool:
    cfg = getattr(model_metadata(model), "cfg", None)
    return bool(getattr(cfg, "pose_native", False)) and getattr(batch, "pose_deltas_gt", None) is not None


def rigid_count(batch) -> int:
    return int((batch.body_type[0] == 1).sum())


def schedule_timesteps(diffusion, F: int, steps: int, schedule: str, uncertainty_scale: float, device,
                       stream_keep: int | None = None):
    """[M, F] integer training timesteps, exactly as the production sampler maps them (the stream
    staircase is fractional and ROUNDS to the training clock; the integer schedules truncate)."""
    if schedule == "stream" and stream_keep is None:
        raise ValueError("schedule='stream' needs stream_keep (the rows kept clean, c)")
    K = generate_scheduling_matrix(F, steps, schedule, uncertainty_scale, keep_clean=int(stream_keep or 0))
    K_float = K.astype(np.float64) / steps * diffusion.num_train_timesteps
    K_int = np.rint(K_float).astype(np.int64) if schedule == "stream" else K_float.astype(np.int64)
    K_timesteps = np.clip(K_int, 0, diffusion.num_train_timesteps - 1)
    return torch.from_numpy(K_timesteps).to(device)


def context_predicted(model, batch) -> bool:
    """The checkpoint denoises its context rows (train.pose_noise_context): the sampler generates
    them with the future, from their own noise, instead of pinning a clean history."""
    return bool(getattr(model_metadata(model), "_context_predicted", False)) and batch.context_len > 1


def draw_noise(model, batch, generator: torch.Generator | None = None):
    """Initial noise in the production sampler's draw order: particles, then poses."""
    B, F, N = batch.batch_size, batch.num_future_frames, batch.num_particles
    x_t = torch.randn((B, F, N, 3), device=batch.device, generator=generator)
    x_pose = None
    if pose_native(model, batch):
        x_pose = torch.randn((B, F, rigid_count(batch), 6), device=batch.device, generator=generator)
    if context_predicted(model, batch):
        # A context-predicting checkpoint: (x_t, x_pose, x_ctx, x_pose_ctx), the production
        # sampler's order, so the same generator gives the same window.
        C_ctx = batch.context_len - 1
        x_ctx = torch.randn((B, C_ctx, N, 3), device=batch.device, generator=generator)
        x_pose_ctx = (torch.randn((B, C_ctx, rigid_count(batch), 6), device=batch.device, generator=generator)
                      if x_pose is not None else None)
        return x_t, x_pose, x_ctx, x_pose_ctx
    return x_t, x_pose


def sample_window(model, diffusion, batch, noise, *, steps: int, schedule: str = "pyramid",
                  uncertainty_scale: float = 1.0, use_checkpoint: bool = True,
                  guidance_fn=None, guidance_scale: float = 5., guidance_threshold: float = .35,
                  guidance_max_rms: float = .03, guidance_trace=None,known_pose=None,known_mask=None,
                  stream_keep: int | None = None):
    """Denoise one window from `noise` = (x_t, x_pose or None).

    Returns (x0_particles [B, F, N, 3], x0_pose [B, F, Kr, 6] or None), both
    normalized. Under torch.no_grad this equals the production sampler bitwise
    for the same noise (tests/test_guidance_core.py).

    A context-predicting checkpoint (context_predicted) takes `noise` =
    (x_t, x_pose, x_ctx, x_pose_ctx) from draw_noise and denoises the C-1 context rows
    with the future under one schedule over C-1+F rows; it then returns
    (x0_particles [B, C-1+F, N, 3], x0_pose [B, C-1+F, Kr, 6]) -- the whole window as
    generated transitions from frame 0 (objectives.materialize integrates such an output
    from frame 0). Local pose guidance and inpainting are not ported to that path.
    """
    if not isinstance(diffusion, TrajectoryFlowMatching) or batch.prediction_target != "delta":
        raise ValueError("Guided sampling requires flow matching and delta targets")
    if steps < 1 or batch.num_future_frames < 1:
        raise ValueError("Sampling requires positive inference steps and future frames")
    B, F = batch.batch_size, batch.num_future_frames
    predict_ctx = context_predicted(model, batch)
    C_ctx = batch.context_len - 1
    if predict_ctx:
        if len(noise) != 4:
            raise ValueError("a context-predicting checkpoint needs (x_t, x_pose, x_ctx, x_pose_ctx) noise")
        if guidance_fn is not None or known_pose is not None:
            raise NotImplementedError("local guidance / inpainting are not ported to the context-generating sampler")
        if bool((batch.body_type[0] == 0).any()):
            raise ValueError("the context-generating sampler covers entirely rigid scenes")
        x_t, x_pose, x_ctx, x_pose_ctx = noise
    else:
        x_t, x_pose = noise[:2]
        x_ctx = x_pose_ctx = None
    native = pose_native(model, batch)
    if guidance_fn is not None and (not native or torch.is_grad_enabled()):
        raise ValueError('Local pose guidance requires pose-native sampling inside torch.no_grad')
    if guidance_fn is not None and (guidance_scale < 0 or guidance_threshold <= 0 or guidance_max_rms <= 0):
        raise ValueError('Invalid local-guidance settings')
    if native != (x_pose is not None):
        raise ValueError("Pose noise must be given exactly when the model is pose-native")
    if (known_pose is None)!=(known_mask is None):
        raise ValueError('Supply both clean known pose channels and their mask')
    if known_pose is not None:
        if not native or known_pose.shape!=x_pose.shape or not bool(torch.isfinite(known_pose).all()):
            raise ValueError('Known pose deltas must be finite and match the full window noise shape')
        if guidance_fn is not None:
            raise ValueError('Combined local guidance and inpainting has not been qualified')
        known_pose=known_pose.to(x_pose)
        known_mask=known_mask.to(device=x_pose.device,dtype=torch.bool).expand_as(x_pose)
        known_noise=x_pose
    pose_mask = None
    if native:
        from utils.pose_native import pose_channel_mask
        pose_mask = pose_channel_mask(batch, (batch.body_type[0] == 1).nonzero(as_tuple=True)[0])
        if pose_mask is not None:                       # jointed bodies: only their joint channel
            pose_mask = pose_mask.to(x_pose)
            if known_pose is not None and bool((known_mask & ~pose_mask.bool()).any()):
                raise ValueError('Cannot prescribe inactive articulation channels')
            x_pose = x_pose * pose_mask
            if x_pose_ctx is not None:
                x_pose_ctx = x_pose_ctx * pose_mask
    n_rows = C_ctx + F if predict_ctx else F
    K = schedule_timesteps(diffusion, n_rows, steps, schedule, uncertainty_scale, batch.device, stream_keep=stream_keep)
    noisy_context, context_timesteps = (None, None) if predict_ctx else clean_context(model, batch)
    grad = torch.is_grad_enabled() and use_checkpoint
    _base = model_metadata(model)

    def call(x_t, x_pose, t_batch, x_ctx=None, x_pose_ctx=None, t_ctx=None):
        if predict_ctx:
            # the context rows are inputs at their own levels; the wrapper stashes their pose
            # prediction, returned here as a THIRD output so activation checkpointing sees it
            pred, pose_pred = model(batch, x_t, t_batch, noisy_context=x_ctx, context_timesteps=t_ctx,
                                    noisy_pose=x_pose, noisy_ctx_pose=x_pose_ctx)
            return pred, pose_pred, _base._last_ctx_pose_pred
        if native:
            return model(batch, x_t, t_batch, noisy_context=noisy_context,
                         context_timesteps=context_timesteps, noisy_pose=x_pose)
        return model(batch, x_t, t_batch, noisy_context=noisy_context,
                     context_timesteps=context_timesteps), None

    for m in range(len(K) - 1):
        curr_k, next_k = K[m], K[m + 1]
        if (curr_k == next_k).all():
            continue
        if predict_ctx:
            t_ctx = curr_k[:C_ctx].unsqueeze(0).expand(B, -1).long()
            dt_c = ((curr_k[:C_ctx] - next_k[:C_ctx]).float() / diffusion.num_train_timesteps)[None, :, None, None]
            curr_k, next_k = curr_k[C_ctx:], next_k[C_ctx:]
            if grad:
                pred, pose_pred, ctx_pose_pred = checkpoint(call, x_t, x_pose, curr_k.unsqueeze(0).expand(B, -1).long(),
                                                            x_ctx, x_pose_ctx, t_ctx, use_reentrant=False)
            else:
                pred, pose_pred, ctx_pose_pred = call(x_t, x_pose, curr_k.unsqueeze(0).expand(B, -1).long(),
                                                      x_ctx, x_pose_ctx, t_ctx)
            t_batch = curr_k.unsqueeze(0).expand(B, -1).long()
            dt = ((curr_k - next_k).float() / diffusion.num_train_timesteps)[None, :, None, None]
            if pred is not None:
                x_t = x_t - dt * diffusion.as_velocity(pred, x_t, t_batch)
            x_pose = x_pose - dt * diffusion.as_velocity(pose_pred, x_pose, t_batch)
            # rigid context rows are rebuilt from the context POSE every forward: evolving that
            # pose is what moves them (x_ctx would carry deformable rows, excluded above)
            x_pose_ctx = x_pose_ctx - dt_c * diffusion.as_velocity(ctx_pose_pred, x_pose_ctx, t_ctx)
            if pose_mask is not None:
                x_pose = x_pose * pose_mask
                x_pose_ctx = x_pose_ctx * pose_mask
            continue
        t_batch = curr_k.unsqueeze(0).expand(B, -1).long()
        if known_pose is not None:
            marginal=diffusion.add_noise(known_pose,known_noise,t_batch)
            x_pose=torch.where(known_mask,marginal,x_pose)
        dt = ((curr_k - next_k).float() / diffusion.num_train_timesteps)[None, :, None, None]
        correction = None
        if guidance_fn is not None and guidance_scale > 0:
            # Accumulated normalized transition-noise proxy, not calibrated uncertainty.
            ready = (curr_k.float().div(diffusion.num_train_timesteps).square().cumsum(0).sqrt() <= guidance_threshold)
            active = ready & (curr_k > next_k)
        else:
            active = torch.zeros(F,dtype=torch.bool,device=batch.device)
        if bool(active.any()):
            with torch.enable_grad():
                variable = x_pose.detach().requires_grad_(True)
                pred, pose_pred = call(x_t.detach(),variable,t_batch)
                clean_particles = diffusion.get_x0_est(pred,t_batch,x_t.detach()) if pred is not None else x_t.detach()
                clean_pose = diffusion.get_x0_est(pose_pred,t_batch,variable)
                if pose_mask is not None: clean_pose = clean_pose*pose_mask
                loss = guidance_fn(clean_particles,clean_pose,int(ready.sum()))
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite clean-estimate guidance loss')
                gradient, = torch.autograd.grad(loss,variable)
                if not bool(torch.isfinite(gradient).all()): raise FloatingPointError('Nonfinite clean-estimate guidance gradient')
                mask = active[None,:,None,None].expand_as(variable)
                if pose_mask is not None: mask = mask & pose_mask.bool()
                correction = -guidance_scale*dt*gradient*mask
                rms = correction.masked_select(mask).square().mean().sqrt()
                correction = (correction*(guidance_max_rms/rms.clamp_min(1e-12)).clamp(max=1.)).detach()
                if guidance_trace is not None:
                    guidance_trace.append(dict(step=m,ready_frames=int(ready.sum()),loss=float(loss.detach()),
                        gradient_norm=float(gradient.norm()),correction_rms=float(correction.masked_select(mask).square().mean().sqrt())))
            pred = pred.detach() if pred is not None else None
            pose_pred = pose_pred.detach()
        elif grad:
            pred, pose_pred = checkpoint(call, x_t, x_pose, t_batch, use_reentrant=False)
        else:
            pred, pose_pred = call(x_t, x_pose, t_batch)
        if pred is not None:
            x_t = x_t - dt * diffusion.as_velocity(pred, x_t, t_batch)
        if native:
            x_pose = x_pose - dt * diffusion.as_velocity(pose_pred, x_pose, t_batch)
            if correction is not None: x_pose = x_pose+correction
            if pose_mask is not None:
                x_pose = x_pose * pose_mask
            if known_pose is not None:
                marginal=diffusion.add_noise(known_pose,known_noise,next_k[None].expand(B,-1))
                x_pose=torch.where(known_mask,marginal,x_pose)
    if predict_ctx:
        return torch.cat([x_ctx, x_t], dim=1), torch.cat([x_pose_ctx, x_pose], dim=1)
    return x_t, x_pose
