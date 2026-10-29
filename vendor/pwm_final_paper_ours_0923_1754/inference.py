"""Shared unified flow-matching samplers and rollouts.

windowed_rollout is the rollout every evaluator uses (docs/EVALUATION.md).
autoregressive_rollout keeps the strided chunks of the paper MPM protocol and
dispatches to streaming_rollout, a different protocol again.
Historical standalone/guided inference lives at the pre-unified-only revision.
"""

from typing import Dict, List
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from data.test_batch import LongTrajectoryBatch
from data.batch import TrajectoryBatch
from diffusion import TrajectoryFlowMatching


@dataclass
class AutoregressiveConfig:
    """Configuration for autoregressive rollout."""
    context_len: int = 2          # Number of context frames (C)
    future_len: int = 22          # Number of future frames per step (F) - matches T=24, C=2
    num_inference_steps: int = 25 # Diffusion/flow inference steps
    diffusion_forcing: bool = True  # Use per-frame scheduling (DF)
    df_schedule: str = "pyramid"     # DF schedule: "full_sequence" or "pyramid"
    df_uncertainty_scale: float = 1.0  # Controls pyramid slope
    stride: int = -1              # Frames to advance per step (-1 = future_len, no overlap)
    streaming: bool = False       # True = rolling buffer (no wasted work), requires diffusion_forcing=True


@torch.no_grad()
def autoregressive_rollout(
    model: nn.Module,
    diffusion,
    sequence_data: LongTrajectoryBatch,
    config: AutoregressiveConfig,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Strided autoregression on a long sequence: the paper Table-1 MPM protocol.

    Chunks of `stride` frames, and the last chunk is SHORTER than the trained
    window. New checkpoints are evaluated with windowed_rollout, whose windows all
    keep the trained length; the two are not interchangeable past the first window.
    Dispatches to streaming_rollout when config.streaming=True.

    Args:
        model: Trained TrajectoryTransformer
        diffusion: Diffusion/FlowMatching process
        sequence_data: Long sequence data (e.g., 96 frames)
        config: Autoregressive configuration
        verbose: Print progress

    Returns:
        dict with:
            - predicted_positions: [B, T, N, 3] full predicted trajectory
            - gt_positions: [B, T, N, 3] ground truth
            - per_frame_mse: [T-C] MSE for each future frame
    """
    if not config.diffusion_forcing or not isinstance(diffusion, TrajectoryFlowMatching):
        raise ValueError("Unified autoregression requires flow matching with diffusion forcing")
    if config.context_len < 2 or config.future_len < 1 or config.context_len >= sequence_data.num_frames:
        raise ValueError("Invalid autoregressive context or future window")
    if config.num_inference_steps < 1 or config.stride == 0 or config.stride > config.future_len:
        raise ValueError("Invalid inference step count or autoregressive stride")
    if config.streaming:
        return streaming_rollout(model, diffusion, sequence_data, config, verbose)

    B = sequence_data.batch_size
    T = sequence_data.num_frames
    N = sequence_data.num_particles
    device = sequence_data.device

    C = config.context_len
    F = config.future_len
    stride = config.stride if config.stride > 0 else F  # -1 = no overlap

    # Initialize predicted positions with ground truth context
    predicted_positions = torch.zeros(B, T, N, 3, device=device)
    predicted_positions[:, :C] = sequence_data.positions[:, :C]

    # Track per-frame MSE
    per_frame_mse = torch.zeros(T - C)

    current_frame = C  # Next frame to predict

    step = 0
    while current_frame < T:
        # Determine how many frames to predict this step
        frames_this_step = min(F, T - current_frame)

        if verbose:
            print(f"Step {step}: predicting frames [{current_frame}, {current_frame + frames_this_step - 1}]")

        # Get window for this step
        # Context comes from predicted positions (autoregressive)
        context_start = current_frame - C
        batch = sequence_data.get_window(
            start_frame=context_start,
            context_len=C,
            future_len=frames_this_step,
            predicted_positions=predicted_positions,
        )

        # Run diffusion inference
        pred_deltas = run_diffusion_inference_df(
            model=model,
            batch=batch,
            diffusion=diffusion,
            num_inference_steps=config.num_inference_steps,
            schedule=config.df_schedule,
            uncertainty_scale=config.df_uncertainty_scale,
        )
        # Convert normalized output to positions
        pred_raw = batch.prediction_stats.denormalize(pred_deltas)
        pred_positions = batch.predictions_to_positions(pred_raw)  # [B, F, N, 3]

        # Only keep first `stride` frames (or fewer at end of sequence)
        keep = min(stride, frames_this_step)

        # Store predictions
        predicted_positions[:, current_frame:current_frame + keep] = pred_positions[:, :keep]

        # Compute per-frame MSE (on valid particles)
        gt_positions = sequence_data.positions[:, current_frame:current_frame + keep]
        valid_mask = sequence_data.get_valid_mask()  # [B, N]
        for f_idx in range(keep):
            frame_valid = valid_mask[:, :, None].expand(B, N, 3)
            frame_mse = ((pred_positions[:, f_idx] - gt_positions[:, f_idx]) ** 2)[frame_valid].mean().item()
            per_frame_mse[current_frame - C + f_idx] = frame_mse

        if verbose:
            avg_mse = per_frame_mse[current_frame - C:current_frame - C + keep].mean().item()
            print(f"  MSE (kept {keep} frames): {avg_mse:.6f}")

        # Move to next window
        current_frame += keep
        step += 1

    return {
        'predicted_positions': predicted_positions,
        'gt_positions': sequence_data.positions,
        'per_frame_mse': per_frame_mse.tolist(),
    }


@torch.no_grad()
def streaming_rollout(
    model: nn.Module,
    diffusion,
    sequence_data: LongTrajectoryBatch,
    config: AutoregressiveConfig,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Streaming inference with staggered rolling buffer.

    Buffer is initialized with pyramid-staggered noise levels so that
    frame[0] finishes first, then frame[1], etc. — one commit per
    ~scale steps. After commit, the buffer shifts and a new frame
    enters at max noise, maintaining the clean→noisy gradient.

    Args:
        model: TrajectoryTransformer
        diffusion: Diffusion/FlowMatching process
        sequence_data: Long sequence data
        config: Autoregressive configuration (future_len = buffer size)

    Note: Only supports prediction_target='delta'. Anchor_offset predictions
    become stale as the context window slides; position mode is untested.
    """
    assert sequence_data.prediction_target == "delta", (
        f"Streaming inference only supports prediction_target='delta', "
        f"got '{sequence_data.prediction_target}'"
    )

    B = sequence_data.batch_size
    T = sequence_data.num_frames
    N = sequence_data.num_particles
    device = sequence_data.device

    C = config.context_len
    F = min(config.future_len, T - C)  # buffer size: cap at actual future frames
    S = config.num_inference_steps
    scale = config.df_uncertainty_scale

    num_train_ts = diffusion.num_train_timesteps

    # Output
    predicted_positions = torch.zeros(B, T, N, 3, device=device)
    predicted_positions[:, :C] = sequence_data.positions[:, :C]
    per_frame_mse = torch.zeros(T - C)

    # Buffer: F frames with STAGGERED noise (pyramid initialization)
    # frame[i] starts at noise = S + int(i * scale)
    # frame[0] starts at S, frame[1] at S + scale, etc.
    # This means frame[0] reaches 0 first (after S steps),
    # frame[1] reaches 0 ~scale steps later, etc.
    buffer = torch.randn(B, F, N, 3, device=device)
    buffer_noise = torch.tensor(
        [S + int(i * scale) for i in range(F)],
        dtype=torch.long, device=device,
    )

    next_commit = C  # next frame index to commit

    if verbose:
        print(f"Streaming: {T-C} frames, buffer={F}, S={S}, scale={scale}")
        print(f"  Initial buffer_noise: {buffer_noise.tolist()}")
        print(f"  Buffer covers frames [{C}, {C + F - 1}]")

    step = 0
    while next_commit < T and buffer_noise.shape[0] > 0:
        # ── Phase 1: Commit any clean frames ──
        while len(buffer_noise) > 0 and buffer_noise[0] == 0 and next_commit < T:
            committed_delta_norm = buffer[:, 0:1]  # [B, 1, N, 3]

            # Reconstruct position
            committed_raw = sequence_data.delta_stats.denormalize(committed_delta_norm)
            anchor_pos = predicted_positions[:, next_commit - 1]
            predicted_positions[:, next_commit] = anchor_pos + committed_raw[:, 0]

            # Per-frame MSE
            gt_pos = sequence_data.positions[:, next_commit]
            valid_mask = sequence_data.get_valid_mask()
            frame_mse = (
                (predicted_positions[:, next_commit] - gt_pos) ** 2
                * valid_mask[:, :, None].float()
            ).sum() / valid_mask.sum().clamp(min=1) / 3
            per_frame_mse[next_commit - C] = frame_mse.item()

            if verbose:
                print(f"  step {step:3d}: COMMIT frame {next_commit}, "
                      f"MSE={frame_mse.item():.6f}")

            # Shift buffer left, add new frame at max staggered noise
            remaining = buffer[:, 1:]
            remaining_noise = buffer_noise[1:]
            if next_commit + F < T:
                new_noise = S + int((F - 1) * scale)
                buffer = torch.cat([
                    remaining,
                    torch.randn(B, 1, N, 3, device=device),
                ], dim=1)
                buffer_noise = torch.cat([
                    remaining_noise,
                    torch.tensor([new_noise], device=device),
                ])
            else:
                buffer = remaining
                buffer_noise = remaining_noise

            next_commit += 1

        # Done?
        if next_commit >= T or buffer.shape[1] == 0:
            break

        # ── Phase 2: Run model ──
        actual_F = buffer.shape[1]
        ctx_start = next_commit - C

        batch = sequence_data.get_window(
            start_frame=ctx_start,
            context_len=C,
            future_len=min(actual_F, T - next_commit),
            predicted_positions=predicted_positions,
        )

        # Map scheduling noise levels to training timesteps (clamp to valid range)
        curr_t = (buffer_noise[:actual_F].float() / S * num_train_ts).clamp(
            0, num_train_ts - 1
        ).long()
        next_noise = (buffer_noise[:actual_F] - 1).clamp(min=0)
        next_t = (next_noise.float() / S * num_train_ts).clamp(
            0, num_train_ts - 1
        ).long()

        t_batch = curr_t.unsqueeze(0).expand(B, -1)  # [B, actual_F]

        # Prepare context for all_relative models
        from utils.conditioning import clean_context
        noisy_context, context_timesteps_clean = clean_context(model, batch)

        pred = model(batch, buffer[:, :actual_F], t_batch,
                     noisy_context=noisy_context, context_timesteps=context_timesteps_clean)

        # ── Phase 3: Denoise each frame by one step ──
        update_mask = (buffer_noise[:actual_F] > 0)

        dt = (curr_t - next_t).float() / num_train_ts
        dt = dt[None, :, None, None]
        v = diffusion.as_velocity(pred, buffer[:, :actual_F], t_batch)
        new_vals = buffer[:, :actual_F] - dt * v
        mask = update_mask[None, :, None, None]
        buffer[:, :actual_F] = torch.where(mask, new_vals, buffer[:, :actual_F])
        buffer_noise[:actual_F] = next_noise

        if verbose:
            noise_str = "[" + ", ".join(f"{n:2d}" for n in buffer_noise.tolist()) + "]"
            frames_end = next_commit + buffer_noise.shape[0] - 1
            print(f"  step {step:3d}: denoise → noise={noise_str}  "
                  f"[frames {next_commit}..{frames_end}]")

        step += 1

    return {
        'predicted_positions': predicted_positions,
        'gt_positions': sequence_data.positions,
        'per_frame_mse': per_frame_mse.tolist(),
    }


@torch.no_grad()
def run_diffusion_inference_df(
    model: nn.Module,
    batch: TrajectoryBatch,
    diffusion,
    num_inference_steps: int = 25,
    schedule: str = "pyramid",
    uncertainty_scale: float = 1.0,
    stream_keep: int | None = None,
) -> torch.Tensor:
    """
    Run inference with per-frame scheduling (Diffusion Forcing).

    Uses manual per-frame steps since the built-in scheduler.step()
    cannot handle per-frame [B, F] timesteps.

    Flow matching uses Euler updates under the checkpoint's time convention.

    Args:
        model: TrajectoryTransformer
        batch: Input batch with context, actions, etc.
        diffusion: TrajectoryFlowMatching
        num_inference_steps: Number of denoising steps
        schedule: "full_sequence", "pyramid" or "stream"
        uncertainty_scale: Controls pyramid slope
        stream_keep: schedule="stream" only -- the rows a streaming rollout keeps clean at the
            window front (c). This matrix is ONE window of that rollout, so it is the sampler a
            stream-trained checkpoint should be judged by; pyramid is a different staircase

    Returns:
        predicted_deltas: [B, F, N, 3] normalized deltas
    """
    from diffusion.scheduling import generate_scheduling_matrix

    B = batch.batch_size
    F = batch.num_future_frames
    N = batch.num_particles
    device = batch.device

    if not isinstance(diffusion, TrajectoryFlowMatching) or batch.prediction_target != "delta":
        raise ValueError("Unified sampling requires flow matching and delta targets")
    if num_inference_steps < 1 or F < 1:
        raise ValueError("Sampling requires positive inference steps and future frames")
    from utils.conditioning import clean_context, model_metadata
    _base = model_metadata(model)
    C_ctx = batch.context_len - 1
    # Context handling follows the CHECKPOINT (models.factory.context_predicted): a model
    # trained only to READ its history keeps it pinned clean while the future denoises; one
    # trained to PREDICT its context (train.pose_noise_context) has never seen a clean context
    # beside a noisy future, so the ladder spans the whole [context | future] window and the
    # context rows are generated too.
    predict_ctx = bool(getattr(_base, "_context_predicted", False)) and C_ctx > 0
    n_rows = C_ctx + F if predict_ctx else F
    # Generate scheduling matrix
    if schedule == "stream" and stream_keep is None:
        raise ValueError("schedule='stream' needs stream_keep (the rows kept clean, c)")
    K = generate_scheduling_matrix(n_rows, num_inference_steps, schedule, uncertainty_scale,
                                   keep_clean=int(stream_keep or 0))
    # K: [M, n_rows] integer noise levels in [0, num_inference_steps]

    # Map to training timestep integers
    K_float = K.astype(np.float64) / num_inference_steps * diffusion.num_train_timesteps
    # The stream matrix is fractional (S/m is not an integer), so it ROUNDS to the training clock;
    # truncating it, as the integer schedules do, would bias every level 1/T toward clean.
    K_timesteps = np.clip(np.rint(K_float).astype(np.int64) if schedule == "stream"
                          else K_float.astype(np.int64), 0, diffusion.num_train_timesteps - 1)
    K_timesteps = torch.from_numpy(K_timesteps).to(device)

    # Start from pure noise
    x_t = torch.randn((B, F, N, 3), device=device)

    # Tier-2 v2 (docs/tier2_v2_design.md): pose channels jointly denoised under
    # the SAME per-frame schedule. Returned particle x0's RIGID rows are
    # meaningless under pose_native — consumers rebuild them as
    # template @ integrate(x0_pose). Returns (x0_particles, x0_pose) then.
    pose_native = (getattr(getattr(_base, "cfg", None), "pose_native", False)
                   and getattr(batch, "pose_deltas_gt", None) is not None)
    x_pose = pose_mask = x_ctx = x_pose_ctx = None
    if pose_native:
        Kr = int((batch.body_type[0] == 1).sum())
        x_pose = torch.randn((B, F, Kr, 6), device=device)
        from utils.pose_native import pose_channel_mask
        pose_mask = pose_channel_mask(batch, (batch.body_type[0] == 1).nonzero(as_tuple=True)[0])
        if pose_mask is not None:                       # jointed bodies: only their joint channel
            pose_mask = pose_mask.to(x_pose)
            x_pose = x_pose * pose_mask

    # model.unet_pose_action_mode=clean: the conditioning IS a given actuated body arriving unnoised, so
    # its pose rows are held at their clean value -- at init and after every step, future and context --
    # exactly as in training. Read from the MODEL, so a clean-trained checkpoint cannot be sampled without
    # it. The pinned set is batch.pose_action_mask, so hide_actions(batch, 1.0) is the unconditional case.
    pin_pose = pin_ctx_pose = None
    # _base is the UNetWrapper: its .cfg is the ModelConfig (unet_pose_action_mode); the UNetConfig field
    # (pose_action_mode) sits on .unet_cfg. Read both -- checking only .cfg silently disabled the pin.
    _pam = (getattr(getattr(_base, "unet_cfg", None), "pose_action_mode", None)
            or getattr(getattr(_base, "cfg", None), "unet_pose_action_mode", None) or "skip")
    if pose_native and str(_pam) == "clean":
        _ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
        _given = batch.pose_action_mask[:, _ridx][:, None, :, None]
        _clean = batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, _ridx], clip=False).to(x_pose)
        if pose_mask is not None:
            _clean = _clean * pose_mask
        pin_pose = lambda xp: torch.where(_given, _clean.to(xp.dtype), xp)
        x_pose = pin_pose(x_pose)
        if predict_ctx:
            from utils.pose_native import context_pose_deltas
            _cclean = batch.pose_delta_stats.normalize(context_pose_deltas(batch, _ridx), clip=False).to(x_pose)
            if pose_mask is not None:
                _cclean = _cclean * pose_mask
            pin_ctx_pose = lambda xp: torch.where(_given, _cclean.to(xp.dtype), xp)

    if predict_ctx:
        # The context is generated, so it starts from noise like the future. This branch
        # denoises it through the rigid POSES, the only context channel this tree supervises:
        # a deformable row's context particles would stay at pure noise for the whole ladder.
        if not pose_native:
            raise ValueError("a context-predicting checkpoint needs the pose-native sampling path "
                             "(an emit_sidecar batch with pose_deltas_gt)")
        deform_ctx = bool((batch.body_type[0] == 0).any())
        if deform_ctx and not getattr(getattr(_base, "cfg", None), "unet_deformable_context", False):
            raise ValueError("a context-predicting checkpoint generates its context through the rigid "
                             "poses; this scene has deformable particles, whose context rows nothing "
                             "would denoise")
        x_ctx = torch.randn((B, C_ctx, N, 3), device=device)
        if pose_native:
            x_pose_ctx = torch.randn((B, C_ctx, Kr, 6), device=device)
            if pose_mask is not None:
                x_pose_ctx = x_pose_ctx * pose_mask
            if pin_ctx_pose is not None:
                x_pose_ctx = pin_ctx_pose(x_pose_ctx)
        noisy_context, context_timesteps_clean = None, None
    else:
        noisy_context, context_timesteps_clean = clean_context(model, batch)

    def _euler(x, pred, t_row, dt_row):
        return x - dt_row * diffusion.as_velocity(pred, x, t_row)

    for m in range(len(K_timesteps) - 1):
        curr_k = K_timesteps[m]      # [n_rows]
        next_k = K_timesteps[m + 1]  # [n_rows]

        # Skip if no frame needs updating this step
        if (curr_k == next_k).all():
            continue

        # Model forward with per-frame timesteps
        curr_f, next_f = (curr_k[C_ctx:], next_k[C_ctx:]) if predict_ctx else (curr_k, next_k)
        t_batch = curr_f.unsqueeze(0).expand(B, -1).long()  # [B, F]
        kwargs = dict(noisy_context=noisy_context, context_timesteps=context_timesteps_clean)
        if predict_ctx:
            t_ctx = curr_k[:C_ctx].unsqueeze(0).expand(B, -1).long()
            kwargs = dict(noisy_context=x_ctx, context_timesteps=t_ctx)
            if pose_native:
                kwargs["noisy_ctx_pose"] = x_pose_ctx
        if pose_native:
            pred, pose_pred = model(batch, x_t, t_batch, noisy_pose=x_pose, **kwargs)
        else:
            pred = model(batch, x_t, t_batch, **kwargs)

        # Euler step: x_{t-dt} = x_t - dt * velocity
        dt = (curr_f - next_f).float() / diffusion.num_train_timesteps  # [F]
        dt = dt[None, :, None, None]  # [1, F, 1, 1]
        if pred is not None:
            x_t = _euler(x_t, pred, t_batch, dt)
        if pose_native:
            x_pose = _euler(x_pose, pose_pred, t_batch, dt)
            if pose_mask is not None:
                x_pose = x_pose * pose_mask
            if pin_pose is not None:
                x_pose = pin_pose(x_pose)
        if predict_ctx:
            dt_c = ((curr_k[:C_ctx] - next_k[:C_ctx]).float()
                    / diffusion.num_train_timesteps)[None, :, None, None]
            # A rigid body's context particle rows are rebuilt from its context POSE every
            # forward, so evolving that pose is what moves them; x_ctx carries deformable rows.
            if pose_native:
                ctx_pose_pred = getattr(_base, "_last_ctx_pose_pred", None)
                if ctx_pose_pred is not None:
                    x_pose_ctx = _euler(x_pose_ctx, ctx_pose_pred, t_ctx, dt_c)
                    if pose_mask is not None:
                        x_pose_ctx = x_pose_ctx * pose_mask
                    if pin_ctx_pose is not None:
                        x_pose_ctx = pin_ctx_pose(x_pose_ctx)
            if deform_ctx:                              # model.unet_deformable_context
                x_ctx = _euler(x_ctx, _base._last_ctx_part_pred, t_ctx, dt_c)
    if pose_native:
        return x_t, x_pose
    return x_t


def window_bounds(committed: int, target_len: int, future_len: int, context_len: int):
    """(start, length, skip) of a window anchored at the latest committed state.

    The final window contains only the remaining frames. Sliding it backward and
    discarding a regenerated prefix would splice states integrated from different
    histories, introducing a jump. Evaluation and receding-horizon planning share
    this contract; the retained skip field is always zero.
    """
    start = max(committed, context_len)
    return start, min(future_len, target_len - start), 0


def rollout_window(batch, committed, start, length, *, rigid_idx=None, R_track=None, T_track=None):
    """The model input predicting frames [start, start + length) of `batch`.

    Context particles come from `committed`, the rollout's own output. With a carried
    pose track (R_track, T_track over rigid_idx) every context pose of those bodies
    comes from it too: the model reads context pose deltas, so a ground-truth pose at
    C-2 next to a predicted one at C-1 would encode the accumulated drift as a
    spurious body velocity. Actions are sliced from `batch`; its future positions and
    poses are ground truth that only metrics read."""
    C = batch.context_len
    positions = torch.cat([committed[:, start - C:start], batch.positions[:, start:start + length]], dim=1)
    poses = None
    if batch.body_poses is not None:
        poses = batch.body_poses[:, start - C:start + length].clone()
        if R_track is not None:
            for ci in range(C):
                poses[:, ci, rigid_idx, :3, :3] = R_track[:, start - C + ci]
                poses[:, ci, rigid_idx, :3, 3] = T_track[:, start - C + ci]
    frames = slice(start - C, start - C + length)
    activity={}
    if batch.action_active is not None:
        full_gate=torch.cat([batch.context_action_active,batch.action_active],dim=1)
        activity=dict(action_active=batch.action_active[:,frames],
                      context_action_active=full_gate[:,start-C:start-1])
    return replace(
        batch, positions=positions, body_poses=poses, actions=batch.actions[:, frames],
        pose_deltas_gt=None if batch.pose_deltas_gt is None else batch.pose_deltas_gt[:, frames],
        pose_actions=None if batch.pose_actions is None else batch.pose_actions[:, frames],**activity)


@torch.no_grad()
def windowed_rollout(model, diffusion, batch, *, horizon: int, future_len: int,
                     num_inference_steps: int = 25, schedule: str = "pyramid",
                     uncertainty_scale: float = 1.0, clamp_actuated: bool = False,
                     hide_given_actions: bool = False) -> Dict:
    """Forward-window autoregression: the rollout protocol of every evaluator.

    Predicts frames [C, C + H), H = min(horizon, T - C), one model window at a time
    (window_bounds). After the first window the context is the model's own committed
    output: particles and, for pose-native checkpoints, the carried rigid-pose track.
    The final window is shortened to the remaining horizon, so each new prediction
    integrates from the latest committed state without regenerating an overlap.
    Nothing is reset to ground truth mid-rollout; ground truth enters only as the
    actuator's actions. Routed (predict_body_poses) checkpoints Kabsch-fit their anchor
    poses from committed particles. clamp_actuated materializes actuated bodies exactly
    from their given pose actions. hide_given_actions samples every window with its
    actions hidden (utils.conditioning.hide_actions), as a checkpoint trained with
    train.action_dropout=1 expects; materialization and metrics keep the unhidden
    window. The caller seeds the sampler.

    Returns dict(positions [B, C + H, N, 3], horizon H, windows [(start, length, skip)],
    R [B, C + H, Kr, 3, 3] / T [B, C + H, Kr, 3] / rigid_idx for pose-native checkpoints
    (None otherwise)).
    """
    from utils.conditioning import hide_actions
    from utils.evaluation import validate_prediction
    from utils.pose_native import clamp_actuated_pose, materialize_pose_native
    if batch.action_active is not None and hide_given_actions:
        raise ValueError('Pose-only hide_given_actions cannot hide scheduled MPM commands')
    C = batch.context_len
    H = min(int(horizon), batch.positions.shape[1] - C)
    if H < 1 or future_len < 1:
        raise ValueError("windowed_rollout needs a positive window and future frames beyond the context")
    target_len = C + H
    model_cfg = getattr(model, "cfg", None)
    pose_track = bool(getattr(model_cfg, "pose_native", False)) and batch.body_poses is not None
    routed = bool(getattr(model_cfg, "predict_body_poses", False))
    committed = batch.positions[:, :C].clone()
    ridx = R_track = T_track = None
    if pose_track:
        ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
        R_track = batch.body_poses[:, :C, ridx, :3, :3].float().clone()
        T_track = batch.body_poses[:, :C, ridx, :3, 3].float().clone()
    windows = []
    while committed.shape[1] < target_len:
        start, length, skip = window_bounds(committed.shape[1], target_len, future_len, C)
        win = rollout_window(batch, committed, start, length, rigid_idx=ridx, R_track=R_track, T_track=T_track)
        if routed and not pose_track and win.body_poses is not None:
            from utils.rigid_ops import kabsch_batched
            valid = batch.get_valid_mask()
            for k in range(batch.body_type.shape[1]):
                if int(batch.body_type[0, k]) != 1:
                    continue
                for e in range(batch.batch_size):
                    sel = valid[e] & (batch.particle_body[e] == k)
                    template = batch.body_templates[e, k, :int(batch.body_template_mask[e, k].sum())]
                    R, t = kabsch_batched(template, committed[e, start - 1, sel])
                    win.body_poses[e, C - 1, k, :3, :3] = R
                    win.body_poses[e, C - 1, k, :3, 3] = t
        out = run_diffusion_inference_df(model, hide_actions(win, 1.0) if hide_given_actions else win,
                                         diffusion, num_inference_steps=num_inference_steps,
                                         schedule=schedule, uncertainty_scale=uncertainty_scale)
        if isinstance(out, tuple):
            if clamp_actuated:
                out = (out[0], clamp_actuated_pose(win, out[1]))
            pred, R_win, T_win, _ = materialize_pose_native(win, *out, stabilize_joints=True)
        else:
            pred = win.predictions_to_positions(win.prediction_stats.denormalize(out))
        validate_prediction(pred, win.future, win.get_valid_mask())
        committed = torch.cat([committed, pred[:, skip:]], dim=1)
        if pose_track:
            R_track = torch.cat([R_track, R_win[:, skip:]], dim=1)
            T_track = torch.cat([T_track, T_win[:, skip:]], dim=1)
        windows.append((start, length, skip))
    return {"positions": committed, "horizon": H, "windows": windows,
            "R": R_track, "T": T_track, "rigid_idx": ridx}


def compute_metrics(
    predicted: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    context_len: int = 2,
    action_mask: torch.Tensor = None,
    objects_only: bool = False,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predicted: [B, T, N, 3] predicted positions
        gt: [B, T, N, 3] ground truth positions
        valid_mask: [B, N] valid particle mask
        context_len: Number of context frames (excluded from metrics)
        action_mask: [B, N] True for gripper/actuated particles (optional)
        objects_only: If True and action_mask provided, also compute object-only metrics

    Returns:
        dict of metrics
    """
    B, T, N, _ = predicted.shape
    F = T - context_len

    # Expand valid mask
    valid_expanded = valid_mask[:, None, :, None].expand(B, F, N, 3)

    # Only evaluate future frames
    pred_future = predicted[:, context_len:]
    gt_future = gt[:, context_len:]

    # MSE on valid particles
    mse = ((pred_future - gt_future) ** 2)[valid_expanded].mean().item()

    # Per-frame MSE
    per_frame_mse = []
    for f in range(F):
        frame_valid = valid_mask[:, None, :, None].expand(B, 1, N, 3)
        frame_mse = ((pred_future[:, f:f+1] - gt_future[:, f:f+1]) ** 2)[frame_valid].mean().item()
        per_frame_mse.append(frame_mse)

    # Final frame error (how bad is accumulation?)
    final_mse = per_frame_mse[-1]

    results = {
        'mse': mse,
        'final_frame_mse': final_mse,
        'per_frame_mse': per_frame_mse,
    }

    # Object-only metrics (excluding gripper particles)
    if objects_only and action_mask is not None:
        object_mask = valid_mask & ~action_mask  # Valid particles that are NOT gripper
        object_expanded = object_mask[:, None, :, None].expand(B, F, N, 3)

        # Check if there are any object particles
        if object_expanded.any():
            object_mse = ((pred_future - gt_future) ** 2)[object_expanded].mean().item()

            # Per-frame object MSE
            per_frame_object_mse = []
            for f in range(F):
                frame_object = object_mask[:, None, :, None].expand(B, 1, N, 3)
                frame_mse = ((pred_future[:, f:f+1] - gt_future[:, f:f+1]) ** 2)[frame_object].mean().item()
                per_frame_object_mse.append(frame_mse)

            results['object_mse'] = object_mse
            results['object_final_frame_mse'] = per_frame_object_mse[-1]
            results['per_frame_object_mse'] = per_frame_object_mse

    return results


def plot_error_vs_frames(
    all_per_frame_mse: List[List[float]],
    context_len: int = 2,
    future_len: int = 22,
    save_path: str = None,
    title: str = "Prediction Error vs Frame",
    ylabel: str = "MSE",
):
    """
    Plot MSE vs frame number with autoregressive boundary markers.
    Averages across all samples with std dev error bands.

    Args:
        all_per_frame_mse: List of per_frame_mse lists from each batch
        context_len: Number of context frames (C)
        future_len: Frames predicted per step (F) - boundary at C + F
        save_path: Path to save figure (optional)
        title: Plot title
    """
    # Stack all per-frame MSE (assuming same length)
    # Handle variable lengths by padding with NaN
    max_frames = max(len(mse_list) for mse_list in all_per_frame_mse)

    mse_array = np.full((len(all_per_frame_mse), max_frames), np.nan)
    for i, mse_list in enumerate(all_per_frame_mse):
        mse_array[i, :len(mse_list)] = mse_list

    # Compute mean and std (ignoring NaN)
    mean_mse = np.nanmean(mse_array, axis=0)
    std_mse = np.nanstd(mse_array, axis=0)

    # Frame indices (future frames start after context)
    frame_indices = np.arange(context_len, context_len + max_frames)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean MSE with std band
    ax.plot(frame_indices, mean_mse, 'b-', linewidth=2, label=f'Mean {ylabel}')
    ax.fill_between(
        frame_indices,
        mean_mse - std_mse,
        mean_mse + std_mse,
        alpha=0.3,
        color='blue',
        label='±1 Std Dev'
    )

    # Mark autoregressive boundaries (every future_len frames after first window)
    ar_boundaries = []
    boundary = context_len + future_len  # First boundary (e.g., frame 24)
    while boundary <= frame_indices[-1]:
        ar_boundaries.append(boundary)
        boundary += future_len

    # Add vertical lines at boundaries
    for i, boundary in enumerate(ar_boundaries):
        label = 'Autoregressive boundary' if i == 0 else None
        ax.axvline(x=boundary, color='r', linestyle='--', linewidth=2, alpha=0.7, label=label)

    # Shade regions
    # Region 1: First window uses GT context
    ax.axvspan(context_len, context_len + future_len, alpha=0.1, color='green', label='GT context window')

    # Region 2+: Autoregressive (uses predicted context)
    if ar_boundaries:
        ax.axvspan(ar_boundaries[0], frame_indices[-1] + 1, alpha=0.1, color='red', label='Autoregressive rollout')

    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Set x-axis ticks
    tick_interval = max(1, max_frames // 10)
    ax.set_xticks(frame_indices[::tick_interval])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    plt.close(fig)
    return fig, mean_mse, std_mse


def load_model_and_diffusion(checkpoint_path: str, device: torch.device):
    """
    Load model and diffusion process from checkpoint.

    Returns:
        model, diffusion, config
    """
    from utils.checkpoint import read_checkpoint, checkpoint_config
    ckpt = read_checkpoint(checkpoint_path)
    cfg = checkpoint_config(ckpt)

    # Back-fill fields added after some checkpoints were saved, so downstream
    # direct-access (cfg.flow_matching.prediction_type) works on older checkpoints.
    if 'flow_matching' in cfg and 'prediction_type' not in cfg.flow_matching:
        cfg.flow_matching.prediction_type = 'velocity'

    from models.factory import build_particle_model
    from utils.state_dict import load_model_state
    model = build_particle_model(cfg, checkpoint=True)
    weights_key = "ema" if ckpt.get("ema") else "model"
    regenerated = load_model_state(model, ckpt[weights_key])
    print(f"Loaded {weights_key} weights ({len(regenerated)} deterministic buffers regenerated)")

    model = model.to(device)
    model.eval()

    # Create diffusion/flow matching
    denoise_method = cfg.denoise_method
    if denoise_method != "flow_matching":
        raise ValueError("Only unified flow-matching checkpoints are supported")
    diffusion = TrajectoryFlowMatching(
        num_train_timesteps=cfg.flow_matching.num_train_timesteps,
        shift=cfg.flow_matching.shift,
        prediction_type=cfg.flow_matching.prediction_type,
    )
    inference_steps = cfg.flow_matching.inference_steps

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Denoise method: {denoise_method}")
    print(f"  Model dim: {cfg.model.get('dim', cfg.model.get('unet_widths', '?'))}, "
          f"depth: {cfg.model.get('depth', cfg.model.get('unet_enc_depths', '?'))}")
    print(f"  Inference steps: {inference_steps}")

    return model, diffusion, cfg, inference_steps
