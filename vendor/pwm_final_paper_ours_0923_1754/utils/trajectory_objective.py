"""The particle/pose training objective, independent of optimizer and run I/O.

Sample FM uses clean-target loss for particles but velocity-space loss for native
poses (implicit 1/t^2 weighting of pose clean errors). This is preserved for
checkpoint compatibility, not silently changed during structural cleanup.
train.pose_loss_space=sample (opt-in, sample heads) scores poses in clean-target space
like particles; train.velocity_weight_cap (opt-in, on top of it) multiplies both terms by
one capped 1/t^2 weight; train.timestep_schedule (opt-in) shapes the future frames' joint
noise levels (diffusion/timestep_schedules.py).
"""
import math
from typing import Optional

import torch
from omegaconf import OmegaConf

from diffusion import TrajectoryFlowMatching


def _flow_matching_x0(
    pred_velocity: torch.Tensor,
    x_t: torch.Tensor,
    timesteps: torch.Tensor,
    num_train_timesteps: int,
) -> torch.Tensor:
    """Recover predicted clean delta from velocity (FM: x_0 = x_t - t * v)."""
    B, F = pred_velocity.shape[:2]
    if timesteps.ndim == 1:
        t_norm = (timesteps.float() / num_train_timesteps).view(B, 1, 1, 1)
    elif timesteps.ndim == 2:
        t_norm = (timesteps.float() / num_train_timesteps).view(B, F, 1, 1)
    else:
        raise ValueError("Unified FM timesteps must have shape [B] or [B,F]")
    return x_t - t_norm * pred_velocity


def compute_mesh_pair_loss(
    pred_x0: torch.Tensor,      # [B, F, N, 3] normalized predicted delta
    x_0: torch.Tensor,          # [B, F, N, 3] normalized target delta
    batch,                      # TrajectoryBatch (for anchor, delta_stats, object_ids)
    K: int = 8,
) -> torch.Tensor:
    """Mesh edge-distance preservation loss for cloth.

    Defines K-nearest-neighbor edges per cloth particle on the ANCHOR frame.
    Penalizes |pred_dist(i,j) - gt_dist(i,j)|^2 between those pairs at every
    predicted frame. Output is in raw m^2 — caller scales by a lambda that
    balances against the normalized main v-loss (typical λ_mesh ~ 1000-5000).
    """
    from pytorch3d.ops import knn_points

    pred_delta = batch.delta_stats.denormalize(pred_x0)
    gt_delta   = batch.delta_stats.denormalize(x_0)
    anchor     = batch.anchor                                          # [B, N, 3]
    pred_pos = anchor.unsqueeze(1) + torch.cumsum(pred_delta, dim=1)   # [B, F, N, 3]
    gt_pos   = anchor.unsqueeze(1) + torch.cumsum(gt_delta, dim=1)

    B = pred_x0.shape[0]
    obj_ids = batch.object_ids
    losses = []
    for b in range(B):
        cloth_idx = (obj_ids[b] == 0).nonzero(as_tuple=True)[0]
        if cloth_idx.numel() < K + 1:
            continue
        anchor_cloth = anchor[b, cloth_idx].unsqueeze(0).float()        # [1, n_cloth, 3]
        _, nn_idx, _ = knn_points(anchor_cloth, anchor_cloth, K=K + 1)
        edges = nn_idx.squeeze(0)[:, 1:]                                # drop self

        pred_c = pred_pos[b, :, cloth_idx, :]                           # [F, n_cloth, 3]
        gt_c   = gt_pos[b, :, cloth_idx, :]
        nb_pred = pred_c[:, edges, :]                                   # [F, n_cloth, K, 3]
        nb_gt   = gt_c[:, edges, :]
        d_pred = (nb_pred - pred_c.unsqueeze(2)).norm(dim=-1)
        d_gt   = (nb_gt - gt_c.unsqueeze(2)).norm(dim=-1)
        losses.append(((d_pred - d_gt) ** 2).mean())

    if not losses:
        return torch.tensor(0.0, device=pred_x0.device)
    return torch.stack(losses).mean()


def compute_aux_losses(
    pred_x0: torch.Tensor,        # [B, F, N, 3] predicted clean delta (normalized)
    x_0: torch.Tensor,             # [B, F, N, 3] target clean delta (normalized)
    valid_mask: torch.Tensor,      # [B, F, N, 3] bool, already expanded
    obj_ids: Optional[torch.Tensor],  # [B, N] integer object id per particle
    weights: dict,                 # {pos_cum, accel, com, spread}: float weights, 0 disables
    batch=None,                    # TrajectoryBatch — required for the phys terms
) -> dict:
    """
    Auxiliary losses on top of per-frame velocity MSE.

    Position/acceleration/COM/spread terms use normalized delta space. Physics
    terms explicitly denormalize to metres and retain their historical units.
    Returns a dict {name: scalar tensor}, only populated for enabled aux losses.

    - pos_cum: MSE on cumulatively integrated trajectory (penalizes drift over F frames)
    - accel:   MSE on frame-to-frame change in delta (acceleration target signal)
    - com:     MSE on per-object center-of-mass trajectory
    - spread:  MSE on per-object spatial std (sqrt of mean sq dist to COM)

    PhysCtrl-style physics terms (utils/phys_losses.py; need `batch`, delta
    target, full particle set — designed + lambda-calibrated 2026-08-04 on the
    unetstd run, see reports/aux_loss_calibration_unetstd_final.json):
    - strain_inc: L1 on per-frame incremental deformation gradients (spatial
                  form: GT prev-frame kNN graphs), ALL materials
    - strain_cum: L1 on frame-0 -> final-frame cumulative deformation gradient
                  (frame-0 graph), elastic+plasticine only (train.aux_strain_cum_mats)
    - floor:      hinge below the floor plane on integrated positions
                  (train.aux_floor_z, default 0.091 for robert_mpm_v2)
    """
    out: dict = {}
    B, F, N, _ = pred_x0.shape
    device = pred_x0.device

    want_pos = weights.get("pos_cum", 0.0) > 0
    want_acc = weights.get("accel", 0.0) > 0
    want_com = weights.get("com", 0.0) > 0
    want_spr = weights.get("spread", 0.0) > 0
    want_spr_axes = weights.get("spread_axes", 0.0) > 0   # v7: per-axis spread
    if want_pos or want_com or want_spr or want_spr_axes:
        pred_pos = torch.cumsum(pred_x0, dim=1)
        gt_pos = torch.cumsum(x_0, dim=1)

    if want_pos:
        out["pos_cum"] = ((pred_pos - gt_pos) ** 2)[valid_mask].mean()

    if want_acc:
        pred_acc = pred_x0[:, 1:] - pred_x0[:, :-1]
        gt_acc = x_0[:, 1:] - x_0[:, :-1]
        out["accel"] = ((pred_acc - gt_acc) ** 2)[valid_mask[:, 1:]].mean()

    if (want_com or want_spr or want_spr_axes) and obj_ids is not None:
        K = int(obj_ids.max().item()) + 1
        K_range = torch.arange(K, device=device)
        obj_onehot = (obj_ids.unsqueeze(-1) == K_range).float()  # [B, N, K]
        particle_valid = valid_mask[:, 0, :, 0].float()           # [B, N]
        obj_onehot = obj_onehot * particle_valid.unsqueeze(-1)
        count = obj_onehot.sum(dim=1).clamp(min=1)                # [B, K]
        obj_present = (count > 1).float()                          # [B, K]
        n_obj_present = obj_present.sum().clamp(min=1)

        pred_com = torch.einsum('bfnd,bnk->bfkd', pred_pos, obj_onehot) / count[:, None, :, None]
        gt_com = torch.einsum('bfnd,bnk->bfkd', gt_pos, obj_onehot) / count[:, None, :, None]

        if want_com:
            com_se = ((pred_com - gt_com) ** 2).sum(dim=-1)        # [B, F, K]
            com_se = com_se * obj_present.unsqueeze(1)
            out["com"] = com_se.sum() / (n_obj_present * F * 3)

        if want_spr or want_spr_axes:
            obj_idx = obj_ids.long().clamp(min=0, max=K - 1)
            obj_idx_3 = obj_idx[:, None, :, None].expand(B, F, N, 3)
            pred_com_pp = pred_com.gather(2, obj_idx_3)             # [B, F, N, 3]
            gt_com_pp = gt_com.gather(2, obj_idx_3)
            # Per-axis squared deviation from object COM
            pred_dev2_axes = (pred_pos - pred_com_pp) ** 2 \
                              * particle_valid[:, None, :, None]    # [B, F, N, 3]
            gt_dev2_axes = (gt_pos - gt_com_pp) ** 2 \
                            * particle_valid[:, None, :, None]      # [B, F, N, 3]
            # Per-object per-axis variance [B, F, K, 3]
            pred_var_axes = torch.einsum('bfnd,bnk->bfkd', pred_dev2_axes, obj_onehot) \
                            / count[:, None, :, None]
            gt_var_axes = torch.einsum('bfnd,bnk->bfkd', gt_dev2_axes, obj_onehot) \
                          / count[:, None, :, None]

            if want_spr:
                # v5/v6 legacy: collapse axes to a single scalar per object
                pred_std = torch.sqrt(pred_var_axes.sum(dim=-1).clamp(min=0) + 1e-8)
                gt_std = torch.sqrt(gt_var_axes.sum(dim=-1).clamp(min=0) + 1e-8)
                spr_se = ((pred_std - gt_std) ** 2) * obj_present.unsqueeze(1)
                out["spread"] = spr_se.sum() / (n_obj_present * F)

            if want_spr_axes:
                # v7: per-axis std, MSE summed over 3 axes.
                # Catches anisotropic distortions that the scalar version is
                # mathematically blind to (e.g., sphere -> pancake preserving
                # total variance). Material-agnostic: compares to GT per-frame,
                # so sand legitimately spreading != penalized.
                pred_std_axes = torch.sqrt(pred_var_axes.clamp(min=0) + 1e-8)
                gt_std_axes = torch.sqrt(gt_var_axes.clamp(min=0) + 1e-8)
                spr_se_ax = ((pred_std_axes - gt_std_axes) ** 2) \
                            * obj_present.unsqueeze(1).unsqueeze(-1)
                out["spread_axes"] = spr_se_ax.sum() / (n_obj_present * F * 3)

    # Physics auxiliaries use raw metres, unlike the normalized terms above.
    want_si = weights.get("strain_inc", 0.0) > 0
    want_sc = weights.get("strain_cum", 0.0) > 0
    want_fl = weights.get("floor", 0.0) > 0
    if weights.get("vel", 0.0) or weights.get("accel_f0", 0.0):
        raise ValueError("vel/accel_f0 auxiliaries belonged to retired frame0_offset targets")
    if want_si or want_sc or want_fl:
        from utils.phys_losses import (DEFAULT_FLOOR_Z,
                                       DEFAULT_STRAIN_CUM_MATS, floor_loss,
                                       strain_cum_loss, strain_inc_loss)
        if batch is None or batch.prediction_target != "delta":
            raise ValueError("Physics auxiliaries require a unified delta-target batch")
        pred_raw = batch.prediction_stats.denormalize(pred_x0.float())
        pred_deltas = pred_raw
        pred_disp = pred_raw.sum(dim=1)
        gt_disp = batch.prediction_x0.sum(dim=1)
        if want_si:
            out["strain_inc"] = strain_inc_loss(batch, pred_deltas)
        if want_sc:
            mats = weights.get("strain_cum_mats", DEFAULT_STRAIN_CUM_MATS)
            out["strain_cum"] = strain_cum_loss(batch, pred_disp, gt_disp, mats=mats)
        if want_fl:
            fz = float(weights.get("floor_z", DEFAULT_FLOOR_Z))
            out["floor"] = floor_loss(batch, pred_raw, floor_z=fz, valid_mask=valid_mask[..., 0])

    return out


def capped_velocity_weight(timesteps, num_train_timesteps, cap):
    """min(1 / t^2, cap) per frame, t = clock / T: the weight of an x0-space error.

    A velocity error is the clean-target error over t (v = (x_t - x_0) / t), so the velocity
    loss weights x0 errors by 1 / t^2, singular at the clean end. The cap keeps that emphasis
    on nearly clean frames without the blow-up. Unlike min-SNR it never suppresses noisy
    frames: 1 / t^2 >= 1, so a frame at pure noise keeps weight 1. (Ported from mb/pick_place.)
    """
    time = timesteps.float() / float(num_train_timesteps)
    return (1.0 / time.clamp_min(1e-6) ** 2).clamp_max(float(cap))


def timestep_schedule_options(cfg):
    """train.timestep_schedule (with its ramp knobs) and train.velocity_weight_cap as objective
    keyword arguments. Empty for the defaults, so every existing configuration resolves to the
    options it always had. Fails closed on a bad name, weight, slope, jitter or cap."""
    from diffusion.timestep_schedules import resolve_mixture, resolve_slope
    # Partial configurations (the overfit plots compare objective contracts from `train` alone)
    # resolve like any default: nothing outside `train` is read unless an option needs it.
    t, fm = cfg.train, cfg.get("flow_matching") or {}
    options = {}
    spec = t.get("timestep_schedule", "independent")
    schedule = resolve_mixture(OmegaConf.to_container(spec, resolve=True) if OmegaConf.is_config(spec) else spec)
    if "ramp" not in schedule and any(t.get(name) is not None for name in ("ramp_slope", "ramp_jitter")):
        raise ValueError("train.ramp_slope and train.ramp_jitter are inactive unless "
                         "train.timestep_schedule includes ramp")
    if set(schedule) != {"independent"}:
        if t.get("noise_context", False) and not t.get("pose_noise_context", False):
            raise ValueError("train.timestep_schedule shapes a window whose context is either clean or "
                             "GENERATED; train.noise_context on its own is retired (it noises rigid "
                             "particles away from their pose) — add train.pose_noise_context")
        options["timestep_schedule"] = schedule
    if "ramp" in schedule:
        if fm.get("time_sampler", "uniform") != "uniform":
            raise ValueError("timestep_schedule ramp requires flow_matching.time_sampler=uniform")
        slope = t.get("ramp_slope")
        if slope is None:
            # The staircase of the run's own pyramid sampler: uncertainty scale over inference steps.
            slope = float(cfg.get("df_uncertainty_scale", 1.0)) / int(fm.get("inference_steps", 25))
        elif OmegaConf.is_config(slope):
            slope = OmegaConf.to_container(slope, resolve=True)
        slope = resolve_slope(slope)
        options["ramp_slope"] = list(slope) if isinstance(slope, tuple) else slope
        jitter = t.get("ramp_jitter")
        if jitter is not None:
            if isinstance(jitter, bool) or not isinstance(jitter, (int, float)) \
                    or not math.isfinite(jitter) or jitter < 0:
                raise ValueError("train.ramp_jitter must be a finite nonnegative number")
            options["ramp_jitter"] = float(jitter)
    if "stream" in schedule:
        # The windows a streaming rollout shows the model, read from a table. Only when used: the
        # resume contract compares this dict key by key.
        from diffusion.timestep_schedules import resolve_stream_options
        shares = t.get("stream_phase_shares")
        if OmegaConf.is_config(shares):
            shares = OmegaConf.to_container(shares, resolve=True)
        if not t.get("noise_context", False):
            raise ValueError("train.timestep_schedule stream covers the [context | future] rows of a "
                             "window whose context is generated; it needs train.noise_context with "
                             "train.pose_noise_context")
        options["stream_options"] = resolve_stream_options(
            t.get("stream_states"), phase_shares=shares, jitter=t.get("stream_jitter", 0.048),
            keep=t.get("stream_keep"), given=t.get("stream_given"))
    elif any(t.get(name) is not None for name in
             ("stream_states", "stream_phase_shares", "stream_jitter", "stream_keep", "stream_given")):
        raise ValueError("train.stream_* is inactive unless train.timestep_schedule includes stream")
    cap = t.get("velocity_weight_cap")
    if cap is not None and cap != 0:
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap) or cap < 1:
            raise ValueError("train.velocity_weight_cap must be 0 (off) or a finite number >= 1")
        if fm.get("prediction_type", "velocity") != "sample":
            raise ValueError("train.velocity_weight_cap weights x0-space errors; it requires "
                             "flow_matching.prediction_type=sample (a velocity head already carries "
                             "the uncapped 1/t^2 weight)")
        if t.get("supervise_context", False):
            raise ValueError("train.velocity_weight_cap does not cover the retired supervise_context term")
        if (cfg.get("model") or {}).get("pose_native", True) and t.get("pose_loss_space", "velocity") != "sample":
            raise ValueError("train.velocity_weight_cap weights clean-target errors: a pose-native model also "
                             "needs train.pose_loss_space=sample (the velocity-space pose term already carries "
                             "the uncapped 1/t^2 weight)")
        options["velocity_weight_cap"] = float(cap)
    return options


def trajectory_loss_options(cfg):
    """Resolve the objective once for training, validation and overfit diagnostics."""
    from utils.pose_native import pose_noise_scope
    t = cfg.train
    anchor = t.get("light_pose_noise_anchor", False)
    if not isinstance(anchor, bool):
        raise ValueError("train.light_pose_noise_anchor must be a boolean")
    retired = {name: t[name] for name, default in {
        "lambda_pose": 0.0, "lambda_body_pose": 0.0, "gripper_loss_weight": 1.0,
    }.items() if name in t and t[name] != default}
    if cfg.get("f0off_delta_loss", False):
        retired["f0off_delta_loss"] = True
    if retired:
        raise ValueError(f"Retired objective settings: {retired}")
    options = dict(
        diffusion_forcing=bool(cfg.get("diffusion_forcing", True)),
        loss_weighting=cfg.get("loss_weighting", "uniform"),
        noise_context=bool(t.get("noise_context", True)),
        supervise_context=bool(t.get("supervise_context", False)),
        light_context_noise_std=float(t.get("light_context_noise_std", 0.0)),
        light_pose_noise_std=float(t.get("light_pose_noise_std", 0.0)),
        light_pose_noise_anchor=anchor,
        light_pose_noise_target=t.get("light_pose_noise_target", "restore"),
        light_pose_noise_scope=pose_noise_scope(anchor, t.get("light_pose_noise_scope", "auto")),
        light_pose_noise_type=t.get("light_pose_noise_type", "iid"),
        pose_loss_space=str(t.get("pose_loss_space", "velocity")),
        clean_ctx_timesteps=bool(t.get("clean_ctx_timesteps", False)),
        aux_loss_weights=OmegaConf.to_container(t.aux_losses, resolve=True) if t.get("aux_losses") else None,
        lambda_mesh=float(t.get("lambda_mesh", 0.0)),
        lambda_pose_native=float(t.get("lambda_pose_native", 1.0)),
        mesh_pair_K=int(t.get("mesh_pair_K", 8)), clip_deltas=bool(t.get("clip_deltas", True)),
        pose_action_mode=str(((cfg.get("model") or {}).get("unet_pose_action_mode", "skip")) or "skip"),
        clip_pose_deltas=bool(t.get("clip_pose_deltas", True)),
        action_dropout=float(t.get("action_dropout", 0.0)),
        **timestep_schedule_options(cfg))
    if t.get("pose_noise_context", False):
        # Only when used: the resume contract compares this dict key by key, so a recipe that
        # reads its context keeps exactly the options it always had.
        options["pose_noise_context"] = True
    if bool(((cfg.get("model") or {}).get("unet_deformable_context", False))):
        options["deformable_context"] = True          # same rule: only when used
    return options


def trajectory_objective(
    model, batch, diffusion, *,
    diffusion_forcing=True, loss_weighting="uniform",
    noise_context=False, pose_noise_context=False, supervise_context=False, light_context_noise_std=0.0,
    deformable_context=False,
    clean_ctx_timesteps=False, aux_loss_weights=None, lambda_pose_native=1.0,
    lambda_mesh=0.0, mesh_pair_K=8, clip_deltas=True, clip_pose_deltas=True, action_dropout=0.0,
    light_pose_noise_std=0.0, light_pose_noise_anchor=False,
    light_pose_noise_target="restore", light_pose_noise_scope="auto", light_pose_noise_type="iid",
    pose_loss_space="velocity",
    timestep_schedule="independent", ramp_slope=None, ramp_jitter=None, stream_options=None,
    velocity_weight_cap=0.0, pose_action_mode="skip",
):
    """Unified delta-space FM/DF loss without optimizer or run-I/O side effects.

    Returns loss, scalar diagnostics and the sampled draw. Future particle MSE
    averages valid deformable coordinates; native pose MSE averages rigid-body
    coordinates separately, then is added with lambda_pose_native. It is NOT a
    single particle-count-weighted average. Sample predictions retain the
    historical velocity-space pose loss (clean pose errors weighted by 1/t^2)
    unless pose_loss_space="sample" (train.pose_loss_space), which compares the
    clean pose output with the clean pose target, flat in t like the particle term.

    timestep_schedule chooses WHICH joint patterns of per-frame noise levels training
    sees: 'independent' (iid, the default, bitwise the historical draw), or a shaped
    schedule or mixture from diffusion/timestep_schedules ('flat' is the full_sequence
    sampler's states, 'ramp' the pyramid sampler's, 'stream' the windows a streaming rollout
    produces, read from a table). With pose_noise_context the schedule
    covers the [context | future] rows jointly, because the context is generated too.
    velocity_weight_cap > 0 (sample
    heads) multiplies every clean-target error by min(1/t^2, cap) per frame: the velocity
    loss without its singularity at the clean end. It is a weight on clean-target terms,
    so with rigid bodies it requires pose_loss_space="sample"; the velocity-space pose
    term already carries the uncapped weight.
    """
    if pose_loss_space not in ("velocity", "sample"):
        raise ValueError("train.pose_loss_space must be velocity or sample")
    if pose_loss_space == "sample" and getattr(diffusion, "prediction_type", None) != "sample":
        raise ValueError("train.pose_loss_space=sample requires flow_matching.prediction_type=sample")
    if not isinstance(diffusion, TrajectoryFlowMatching):
        raise TypeError("Unified objectives require TrajectoryFlowMatching")
    if batch.prediction_target != "delta":
        raise ValueError("Unified objectives require delta targets")
    if not diffusion_forcing:
        raise ValueError("Unified training requires diffusion_forcing=True")
    if loss_weighting != "uniform":
        raise ValueError("Unified training requires uniform loss weighting")
    if not isinstance(light_pose_noise_anchor, bool):
        raise ValueError("train.light_pose_noise_anchor must be a boolean")
    if light_pose_noise_anchor and light_pose_noise_std <= 0:
        raise ValueError("train.light_pose_noise_anchor needs a positive train.light_pose_noise_std")
    if light_pose_noise_target not in ("restore", "carry") or (light_pose_noise_target == "carry" and not light_pose_noise_anchor):
        raise ValueError("train.light_pose_noise_target must be restore, or carry with anchor=true")
    if light_pose_noise_type not in ("iid", "random_walk"):
        raise ValueError("train.light_pose_noise_type must be iid or random_walk")
    if velocity_weight_cap:
        if diffusion.prediction_type != "sample":
            raise ValueError("velocity_weight_cap weights x0-space errors; it requires a sample head")
        if not math.isfinite(velocity_weight_cap) or velocity_weight_cap < 1:
            raise ValueError("velocity_weight_cap must be 0 (off) or a finite number >= 1")
        if supervise_context:
            raise ValueError("velocity_weight_cap does not cover the retired supervise_context term")
    if timestep_schedule != "independent" and noise_context and not pose_noise_context:
        raise ValueError("timestep_schedule shapes a window whose context is clean or generated; "
                         "noise_context alone is the retired recipe — add pose_noise_context")
    if pose_noise_context and not noise_context:
        raise ValueError("pose_noise_context makes the context a generated channel: it requires "
                         "noise_context (the particle side of the same context stream)")
    from utils.pose_native import pose_noise_scope
    light_pose_noise_scope = pose_noise_scope(light_pose_noise_anchor, light_pose_noise_scope)
    if action_dropout > 0:
        # train.action_dropout: per element, the actuated bodies' given future motion is
        # replaced by the no-action input, so the model must generate it. The pose target
        # still covers every rigid body (utils/conditioning.hide_actions).
        from utils.conditioning import hide_actions
        batch = hide_actions(batch, action_dropout)

    from utils.conditioning import model_metadata
    base_m = model_metadata(model)
    pose_native = (getattr(base_m.cfg, "pose_native", False)
                   and batch.pose_deltas_gt is not None)
    if velocity_weight_cap and pose_native and pose_loss_space != "sample":
        raise ValueError("velocity_weight_cap weights clean-target errors: with rigid bodies it requires "
                         "pose_loss_space=sample (the velocity-space pose term already carries the uncapped weight)")
    if light_pose_noise_std > 0 and pose_native:
        # Whole-body history corruption, optionally with a moved passive anchor. Restore
        # relands on recorded poses; carry preserves world/joint increments from that anchor.
        if light_pose_noise_anchor and clip_pose_deltas:
            raise ValueError("train.light_pose_noise_anchor requires train.clip_pose_deltas=false "
                             "to preserve its restore/carry pose targets")
        if noise_context:
            raise ValueError("train.light_pose_noise_std cannot be combined with the retired noise_context")
        from utils.pose_native import jitter_rigid_history
        batch = jitter_rigid_history(batch, light_pose_noise_std, anchor=light_pose_noise_anchor,
                                     target=light_pose_noise_target, scope=light_pose_noise_scope,
                                     noise_type=light_pose_noise_type)
    B = batch.batch_size
    F = batch.num_future_frames
    device = batch.device
    C_ctx = batch.context_len - 1  # all_relative uses C-1 context deltas

    valid_mask = batch.get_valid_mask()  # [B, N]
    valid_mask = valid_mask[:, None, :, None].expand(B, F, -1, 3)  # [B, F, N, 3]

    # existing future noising. clip_deltas=False trains on the faithful (un-clipped) delta tail —
    # cloth Phase C clipping A/B (default True = byte-identical to the deployed v8 recipe).
    raw_x0 = batch.prediction_x0
    x_0 = batch.prediction_stats.normalize(raw_x0, clip=clip_deltas)
    noise = torch.randn_like(x_0)
    # 'independent' keeps the historical call, and so the RNG stream, exactly; a shaped
    # schedule or a mixture draws the levels jointly — over the WHOLE window when the
    # context is generated (pose_noise_context), over the future rows alone otherwise.
    shaped_ctx = None
    if timestep_schedule == "independent":
        timesteps = diffusion.sample_timesteps(B, device, num_frames=F, low=1)
    else:
        from diffusion.timestep_schedules import sample_window_timesteps
        shaped_ctx, timesteps = sample_window_timesteps(
            diffusion, B, device, C_ctx if pose_noise_context else 0, F, timestep_schedule,
            ramp_slope=ramp_slope, ramp_jitter=ramp_jitter, low=1, stream_options=stream_options)
    x_t = diffusion.add_noise(x_0, noise, timesteps)
    row_weight = ctx_row_weight = None
    if velocity_weight_cap:
        row_weight = capped_velocity_weight(
            timesteps, diffusion.num_train_timesteps, velocity_weight_cap)[:, :, None, None]   # [B, F, 1, 1]

    # context noising — only when all_relative
    noisy_context = None
    context_timesteps = None
    if noise_context:
        # The context is a generated channel: its rows carry their own noise levels and are
        # predicted like future rows. On its own (the retired recipe, refused by
        # utils/config.py) this noises every particle's history independently and puts a rigid
        # body's particles away from its pose; with train.pose_noise_context below, the rigid
        # rows are REBUILT from the noisy context pose instead, so the body stays rigid.
        ctx_raw = batch.context_deltas          # [B, C-1, N, 3]
        ctx_norm = batch.delta_stats.normalize(ctx_raw)
        ctx_noise = torch.randn_like(ctx_norm)
        context_timesteps = (shaped_ctx if shaped_ctx is not None
                             else diffusion.sample_timesteps(B, device, num_frames=C_ctx))
        noisy_context = diffusion.add_noise(ctx_norm, ctx_noise, context_timesteps)
        # noisy_context: [B, C-1, N, 3] — already C_ctx shaped, already normalized
        if velocity_weight_cap:
            ctx_row_weight = capped_velocity_weight(
                context_timesteps, diffusion.num_train_timesteps, velocity_weight_cap)[:, :, None, None]
    elif light_context_noise_std > 0:
        # Pushforward-style: add small Gaussian to normalized context to simulate
        # AR-rollout prediction error without the OOD "pure noise" regime of
        # DFoT-style full timestep noising. No context_timesteps emitted — model
        # treats this as clean context. Rigid rows of a pose_native batch stay clean:
        # they are template @ pose, and their rollout error is a whole-body pose error.
        ctx_raw = batch.context_deltas
        ctx_norm = batch.delta_stats.normalize(ctx_raw)
        noisy_context = ctx_norm + torch.randn_like(ctx_norm) * light_context_noise_std
        if pose_native:
            rigid_rows = batch.body_type.gather(1, batch.particle_body) == 1      # [B, N]
            noisy_context = torch.where(rigid_rows[:, None, :, None], ctx_norm, noisy_context)

    # train.clean_ctx_timesteps (opt-in, default OFF => byte-identical to before).
    # With noise_context=false no context timesteps are emitted. The Transformer
    # then uses a literal zero embedding (adaln_proj(0)); the UNet wrapper maps
    # absent context timesteps to t=0. This instead emits t=1 for time_embed(1):
    # the CLEAN end of the same manifold the future frames live on (matches
    # pwm_fresh, which pins clean history frames at t=1). Context stays clean.
    if clean_ctx_timesteps and context_timesteps is None:
        context_timesteps = torch.ones(B, batch.context_len - 1, device=device, dtype=torch.long)

    # Tier-2 v2 (docs/tier2_v2_design.md): pose deltas of rigid bodies are jointly
    # noised with the particles under the SAME timesteps and supervised by the
    # same main v-loss — no auxiliaries.
    noisy_pose = pose_x0 = pose_noise = pose_mask = None
    if pose_native:
        ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
        # clip_pose_deltas=False trains on the faithful (un-clipped) pose tail: needed when a
        # domain's rigid motion exceeds the frozen scales (native episodes move the gripper and
        # the object at ~2x the pickup p99, and clipped targets cap the learned speed there)
        pose_x0 = batch.pose_delta_stats.normalize(
            batch.pose_deltas_gt[:, :, ridx], clip=clip_pose_deltas)
        pose_noise = torch.randn_like(pose_x0)
        from utils.pose_native import pose_channel_mask
        pose_mask = pose_channel_mask(batch, ridx)
        if pose_mask is not None:                       # jointed bodies: only their joint channel
            pose_mask = pose_mask.to(pose_x0)
            pose_x0, pose_noise = pose_x0 * pose_mask, pose_noise * pose_mask
        noisy_pose = diffusion.add_noise(pose_x0, pose_noise, timesteps)
        if pose_action_mode == "clean":
            # model.unet_pose_action_mode=clean: a GIVEN actuated body is shown unnoised. `batch` has
            # already been through hide_actions, so pose_action_mask IS the given set. pose_native_inputs
            # rebuilds each rigid body's particle rows from its pose, so the body's particles come out
            # clean too. Its pose target stays supervised (it can simply copy the clean input).
            _given = batch.pose_action_mask[:, ridx][:, None, :, None]           # [B, 1, Kr, 1]
            noisy_pose = torch.where(_given, pose_x0, noisy_pose)
    ctx_pose_x0 = ctx_pose_noise = noisy_ctx_pose = None
    if pose_native and pose_noise_context:
        if bool((batch.body_type[0] == 0).any()) and not deformable_context:
            raise ValueError("pose_noise_context generates the context through the rigid POSES; a scene "
                             "with deformable particles would leave their context rows at pure noise, "
                             "unsupervised (set model.unet_deformable_context=true to supervise them)")
        # train.pose_noise_context: the C-1 context pose transitions are a noised, supervised
        # channel too, at the context rows' own levels (a uniform window — no clean context pose
        # reaches the model). Jointed bodies: only their joint channel, as for the future.
        from utils.pose_native import context_pose_deltas
        ctx_pose_x0 = batch.pose_delta_stats.normalize(
            context_pose_deltas(batch, ridx), clip=clip_pose_deltas)
        ctx_pose_noise = torch.randn_like(ctx_pose_x0)
        if pose_mask is not None:
            ctx_pose_x0, ctx_pose_noise = ctx_pose_x0 * pose_mask, ctx_pose_noise * pose_mask
        noisy_ctx_pose = diffusion.add_noise(ctx_pose_x0, ctx_pose_noise, context_timesteps)
        if pose_action_mode == "clean":                 # the given body is clean in the CONTEXT rows too
            noisy_ctx_pose = torch.where(batch.pose_action_mask[:, ridx][:, None, :, None],
                                         ctx_pose_x0, noisy_ctx_pose)

    if pose_native:
        _ctx_kw = {} if noisy_ctx_pose is None else {"noisy_ctx_pose": noisy_ctx_pose}
        pred, pose_pred = model(batch, x_t, timesteps,
                                noisy_context=noisy_context,
                                context_timesteps=context_timesteps,
                                noisy_pose=noisy_pose, **_ctx_kw)
    else:
        pred = model(batch, x_t, timesteps,
                     noisy_context=noisy_context,
                     context_timesteps=context_timesteps)

    # Sample-parameterized FM trains particles DIRECTLY in x0 space: compare the
    # model's x0 output to GT x0 — no velocity conversion, no 1/t^2, flat in t.
    # Velocity mode keeps the v-loss (as_velocity is a no-op there).
    _fm_x0_loss = diffusion.prediction_type == "sample"
    obj_ids_loss = batch.object_ids
    if pred is not None:
        target = x_0 if _fm_x0_loss else diffusion.get_target(x_0, noise, timesteps)
        if not _fm_x0_loss:
            pred = diffusion.as_velocity(pred, x_t, timesteps)

        mse = (pred - target) ** 2
        if row_weight is not None:
            mse = mse * row_weight
    else:
        # pose_native pure-rigid batch (pickup/pushing): the particle decode was
        # skipped — the pose term below IS the loss.
        mse = None


    # pose_native: the particle term counts DEFORMABLE rows only — rigid rows'
    # particle representation is a deterministic function of the pose channels,
    # which carry their own main-loss term below.
    if mse is not None and pose_native and batch.body_type is not None:
        deform = batch.body_type.gather(1, batch.particle_body) == 0     # [B, N]
        mse = mse * deform[:, None, :, None].float()
        valid_mask = valid_mask & deform[:, None, :, None].expand_as(valid_mask)

    if mse is None:
        loss = x_t.sum() * 0.0
    else:
        # Guard: under the routing head a pure-rigid scene leaves valid_mask empty.
        loss = mse[valid_mask].mean() if bool(valid_mask.any()) else mse.sum() * 0.0

    # model.unet_deformable_context: the deformable rows' generated CONTEXT particles, scored by the
    # future rows' recipe (valid_mask is deformable-only here) and folded in as ONE row-count-weighted
    # mean over the whole [context | future] window, as the context pose term is below.
    if deformable_context and noisy_ctx_pose is not None and mse is not None and bool(valid_mask.any()):
        _base_dc = model.module if hasattr(model, "module") else model
        ctx_part_pred = getattr(_base_dc, "_last_ctx_part_pred", None)
        if ctx_part_pred is None:
            raise RuntimeError("unet_deformable_context: the model did not stash context particle predictions")
        if _fm_x0_loss:
            ctx_part_target = ctx_norm
        else:
            ctx_part_target = diffusion.get_target(ctx_norm, ctx_noise, context_timesteps)
            ctx_part_pred = diffusion.as_velocity(ctx_part_pred, noisy_context, context_timesteps)
        ctx_part_mse = (ctx_part_pred.float() - ctx_part_target) ** 2          # [B, C-1, N, 3]
        if ctx_row_weight is not None:
            ctx_part_mse = ctx_part_mse * ctx_row_weight
        ctx_valid = valid_mask[:, :1].expand(-1, ctx_part_mse.shape[1], -1, -1)
        n_fut, n_ctx = valid_mask.sum(), ctx_valid.sum()
        loss = (loss * n_fut + ctx_part_mse[ctx_valid].mean() * n_ctx) / (n_fut + n_ctx)

    # Tier-2 v2 pose term: velocity-space MSE by default, even when the particle
    # sample-mode term is clean x0 MSE. Sample-to-velocity conversion then weights
    # pose x0 errors by 1/t^2 (about 1,000 on average for t down to 0.001): the pose
    # term sets the clipped gradient and starves the particle term (folding, measured
    # 2026-09-19). pose_loss_space="sample" compares clean pose to clean pose instead.
    # The default preserves the historical checkpoint recipe explicitly.
    pose_native_loss_val = 0.0
    if pose_native:
        if pose_loss_space == "sample":
            pose_target = pose_x0
        else:
            pose_target = diffusion.get_target(pose_x0, pose_noise, timesteps)
            pose_pred = diffusion.as_velocity(pose_pred, noisy_pose, timesteps)
        pose_mse = (pose_pred - pose_target) ** 2                        # [B, F, Kr, 6]
        if row_weight is not None:                      # clean-target space, checked above
            pose_mse = pose_mse * row_weight
        def _pose_mean(mse):                            # mean over the channels the bodies use
            if pose_mask is None:
                return mse.mean()
            return (mse * pose_mask).sum() / (pose_mask.sum() * mse.shape[0] * mse.shape[1])
        pose_native_loss = _pose_mean(pose_mse)
        if noisy_ctx_pose is not None:
            # The generated context rows are scored by the same recipe as the future rows and
            # folded in as ONE row-count-weighted mean over the whole [context | future] window.
            _base_pc = model.module if hasattr(model, "module") else model
            ctx_pose_pred = getattr(_base_pc, "_last_ctx_pose_pred", None)
            if ctx_pose_pred is None:
                raise RuntimeError("pose_noise_context: the model did not stash context pose "
                                   "predictions (not the UNet backbone?)")
            if pose_loss_space == "sample":
                ctx_pose_target = ctx_pose_x0
            else:
                ctx_pose_target = diffusion.get_target(ctx_pose_x0, ctx_pose_noise, context_timesteps)
                ctx_pose_pred = diffusion.as_velocity(ctx_pose_pred, noisy_ctx_pose, context_timesteps)
            ctx_pose_mse = (ctx_pose_pred - ctx_pose_target) ** 2        # [B, C-1, Kr, 6]
            if ctx_row_weight is not None:
                ctx_pose_mse = ctx_pose_mse * ctx_row_weight
            ctx_pose_loss = _pose_mean(ctx_pose_mse)
            n_fut, n_ctx = pose_mse.shape[1], ctx_pose_mse.shape[1]
            pose_native_loss = (pose_native_loss * n_fut + ctx_pose_loss * n_ctx) / (n_fut + n_ctx)
        loss = loss + lambda_pose_native * pose_native_loss
        pose_native_loss_val = float(pose_native_loss.item())

    # DF-on-context supervision (unet): the wrapper stashed the context-token
    # predictions this forward; supervise them with the same FM target recipe
    # as the future tokens so context frames train exactly like DF frames.
    # Combined as a token-count-weighted mean — identical to concatenating
    # [ctx ; future] along the frame dim and taking one masked mean.
    ctx_loss_val = None
    if supervise_context:
        assert noise_context, \
            "supervise_context: uniform-loss full-batch recipe only"
        _base_sc = model.module if hasattr(model, "module") else model
        ctx_pred = getattr(_base_sc, "_last_ctx_pred", None)
        assert ctx_pred is not None, \
            "supervise_context: model did not stash context predictions " \
            "(model.unet_supervise_context off?)"
        # Same parameterization switch as the future target above: sample-
        # parameterized FM supervises ctx tokens directly in x0 space (the
        # CLEAN context deltas); velocity mode uses the FM velocity target.
        ctx_target = ctx_norm if _fm_x0_loss \
            else diffusion.get_target(ctx_norm, ctx_noise, context_timesteps)
        ctx_mse = (ctx_pred.float() - ctx_target) ** 2
        ctx_valid = valid_mask[:, :1].expand(-1, ctx_mse.shape[1], -1, -1)
        ctx_loss = ctx_mse[ctx_valid].mean()
        n_fut, n_ctx = valid_mask.sum(), ctx_valid.sum()
        loss = (loss * n_fut + ctx_loss * n_ctx) / (n_fut + n_ctx)
        ctx_loss_val = ctx_loss.item()

    # Auxiliary losses (FM only): pos_cum, accel, com, spread. No-op if disabled.
    # pred is None on pure-rigid pose_native batches (particle decode skipped).
    aux_loss_values: dict = {}
    if aux_loss_weights and pred is not None:
        pred_x0 = pred if _fm_x0_loss else _flow_matching_x0(pred, x_t, timesteps, diffusion.num_train_timesteps)
        aux_loss_values = compute_aux_losses(
            pred_x0=pred_x0, x_0=x_0, valid_mask=valid_mask,
            obj_ids=obj_ids_loss, weights=aux_loss_weights, batch=batch,
        )
        for name, val in aux_loss_values.items():
            w = float(aux_loss_weights.get(name, 0.0))
            loss = loss + w * val

    # x0-space diagnostic in RAW physical units — comparable across runs regardless
    # of normalization stats, prediction target/space, or loss recipe. FM only.
    x0_mse_raw_val = None
    if pred is not None and bool(valid_mask.any()):   # empty on pure-rigid batches
        with torch.no_grad():
            _px0 = pred.detach() if _fm_x0_loss else _flow_matching_x0(pred.detach(), x_t, timesteps, diffusion.num_train_timesteps)
            _pred_raw = batch.prediction_stats.denormalize(_px0)
            _raw_x0 = batch.prediction_x0
            x0_mse_raw_val = ((_pred_raw - _raw_x0) ** 2)[valid_mask].mean().item()


    # Mesh-pair loss: preserves cloth mesh structure by penalizing edge-distance
    # divergence vs GT (v7+). Requires pred_x0; for sample mode that's `pred`,
    # for velocity mode we recover x_0 = x_t - t*v.
    aux_mesh_loss_val = 0.0
    if lambda_mesh > 0 and pred is not None:
        if diffusion.prediction_type == "velocity":
            t_norm = (timesteps.float() / diffusion.num_train_timesteps)
            while t_norm.ndim < 4:
                t_norm = t_norm.unsqueeze(-1)
            pred_x0_for_mesh = x_t - t_norm * pred
        else:
            pred_x0_for_mesh = pred
        aux_mesh_loss = compute_mesh_pair_loss(pred_x0_for_mesh.float(), x_0, batch, K=mesh_pair_K)
        loss = loss + lambda_mesh * aux_mesh_loss
        aux_mesh_loss_val = aux_mesh_loss.item()

    # Metrics
    metrics = {"loss": loss.item()}
    if ctx_loss_val is not None:
        metrics["loss_ctx"] = ctx_loss_val
    if x0_mse_raw_val is not None:
        metrics["x0_mse_raw"] = x0_mse_raw_val
    for name, val in aux_loss_values.items():
        metrics[f"loss_{name}"] = val.item()
    if pose_native:
        metrics["loss_pose_native"] = pose_native_loss_val
    if aux_mesh_loss_val > 0.0 or lambda_mesh > 0:
        metrics["loss_mesh"] = aux_mesh_loss_val
    # Log first sample's timestep
    if timesteps.ndim == 1:
        metrics["timestep"] = timesteps[0].item()
    else:
        metrics["timestep"] = timesteps[0, 0].item()

    draw = dict(
        batch=batch,  # Effective geometry/actions/targets for diagnostics; the caller's batch may differ.
        timesteps=timesteps, x0=x_0, x_t=x_t, valid_mask=valid_mask,
        context=noisy_context, context_timesteps=context_timesteps,
        pose_native=pose_native, pose_x0=pose_x0,
        prediction=pred, target=target if pred is not None else None,
        pose_prediction=pose_pred if pose_native else None,
        pose_target=pose_target if pose_native else None,
    )
    if pose_mask is not None:                  # jointed bodies only; consumers use draw.get()
        draw["pose_mask"] = pose_mask
    return loss, metrics, draw
