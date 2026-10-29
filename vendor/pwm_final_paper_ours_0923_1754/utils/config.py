"""Small shared defaults for the two unified training recipes.

YAML files hold backbone/experiment differences. The fully resolved configuration,
including a numerical normalization snapshot, is saved with every new run.
This is not a configuration inheritance framework.
"""
from pathlib import Path
from dataclasses import fields
from numbers import Real
import math

from omegaconf import OmegaConf


DEFAULTS = {
    "seed": 42,
    "resume": None,
    "normalization": {"profile": "xy_p99_per_domain"},
    "data": {
        "format": "unified", "wds_location": "unified://v1",
        "domains": ["mpm", "pickup", "cloth", "pushing"],
        "context_len": 4, "future_len": 16, "num_workers": 2,
        "num_collate": 1, "batch_size": 4, "max_objects": -1,
        "emit_sidecar": True, "bucket_max_items": 64,
    },
    "model": {
        "all_relative": True, "pose_native": True, "dropout": 0.1,
        "num_mat_types": 6,
    },
    "prediction_target": "delta",
    "denoise_method": "flow_matching",
    "loss_weighting": "uniform",
    "diffusion_forcing": True,
    "df_schedule": "pyramid",
    "df_uncertainty_scale": 1.0,
    "flow_matching": {
        "num_train_timesteps": 1000, "shift": 1.0,
        "inference_steps": 25, "prediction_type": "velocity",
    },
    "train": {
        "steps": 200000, "warmup_steps": 5000, "lr": 1e-4,
        "weight_decay": 0.001, "grad_clip": 1.0, "ema_decay": 0.99,
        "overfit_batches": -1, "max_batch_size": -1,
        "mixed_precision": "bf16", "grad_accum": 1, "compile": False,
        "noise_context": False, "loss_spike_threshold": 10.0,
        "grad_norm_spike_threshold": 50.0, "spike_grace_steps": 500,
        "lambda_pose_native": 1.0, "stop_after_step": 0,
    },
    "log": {
        "dir": "./runs", "project": "particle_world_model", "entity": None,
        "every": 100, "val_every": 1000, "val_batches": 40,
        "sampler_batches": 8, "save_every": 20000, "sample_every": 0,
        "best_gate": "ema", "best_metric": "sample_mm",
    },
}

# Optional, implemented controls. Unknown nested keys (including CLI typos) are
# errors, not knobs silently accepted by a permissive dictionary merge.
OPTIONAL = {
    "data": {"eval_domains", "domain_weights", "seed", "first_window", "obj0_height_color", "roots",
             "window_motion", "batch_across_episodes"},
    "train": {"compile_regional", "compile_dynamic", "compile_mode", "compile_l0",
              "ddp_static_graph", "nccl_timeout_min", "adam_betas", "adam_eps",
              "lr_schedule", "wsd_warmup_frac", "wsd_decay_frac", "ema_warmup",
              "rot_aug", "supervise_context", "light_context_noise_std", "light_pose_noise_std",
              "light_pose_noise_anchor", "light_pose_noise_target", "light_pose_noise_scope",
              "light_pose_noise_type", "clean_ctx_timesteps", "pose_loss_space",
              "pose_noise_context",                               # the context is generated, not read
              "aux_losses", "lambda_mesh", "mesh_pair_K", "clip_deltas", "clip_pose_deltas",
              "action_dropout", "init_from",
              "timestep_schedule", "ramp_slope", "ramp_jitter",   # joint per-frame noise-level patterns
              "stream_states", "stream_phase_shares", "stream_jitter",   # the streaming rollout's own windows
              "stream_keep", "stream_given",                             # ... restricted to one rollout setting
              "velocity_weight_cap"},                             # min(1/t^2, cap) on x0-space errors
    "log": {"max_viz_samples", "num_vis", "viz_first_window", "viz_weights",
            "save_best", "save_pre_decay", "save_final",
            "stream_rollout_keep"},                      # logged rollouts walk the streaming staircase
    "flow_matching": {"time_sampler", "beta_alpha", "beta_beta", "beta_t_min"},
}


def _number(value, name, *, minimum=None, positive=False, maximum=None, integer=False, allow_inf=False):
    if isinstance(value, bool) or not isinstance(value, Real) or (integer and not isinstance(value, int)):
        raise ValueError(f"{name} must be a {'whole integer' if integer else 'number'}")
    if math.isnan(value) or (not allow_inf and not math.isfinite(value)):
        raise ValueError(f"{name} must be finite")
    if ((minimum is not None and value < minimum) or (positive and value <= 0)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} is outside its supported range")
    return value


def _validate_data_roots(cfg):
    """Explicit dataset releases must resolve before model allocation or corpus access."""
    roots = cfg.data.get("roots")
    if roots is None:
        return
    if not OmegaConf.is_dict(roots):
        raise ValueError("data.roots must be a mapping of domain -> {train, val, exclusions}")
    from data.unified_dataset import resolve_data_roots
    resolved=resolve_data_roots(OmegaConf.to_container(roots, resolve=True))
    if resolved.get('mpm',{}).get('task')=='drag' or 'mpm_drag' in resolved:
        if cfg.model.get('no_action_cond',False) or (
                cfg.model.architecture=='unet' and not cfg.model.get('unet_actions',False)):
            raise ValueError('MPM drag requires particle action conditioning (UNet: unet_actions=true)')
        if cfg.train.get('action_dropout',0):
            raise ValueError('Pose-only action dropout does not support MPM drag')


def _known_sections(cfg):
    unknown = set(cfg) - (set(DEFAULTS) | {"run_name"})
    if unknown:
        raise ValueError(f"Unknown training configuration sections: {sorted(unknown)}")
    for section, extra in OPTIONAL.items():
        values = cfg.get(section)
        if not OmegaConf.is_dict(values):
            raise ValueError(f"{section} must be a mapping")
        unknown = set(values) - (set(DEFAULTS[section]) | extra)
        if unknown:
            raise ValueError(f"Unknown or retired {section} options: {sorted(unknown)}")
    if not OmegaConf.is_dict(cfg.model):
        raise ValueError("model must be a mapping")


def _validate_scalars(cfg):
    _number(cfg.seed, "seed", minimum=0, integer=True)
    if not isinstance(cfg.run_name, str) or not cfg.run_name or Path(cfg.run_name).name != cfg.run_name \
            or cfg.run_name in (".", ".."):
        raise ValueError("run_name must be a nonempty directory basename")
    if cfg.resume is not None and (not isinstance(cfg.resume, str) or not cfg.resume):
        raise ValueError("resume must be null or a nonempty checkpoint path")
    integer_ranges = {
        "data": {"context_len": 2, "future_len": 1, "num_workers": 0, "batch_size": 1,
                 "bucket_max_items": 0, "seed": 0},
        "train": {"steps": 1, "warmup_steps": 0, "grad_accum": 1, "spike_grace_steps": 0,
                  "stop_after_step": 0, "nccl_timeout_min": 1, "mesh_pair_K": 1},
        "log": {"every": 1, "val_every": 0, "val_batches": 1, "sampler_batches": 0,
                "save_every": 0, "sample_every": 0, "max_viz_samples": 1, "num_vis": 1},
        "flow_matching": {"num_train_timesteps": 2, "inference_steps": 1},
    }
    for section, entries in integer_ranges.items():
        for name, minimum in entries.items():
            if name in cfg[section]:
                _number(cfg[section][name], f"{section}.{name}", minimum=minimum, integer=True)
    for name in ("overfit_batches", "max_batch_size"):
        value = _number(cfg.train[name], f"train.{name}", minimum=-1, integer=True)
        if value == 0:
            raise ValueError(f"train.{name} must be -1 (disabled) or positive")
    if cfg.data.num_collate != 1 or isinstance(cfg.data.num_collate, bool) or cfg.data.max_objects != -1:
        raise ValueError("Unified data requires num_collate=1 and max_objects=-1; use batch_size and domains")
    _number(cfg.data.num_collate, "data.num_collate", integer=True)
    _number(cfg.data.max_objects, "data.max_objects", integer=True)
    bool_fields = {
        "data": ("emit_sidecar", "first_window", "obj0_height_color"),
        "train": ("compile", "compile_regional", "compile_l0", "ddp_static_graph", "ema_warmup",
                  "noise_context", "pose_noise_context", "rot_aug", "supervise_context",
                  "clean_ctx_timesteps", "clip_deltas",
                  "clip_pose_deltas", "light_pose_noise_anchor"),
        "log": ("viz_first_window", "save_best", "save_pre_decay", "save_final"),
    }
    for section, names in bool_fields.items():
        for name in names:
            if name in cfg[section] and not isinstance(cfg[section][name], bool):
                raise ValueError(f"{section}.{name} must be a boolean")
    if "compile_dynamic" in cfg.train and cfg.train.compile_dynamic is not None \
            and not isinstance(cfg.train.compile_dynamic, bool):
        raise ValueError("train.compile_dynamic must be true, false, or null")
    if cfg.train.mixed_precision not in ("no", "fp16", "bf16"):
        raise ValueError("train.mixed_precision must be no, fp16 or bf16")
    for name in ("lr", "adam_eps"):
        _number(cfg.train.get(name, 1e-8), f"train.{name}", positive=True)
    for name in ("weight_decay", "lambda_pose_native", "lambda_mesh", "light_context_noise_std",
                 "light_pose_noise_std"):
        _number(cfg.train.get(name, 0), f"train.{name}", minimum=0)
    if cfg.train.get("light_pose_noise_anchor", False) and not cfg.train.get("light_pose_noise_std", 0):
        raise ValueError("train.light_pose_noise_anchor needs a positive train.light_pose_noise_std")
    anchor = cfg.train.get("light_pose_noise_anchor", False)
    target = cfg.train.get("light_pose_noise_target", "restore")
    if target not in ("restore", "carry") or (target == "carry" and not anchor):
        raise ValueError("train.light_pose_noise_target must be restore, or carry with anchor=true")
    from utils.pose_native import pose_noise_scope
    pose_noise_scope(anchor, cfg.train.get("light_pose_noise_scope", "auto"))
    if cfg.train.get("light_pose_noise_type", "iid") not in ("iid", "random_walk"):
        raise ValueError("train.light_pose_noise_type must be iid or random_walk")
    if cfg.train.get("pose_loss_space", "velocity") not in ("velocity", "sample"):
        raise ValueError("train.pose_loss_space must be velocity or sample")
    if (cfg.train.get("pose_loss_space", "velocity") == "sample"
            and cfg.flow_matching.get("prediction_type", "velocity") != "sample"):
        raise ValueError("train.pose_loss_space=sample requires flow_matching.prediction_type=sample")
    if anchor and cfg.train.get("clip_pose_deltas", True):
        raise ValueError("train.light_pose_noise_anchor requires train.clip_pose_deltas=false "
                         "to preserve its restore/carry pose targets")
    for name in ("loss_spike_threshold", "grad_norm_spike_threshold"):
        _number(cfg.train[name], f"train.{name}", positive=True, allow_inf=True)
    if cfg.train.grad_clip is not None:
        _number(cfg.train.grad_clip, "train.grad_clip", positive=True)
    decay = _number(cfg.train.ema_decay, "train.ema_decay", minimum=0, maximum=1)
    if decay == 1:
        raise ValueError("train.ema_decay must be below 1; otherwise EMA never updates")
    betas = cfg.train.get("adam_betas", [0.9, 0.999])
    if betas is not None:
        if (not OmegaConf.is_list(betas) and not isinstance(betas, (list, tuple))) or len(betas) != 2:
            raise ValueError("train.adam_betas must be two values in [0,1)")
        if any(_number(value, "train.adam_betas", minimum=0, maximum=1) == 1 for value in betas):
            raise ValueError("train.adam_betas must be below 1")
    if cfg.df_schedule not in ("pyramid", "full_sequence"):
        raise ValueError("df_schedule must be pyramid or full_sequence")
    _number(cfg.df_uncertainty_scale, "df_uncertainty_scale", positive=True)
    if cfg.log.best_gate not in ("ema", "raw") or cfg.log.get("viz_weights", "ema") not in ("ema", "raw"):
        raise ValueError("log.best_gate and log.viz_weights must be ema or raw")
    _metric = cfg.log.get("best_metric", "sample_mm")
    if _metric not in ("sample_mm", "loss"):
        raise ValueError("log.best_metric must be sample_mm or loss")
    # sample_mm is the full-sampler error, which only the EMA pass produces and
    # only when that pass runs. Both are configuration, so both fail here.
    if _metric == "sample_mm" and bool(cfg.log.get("save_best", True)) and cfg.log.val_every > 0:
        if cfg.log.sampler_batches <= 0:
            raise ValueError("log.best_metric=sample_mm requires log.sampler_batches > 0")
        if cfg.log.best_gate != "ema":
            raise ValueError("log.best_metric=sample_mm scores the EMA sampler pass; "
                             "set log.best_gate=ema or log.best_metric=loss")
    if not isinstance(cfg.log.dir, str) or not cfg.log.dir:
        raise ValueError("log.dir must be a nonempty path")


def _validate_training_policy(cfg):
    from data.unified_dataset import selected_domains
    domains, _ = selected_domains(cfg.data.domains, cfg.data.get("domain_weights"))
    eval_domains, _ = selected_domains(cfg.data.eval_domains) if cfg.data.get("eval_domains") is not None else (domains, {})
    t, fm = cfg.train, cfg.flow_matching
    if t.get("lr_schedule", "cosine") not in ("cosine", "constant", "wsd"):
        raise ValueError("train.lr_schedule must be cosine, constant or wsd")
    if t.get("lr_schedule", "cosine") == "wsd":
        ramp = _number(t.get("wsd_warmup_frac", .05), "train.wsd_warmup_frac", minimum=0, maximum=1)
        decay = _number(t.get("wsd_decay_frac", .1), "train.wsd_decay_frac", positive=True, maximum=1)
        if ramp >= 1 or decay > 1 - ramp:
            raise ValueError("WSD warmup/decay fractions must fit within the step budget")
    elif t.warmup_steps >= t.steps:
        raise ValueError("train.warmup_steps must be below train.steps")
    if t.stop_after_step >= t.steps:
        raise ValueError("train.stop_after_step must be 0 (disabled) or below train.steps")
    if fm.get("time_sampler", "uniform") not in ("uniform", "beta"):
        raise ValueError("flow_matching.time_sampler must be uniform or beta")
    if fm.get("time_sampler", "uniform") == "beta":
        for name, default in (("beta_alpha", 1.5), ("beta_beta", 1.0)):
            _number(fm.get(name, default), f"flow_matching.{name}", positive=True)
        if _number(fm.get("beta_t_min", .001), "flow_matching.beta_t_min", minimum=0, maximum=1) == 1:
            raise ValueError("flow_matching.beta_t_min must be below 1")
    elif any(name in fm for name in ("beta_alpha", "beta_beta", "beta_t_min")):
        raise ValueError("Beta parameters are inactive unless flow_matching.time_sampler=beta")
    if t.noise_context and not t.get("pose_noise_context", False):
        raise ValueError("train.noise_context is retired on its own: it noised every particle's history "
                         "independently, putting rigid bodies' particles away from their pose. Checkpoints "
                         "trained with it still load, sample and serve as train.init_from; use clean "
                         "context, train.light_context_noise_std, or train.pose_noise_context, which noises "
                         "the context POSES and rebuilds each rigid body's particle rows from them")
    if t.get("pose_noise_context", False):
        # The context becomes a generated channel: its rows carry their own noise levels and are
        # predicted. Rigid rows are rebuilt from the noisy context pose (utils/pose_native.py), so
        # this needs the pose-native UNet, and the particle side of the same stream.
        if not t.noise_context:
            raise ValueError("train.pose_noise_context requires train.noise_context (the particle side "
                             "of the generated context stream)")
        from models.factory import parse_model_config
        _arch, _model = parse_model_config(cfg.model)
        if _arch != "unet" or not _model.pose_native:
            raise ValueError("train.pose_noise_context needs model.architecture=unet with pose_native "
                             "(the context pose rewrite and the stashed context predictions are the "
                             "UNet's pose-native path)")
    if t.get("compile_regional", False) and cfg.model.architecture != "unet":
        raise ValueError("Regional compilation requires model.architecture=unet")
    if t.get("compile_mode") not in (None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"):
        raise ValueError("Unsupported train.compile_mode")
    if t.get("compile_mode") is not None and not t.get("compile_regional", False):
        raise ValueError("train.compile_mode is only consumed by regional compilation")
    if t.get("compile_l0", False) and not t.get("compile_regional", False):
        raise ValueError("train.compile_l0 requires regional compilation")
    if t.get("rot_aug", False) and domains != ("mpm",):
        raise ValueError("rot_aug currently rotates particle channels only; use MPM-only training")
    if t.get("ddp_static_graph", False) and domains != ("mpm",):
        raise ValueError("ddp_static_graph requires MPM-only training; mixed particle/pose branches change used parameters")
    if t.get("lambda_mesh", 0) > 0 and set(domains + eval_domains) != {"cloth"}:
        raise ValueError("lambda_mesh currently assumes object 0 is cloth; use cloth-only domains")
    if t.get("supervise_context", False):
        raise ValueError("train.supervise_context is retired: it supervised noisy context frames, "
                         "which require the retired train.noise_context")
    from utils.trajectory_objective import timestep_schedule_options
    timestep_schedule_options(cfg)            # fails closed on a bad schedule, slope, jitter or cap
    keep = cfg.log.get("stream_rollout_keep")
    if keep is not None:
        # One window of the streaming rollout, c rows kept clean. It is a sampler setting, so it
        # does not have to match train.stream_keep, but it must fit the window.
        rows = int(cfg.data.context_len) - 1 + int(cfg.data.future_len)
        if isinstance(keep, bool) or not isinstance(keep, int) or not 0 <= keep < rows:
            raise ValueError(f"log.stream_rollout_keep must be a whole number of rows in [0, {rows})")
    dropout = _number(t.get("action_dropout", 0.0), "train.action_dropout", minimum=0, maximum=1)
    if dropout > 0:
        from models.factory import parse_model_config
        _, model = parse_model_config(cfg.model)
        if cfg.model.architecture != "unet" or not model.pose_native or model.unet_actions:
            raise ValueError("train.action_dropout hides the pose-token action stream only; it requires "
                             "model.architecture=unet with pose_native and unet_actions=false")
    _pam = str(((cfg.get("model") or {}).get("unet_pose_action_mode", "skip")) or "skip")
    if _pam not in ("skip", "clean"):
        raise ValueError(f"model.unet_pose_action_mode must be skip or clean, got {_pam!r}")
    if _pam == "clean":
        from models.factory import parse_model_config
        _, model = parse_model_config(cfg.model)
        if cfg.model.architecture != "unet" or not model.pose_native:
            raise ValueError("model.unet_pose_action_mode=clean conditions THROUGH the pose tokens: "
                             "it requires model.architecture=unet with pose_native")
        if model.unet_actions:
            raise ValueError("model.unet_pose_action_mode=clean passes no action vector: set "
                             "model.unet_actions=false, or the model reads the given motion twice")
    if bool(((cfg.get("model") or {}).get("unet_deformable_context", False))):
        from models.factory import parse_model_config
        _, model = parse_model_config(cfg.model)
        if cfg.model.architecture != "unet" or not model.pose_native or not t.get("pose_noise_context", False):
            raise ValueError("model.unet_deformable_context extends train.pose_noise_context's generated "
                             "context to deformable rows: it requires model.architecture=unet, pose_native "
                             "and train.pose_noise_context=true")
    init_from = t.get("init_from")
    if init_from is not None and (not isinstance(init_from, str) or not Path(init_from).is_absolute()
                                  or not Path(init_from).is_file()):
        raise ValueError("train.init_from must be null or an absolute path to an existing checkpoint")
    aux = t.get("aux_losses") or {}
    if t.get("aux_losses") is not None and not OmegaConf.is_dict(t.aux_losses):
        raise ValueError("train.aux_losses must be a mapping")
    allowed = {"pos_cum", "accel", "com", "spread", "spread_axes", "strain_inc", "strain_cum", "floor",
               "strain_cum_mats", "floor_z"}
    if set(aux) - allowed:
        raise ValueError(f"Unknown or unsupported train.aux_losses: {sorted(set(aux) - allowed)}")
    for name, value in aux.items():
        if name not in ("strain_cum_mats", "floor_z"):
            _number(value, f"train.aux_losses.{name}", minimum=0)
    if aux.get("accel", 0) > 0 and cfg.data.future_len < 2:
        raise ValueError("Acceleration supervision requires at least two future frames")
    if (aux.get("strain_inc", 0) > 0 or aux.get("strain_cum", 0) > 0) and set(domains + eval_domains) != {"mpm"}:
        raise ValueError("Strain auxiliaries require MPM-only training and evaluation: full-valid equal-sized object blocks")
    if aux.get("floor", 0) > 0 and aux.get("floor_z") != 0:
        raise ValueError("Canonical coordinates require explicit train.aux_losses.floor_z=0")
    if "floor_z" in aux:
        _number(aux["floor_z"], "train.aux_losses.floor_z")
    if "strain_cum_mats" in aux:
        values = aux["strain_cum_mats"]
        if not OmegaConf.is_list(values) or not values:
            raise ValueError("strain_cum_mats must be a nonempty list of material IDs")
        for value in values:
            _number(value, "strain_cum_mats", minimum=0, maximum=5, integer=True)


def _validate_model(cfg):
    from models.factory import parse_model_config, validate_particle_training_config
    architecture, model = parse_model_config(cfg.model)
    for field in fields(model):
        if field.name not in cfg.model:
            continue
        value, default = cfg.model[field.name], field.default
        if isinstance(default, bool) and field.name != "gradient_checkpointing" and not isinstance(value, bool):
            raise ValueError(f"model.{field.name} must be a boolean")
        if isinstance(default, int) and not isinstance(default, bool):
            _number(value, f"model.{field.name}", integer=True)
        elif isinstance(default, float):
            _number(value, f"model.{field.name}")
    _number(model.dropout, "model.dropout", minimum=0, maximum=1)
    if model.dropout == 1:
        raise ValueError("model.dropout must be below 1")
    if model.num_mat_types < 6:
        raise ValueError("Unified material vocabulary requires model.num_mat_types >= 6")
    if architecture == "transformer":
        if any(name.startswith("unet_") for name in cfg.model):
            raise ValueError("UNet model options are inactive on the Transformer")
        for name in ("dim", "depth", "heads", "num_latents", "k_neighbors"):
            _number(getattr(model, name), f"model.{name}", positive=True, integer=True)
        if model.dim % model.heads:
            raise ValueError("model.dim must be divisible by model.heads")
        if not isinstance(model.gradient_checkpointing, bool) and model.gradient_checkpointing != "full":
            raise ValueError("model.gradient_checkpointing must be true, false or full")
        for name in ("rope_position_scale", "rope_spatial_base"):
            _number(getattr(model, name), f"model.{name}", positive=True)
        if model.use_windowed_temporal:
            _number(model.temporal_window_size, "model.temporal_window_size", positive=True, integer=True)
        for name in ("rope_f_min", "rope_f_max", "rope_rotary_fraction"):
            if getattr(model, name) is not None:
                _number(getattr(model, name), f"model.{name}", positive=True)
        if model.rope_rotary_fraction is not None and model.rope_rotary_fraction > 1:
            raise ValueError("model.rope_rotary_fraction cannot exceed 1")
        if model.rope_f_min is not None and model.rope_f_max is not None and model.rope_f_min > model.rope_f_max:
            raise ValueError("model.rope_f_min cannot exceed rope_f_max")
    else:
        shared = {"architecture", "context_len", "all_relative", "pose_native", "articulated_bodies", "dropout",
                  "num_mat_types", "normalize_physics_features"}
        inactive = [name for name in cfg.model if name not in shared and not name.startswith("unet_")]
        if inactive:
            raise ValueError(f"Transformer model options are inactive on the UNet: {inactive}")
        _number(model.unet_geo_r_m, "model.unet_geo_r_m", positive=True)
        for name in ("unet_head_dim", "unet_expansion", "unet_t_dim", "unet_rope_f_min",
                     "unet_pool_k", "unet_interp_k", "unet_register_objs", "unet_serial_patch"):
            _number(getattr(model, name), f"model.{name}", positive=True)
        for name in ("unet_strides", "unet_widths", "unet_enc_depths", "unet_dec_depths"):
            value = getattr(model, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"model.{name} requires a nonempty comma-separated integer string")
            try:
                entries = [int(v.strip()) for v in value.split(",")]
            except ValueError as error:
                raise ValueError(f"model.{name} requires a comma-separated integer string") from error
            if any(v < 1 for v in entries):
                raise ValueError(f"model.{name} entries must be positive")
        from models.unet_std import build_unet_config
        try:
            ladder = build_unet_config(model)
        except (AssertionError, TypeError, ZeroDivisionError) as error:
            raise ValueError(f"Invalid UNet ladder: {error}") from error
        if any(width % ladder.head_dim for width in ladder.widths):
            raise ValueError("Every UNet width must be divisible by unet_head_dim")
        if not ladder.stratified:
            raise ValueError("Unified native counts require a stratified UNet L1 budget")
        from data.unified_dataset import selected_domains
        domains, _ = selected_domains(cfg.data.domains, cfg.data.get("domain_weights"))
        evaluation = cfg.data.get("eval_domains") or domains
        material_ids = {"mpm": {0, 1, 2}, "pickup": {3, 4}, "cloth": {4, 5}, "pushing": {3}, "water": {6},
                        "rl_push": {3, 4}, "mpm_fresh": {0, 1, 2}, "mpm_drag": {0, 1},
                        "folding": {4, 5}, "yikai": {3, 4}, "dex": {3, 4}, "mould": {3, 7}}
        needed = set().union(*(material_ids[d] for d in set(domains) | set(evaluation)))
        if "mould" in set(domains) | set(evaluation) and int(model.num_mat_types) <= 7:
            raise ValueError("the mould domain's dough is material id 7: set model.num_mat_types >= 8")
        if not ladder.l1_default and not ladder.l1_rule and needed - set(ladder.l1_by_mat):
            raise ValueError(f"UNet L1 table is missing selected-domain materials: {sorted(needed - set(ladder.l1_by_mat))}")
    validate_particle_training_config(cfg)


def _validate_water(cfg):
    """The opt-in water domain: material id 6 needs a 7-row material table, and the
    per-domain profile needs an artifact that carries a water entry (the frozen
    stats/normalization.json does not; stats/normalization_water3d.json does)."""
    from data.unified_dataset import MAT_WATER, selected_domains
    domains, _ = selected_domains(cfg.data.domains, cfg.data.get("domain_weights"))
    evaluation = selected_domains(cfg.data.eval_domains)[0] if cfg.data.get("eval_domains") is not None else domains
    if "water" not in set(domains) | set(evaluation):
        return
    if int(cfg.model.num_mat_types) <= MAT_WATER:
        raise ValueError(f"The water domain uses material id {MAT_WATER}; set model.num_mat_types={MAT_WATER + 1}")
    scales = cfg.normalization["scales"]
    if cfg.normalization["profile"] == "xy_p99_per_domain" and "water" not in scales["particle"]:
        raise ValueError("The water domain needs a normalization artifact with a water entry: "
                         "normalization.artifact=stats/normalization_water3d.json")


def _validate_rl_push(cfg):
    """The opt-in rl_push domain: the per-domain profile needs an artifact whose rl_push entry carries
    particle and pose scales (the frozen stats/normalization.json has none)."""
    from data.unified_dataset import selected_domains
    domains, _ = selected_domains(cfg.data.domains, cfg.data.get("domain_weights"))
    evaluation = selected_domains(cfg.data.eval_domains)[0] if cfg.data.get("eval_domains") is not None else domains
    if "rl_push" not in set(domains) | set(evaluation):
        return
    if cfg.normalization["profile"] == "xy_p99_per_domain" and "rl_push" not in cfg.normalization["scales"]["particle"]:
        raise ValueError("The rl_push domain needs a normalization artifact with an rl_push entry: "
                         "normalization.artifact=stats/normalization_rl_push.json")


def resolve_config(options, overrides=None):
    """Resolve a fresh recipe; checkpoint loading never injects these defaults."""
    from data.normalization import resolve_normalization
    from models.config import ModelConfig
    DEFAULT_UNET_ROPE_F_MIN = ModelConfig().unet_rope_f_min

    options = OmegaConf.create(options)
    if not OmegaConf.is_dict(options):
        raise ValueError("Training configuration must be a mapping")
    cfg = OmegaConf.merge(OmegaConf.create(DEFAULTS), options)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    _known_sections(cfg)
    if not cfg.get("run_name"):
        raise ValueError("run_name is required; choose a distinct output for each experiment")
    if cfg.model.get("architecture") not in ("transformer", "unet"):
        raise ValueError("Fresh recipes require model.architecture: transformer or unet")
    if cfg.data.format != "unified" or cfg.prediction_target != "delta":
        raise ValueError("Fresh training supports unified delta trajectories only")
    if cfg.denoise_method != "flow_matching" or cfg.diffusion_forcing is not True:
        raise ValueError("Unified training requires flow_matching and diffusion_forcing")
    if cfg.loss_weighting != "uniform":
        raise ValueError("Unified training requires uniform loss weighting")
    _validate_scalars(cfg)
    _validate_data_roots(cfg)
    cfg.model.setdefault("context_len", cfg.data.context_len)
    if int(cfg.model.context_len) != int(cfg.data.context_len):
        raise ValueError("model.context_len and data.context_len must agree")
    _validate_training_policy(cfg)
    _validate_model(cfg)
    if cfg.model.get("architecture") == "unet":
        # Stamp the RoPE band into the resolved config so every checkpoint this run saves names the
        # value it trained with. Without this, a recipe written before the 0.5 -> 0.15 default change
        # would train at 0.15 and resume at 0.5, because models/factory pins an ABSENT key to the
        # historical 0.5. Recipes that state a value keep it.
        cfg.model.unet_rope_f_min = float(cfg.model.get("unet_rope_f_min", DEFAULT_UNET_ROPE_F_MIN))
    spec = OmegaConf.to_container(cfg.normalization, resolve=True) if OmegaConf.is_config(cfg.normalization) else cfg.normalization
    cfg.normalization = resolve_normalization(spec)
    _validate_water(cfg)
    _validate_rl_push(cfg)
    _validate_indexed_domains(cfg)
    _validate_window_sampling(cfg)
    return cfg


def _validate_window_sampling(cfg):
    """Training window policy is a recipe decision: motion-weighted starts and cross-episode MPM batches."""
    from data.unified_dataset import validate_window_sampling
    motion, across = cfg.data.get("window_motion"), cfg.data.get("batch_across_episodes")
    if motion is not None and not OmegaConf.is_dict(motion):
        raise ValueError("data.window_motion must be a mapping of domain -> {alpha, floor_mm}")
    if across is not None and not OmegaConf.is_list(across):
        raise ValueError("data.batch_across_episodes must be a list of domains")
    validate_window_sampling(None if motion is None else OmegaConf.to_container(motion, resolve=True),
                             None if across is None else OmegaConf.to_container(across, resolve=True))


def _validate_indexed_domains(cfg):
    from data.unified_dataset import INDEXED_DOMAINS, selected_domains
    domains = selected_domains(cfg.data.domains, cfg.data.get("domain_weights"))[0]
    evaluation = selected_domains(cfg.data.eval_domains)[0] if cfg.data.get("eval_domains") is not None else domains
    selected = set(domains + evaluation) & set(INDEXED_DOMAINS)
    roots = cfg.data.get("roots") or {}
    for domain in sorted(selected):
        if domain not in roots:
            raise ValueError(f"{domain} requires explicit frozen data.roots.{domain} train/val indices")
        if (cfg.normalization.profile == "xy_p99_per_domain"
                and domain not in cfg.normalization.scales.particle):
            raise ValueError(f"{domain} requires its own entry in the selected normalization artifact")


def load_config(path: str | Path, overrides=None):
    return resolve_config(OmegaConf.load(path), overrides)
