"""One construction contract for unified training and checkpoint inference."""
from dataclasses import fields
import warnings

from models.config import ModelConfig


HISTORICAL_REVISION = "c5fac334540abec89c62824a528e60985e75a7c7"
_CHECKPOINT_ALIASES = {"classic": "transformer", "unet_std": "unet"}
# These belonged to retired learned paths. Only their disabled state is safe to
# discard. Never partially load a retired architecture into a retained model.
_DISABLED_LEGACY_FIELDS = {
    "predict_rigid_pose": False, "predict_body_poses": False,
    "pose_action_dim": 0, "gripper_template_path": None,
    "commanded_gripper_tokens_enabled": False,
    "num_commanded_gripper_tokens_per_gripper": 32, "per_object_noise": False,
}
# None of these ever affected either retained backbone's arithmetic. In
# particular the batched stratification flags applied only to the joint model.
_INERT_FIELDS = {
    "use_continuous_physics_adaln", "use_cross_object_edge", "causal_context",
    "batched_stratified_encode", "batched_stratified_decode", "latents_per_object",
    "register_token", "concat_noisy_z", "rope_shift_frame_start",
    "flow_matching_prediction_type", "num_train_timesteps", "fps_random_start",
}
_INERT_UNET_FIELDS = {
    "unet_mixer": "full", "unet_k": 16, "unet_unpool_k": 8,
    "unet_block_graph": "noisy", "unet_unpool_graph": "noisy",
    "unet_unpool_edge": "std",
}
_HISTORICAL_UNET_ROPE_F_MIN = 0.5       # what a checkpoint that does not name the key trained with


def strict_dataclass(config_type, values):
    known = {field.name for field in fields(config_type)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown or retired {config_type.__name__} options: {sorted(unknown)}. "
            f"Historical workflows remain at revision {HISTORICAL_REVISION}.")
    return config_type(**values)


def parse_model_config(options, *, checkpoint=False):
    values = dict(options)
    architecture = values.pop("architecture", "transformer")
    if checkpoint:
        architecture = _CHECKPOINT_ALIASES.get(architecture, architecture)
    if architecture not in ("transformer", "unet"):
        raise ValueError(
            f"Unsupported model.architecture {architecture!r}; expected transformer or unet. "
            f"Historical workflows remain at revision {HISTORICAL_REVISION}.")
    retired = []
    if checkpoint:
        for name, disabled in _DISABLED_LEGACY_FIELDS.items():
            if name not in values:
                continue
            if values[name] != disabled:
                raise ValueError(
                    f"Checkpoint uses retired model option {name}={values[name]!r}; "
                    f"reproduce it at revision {HISTORICAL_REVISION}.")
            values.pop(name)
            retired.append(name)
        for name in sorted(_INERT_FIELDS & values.keys()):
            values.pop(name)
            retired.append(name)
        if values.get("rigid_delta_frame") in ("body", "world"):
            values.pop("rigid_delta_frame")
            retired.append("rigid_delta_frame")
        if architecture == "unet":
            for name, expected in _INERT_UNET_FIELDS.items():
                if name in values and values[name] == expected:
                    values.pop(name)
                    retired.append(name)
            # HISTORICAL DEFAULT. The fresh-run default moved from 0.5 to 0.15 (5.58 m unique range)
            # on 2026-09-20; checkpoints saved before that never wrote the key, and rebuilding them at
            # the new default would silently re-band their positions. Absent means the value they
            # trained with.
            values.setdefault("unet_rope_f_min", _HISTORICAL_UNET_ROPE_F_MIN)
    if retired:
        warnings.warn(f"Ignoring verified inert or disabled historical model fields: {retired}",
                      stacklevel=2)
    return architecture, strict_dataclass(ModelConfig, values)


def context_timesteps_trained(cfg):
    """Whether training supplied context timesteps; their default is backbone-specific."""
    train = cfg.get("train", {})
    default = bool(cfg.get("diffusion_forcing", False) and cfg.model.get("all_relative", True))
    return bool(train.get("noise_context", default) or train.get("clean_ctx_timesteps", False))


def context_predicted(cfg):
    """True when the checkpoint was trained to DENOISE its context rows, not merely read
    them (train.pose_noise_context). Such a model has never seen a clean context beside a
    noisy future, so its sampler must generate the context alongside the future instead of
    pinning it at timestep zero (inference.run_diffusion_inference_df)."""
    return bool(cfg.get("train", {}).get("pose_noise_context", False))


def clean_context_timestep(cfg):
    train = cfg.get("train", {})
    default = bool(cfg.get("diffusion_forcing", False) and cfg.model.get("all_relative", True))
    if train.get("noise_context", default):
        return 0
    return 1 if train.get("clean_ctx_timesteps", False) else None


def validate_particle_training_config(cfg):
    architecture, model_cfg = parse_model_config(cfg.model)
    if model_cfg.context_len != int(cfg.data.context_len):
        raise ValueError("model.context_len and data.context_len must agree")
    if cfg.data.format != "unified":
        raise ValueError("Fresh training supports data.format='unified' only")
    if cfg.get("prediction_target", "delta") != "delta":
        raise ValueError("Unified datasets support prediction_target='delta' only")
    if not model_cfg.all_relative or model_cfg.context_len < 2:
        raise ValueError("Unified training requires all_relative and at least two context frames")
    if not model_cfg.pose_native:
        raise ValueError("Unified training requires model.pose_native=true")
    if not cfg.data.get("emit_sidecar", True):
        raise ValueError("Unified training requires data.emit_sidecar=true")
    if cfg.get("denoise_method") != "flow_matching":
        raise ValueError("Unified training supports flow_matching only")
    if cfg.flow_matching.get("prediction_type", "velocity") not in ("velocity", "sample"):
        raise ValueError("Unified flow_matching.prediction_type must be velocity or sample")
    if not cfg.get("diffusion_forcing", True):
        raise ValueError("Unified training requires diffusion_forcing=True")
    if cfg.get("loss_weighting", "uniform") != "uniform":
        raise ValueError("Unified training requires uniform loss weighting")
    if cfg.get("diffusion_forcing", False) and float(cfg.flow_matching.get("shift", 1.0)) != 1.0:
        raise ValueError("Diffusion-forcing sampling uses an unshifted grid; shift must be 1")
    if architecture == "transformer":
        if model_cfg.decoder_query_self_attn:
            raise ValueError("decoder_query_self_attn is inactive under unified stratified decoding")
        if not (model_cfg.use_stratified_encoding and model_cfg.use_stratified_decoding
                and model_cfg.use_rotary and not model_cfg.bypass_compression):
            raise ValueError("Unified Transformer requires stratified encoding/decoding, rotary "
                             "positions and bypass_compression=false")
        return
    checks = {
        "diffusion forcing required": bool(cfg.get("diffusion_forcing", False)),
        "uniform loss weighting required": cfg.get("loss_weighting", "uniform") == "uniform",
        "subset supervision unsupported": cfg.train.get("supervised_points") in (None, 0, -1),
    }
    if cfg.train.get("supervise_context", False):
        checks["context supervision requires noise_context and unet_supervise_context"] = bool(
            cfg.train.get("noise_context", False) and model_cfg.unet_supervise_context)
    allowed_aux = {"pos_cum", "strain_inc", "strain_cum", "floor", "accel",
                   "strain_cum_mats", "floor_z"}
    unsupported_aux = [name for name, value in (cfg.train.get("aux_losses") or {}).items()
                       if name not in allowed_aux and isinstance(value, (float, int)) and value != 0]
    failures = [message for message, valid in checks.items() if not valid]
    if unsupported_aux:
        failures.append(f"unsupported auxiliary losses {unsupported_aux}")
    if failures:
        raise ValueError("UNet recipe: " + "; ".join(failures))


def build_particle_model(cfg, *, checkpoint=False):
    architecture, model_cfg = parse_model_config(cfg.model, checkpoint=checkpoint)
    if architecture == "transformer":
        from models.transformer import TrajectoryTransformer
        model = TrajectoryTransformer(model_cfg)
    else:
        from models.unet_std import UNetWrapper
        model = UNetWrapper(model_cfg)
    model._ctx_timesteps_trained = context_timesteps_trained(cfg)
    model._clean_context_timestep = clean_context_timestep(cfg)
    model._context_predicted = context_predicted(cfg)
    return model
