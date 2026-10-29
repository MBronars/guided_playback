"""Retained model scope and non-vacuous parity with the round-three baseline.

Reference sources come from the frozen Git revision, not a second maintained
implementation. This checks the edited model modules at identical saved weights;
fresh Transformer modulation now starts at zero. The corrected SO(3) logarithm
is supplied to both sides of backbone parity.
End-to-end data/objective parity is covered by the separate audit receipts.
"""
from dataclasses import asdict, fields
import builtins
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import types

from omegaconf import OmegaConf
import pytest
import torch

from models.config import ModelConfig
from models.factory import HISTORICAL_REVISION, parse_model_config, validate_particle_training_config
from models.transformer import TrajectoryTransformer
from models.unet_std import UNetWrapper
from tests.test_unet_std_pose_native import _model_cfg, _noisy_pose_gt, make_batch


@lru_cache(None)
def _reference_module(path):
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("Frozen-model parity requires a source Git checkout")
    result = subprocess.run(["git", "show", f"{HISTORICAL_REVISION}:{path}"],
                            cwd=root, check=True, text=True, capture_output=True)
    name = "_round3_reference_" + path.replace("/", "_").replace(".", "_")
    module = types.ModuleType(name)
    module.__file__ = f"git:{HISTORICAL_REVISION}:{path}"
    sys.modules[name] = module  # dataclasses resolves its module through sys.modules
    def frozen_import(name, globals=None, locals=None, fromlist=(), level=0):
        if fromlist and not level and name.split(".")[0] in {"models", "diffusion", "utils"}:
            relative = name.replace(".", "/")
            # Package initializers and all internal numerical dependencies come
            # from the frozen revision as well (including retired pose constants).
            is_package = name in {"models", "models.components", "diffusion", "utils"}
            return _reference_module(relative + ("/__init__.py" if is_package else ".py"))
        return builtins.__import__(name, globals, locals, fromlist, level)
    module.__dict__["__builtins__"] = dict(vars(builtins), __import__=frozen_import)
    exec(compile(result.stdout, module.__file__, "exec"), module.__dict__)
    return module


def _options(architecture):
    if architecture == "unet":
        return _model_cfg(unet_actions=True)
    return ModelConfig(
        dim=32, depth=2, heads=4, num_latents=24, k_neighbors=8, dropout=0,
        context_len=3, all_relative=True, num_mat_types=6, pose_native=True,
        use_stratified_encoding=True, use_stratified_decoding=True,
        bypass_compression=False, per_frame_knn=True, concat_anchor_z=True,
        embed_actuatable=True, causal_temporal=True)


def _construct(architecture, reference):
    cfg = _options(architecture)
    if reference:
        # Fields added after the frozen revision may be dropped only while they
        # sit at their default: the reference must be the current model's arithmetic.
        reference_config = _reference_module("models/config.py").ModelConfig
        known = {field.name for field in fields(reference_config)}
        values = asdict(cfg)
        for name in set(values) - known:
            assert values[name] == getattr(ModelConfig(), name), \
                f"{name} is not at its default; the frozen reference cannot represent it"
            values.pop(name)
        cfg = reference_config(**values)
        cls = (_reference_module("models/transformer.py").TrajectoryTransformer
               if architecture == "transformer" else
               _reference_module("models/unet_std.py").UNetWrapper)
    else:
        cls = TrajectoryTransformer if architecture == "transformer" else UNetWrapper
    torch.manual_seed(29417)
    model = cls(cfg).eval()
    # Exercise the full trunk and pose readout, not equality of zero heads.
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.025)
    return model


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("scene", ["deformable", "mixed", "rigid"])
def test_retained_backbone_state_forward_gradient_and_update_parity(architecture, scene, monkeypatch):
    from utils.pose_ops import so3_log
    # Isolate the backbone from the intentional shared half-turn correction.
    # Keep all other frozen arithmetic and the exact equality checks below.
    monkeypatch.setattr(_reference_module("utils/pose_ops.py"), "so3_log", so3_log)
    options = {"deformable": dict(rigid=(), sidecar=False), "mixed": {},
               "rigid": dict(deform_n=0, rigid=((1, 24), (2, 16)))}[scene]
    batch = make_batch(**options)
    if scene == "deformable":
        batch.mat_type.zero_()
    current, reference = _construct(architecture, False), _construct(architecture, True)
    assert list(current.state_dict()) == list(reference.state_dict())
    if architecture == "transformer":
        # Initialization intentionally changed; saved-weight forward, gradients
        # and updates must still agree exactly with the frozen implementation.
        current.load_state_dict(reference.state_dict(), strict=True)
    torch.testing.assert_close(current.state_dict(), reference.state_dict(), rtol=0, atol=0)
    initial_state = {name: value.detach().clone() for name, value in current.state_dict().items()}
    torch.manual_seed(7293)
    noisy = torch.randn_like(batch.future_deltas)
    noisy_pose = None if scene == "deformable" else _noisy_pose_gt(batch) + 0.15
    times = torch.full(noisy.shape[:2], 500, dtype=torch.long)
    context = batch.delta_stats.normalize(batch.context_deltas)
    context_times = torch.zeros(context.shape[:2], dtype=torch.long)
    results = []
    for model in (current, reference):
        particle_input = noisy.detach().clone().requires_grad_()
        pose_input = None if noisy_pose is None else noisy_pose.detach().clone().requires_grad_()
        output = model(batch, particle_input, times, noisy_context=context,
                       context_timesteps=context_times, noisy_pose=pose_input)
        output = output if isinstance(output, tuple) else (output,)
        tensors = [value for value in output if value is not None]
        assert tensors and all(torch.isfinite(value).all() for value in tensors)
        assert all(torch.count_nonzero(value) > 0 for value in tensors)
        loss = sum((index + 1) * value.square().mean() for index, value in enumerate(tensors))
        loss.backward()
        grads = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters()}
        assert any(value is not None and torch.count_nonzero(value) > 0 for value in grads.values())
        inputs = [particle_input.grad, None if pose_input is None else pose_input.grad]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3, foreach=False)
        optimizer.step()
        results.append((output, loss, grads, inputs, model.state_dict()))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    traces = []
    for model in (current, reference):
        model.load_state_dict(initial_state, strict=True)
        particle_state = noisy.detach().clone()
        pose_state = None if noisy_pose is None else noisy_pose.detach().clone()
        trace = []
        with torch.no_grad():
            for index in range(4):
                clock = torch.full(noisy.shape[:2], 1000 - 250 * index, dtype=torch.long)
                output = model(batch, particle_state, clock, noisy_context=context,
                               context_timesteps=context_times, noisy_pose=pose_state)
                particle_v, pose_v = output if isinstance(output, tuple) else (output, None)
                if particle_v is not None:
                    particle_state = particle_state - 0.25 * particle_v
                if pose_v is not None:
                    pose_state = pose_state - 0.25 * pose_v
                trace.append((particle_state.clone(), None if pose_state is None else pose_state.clone()))
        traces.append(trace)
    torch.testing.assert_close(traces[0], traces[1], rtol=0, atol=0)


@pytest.mark.parametrize("old,new", [("classic", "transformer"), ("unet_std", "unet")])
def test_historical_names_are_checkpoint_aliases_only(old, new):
    assert parse_model_config({"architecture": old}, checkpoint=True)[0] == new
    assert parse_model_config({"architecture": new})[0] == new
    with pytest.raises(ValueError, match="expected transformer or unet"):
        parse_model_config({"architecture": old})


@pytest.mark.parametrize("options", [
    {"architecture": "joint"}, {"predict_rigid_pose": True}, {"predict_body_poses": True},
    {"pose_action_dim": 20}, {"commanded_gripper_tokens_enabled": True}, {"per_object_noise": True},
])
def test_retired_learned_architectures_are_never_partially_migrated(options):
    with pytest.raises(ValueError, match="Historical workflows|retired model option"):
        parse_model_config(options, checkpoint=True)


@pytest.mark.parametrize("field,value", [
    ("batched_stratified_encode", True), ("batched_stratified_decode", True),
    ("flow_matching_prediction_type", "velocity"), ("num_train_timesteps", 1000),
    ("per_object_noise", False), ("predict_body_poses", False), ("pose_action_dim", 0),
    ("fps_random_start", True),
])
def test_verified_inert_fields_are_checkpoint_only(field, value):
    with pytest.warns(UserWarning, match="verified inert or disabled"):
        _, cfg = parse_model_config({field: value}, checkpoint=True)
    assert not hasattr(cfg, field)
    with pytest.raises(ValueError, match="Unknown or retired"):
        parse_model_config({field: value})


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("prediction_type", ["velocity", "sample"])
def test_unified_prediction_parameterizations_remain_explicit(architecture, prediction_type):
    model = dict(asdict(_options(architecture)), architecture=architecture)
    cfg = OmegaConf.create(dict(
        model=model, data=dict(format="unified", context_len=3, emit_sidecar=True),
        prediction_target="delta", denoise_method="flow_matching", diffusion_forcing=True,
        flow_matching=dict(prediction_type=prediction_type, shift=1), train={}))
    validate_particle_training_config(cfg)


def test_inert_stratified_query_attention_cannot_be_requested_for_fresh_training():
    options = dict(asdict(_options("transformer")), decoder_query_self_attn=True)
    cfg = OmegaConf.create(dict(model=options, data=dict(format="unified", context_len=3),
        denoise_method="flow_matching", flow_matching=dict(prediction_type="velocity"), train={}))
    with pytest.raises(ValueError, match="inactive under unified stratified"):
        validate_particle_training_config(cfg)
    # Checkpoint construction preserves registered learned keys. It does not
    # silently strip this historical field or pretend it affected this decoder.
    _, restored = parse_model_config(options, checkpoint=True)
    assert restored.decoder_query_self_attn


def test_yaw_invariance_and_transformer_fps_determinism():
    """The retained Transformer has always used deterministic FPS, unlike joint."""
    from data.rotation_aug import apply_yaw_aug, xy_centroid

    batch = make_batch(rigid=(), sidecar=False)
    batch.mat_type.zero_()
    original = batch.positions.clone()
    deltas = batch.prediction_x0.clone()
    center = xy_centroid(batch.positions, batch.valid_lengths)
    torch.manual_seed(11)
    apply_yaw_aug(batch, center)
    torch.testing.assert_close(batch.positions[..., 2], original[..., 2])
    before_radius = torch.linalg.vector_norm(original[..., :2] - center[:, None, None], dim=-1)
    after_radius = torch.linalg.vector_norm(batch.positions[..., :2] - center[:, None, None], dim=-1)
    torch.testing.assert_close(before_radius, after_radius, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(batch.prediction_x0[..., 2], deltas[..., 2])
    torch.testing.assert_close(batch.prediction_x0.norm(dim=-1), deltas.norm(dim=-1),
                               atol=1e-5, rtol=1e-5)
    assert not torch.allclose(batch.prediction_x0, deltas)
    model = _construct("transformer", False)
    noisy = torch.randn_like(batch.future_deltas)
    times = torch.full(noisy.shape[:2], 500, dtype=torch.long)
    with torch.no_grad():
        fixed_a, fixed_b = model(batch, noisy, times), model(batch, noisy, times)
    torch.testing.assert_close(fixed_a, fixed_b, rtol=0, atol=0)


def test_unet_fps_random_start_is_an_implemented_ablation():
    batch = make_batch(rigid=(), sidecar=False)
    batch.mat_type.zero_()
    model = _construct("unet", False)
    noisy = torch.randn_like(batch.future_deltas)
    times = torch.full(noisy.shape[:2], 500, dtype=torch.long)
    with torch.no_grad():
        model.unet_cfg.fps_random_start = True
        random_a, random_b = model(batch, noisy, times), model(batch, noisy, times)
        model.unet_cfg.fps_random_start = False
        fixed_a, fixed_b = model(batch, noisy, times), model(batch, noisy, times)
    assert not torch.allclose(random_a, random_b)
    torch.testing.assert_close(fixed_a, fixed_b, rtol=0, atol=0)


@pytest.mark.parametrize("options", [
    {"banded_rope": True}, {"decoder_anchor_identity": True},
    {"normalize_physics_features": True}, {"cloth_attends_all_latents": True},
])
def test_retained_transformer_ablations_have_finite_nonzero_gradients(options):
    cfg = _options("transformer")
    for key, value in options.items():
        setattr(cfg, key, value)
    batch = make_batch()
    model = TrajectoryTransformer(cfg)
    with torch.no_grad():
        model.pose_v_head.weight.normal_(0, 0.025)
    output = model(batch, torch.randn_like(batch.future_deltas),
                   torch.full(batch.future_deltas.shape[:2], 500, dtype=torch.long),
                   noisy_pose=_noisy_pose_gt(batch))
    assert all(torch.isfinite(value).all() for value in output)
    sum(value.square().mean() for value in output).backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
    assert model.head.weight.grad.abs().sum() > 0
    assert model.pose_v_head.weight.grad.abs().sum() > 0
