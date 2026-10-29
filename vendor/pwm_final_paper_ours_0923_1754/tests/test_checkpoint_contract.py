"""Production loading must not silently evaluate a partly random model."""
from dataclasses import asdict

from omegaconf import OmegaConf
import pytest
import torch

from inference import load_model_and_diffusion
from models.config import ModelConfig
from models.transformer import TrajectoryTransformer


def checkpoint(tmp_path, *, model_options=None, train_options=None, mutate=None, ema=False):
    options = dict(dim=32, depth=1, heads=4, num_latents=8, k_neighbors=4, dropout=0.0,
                   context_len=2, all_relative=True)
    model = TrajectoryTransformer(ModelConfig(**options))
    weights = dict(model.named_parameters()) if ema else model.state_dict()
    if mutate:
        mutate(weights, model)
    options.update(model_options or {})
    cfg = dict(model=options, data=dict(context_len=2, format="unified", stats_location="unified://xy_p99"), train=train_options or {},
               diffusion_forcing=True, denoise_method="flow_matching",
               flow_matching=dict(num_train_timesteps=1000, shift=1.0,
                                  prediction_type="velocity", inference_steps=4))
    path = tmp_path / "checkpoint.pt"
    torch.save({"cfg": cfg, "ema" if ema else "model": weights}, path)
    return path


def test_missing_learned_weight_is_fatal(tmp_path):
    path = checkpoint(tmp_path, mutate=lambda sd, m: sd.pop(next(iter(dict(m.named_parameters())))))
    with pytest.raises(RuntimeError, match="Missing|missing"):
        load_model_and_diffusion(path, torch.device("cpu"))


def test_unexpected_weight_is_fatal(tmp_path):
    path = checkpoint(tmp_path, mutate=lambda sd, m: sd.update(unknown_weight=torch.ones(2)))
    with pytest.raises(RuntimeError, match="Unexpected|unexpected"):
        load_model_and_diffusion(path, torch.device("cpu"))


@pytest.mark.parametrize("extra", [{"depht": 5}, {"encoder_temporal": True},
                                  {"encoder_token_temporal": True}, {"architecture": "joint"}])
def test_unknown_or_retired_model_options_are_fatal(tmp_path, extra):
    path = checkpoint(tmp_path, model_options=extra)
    with pytest.raises(ValueError):
        load_model_and_diffusion(path, torch.device("cpu"))


@pytest.mark.parametrize("options,expected", [({}, True), ({"noise_context": False}, False),
    ({"noise_context": True}, True), ({"noise_context": False, "clean_ctx_timesteps": True}, True)])
def test_offline_context_timestep_contract_matches_training(tmp_path, options, expected):
    path = checkpoint(tmp_path, train_options=options)
    model, _, _, _ = load_model_and_diffusion(path, torch.device("cpu"))
    assert model._ctx_timesteps_trained is expected
    from utils.conditioning import clean_context
    from tests.test_train import create_dummy_batch
    context, times = clean_context(model, create_dummy_batch(B=1))
    assert context is not None
    if not expected:
        assert times is None
    else:
        value = 1 if options.get("clean_ctx_timesteps") and not options.get("noise_context", True) else 0
        assert (times == value).all()


def test_ema_can_omit_only_deterministic_buffers(tmp_path):
    path = checkpoint(tmp_path, ema=True)
    model, _, _, _ = load_model_and_diffusion(path, torch.device("cpu"))
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_supported_recipes_have_valid_model_options():
    from pathlib import Path
    from utils.config import load_config
    from models.factory import parse_model_config, validate_particle_training_config
    recipes = sorted(Path("configs").glob("*.yaml"))
    assert {p.name for p in recipes} == {"transformer.yaml", "unet.yaml", "unet_adaptive.yaml"}
    for path in recipes:
        cfg = load_config(path)
        _, parsed = parse_model_config(cfg.model)
        validate_particle_training_config(cfg)
        assert isinstance(asdict(parsed), dict), path


def test_only_observed_inert_unet_fields_are_compatible():
    from models.factory import parse_model_config
    with pytest.warns(UserWarning, match="verified inert"):
        architecture, _ = parse_model_config({"architecture": "unet_std", "unet_mixer": "full"}, checkpoint=True)
    assert architecture == "unet"
    for value in ("full", "edge"):
        with pytest.raises(ValueError):
            parse_model_config({"architecture": "unet_std", "unet_mixer": value})
    with pytest.raises(ValueError):
        parse_model_config({"architecture": "unet_std", "unet_mixer": "edge"}, checkpoint=True)


def test_state_validation_happens_before_mutating_parameters():
    from utils.state_dict import load_model_state
    model = torch.nn.Linear(2, 3)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="Wrong tensor shapes"):
        load_model_state(model, {"weight": torch.zeros(3, 2), "bias": torch.zeros(4)})
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


@pytest.mark.parametrize("modern", [True, False])
def test_checkpoint_storage_loading_is_cpu_only_and_never_writes_through(tmp_path, monkeypatch, modern):
    from utils.checkpoint import read_checkpoint
    from utils.reproducibility import file_sha256

    path = tmp_path / "storage.pt"
    torch.save({"model": {"weight": torch.ones(8)}, "optimizer": {"moment": torch.ones(8)}},
               path, _use_new_zipfile_serialization=modern)
    digest = file_sha256(path)
    original = torch.load
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record)
    state = read_checkpoint(path)
    assert calls[-1] == dict(map_location="cpu", weights_only=True, mmap=modern)
    # Optimizer mutation on CPU must be copy-on-write, not alter the input file.
    state["optimizer"]["moment"].add_(10)
    assert file_sha256(path) == digest
    torch.testing.assert_close(original(path, weights_only=True)["optimizer"]["moment"], torch.ones(8))


@pytest.mark.parametrize("profile", ["xy_p99_per_domain", "xy_p99_global"])
def test_normalization_snapshot_is_self_contained_and_checked(tmp_path, monkeypatch, profile):
    from copy import deepcopy
    import data.normalization as normalization
    from utils.checkpoint import checkpoint_config, validate_resume_normalization
    snapshot = normalization.resolve_normalization(profile)
    state = {"cfg": {"data": {"format": "unified"}, "normalization": snapshot},
             "normalization": snapshot}
    # A snapshot must still work when the artifact is absent on the new machine.
    monkeypatch.setattr(normalization, "DEFAULT_ARTIFACT", tmp_path / "missing.json", raising=False)
    cfg = checkpoint_config(state)
    assert cfg.normalization.sha256 == snapshot["sha256"]
    validate_resume_normalization(state, snapshot)
    broken = deepcopy(state)
    broken["normalization"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        checkpoint_config(broken)


@pytest.mark.parametrize("legacy,profile", [("unified://xy_p99", "xy_p99_per_domain"),
                                          ("unified://xy_p99_global", "xy_p99_global")])
def test_legacy_unified_migration_uses_the_frozen_profile(legacy, profile):
    from utils.checkpoint import checkpoint_config
    cfg = checkpoint_config({"cfg": {"data": {"format": "unified", "stats_location": legacy}}})
    assert cfg.normalization.profile == profile


def test_resume_rejects_changed_normalization_before_touching_weights():
    from data.normalization import resolve_normalization
    from utils.checkpoint import load_training_state
    model = torch.nn.Linear(1, 1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = {"cfg": {"data": {"format": "unified"}, "normalization": resolve_normalization()},
             "model": {k: torch.zeros_like(v) for k, v in before.items()}}
    with pytest.raises(ValueError, match="Resume normalization"):
        load_training_state(state, model, normalization="xy_p99_global")
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


def test_new_checkpoint_embeds_normalization_without_mutating_input():
    from copy import deepcopy
    from utils.checkpoint import training_checkpoint, checkpoint_config
    from utils.ema import EMA
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    cfg = {"data": {"format": "unified"}, "normalization": {"profile": "xy_p99_global"}}
    before = deepcopy(cfg)
    state = training_checkpoint(model, EMA(model), optimizer, None, 0, cfg)
    assert cfg == before
    assert state["checkpoint_version"] == 3
    assert state["normalization"] == state["cfg"]["normalization"]
    assert checkpoint_config(state).normalization.profile == "xy_p99_global"


@pytest.mark.parametrize("data", [{"format": "mpm_multi"}, {"format": "unified", "stats_location": "per-domain"}])
def test_retired_checkpoint_contracts_fail_explicitly(data):
    from utils.checkpoint import checkpoint_config
    with pytest.raises(ValueError, match="pre-unified-only"):
        checkpoint_config({"cfg": {"data": data}})


@pytest.mark.parametrize("change", ["flow_matching.prediction_type=sample",
    "flow_matching.num_train_timesteps=2000", "train.grad_accum=2",
    "model.dropout=0.2", "data.future_len=20", "train.light_context_noise_std=0.05",
    "train.light_pose_noise_std=0.05"])
def test_resume_rejects_shape_compatible_semantic_changes(change):
    from utils.config import load_config
    from utils.checkpoint import validate_resume_configuration
    cfg = load_config("configs/transformer.yaml")
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True)}
    changed = load_config("configs/transformer.yaml", [change])
    with pytest.raises(ValueError, match="training contract"):
        validate_resume_configuration(state, changed)


def test_resume_allows_new_output_and_logging_controls():
    from utils.config import load_config
    from utils.checkpoint import validate_resume_configuration
    cfg = load_config("configs/transformer.yaml")
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True)}
    cfg.run_name, cfg.log.every, cfg.train.stop_after_step = "continued", 10, 20
    validate_resume_configuration(state, cfg)


@pytest.mark.parametrize("change", ["train.light_pose_noise_anchor=true",
                                    "train.light_pose_noise_scope=passive"])
def test_resume_rejects_changed_pose_noise_anchor_or_scope(change):
    from utils.config import load_config
    from utils.checkpoint import validate_resume_configuration
    base = ["train.light_pose_noise_std=.05", "train.clip_pose_deltas=false"]
    cfg = load_config("configs/unet.yaml", base)
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True)}
    with pytest.raises(ValueError, match="training contract"):
        validate_resume_configuration(state, load_config("configs/unet.yaml", base+[change]))


def test_resume_rejects_carry_restore_toggle_and_accepts_legacy_defaults():
    from utils.config import load_config
    from utils.checkpoint import validate_resume_configuration
    base = ["train.light_pose_noise_std=.05", "train.light_pose_noise_anchor=true", "train.clip_pose_deltas=false"]
    cfg = load_config("configs/unet.yaml", base)
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True)}
    with pytest.raises(ValueError, match="training contract"):
        validate_resume_configuration(state, load_config("configs/unet.yaml", base+["train.light_pose_noise_target=carry"]))
    # An older checkpoint without new keys means restore/passive when its anchor flag is true.
    validate_resume_configuration(state, load_config("configs/unet.yaml", base+[
        "train.light_pose_noise_target=restore", "train.light_pose_noise_scope=passive"]))
    clean = load_config("configs/unet.yaml")
    validate_resume_configuration({"cfg": OmegaConf.to_container(clean, resolve=True)},
        load_config("configs/unet.yaml", ["train.light_pose_noise_std=0", "train.light_pose_noise_anchor=false"]))


def test_noisy_context_checkpoints_still_initialize_fine_tunes_but_do_not_resume():
    """The 200K pickup runs trained with train.noise_context=true, which new configs refuse.
    Their weights stay valid train.init_from sources (loading and sampling are covered by
    test_offline_context_timestep_contract_matches_training); continuing one would change
    its objective."""
    from utils.config import load_config
    from utils.checkpoint import validate_initialization, validate_resume_configuration
    cfg = load_config("configs/unet.yaml", ["data.domains=[pickup]", "model.articulated_bodies=true"])
    saved = OmegaConf.to_container(cfg, resolve=True)
    saved["train"]["noise_context"] = True
    state = {"cfg": saved}
    validate_initialization(state, cfg)
    with pytest.raises(ValueError, match="training contract"):
        validate_resume_configuration(state, cfg)


def _pushing_release(tmp_path, name):
    from tests.test_data_roots import _shards
    return _shards(tmp_path / name, 10)


def _exclusion_list(path, key):
    import json
    path.write_text(json.dumps({"blocklist": [{"shard": "00000.tar", "key": key}]}))
    return str(path)


def _release_run(tmp_path):
    """A resolved pushing config on its own release (data.roots) with one training exclusion list."""
    from utils.config import load_config
    release = _pushing_release(tmp_path, "push_v3")
    cfg = OmegaConf.to_container(load_config("configs/transformer.yaml", [
        "data.domains=[pushing]", f"data.roots.pushing.train={release}"]), resolve=True)
    cfg["data"]["roots"]["pushing"]["exclusions"] = {"train": [_exclusion_list(tmp_path / "exclude.json", "k0")]}
    return OmegaConf.create(cfg)


def test_resume_refuses_another_release_or_exclusion_list(tmp_path):
    from utils.checkpoint import corpus_identity, validate_resume_configuration
    cfg = _release_run(tmp_path)
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True), "corpus": corpus_identity(cfg)}
    validate_resume_configuration(state, cfg)
    for field, value in (("train", _pushing_release(tmp_path, "push_v4")),
                         ("exclusions", {"train": [_exclusion_list(tmp_path / "other.json", "k0")]})):
        changed = OmegaConf.create(state["cfg"])
        changed.data.roots.pushing[field] = value
        with pytest.raises(ValueError, match="training contract"):
            validate_resume_configuration(state, changed)


@pytest.mark.parametrize("registered", [False, True])
def test_resume_refuses_exclusion_files_edited_in_place(tmp_path, monkeypatch, registered):
    """Same settings and paths, different episodes excluded: the checkpoint's hashes catch it,
    for a release's own list and for a registered v1 list alike."""
    import data.unified_dataset as unified
    from utils.config import load_config
    from utils.checkpoint import corpus_identity, validate_resume_configuration
    edited = tmp_path / ("mpm_blocklist_train.json" if registered else "exclude.json")
    if registered:
        monkeypatch.setattr(unified, "MPM_BLOCKLISTS", [_exclusion_list(edited, "k0")])
        cfg = load_config("configs/transformer.yaml", ["data.domains=[mpm]"])
    else:
        cfg = _release_run(tmp_path)
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True), "corpus": corpus_identity(cfg)}
    validate_resume_configuration(state, cfg)
    _exclusion_list(edited, "k1")
    with pytest.raises(ValueError, match="corpus files differ"):
        validate_resume_configuration(state, cfg)


def test_checkpoints_without_exclusion_hashes_resume_with_a_warning():
    from utils.config import load_config
    from utils.checkpoint import validate_resume_configuration
    cfg = load_config("configs/transformer.yaml", ["data.domains=[pushing]"])
    with pytest.warns(UserWarning, match="exclusion"):
        validate_resume_configuration({"cfg": OmegaConf.to_container(cfg, resolve=True)}, cfg)


def test_initialization_accepts_a_source_trained_on_an_absent_release(tmp_path):
    """train.init_from compares the network only; the source's data need not exist here."""
    from utils.config import load_config
    from utils.checkpoint import validate_initialization
    cfg = load_config("configs/unet.yaml", ["data.domains=[pushing]"])
    saved = OmegaConf.to_container(cfg, resolve=True)
    saved["data"]["roots"] = {"pushing": {"train": str(tmp_path / "release_on_another_machine")}}
    validate_initialization({"cfg": saved}, cfg)


def test_bad_adam_moment_shape_fails_before_model_mutation():
    from utils.checkpoint import load_training_state
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = {"model": {k: torch.zeros_like(v) for k, v in before.items()},
             "optimizer": optimizer.state_dict(), "step": 1}
    first = next(iter(state["optimizer"]["state"].values()))
    first["exp_avg"] = torch.zeros(3, 3)
    with pytest.raises(RuntimeError, match="Optimizer exp_avg shape"):
        load_training_state(state, model, optimizer)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


@pytest.mark.parametrize("next_step", [None, "bad", 1.5, -1, True])
def test_invalid_continuation_boundary_fails_before_model_mutation(next_step):
    from utils.checkpoint import load_training_state
    model = torch.nn.Linear(1, 1)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = {"model": {k: torch.zeros_like(v) for k, v in before.items()},
             "step": 1, "next_step": next_step}
    with pytest.raises(ValueError, match="continuation step"):
        load_training_state(state, model)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


def test_resume_canonicalizes_implicit_adam_defaults():
    from utils.config import load_config
    from utils.checkpoint import validate_resume_configuration
    cfg = load_config("configs/transformer.yaml")
    cfg.train.adam_betas = None
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True)}
    cfg.train.adam_betas = [0.9, 0.999]
    validate_resume_configuration(state, cfg)


class _UnsupportedCheckpointMetadata:
    """Benign fixture class that must never be automatically allowlisted."""


@pytest.mark.parametrize("modern", [True, False])
def test_safe_checkpoint_reader_accepts_real_torch_version_metadata(tmp_path, monkeypatch, modern):
    from torch.torch_version import TorchVersion
    from utils.checkpoint import read_checkpoint
    from utils.reproducibility import file_sha256
    path = tmp_path / "torch-version.pt"
    torch.save({"model": {"value": torch.ones(2)}, "provenance": {"torch": torch.__version__}},
               path, _use_new_zipfile_serialization=modern)
    before_digest = file_sha256(path)
    allowed_before = set(torch.serialization.get_safe_globals())
    original = torch.load
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record)
    state = read_checkpoint(path)
    assert isinstance(state["provenance"]["torch"], TorchVersion)
    assert str(state["provenance"]["torch"]) == str(torch.__version__)
    assert calls == [dict(map_location="cpu", weights_only=True, mmap=modern)]
    assert set(torch.serialization.get_safe_globals()) == allowed_before
    assert file_sha256(path) == before_digest
    state["model"]["value"].add_(1)
    assert file_sha256(path) == before_digest


@pytest.mark.parametrize("modern", [True, False])
@pytest.mark.parametrize("already_allowed", [True, False])
@pytest.mark.parametrize("unsupported", [True, False])
def test_checkpoint_version_allowance_is_scoped_and_never_falls_back_to_unsafe_load(
        tmp_path, monkeypatch, modern, already_allowed, unsupported):
    import pickle
    from torch.torch_version import TorchVersion
    from utils.checkpoint import read_checkpoint
    path = tmp_path / "scoped.pt"
    payload = {"torch": torch.__version__}
    if unsupported:
        payload["unsupported"] = _UnsupportedCheckpointMetadata()
    torch.save(payload, path, _use_new_zipfile_serialization=modern)
    original = torch.load
    calls = []

    def record(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record)
    with torch.serialization.safe_globals([TorchVersion] if already_allowed else []):
        before = set(torch.serialization.get_safe_globals())
        if unsupported:
            with pytest.raises(pickle.UnpicklingError, match="UnsupportedCheckpointMetadata"):
                read_checkpoint(path)
        else:
            assert str(read_checkpoint(path)["torch"]) == str(torch.__version__)
        assert set(torch.serialization.get_safe_globals()) == before
    assert len(calls) == 1 and calls[0]["weights_only"] is True


@pytest.mark.parametrize("eval_rng", ["isolated", "legacy"])
def test_fresh_overfit_save_metadata_uses_plain_torch_version_and_safe_loading(tmp_path, eval_rng):
    """Execute actual metadata/save AST with a tiny CPU model, not a second recipe."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace
    from scripts.unified import overfit_prod
    from utils.checkpoint import atomic_torch_save, capture_runtime, read_checkpoint
    from utils.config import load_config

    tree = ast.parse(Path(overfit_prod.__file__).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    provenance_node = next(node for node in main.body if isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == "provenance" for target in node.targets))
    save_if = next(node for node in main.body if isinstance(node, ast.If) and
        isinstance(node.test, ast.Attribute) and node.test.attr == "save_final")
    save_node = next(node for node in save_if.body if isinstance(node, ast.Expr))
    cfg = load_config("configs/transformer.yaml", ["train.loss_spike_threshold=.inf",
        "train.grad_norm_spike_threshold=.inf"])
    numeric_before = OmegaConf.to_container(cfg, resolve=True)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1)
    acc = SimpleNamespace(unwrap_model=lambda value: value, sync_gradients=True,
        device=torch.device("cpu"), scaler=None, num_processes=1, mixed_precision="bf16")
    scope = {**vars(overfit_prod), "a": SimpleNamespace(steps=0, eval_rng=eval_rng), "cfg": cfg,
             "acc": acc, "model": model, "opt": opt, "sched": sched, "out_dir": str(tmp_path),
             "atomic_torch_save": atomic_torch_save, "capture_runtime": capture_runtime}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[provenance_node, save_node], type_ignores=[])),
                 "actual_overfit_checkpoint_save", "exec"), scope)
    allowed_before = set(torch.serialization.get_safe_globals())
    state = read_checkpoint(tmp_path / "final.pt")
    assert set(torch.serialization.get_safe_globals()) == allowed_before
    assert type(state["provenance"]["torch"]) is str
    assert state["provenance"]["torch"] == str(torch.__version__)
    assert state["provenance"]["eval_rng"] == eval_rng
    assert state["provenance"]["sampling_seed"] == (4242 if eval_rng == "isolated" else None)
    assert state["provenance"]["holdout_sampling_seed"] == 4242
    assert state["cfg"] == numeric_before == OmegaConf.to_container(cfg, resolve=True)
    assert state["cfg"]["train"]["loss_spike_threshold"] == float("inf")
    assert state["runtime"]["version"] == 2 and state["runtime"]["ranks"][0]["torch_cuda"] is None
    # Newly written metadata also works with the unwrapped standard safe reader.
    assert type(torch.load(tmp_path / "final.pt", weights_only=True)["provenance"]["torch"]) is str


# -- effective batch: batch_size x grad_accum x GPUs is pinned, not its factors ------------------------

def _state(cfg, world_size):
    return {"cfg": OmegaConf.to_container(cfg, resolve=True), "runtime": {"world_size": world_size}}


@pytest.mark.parametrize("saved, saved_ws, requested, requested_ws", [
    (["data.batch_size=16", "train.grad_accum=1"], 2, ["data.batch_size=4", "train.grad_accum=4"], 2),   # PRO 6000 -> L40S
    (["data.batch_size=4", "train.grad_accum=4"], 2, ["data.batch_size=8", "train.grad_accum=4"], 1),    # 2 GPUs -> 1
    (["data.batch_size=4", "train.grad_accum=2"], 4, ["data.batch_size=2", "train.grad_accum=8"], 2),
])
def test_resume_accepts_the_same_effective_batch_split_differently(saved, saved_ws, requested, requested_ws):
    from utils.config import load_config
    from utils.checkpoint import validate_resume_configuration
    state = _state(load_config("configs/unet.yaml", saved), saved_ws)
    validate_resume_configuration(state, load_config("configs/unet.yaml", requested), world_size=requested_ws)


@pytest.mark.parametrize("requested, requested_ws", [
    (["data.batch_size=4", "train.grad_accum=2"], 2),     # 16 != 32
    (["data.batch_size=4", "train.grad_accum=4"], 1),     # right split, one GPU short
])
def test_resume_rejects_a_different_effective_batch(requested, requested_ws):
    from utils.config import load_config
    from utils.checkpoint import validate_resume_configuration
    state = _state(load_config("configs/unet.yaml", ["data.batch_size=16", "train.grad_accum=1"]), 2)
    with pytest.raises(ValueError, match="Effective batch"):
        validate_resume_configuration(state, load_config("configs/unet.yaml", requested), world_size=requested_ws)


@pytest.mark.parametrize("change", ["model.unet_ckpt_from_level=0", "model.unet_ckpt_seams=true",
                                    "model.unet_ckpt_ingraph=true", "data.num_workers=7"])
def test_resume_allows_checkpointing_and_worker_changes(change):
    """Recompute-vs-store and loader workers change how a step is computed or fed, not what it trains on."""
    from utils.config import load_config
    from utils.checkpoint import validate_resume_configuration
    cfg = load_config("configs/unet.yaml")
    validate_resume_configuration(_state(cfg, 1), load_config("configs/unet.yaml", [change]), world_size=1)
