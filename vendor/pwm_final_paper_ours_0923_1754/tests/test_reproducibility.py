"""Evaluation seeds and provenance cannot depend on Python process hash state."""
import os
import subprocess
import sys

import pytest


def test_archived_source_does_not_inherit_ambient_parent_checkout_revision(tmp_path,monkeypatch):
    import utils.reproducibility as reproducibility
    root=tmp_path/'snapshot';(root/'utils').mkdir(parents=True)
    revision='a'*40;(root/'SOURCE_REVISION').write_text(revision+'\n')
    monkeypatch.setattr(reproducibility,'__file__',str(root/'utils/reproducibility.py'))
    def no_parent_git(*args,**kwargs):
        pytest.fail('An archive must not search its ambient parent checkout')
    monkeypatch.setattr(reproducibility.subprocess,'check_output',no_parent_git)
    receipt=reproducibility.code_revision()
    assert receipt['git_commit']==revision and receipt['source_snapshot_revision']==revision
    assert receipt['git_dirty'] is None
    assert receipt['source_snapshot_integrity'].startswith('not checked')


@pytest.mark.parametrize('revision',['','deadbee','g'*40,'a'*39+'!'])
def test_malformed_archive_revision_cannot_fall_back_to_parent_git(tmp_path,monkeypatch,revision):
    import utils.reproducibility as reproducibility
    root=tmp_path/'snapshot';(root/'utils').mkdir(parents=True)
    (root/'SOURCE_REVISION').write_text(revision)
    monkeypatch.setattr(reproducibility,'__file__',str(root/'utils/reproducibility.py'))
    with pytest.raises(ValueError,match='full lowercase Git'):
        reproducibility.code_revision()


def test_cpu_seed_matches_torch_without_touching_cuda(monkeypatch):
    import torch
    from utils.reproducibility import seed_for_device

    def no_cuda(*args, **kwargs):
        pytest.fail("CPU seeding must not access any CUDA generator")

    for name in ("manual_seed", "manual_seed_all", "device", "get_rng_state", "get_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(141)
        expected = torch.randn(7)
        seed_for_device(141, "cpu")
        torch.testing.assert_close(torch.randn(7), expected, rtol=0, atol=0)


def test_cuda_seed_selects_only_its_requested_device_without_gpu_allocation(monkeypatch):
    from contextlib import contextmanager
    import torch
    from utils.reproducibility import seed_for_device
    calls = []

    @contextmanager
    def device_context(device):
        calls.append(("enter", device))
        yield
        calls.append(("exit", device))

    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "manual_seed", lambda seed: calls.append(("seed", seed)))
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *a: pytest.fail("Must not seed every GPU"))
    with torch.random.fork_rng(devices=[]):
        seed_for_device(141, "cuda:1")
    assert calls == [("enter", torch.device("cuda:1")), ("seed", 141), ("exit", torch.device("cuda:1"))]


def test_evaluation_seed_is_stable_across_processes_and_batch_specific():
    code = ("from utils.reproducibility import evaluation_seed; "
            "print([evaluation_seed(42, i, name) for i in range(2) "
            "for name in ('easy', 'hard', 'perpendicular')])")
    outputs = [subprocess.check_output([sys.executable, "-c", code], text=True,
               env={**os.environ, "PYTHONHASHSEED": hash_seed}) for hash_seed in ("0", "987")]
    assert outputs[0] == outputs[1]
    import ast
    seeds = ast.literal_eval(outputs[0])
    assert len(set(seeds)) == 6
    assert all(0 <= seed < 2**32 for seed in seeds)


def test_file_digest_streams_actual_contents(tmp_path):
    from utils.reproducibility import file_sha256

    path = tmp_path / "known.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_overfit_import_does_not_launch_and_rejects_tampered_batch(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    import pytest
    from scripts.unified.overfit_prod import main

    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    (batch_dir / "mpm.pt").write_bytes(b"This is not a pickle; digest must fail before unpickling")
    (batch_dir / "mpm.json").write_text(json.dumps({"sha256": "0" * 64, "stats_set": "xy_p99"}))
    config = Path(__file__).resolve().parents[1] / "configs/unet.yaml"
    monkeypatch.setattr(sys, "argv", ["overfit_prod", "--config", str(config), "--domain", "mpm",
        "--batch_dir", str(batch_dir), "--out_root", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        main()


@pytest.mark.parametrize("missing", ["mpm.pt", "mpm.json", "mpm_hold.pt", "mpm_hold.json"])
def test_pinned_overfit_checks_all_inputs_before_data_or_cuda(tmp_path, monkeypatch, missing):
    import torch
    from scripts.unified import overfit_prod

    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    for name in ("mpm.pt", "mpm.json", "mpm_hold.pt", "mpm_hold.json"):
        if name != missing:
            (batch_dir / name).write_text("must not be read")
    monkeypatch.setattr(overfit_prod, "build_domain_dataset", lambda *a, **k: pytest.fail("corpus opened"))
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("partial fixture load"))
    monkeypatch.setattr(torch, "manual_seed", lambda *a, **k: pytest.fail("GPU seeding before preflight"))
    monkeypatch.setattr(sys, "argv", ["overfit_prod", "--config", "configs/transformer.yaml",
        "--domain", "mpm", "--holdout_name", "mpm_hold", "--require_saved_batches",
        "--batch_dir", str(batch_dir), "--out_root", str(tmp_path / "out")])
    with pytest.raises(SystemExit) as error:
        overfit_prod.main()
    assert error.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_overfit_eval_rng_isolated_by_default_and_legacy_consumes_training_stream(legacy):
    import torch
    from scripts.unified.overfit_prod import sample_for_eval

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        before = torch.get_rng_state().clone()
        sample = lambda batch: torch.randn(batch)
        if legacy:
            expected = sample(7)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(before)
            actual = sample_for_eval(sample, 7, "cpu", seed=None)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.equal(torch.get_rng_state(), after)
            assert not torch.equal(after, before)
        else:
            actual = sample_for_eval(sample, 7, "cpu")
            assert torch.equal(torch.get_rng_state(), before)
            torch.testing.assert_close(actual, sample_for_eval(sample, 7, "cpu"), rtol=0, atol=0)
            assert torch.equal(torch.get_rng_state(), before)


def test_overfit_isolated_eval_restores_rng_on_failure():
    import torch
    from scripts.unified.overfit_prod import sample_for_eval

    before = torch.get_rng_state().clone()
    def fail(batch):
        torch.rand(3)
        raise RuntimeError("sampling failed")
    with pytest.raises(RuntimeError, match="sampling failed"):
        sample_for_eval(fail, None, "cpu")
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("mode", ["accumulation", "distributed"])
def test_overfit_rejects_unsupported_execution_before_data_io(tmp_path, monkeypatch, mode):
    from scripts.unified.overfit_prod import main

    command = ["overfit_prod", "--config", "configs/unet.yaml",
               "--domain", "mpm", "--batch_dir", str(tmp_path / "absent")]
    if mode == "accumulation":
        command += ["--override", "train.grad_accum=2"]
    else:
        monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not (tmp_path / "absent").exists()


@pytest.mark.parametrize("domain", ["mpm", "pickup", "cloth", "pushing"])
def test_normalization_ablation_preserves_all_raw_fields(domain):
    from dataclasses import fields
    import torch
    from data.normalization import resolve_normalization
    from tests.test_unet_std_pose_native import make_batch
    from scripts.unified.overfit_prod import bind_batch_normalization
    batch = make_batch(rigid=(), sidecar=False) if domain == "mpm" else make_batch()
    batch.actions = batch.future_deltas * batch.action_mask[:, None, :, None]
    if batch.pose_actions is not None:
        batch.pose_actions = batch.pose_deltas_gt * batch.pose_action_mask[:, None, :, None]
    rebound = bind_batch_normalization(batch, domain, resolve_normalization("xy_p99_global"))
    assert rebound is not batch and rebound.delta_stats is not batch.delta_stats
    for field in fields(batch):
        if torch.is_tensor(value := getattr(batch, field.name)):
            torch.testing.assert_close(getattr(rebound, field.name), value, rtol=0, atol=0)


def test_overfit_refuses_to_overwrite_an_existing_run(tmp_path, monkeypatch):
    from scripts.unified.overfit_prod import main
    previous = tmp_path / "existing"
    previous.mkdir()
    artifact = previous / "final.pt"
    artifact.write_bytes(b"previous experiment")
    monkeypatch.setattr(sys, "argv", ["overfit", "--config", "configs/unet.yaml",
        "--domain", "mpm", "--tag", "existing", "--out_root", str(tmp_path)])
    with pytest.raises(FileExistsError):
        main()
    assert artifact.read_bytes() == b"previous experiment"


def _fixture(domain="mpm"):
    from tests.test_unet_std_pose_native import make_batch
    from scripts.unified.overfit_prod import bind_batch_normalization
    from data.normalization import resolve_normalization
    snapshot = resolve_normalization()
    batch = make_batch(rigid=(), sidecar=False) if domain == "mpm" else make_batch()
    batch.domain = domain
    batch = bind_batch_normalization(batch, domain, snapshot)
    return batch, {"stats_set": "xy_p99", "sha256": "original-file-digest"}, snapshot


def test_fixture_subsets_preserve_raw_rows_and_original_file_receipt():
    import torch
    from scripts.unified.overfit_prod import prepare_fixture
    batch, meta, snapshot = _fixture("pickup")
    selected, receipt = prepare_fixture(batch, meta, "pickup", snapshot, 1)
    assert batch.batch_size == 2 and selected.batch_size == 1
    assert receipt["original_batch_size"] == 2 and receipt["selected_rows"] == [0]
    assert receipt["sha256"] == meta["sha256"] and "selected_rows" not in meta
    torch.testing.assert_close(selected.positions, batch.positions[:1], rtol=0, atol=0)
    torch.testing.assert_close(selected.body_poses, batch.body_poses[:1], rtol=0, atol=0)


@pytest.mark.parametrize("size", [0, -1, 3, True])
def test_fixture_rejects_unavailable_or_invalid_batch_size(size):
    from scripts.unified.overfit_prod import prepare_fixture
    batch, meta, snapshot = _fixture()
    with pytest.raises(ValueError, match="batch_size"):
        prepare_fixture(batch, meta, "mpm", snapshot, size)


@pytest.mark.parametrize("identity", ["batch", "receipt", "holdout", "absent"])
def test_fixture_cannot_be_silently_relabelled(identity):
    from scripts.unified.overfit_prod import prepare_fixture
    batch, meta, snapshot = _fixture()
    if identity == "batch":
        batch.domain = "cloth"
    elif identity == "absent":
        batch.domain = None
    else:
        meta["domain" if identity == "receipt" else "holdout_for"] = "cloth"
    with pytest.raises(ValueError, match="domain identity"):
        prepare_fixture(batch, meta, "mpm", snapshot, 1, rebind=True)


def test_legacy_fixture_can_use_explicit_receipt_domain():
    from scripts.unified.overfit_prod import prepare_fixture
    batch, meta, snapshot = _fixture()
    batch.domain = None
    meta["domain"] = "mpm"
    selected, receipt = prepare_fixture(batch, meta, "mpm", snapshot, 1)
    assert selected.batch_size == 1 and receipt["domain"] == "mpm"


@pytest.mark.parametrize("field", ["delta_stats", "position_stats", "pose_delta_stats"])
def test_matching_profile_label_does_not_hide_wrong_fixture_bounds(field):
    from scripts.unified.overfit_prod import prepare_fixture
    batch, meta, snapshot = _fixture("pickup")
    getattr(batch, field).max_val *= 1.01
    with pytest.raises(ValueError, match="differs from the recipe"):
        prepare_fixture(batch, meta, "pickup", snapshot, 1)
    # The explicit raw rebind is the supported normalization-ablation path.
    selected, _ = prepare_fixture(batch, meta, "pickup", snapshot, 1, rebind=True)
    assert selected.batch_size == 1


def test_overfit_optimizer_uses_all_production_adamw_hyperparameters():
    import torch
    from omegaconf import OmegaConf
    from scripts.unified.overfit_prod import build_optimizer
    config = OmegaConf.create(dict(lr=0.004, weight_decay=0.02,
                                  adam_betas=[0.8, 0.95], adam_eps=1e-6))
    group = build_optimizer(torch.nn.Linear(2, 1), config).param_groups[0]
    assert (group["lr"], group["weight_decay"], group["betas"], group["eps"]) == (
        0.004, 0.02, (0.8, 0.95), 1e-6)
    config.adam_betas = None  # Same explicit-null fallback as train.py.
    assert build_optimizer(torch.nn.Linear(2, 1), config).param_groups[0]["betas"] == (0.9, 0.999)


def test_single_future_frame_velocity_metric_is_explicitly_undefined():
    import json
    import torch
    from scripts.unified.overfit_prod import vel_mm
    points = torch.zeros(1, 1, 3, 3)
    value = vel_mm(points, points, torch.ones(1, 3, dtype=torch.bool))
    assert value is None
    assert json.dumps({"velocity": value}, allow_nan=False) == '{"velocity": null}'


@pytest.mark.parametrize("domain", ["mpm", "pickup", "cloth", "pushing"])
@pytest.mark.parametrize("profile", ["xy_p99_per_domain", "xy_p99_global"])
@pytest.mark.parametrize("receipt_style", ["legacy", "snapshot"])
def test_fixture_profiles_and_receipts_are_exact_and_json_native(domain, profile, receipt_style):
    from dataclasses import fields
    import json
    import torch
    from omegaconf import OmegaConf
    from data.normalization import (bind_batch_normalization, fixture_normalization_label,
                                    prepare_fixture, resolve_normalization)
    batch, _, _ = _fixture(domain)
    snapshot = resolve_normalization(profile)
    batch = bind_batch_normalization(batch, domain, snapshot)
    meta = {"sha256": "immutable-file-digest", "domain": domain}
    meta.update({"stats_set": profile} if receipt_style == "legacy" else {"normalization": snapshot})
    selected, receipt = prepare_fixture(batch, meta, domain, OmegaConf.create(snapshot), 1,
        context_len=batch.context_len, future_len=batch.num_future_frames)
    assert fixture_normalization_label(meta) == profile
    assert receipt["effective_normalization"] == snapshot
    assert receipt["effective_normalization_sha256"] == snapshot["sha256"]
    assert receipt["sha256"] == "immutable-file-digest" and receipt["selected_rows"] == [0]
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt
    for field in fields(batch):
        if torch.is_tensor(value := getattr(batch, field.name)):
            torch.testing.assert_close(getattr(selected, field.name), value[:1], rtol=0, atol=0)


@pytest.mark.parametrize("domain", ["mpm", "pickup", "cloth", "pushing"])
@pytest.mark.parametrize("source_profile", ["xy_p99_per_domain", "xy_p99_global"])
def test_fixture_profile_change_requires_explicit_rebind_even_when_mpm_bounds_equal(domain, source_profile):
    from data.normalization import bind_batch_normalization, prepare_fixture, resolve_normalization
    batch, _, _ = _fixture(domain)
    source = resolve_normalization(source_profile)
    target = resolve_normalization("xy_p99_global" if source_profile == "xy_p99_per_domain"
                                   else "xy_p99_per_domain")
    batch = bind_batch_normalization(batch, domain, source)
    meta = {"normalization": source}
    with pytest.raises(ValueError, match="rebind_normalization"):
        prepare_fixture(batch, meta, domain, target, 1)
    selected, receipt = prepare_fixture(batch, meta, domain, target, 1, rebind=True)
    assert selected.positions.data_ptr() == batch.positions.data_ptr()
    assert receipt["normalization"] == source and receipt["effective_normalization"] == target
    assert receipt["source_normalizer_bounds"]["delta"]["max_val"] == batch.delta_stats.max_val.tolist()


@pytest.mark.parametrize("rebind", [False, True])
@pytest.mark.parametrize("dimension", ["context_len", "future_len"])
def test_fixture_rebind_never_bypasses_checkpoint_window_contract(rebind, dimension):
    from data.normalization import prepare_fixture
    batch, meta, snapshot = _fixture()
    dimensions = {"context_len": batch.context_len, "future_len": batch.num_future_frames}
    dimensions[dimension] += 1
    with pytest.raises(ValueError, match="C/F"):
        prepare_fixture(batch, meta, "mpm", snapshot, 1, rebind=rebind, **dimensions)


@pytest.mark.parametrize("key", ["C", "F", "context_len", "future_len", "shape"])
def test_fixture_receipt_window_and_shape_cannot_contradict_tensors(key):
    from data.normalization import prepare_fixture
    batch, meta, snapshot = _fixture()
    meta[key] = [99, 99, 99, 3] if key == "shape" else 99
    with pytest.raises(ValueError, match="receipt"):
        prepare_fixture(batch, meta, "mpm", snapshot, 1, rebind=True)


@pytest.mark.parametrize("rebind", [False, True])
@pytest.mark.parametrize("corruption", ["dtype", "shape", "nan", "decreasing", "missing"])
def test_fixture_always_checks_actual_bounds(rebind, corruption):
    import torch
    from data.normalization import prepare_fixture
    batch, meta, snapshot = _fixture()
    if corruption == "dtype":
        batch.delta_stats.max_val = batch.delta_stats.max_val.double()
    elif corruption == "shape":
        batch.delta_stats.max_val = batch.delta_stats.max_val[None]
    elif corruption == "nan":
        batch.delta_stats.max_val[0] = torch.nan
    elif corruption == "decreasing":
        batch.delta_stats.max_val = batch.delta_stats.min_val.clone()
    else:
        batch.delta_stats = None
    with pytest.raises(ValueError, match="fixture delta"):
        prepare_fixture(batch, meta, "mpm", snapshot, 1, rebind=rebind)


@pytest.mark.parametrize("rebind", [False, True])
def test_complete_fixture_snapshot_detects_drift_even_during_rebind(rebind):
    from data.normalization import prepare_fixture
    batch, _, snapshot = _fixture("cloth")
    batch.pose_delta_stats.max_val *= 1.01
    with pytest.raises(ValueError, match="source snapshot"):
        prepare_fixture(batch, {"normalization": snapshot}, "cloth", snapshot, 1, rebind=rebind)


@pytest.mark.parametrize("contradiction", ["profile", "digest", "selector", "missing", "tamper"])
def test_fixture_receipt_identity_must_be_self_consistent(contradiction):
    import copy
    from data.normalization import prepare_fixture
    batch, _, snapshot = _fixture()
    meta = {"normalization": copy.deepcopy(snapshot)}
    if contradiction == "profile":
        meta["stats_set"] = "xy_p99_global"
    elif contradiction == "digest":
        meta["normalization_sha256"] = "0" * 64
    elif contradiction == "selector":
        meta["normalization"] = {"profile": "xy_p99_per_domain"}
    elif contradiction == "missing":
        meta = {}
    else:
        meta["normalization"]["scales"]["particle"]["mpm"][0] *= 2
    with pytest.raises(ValueError):
        prepare_fixture(batch, meta, "mpm", snapshot, 1, rebind=True)


def _raw_fixture(domain):
    import torch
    from data.normalization import NormalizationStats
    batch, _, snapshot = _fixture(domain)
    batch.delta_stats = NormalizationStats(-torch.ones(3), torch.ones(3))
    if domain != "mpm":
        batch.pose_delta_stats = NormalizationStats(-torch.ones(6), torch.ones(6))
    meta = {"stats_set": "raw", "domain": domain, "sha256": "original-raw-file-digest",
            "requires_explicit_rebind_normalization": True,
            "normalization_loaded_during_selection": False}
    return batch, meta, snapshot


@pytest.mark.parametrize("domain", ["mpm", "pickup", "cloth", "pushing"])
@pytest.mark.parametrize("profile", ["xy_p99_per_domain", "xy_p99_global"])
def test_truthful_raw_fixture_requires_rebind_and_preserves_all_tensors(domain, profile):
    from dataclasses import fields
    import torch
    from data.normalization import prepare_fixture, resolve_normalization
    batch, meta, _ = _raw_fixture(domain)
    snapshot = resolve_normalization(profile)
    with pytest.raises(ValueError, match="requires --rebind_normalization"):
        prepare_fixture(batch, meta, domain, snapshot, 1)
    selected, receipt = prepare_fixture(batch, meta, domain, snapshot, 1, rebind=True)
    assert receipt["stats_set"] == "raw" and receipt["sha256"] == meta["sha256"]
    assert receipt["effective_normalization"] == snapshot
    for field in fields(batch):
        if torch.is_tensor(value := getattr(batch, field.name)):
            torch.testing.assert_close(getattr(selected, field.name), value[:1], rtol=0, atol=0)
            assert getattr(selected, field.name).data_ptr() == value.data_ptr()


@pytest.mark.parametrize("corruption", ["flag", "snapshot", "bounds", "actions"])
def test_raw_rebind_checks_truthful_placeholder_and_physical_action_contract(corruption):
    from data.normalization import prepare_fixture
    batch, meta, snapshot = _raw_fixture("cloth")
    if corruption == "flag":
        meta["normalization_loaded_during_selection"] = True
    elif corruption == "snapshot":
        meta["normalization"] = snapshot
    elif corruption == "bounds":
        batch.delta_stats.max_val *= 0.5
    else:
        batch.actions += 1
    with pytest.raises((ValueError, AssertionError)):
        prepare_fixture(batch, meta, "cloth", snapshot, 1, rebind=True)


@pytest.mark.parametrize("rebind", [False, True])
@pytest.mark.parametrize("field", ["actions", "pose_actions", "pose_action_mask", "body_templates"])
def test_fixture_raw_action_and_native_pose_contract_is_required_on_both_paths(rebind, field):
    from data.normalization import prepare_fixture
    batch, meta, snapshot = _fixture("cloth")
    if field == "actions":
        batch.actions += 1
    else:
        setattr(batch, field, None)
    with pytest.raises((ValueError, AssertionError)):
        prepare_fixture(batch, meta, "cloth", snapshot, 1, rebind=rebind)


def _checkpoint_cli(tmp_path, monkeypatch, *, raw=False, source_profile="xy_p99_per_domain",
                    checkpoint_profile="xy_p99_per_domain"):
    """Exercise the real CLI/fixture path with CPU tensors and only the model mocked."""
    import json
    from pathlib import Path
    import torch
    from omegaconf import OmegaConf
    import inference
    from data.normalization import bind_batch_normalization, resolve_normalization
    from utils.reproducibility import file_sha256
    from scripts.dev import check_checkpoint
    batch, meta, _ = _raw_fixture("cloth") if raw else _fixture("cloth")
    if not raw:
        source = resolve_normalization(source_profile)
        batch = bind_batch_normalization(batch, "cloth", source)
        meta = {"normalization": source, "domain": "cloth"}  # No legacy stats_set.
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    path = batch_dir / "named-moving.pt"
    torch.save(batch, path)
    meta["sha256"] = file_sha256(path)
    path.with_suffix(".json").write_text(json.dumps(meta))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mocked checkpoint weights; actual digest")
    output = tmp_path / "predictions.pt"
    cfg = OmegaConf.create({"normalization": resolve_normalization(checkpoint_profile),
        "data": {"context_len": batch.context_len, "future_len": batch.num_future_frames}})
    calls, sync = [], []
    monkeypatch.setattr(inference, "load_model_and_diffusion", lambda *a: (object(), object(), cfg, None))

    def sample(model, actual, diffusion, **kwargs):
        calls.append(actual)
        return actual.delta_stats.normalize(actual.future_deltas)

    monkeypatch.setattr(inference, "run_diffusion_inference_df", sample)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: sync.append(device))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo)
    command = ["check_checkpoint", "--repo", str(repo), "--checkpoint", str(checkpoint),
        "--batch-dir", str(batch_dir), "--output", str(output), "--domains", "cloth",
        "--batch_name", "named-moving", "--device", "cpu", "--precision", "fp32"]
    monkeypatch.setattr(sys, "argv", command)
    return check_checkpoint, batch, meta, cfg, calls, sync, command, path, output


@pytest.mark.parametrize("raw", [False, True])
def test_checkpoint_cli_named_fixture_rebind_and_provenance(tmp_path, monkeypatch, raw):
    import json
    import torch
    from data.normalization import domain_normalizers
    from utils.reproducibility import file_sha256
    cli, original, meta, cfg, calls, sync, command, path, output = _checkpoint_cli(
        tmp_path, monkeypatch, raw=raw, checkpoint_profile="xy_p99_global")
    with pytest.raises(ValueError, match="rebind_normalization"):
        cli.main()
    assert not calls and not output.exists()
    command.append("--rebind_normalization")
    cli.main()
    assert len(calls) == 1 and sync == ["cpu"]
    torch.testing.assert_close(calls[0].positions, original.positions, rtol=0, atol=0)
    assert torch.equal(calls[0].delta_stats.max_val, domain_normalizers(cfg.normalization, "cloth")[0].max_val)
    receipt = json.loads(output.with_suffix(".json").read_text())
    assert receipt["checkpoint_sha256"] == file_sha256(tmp_path / "checkpoint.pt")
    assert "git_commit" in receipt and "git_dirty" in receipt
    assert receipt["normalization_sha256"] == cfg.normalization.sha256
    info = receipt["domains"]["cloth"]
    assert info["batch_name"] == "named-moving" and info["fixture"]["selected_rows"] == [0, 1]
    assert info["fixture"]["sha256"] == meta["sha256"] == file_sha256(path)
    assert info["fixture"]["effective_normalization"]["sha256"] == cfg.normalization.sha256
    assert json.loads(path.with_suffix(".json").read_text()) == meta  # Immutable input receipt.


@pytest.mark.parametrize("failure", ["digest", "normalizer", "context", "future", "domain", "missing-domain"])
def test_checkpoint_cli_rejects_incompatible_fixture_before_sampling(tmp_path, monkeypatch, failure):
    import json
    import torch
    from utils.reproducibility import file_sha256
    cli, batch, meta, cfg, calls, _, _, path, output = _checkpoint_cli(tmp_path, monkeypatch)
    if failure == "digest":
        path.write_bytes(b"Not a pickle; integrity check must precede deserialization")
    elif failure == "context":
        cfg.data.context_len += 1
    elif failure == "future":
        cfg.data.future_len += 1
    else:
        if failure == "normalizer":
            batch.delta_stats.max_val *= 1.01
        elif failure == "missing-domain":
            batch.domain = None
            meta.pop("domain")
        else:
            batch.domain = "mpm"
        torch.save(batch, path)
        meta["sha256"] = file_sha256(path)
        path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        cli.main()
    assert not calls and not output.exists()


@pytest.mark.parametrize("domains,names", [
    ("cloth,pickup", "one"), ("cloth", ""), ("cloth", "../moving"),
    ("cloth", "/absolute"), ("cloth", "..\\moving"), ("cloth", "."),
    ("cloth,cloth", "a,b"), ("cloth,pickup", "same,same"), ("unknown", "named")])
def test_checkpoint_cli_rejects_invalid_fixture_names_before_model_load(tmp_path, monkeypatch, domains, names):
    import inference
    cli, _, _, _, _, _, command, _, output = _checkpoint_cli(tmp_path, monkeypatch)
    command[command.index("--domains") + 1] = domains
    command[command.index("--batch_name") + 1] = names
    monkeypatch.setattr(inference, "load_model_and_diffusion", lambda *a: pytest.fail("Must validate names first"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize("named", [False, True])
def test_checkpoint_cli_fresh_fixture_embeds_json_native_snapshot(tmp_path, monkeypatch, named):
    import json
    from scripts.dev import check_checkpoint
    from scripts.unified import eval_unified
    cli, batch, _, cfg, calls, _, command, path, output = _checkpoint_cli(tmp_path, monkeypatch)
    # Select an absent stem: the same CLI draws and pins one validation row.
    if named:
        command[command.index("--batch_name") + 1] = "fresh"
    else:
        at = command.index("--batch_name")
        del command[at:at + 2]

    class Dataset:
        def _process_sample(self, sample):
            return sample

        def __iter__(self):
            self._process_sample({"__url__": "source.tar", "__key__": "record"})
            yield batch

    built = []
    # The checkpoint's own validation split (saved data.roots included), as eval_unified reads it.
    monkeypatch.setattr(eval_unified, "evaluation_dataset",
                        lambda *a: built.append(a) or Dataset())
    check_checkpoint.main()
    assert built == [(cfg, "cloth", cfg.data.context_len, cfg.data.future_len)]
    receipt = json.loads(path.with_name("fresh.json" if named else "cloth.json").read_text())
    assert receipt["domain"] == "cloth" and receipt["future_len"] == cfg.data.future_len
    assert receipt["normalization"]["sha256"] == cfg.normalization.sha256
    assert receipt["sources"] == [{"__url__": "source.tar", "__key__": "record"}]
    assert calls[0].batch_size == 1 and output.exists()


def test_overfit_loads_snapshot_only_receipt_and_passes_recipe_window_before_gpu(tmp_path, monkeypatch, capsys):
    import json
    import torch
    from data.normalization import prepare_fixture
    from utils.reproducibility import file_sha256
    from scripts.unified import overfit_prod
    batch, _, snapshot = _fixture("cloth")
    path = tmp_path / "cloth.pt"
    torch.save(batch, path)
    path.with_suffix(".json").write_text(json.dumps({"normalization": snapshot,
        "domain": "cloth", "sha256": file_sha256(path)}))

    def stop_after_cpu_preflight(*args, **kwargs):
        assert kwargs["context_len"] == batch.context_len
        assert kwargs["future_len"] == batch.num_future_frames
        prepare_fixture(*args, **kwargs)
        raise RuntimeError("CPU fixture preflight complete")

    monkeypatch.setattr(overfit_prod, "prepare_fixture", stop_after_cpu_preflight)
    monkeypatch.setattr(sys, "argv", ["overfit", "--config", "configs/unet.yaml", "--domain", "cloth",
        "--batch_dir", str(tmp_path), "--out_root", str(tmp_path / "out"), "--override",
        f"data.context_len={batch.context_len}", f"data.future_len={batch.num_future_frames}",
        "data.batch_size=1"])
    with pytest.raises(RuntimeError, match="CPU fixture preflight complete"):
        overfit_prod.main()
    assert "stats xy_p99_per_domain" in capsys.readouterr().out


@pytest.mark.parametrize("backbone", ["transformer", "unet"])
def test_config_metadata_encodes_disabled_spikes_without_changing_live_config(backbone):
    import json
    import math
    from omegaconf import OmegaConf
    from utils.config import load_config
    from utils.reproducibility import config_for_json
    cfg = load_config(f"configs/{backbone}.yaml", ["train.loss_spike_threshold=.inf",
        "train.grad_norm_spike_threshold=.inf"])
    before = OmegaConf.to_container(cfg, resolve=True)
    metadata = config_for_json(cfg)
    assert json.loads(json.dumps(metadata, allow_nan=False)) == metadata
    for name in ("loss_spike_threshold", "grad_norm_spike_threshold"):
        assert metadata["train"][name] == "infinity"
        assert math.isinf(cfg.train[name]) and cfg.train[name] > 0
    assert OmegaConf.to_container(cfg, resolve=True) == before
    metadata["train"]["lr"] = 0.9
    assert cfg.train.lr == before["train"]["lr"]


def test_config_metadata_handles_nested_list_infinities_and_preserves_finite_values():
    import copy
    import json
    from utils.reproducibility import config_for_json
    cfg = {"nested": [{"values": [float("inf"), float("-inf"), 1.25, -0.0, 2, True, None, "text"]}]}
    before = copy.deepcopy(cfg)
    metadata = config_for_json(cfg)
    assert metadata == {"nested": [{"values": ["infinity", "-infinity", 1.25, -0.0, 2, True, None, "text"]}]}
    assert cfg == before and json.loads(json.dumps(metadata, allow_nan=False)) == metadata


@pytest.mark.parametrize("cfg", [{"bad": float("nan")}, {"nested": [{"bad": [float("nan")]}]}])
def test_config_metadata_rejects_nan_instead_of_hiding_invalid_values(cfg):
    from utils.reproducibility import config_for_json
    with pytest.raises(ValueError, match="NaN"):
        config_for_json(cfg)


@pytest.mark.parametrize("backbone", ["transformer", "unet"])
def test_finite_config_metadata_is_unchanged_and_objective_consumer_roundtrips(backbone):
    import json
    from omegaconf import OmegaConf
    from utils.config import load_config
    from utils.reproducibility import config_for_json
    from utils.trajectory_objective import trajectory_loss_options
    cfg = load_config(f"configs/{backbone}.yaml")
    assert config_for_json(cfg) == OmegaConf.to_container(cfg, resolve=True)
    cfg.train.loss_spike_threshold = cfg.train.grad_norm_spike_threshold = float("inf")
    cfg.train.clip_deltas = False
    metadata = json.loads(json.dumps(config_for_json(cfg), allow_nan=False))
    # plot_overfit reads objective options from this metadata, not spike settings.
    assert trajectory_loss_options(OmegaConf.create(metadata)) == trajectory_loss_options(cfg)
    assert metadata["train"]["clip_deltas"] is False


def test_config_metadata_does_not_relax_strict_metric_serialization():
    import json
    from utils.reproducibility import config_for_json
    result = {"provenance": {"resolved_config": config_for_json({"threshold": float("inf")})},
              "sample_mean_mm": float("nan")}
    with pytest.raises(ValueError):
        json.dumps(result, allow_nan=False)


def test_overfit_fixture_stream_applies_the_recipe_window_policy():
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from scripts.unified.overfit_prod import apply_window_policy
    cfg = OmegaConf.create(dict(data=dict(window_motion=dict(folding=dict(alpha=1.0, floor_mm=0.02)),
                                          batch_across_episodes=["mpm_fresh"])))
    ds = SimpleNamespace(motion_alpha=0.0, motion_floor_m=0.0, cross_episode=False)
    apply_window_policy(ds, cfg, "folding")
    assert (ds.motion_alpha, ds.cross_episode) == (1.0, False) and abs(ds.motion_floor_m - 2e-5) < 1e-12
    other = SimpleNamespace(motion_alpha=0.0, motion_floor_m=0.0, cross_episode=False)
    apply_window_policy(other, cfg, "mpm_fresh")
    assert (other.motion_alpha, other.cross_episode) == (0.0, True)
    plain = SimpleNamespace(motion_alpha=0.0, motion_floor_m=0.0, cross_episode=False)
    apply_window_policy(plain, OmegaConf.create(dict(data={})), "folding")
    assert (plain.motion_alpha, plain.cross_episode) == (0.0, False)
