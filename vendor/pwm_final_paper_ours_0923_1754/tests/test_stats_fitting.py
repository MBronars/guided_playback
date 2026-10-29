"""Raw calibration and fitting tests use physical fixtures, never old stats files."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data.normalization import load_normalization_artifact, resolve_normalization
from data.stats_fitting import (
    PriorityReservoir, audit_caches, calibration_identity, collect_rows, collection_coverage, fit_artifact,
    json_digest, load_collection_cache, load_collection_index, raw_motion_rows,
    save_collection_cache, write_json_new,
)


def _fixture(domain="pickup", policy="training", seed=8, split="train"):
    gen = torch.Generator().manual_seed(seed)
    positions = torch.randn(2, 5, 5, 3, generator=gen) * 0.01
    valid = torch.ones(2, 5, 5, dtype=torch.bool)
    valid[0, 2, 0] = False  # Both adjacent particle and complete-body transitions disappear.
    batch = SimpleNamespace(
        domain=domain, positions=positions, context_len=2,
        pose_deltas_gt=torch.randn(2, 3, 2, 6, generator=gen) * 0.01,
        body_type=torch.tensor([[1, 0], [0, 1]]),
        particle_body=torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1]]),
        body_template_mask=torch.ones(2, 2, 3, dtype=torch.bool),
        get_valid_mask=lambda: torch.ones(2, 5, dtype=torch.bool),
    )
    if domain == "mpm":
        batch.pose_deltas_gt = batch.body_type = batch.particle_body = batch.body_template_mask = None
    metadata = dict(domain=domain, split=split, source="fixture.tar", key="record0", source_rows=[0, 1],
                    context_len=2, future_len=3, window_starts=[0, 0], window_seed=seed, seed=seed,
                    window_policy=policy, skipped_before=[], valid_particle_mask=valid)
    source = dict(domain=domain, seed=seed, context_len=2, future_len=3, window_policy=policy,
                  split=split, sources=[dict(path="fixture.tar", bytes=123, mtime_ns=1000)],
                  assets=[dict(path="excluded.json", sha256="fixture")], max_records=1)
    source["sha256"] = json_digest(source)
    return batch, metadata, source


def _cache(domain="pickup", policy="training", split="train", capacity=17):
    batch, meta, source = _fixture(domain, policy=policy, split=split)
    return collect_rows(iter([(batch, meta)]), calibration_identity(source, capacity=capacity))


def _all_caches():
    return [_cache(domain, policy) for domain in ("mpm", "pickup", "cloth", "pushing")
            for policy in ("training", "prefix")]


def test_reservoir_is_bounded_chunk_invariant_and_exact_without_truncation():
    rows = np.arange(1800, dtype=np.float32).reshape(600, 3)
    first, second = PriorityReservoir(29, 3, 912, chunk_rows=13), PriorityReservoir(29, 3, 912)
    first.add(rows)
    for block in np.array_split(rows, 19):
        second.add(block)
    assert len(first.rows) == len(second.rows) == 29
    assert first.seen == second.seen == len(rows)
    assert np.array_equal(first.rows, second.rows)
    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.rows, rows[first.indices])
    entire = PriorityReservoir(1000, 3, 912)
    entire.add(rows)
    assert np.array_equal(entire.rows[np.argsort(entire.indices)], rows)


def test_priority_reservoir_has_no_early_row_bias():
    rows = np.arange(30, dtype=np.float32)[:, None]
    counts = np.zeros(len(rows), dtype=int)
    for seed in range(500):
        reservoir = PriorityReservoir(5, 1, seed)
        reservoir.add(rows)
        counts[reservoir.indices] += 1
    # A deterministic sanity check of uniform priorities, not a distributional proof.
    assert counts.min() > 50 and counts.max() < 115


@pytest.mark.parametrize("kwargs", [dict(capacity=0), dict(capacity=True), dict(seed=-1),
                                     dict(seed=False), dict(chunk_rows=0), dict(channels=0)])
def test_invalid_reservoir_settings_fail(kwargs):
    params = dict(capacity=3, channels=3, seed=0)
    params.update(kwargs)
    with pytest.raises(ValueError):
        PriorityReservoir(**params)


def test_physical_row_masks_use_both_frames_and_per_element_rigid_body_types():
    batch, meta, _ = _fixture()
    dp, pose = raw_motion_rows(batch, meta)
    assert dp.shape == (38, 3)  # 2*4*5 minus both transitions adjacent to the invalid frame.
    expected_pose = torch.cat([batch.pose_deltas_gt[0, 2:, 0], batch.pose_deltas_gt[1, :, 1]])
    assert np.array_equal(pose, expected_pose.numpy())
    assert pose.shape == (4, 6)
    batch.positions[0, 2, 0] = float("nan")
    dp2, pose2 = raw_motion_rows(batch, meta)
    assert np.array_equal(dp, dp2)  # Invalid temporal padding cannot contaminate the fit.
    assert np.array_equal(pose, pose2)


def test_supervised_particle_rows_are_the_deformable_rows_of_scenes_with_rigid_bodies():
    # Element 0: particles 0-1 rigid, 2-4 deformable. Element 1: particles 0-1 deformable, 2-4 rigid. The objective
    # supervises deformable rows through the particle channel, so a scale fitted on them ignores commanded rigid points.
    batch, meta, _ = _fixture()
    everything, pose = raw_motion_rows(batch, meta)
    dp, pose_supervised = raw_motion_rows(batch, meta, particle_rows="supervised")
    assert everything.shape == (38, 3) and dp.shape == (20, 3)       # 3 particles x 4 pairs + 2 particles x 4 pairs
    assert np.array_equal(pose, pose_supervised)                     # pose rows never depend on the particle population
    step = batch.positions[:, 1:] - batch.positions[:, :-1]
    expected = torch.cat([step[0][:, 2:].reshape(-1, 3), step[1][:, :2].reshape(-1, 3)])
    assert np.array_equal(np.sort(dp, axis=0), np.sort(expected.numpy(), axis=0))
    # An element without deformable rows keeps every valid particle: its scale only normalizes inputs.
    batch.body_type = torch.tensor([[1, 1], [0, 1]])
    assert raw_motion_rows(batch, meta, particle_rows="supervised")[0].shape == (18 + 8, 3)
    passive, passive_meta, _ = _fixture("mpm")
    assert np.array_equal(raw_motion_rows(passive, passive_meta)[0],
                          raw_motion_rows(passive, passive_meta, particle_rows="supervised")[0])
    with pytest.raises(ValueError, match="particle_rows must be all or supervised"):
        raw_motion_rows(batch, meta, particle_rows="predicted")


def test_batch_valid_lengths_and_absent_templates_also_exclude_rows():
    batch, meta, _ = _fixture()
    batch.get_valid_mask = lambda: torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    meta["valid_particle_mask"] &= batch.get_valid_mask()[:, None, :]
    batch.body_template_mask[1, 1] = False
    dp, pose = raw_motion_rows(batch, meta)
    assert dp.shape == (22, 3)
    assert pose.shape == (1, 6)


@pytest.mark.parametrize("temporally_new", [False, True])
def test_source_validity_outside_batch_prefix_fails_not_silently_truncated(temporally_new):
    batch, meta, _ = _fixture("mpm")
    batch.get_valid_mask = lambda: torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    meta["valid_particle_mask"][0] = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool)
    if temporally_new:
        meta["valid_particle_mask"][0, :2, 3] = False
    with pytest.raises(ValueError, match="outside the batch's representable prefix"):
        raw_motion_rows(batch, meta)


def test_nonfinite_valid_targets_and_missing_rigid_sidecar_fail():
    batch, meta, _ = _fixture()
    batch.positions[1, 1, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        raw_motion_rows(batch, meta)
    batch, meta, _ = _fixture()
    batch.pose_deltas_gt = None
    with pytest.raises(ValueError, match="requires pose"):
        raw_motion_rows(batch, meta)


def test_collection_ledger_and_arrays_are_deterministic_and_masked():
    first, second = _cache(), _cache()
    assert first[1] == second[1]
    for name in first[0]:
        assert np.array_equal(first[0][name], second[0][name])
    assert first[1]["seen"] == dict(particle=38, pose=4)
    assert first[1]["retained"] == dict(particle=17, pose=4)
    assert "valid_particle_mask" not in first[1]["ledger"][0]
    assert len(first[1]["ledger"][0]["valid_mask_sha256"]) == 64


def test_empty_and_inconsistent_streams_fail():
    batch, meta, source = _fixture()
    identity = calibration_identity(source, capacity=10)
    with pytest.raises(ValueError, match="empty"):
        collect_rows(iter([]), identity)
    for key, value in (("split", "val"), ("future_len", 2), ("domain", "mpm")):
        bad = {**meta, key: value}
        with pytest.raises(ValueError):
            collect_rows(iter([(batch, bad)]), identity)


@pytest.mark.parametrize("change", [
    lambda s: s["sources"][0].update(mtime_ns=2000),
    lambda s: s["sources"][0].update(bytes=124),
    lambda s: s["sources"].append(dict(path="another.tar", bytes=1, mtime_ns=1)),
    lambda s: s["assets"][0].update(sha256="changed exclusion"),
    lambda s: s.update(context_len=4), lambda s: s.update(future_len=16),
    lambda s: s.update(seed=9), lambda s: s.update(window_policy="prefix"),
    lambda s: s.update(max_records=2), lambda s: s.update(split="val"),
])
def test_cache_identity_changes_for_source_exclusion_and_sampling_inputs(change):
    _, _, source = _fixture()
    old = calibration_identity(source, capacity=30)
    changed = copy.deepcopy(source)
    change(changed)
    assert calibration_identity(changed, capacity=30)["sha256"] != old["sha256"]
    assert calibration_identity(source, capacity=31)["sha256"] != old["sha256"]


def test_collector_and_geometry_code_fingerprint_invalidate_automatic_reuse(monkeypatch):
    import data.stats_fitting as fitting
    _, _, source = _fixture()
    old = calibration_identity(source, capacity=30)
    assert "data/stats_fitting.py" in old["implementation_sha256"]
    assert "utils/pose_ops.py" in old["implementation_sha256"]
    changed = {**old["implementation_sha256"], "data/stats_fitting.py": "modified quantity logic"}
    monkeypatch.setattr(fitting, "_implementation_fingerprint", lambda: changed)
    new = calibration_identity(source, capacity=30)
    assert new["sha256"] != old["sha256"]


def test_cache_roundtrip_refuses_stale_identity_and_overwrite(tmp_path):
    arrays, meta = _cache()
    path = tmp_path / "cache.npz"
    save_collection_cache(path, arrays, meta)
    loaded, loaded_meta = load_collection_cache(path, expected_identity=meta["identity"])
    assert meta == loaded_meta
    for name in arrays:
        assert np.array_equal(arrays[name], loaded[name])
    wrong = copy.deepcopy(meta["identity"])
    wrong["capacity_per_representation"] = 1
    with pytest.raises(ValueError, match="does not match"):
        load_collection_cache(path, expected_identity=wrong)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_collection_cache(path, arrays, meta)
    assert path.read_bytes() == before


@pytest.mark.parametrize("what", ["array", "metadata"])
def test_cache_tampering_fails(what, tmp_path):
    arrays, meta = _cache()
    if what == "array":
        arrays["particle"][0, 0] += 1
    else:
        meta["ledger"][0]["key"] = "different"
    path = tmp_path / "bad.npz"
    save_collection_cache(path, arrays, meta)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_collection_cache(path)


def test_fit_both_profiles_from_same_fixed_all_domain_rows(tmp_path):
    caches = _all_caches()
    candidate = fit_artifact(caches, artifact_version="fixture-fit-v1")
    path = tmp_path / "candidate.json"
    write_json_new(path, candidate)
    assert load_normalization_artifact(path) == candidate
    per_domain = candidate["profiles"]["xy_p99_per_domain"]
    global4 = candidate["profiles"]["xy_p99_global"]
    for index in (0, 1):
        assert global4["translation"][index] == max(
            *[value[index] for value in per_domain["particle"].values()],
            *[value[index] for value in per_domain["pose"].values()])
        assert global4["rotation"][index] == max(value[index + 2] for value in per_domain["pose"].values())
    assert fit_artifact(list(reversed(caches)), artifact_version="fixture-fit-v1") == candidate
    snapshot = resolve_normalization({"profile": "xy_p99_global", "artifact": str(path)})
    assert snapshot["artifact_version"] == "fixture-fit-v1"
    assert all(row["coverage"]["records"] == 1 and row["coverage"]["episode_windows"] == 2
               for row in candidate["provenance"]["caches"])


def test_coverage_distinguishes_records_windows_and_recorded_shard_visits():
    metadata = dict(identity=dict(source_manifest=dict(
        sources=[dict(path=f"shard{i}.tar") for i in range(4)], max_records=32)), ledger=[
        dict(source="shard0.tar", window_starts=[2, 3], skipped_before=[]),
        dict(source="shard1.tar", window_starts=[0, 0, 4],
             skipped_before=[dict(source="shard2.tar", key="unusable"),
                             dict(source="shard1.tar", key="also-unusable")]),
    ])
    before = copy.deepcopy(metadata)
    coverage = collection_coverage(metadata)
    assert metadata == before
    assert coverage["records"] == 2 and coverage["episode_windows"] == 5
    assert coverage["record_limit"] == 32 and coverage["record_limit_reached"] is False
    assert coverage["shards"] == dict(contributing=2, visited_recorded=3, available=4,
                                      visited_recorded_fraction=0.75)
    assert coverage["skipped_records_recorded"] == 2
    assert "lower bounds" in coverage["meaning"]
    assert "must not be summed" in coverage["meaning"]


def test_coverage_does_not_invent_missing_historical_details():
    _, metadata = _cache()
    row = metadata["ledger"][0]
    del row["window_starts"], row["skipped_before"]
    coverage = collection_coverage(metadata)
    assert coverage["episode_windows"] is None
    assert coverage["skipped_records_recorded"] is None
    assert coverage["records"] == 1 and coverage["record_limit_reached"] is True


def test_coverage_is_derived_without_modifying_older_cache_schema(tmp_path):
    arrays, metadata = _cache()
    assert "coverage" not in metadata
    path = tmp_path / "historical-cache.npz"
    save_collection_cache(path, arrays, metadata)
    before = path.read_bytes()
    loaded_arrays, loaded_metadata = load_collection_cache(path)
    coverage = collection_coverage(loaded_metadata)
    report = audit_caches([(loaded_arrays, loaded_metadata)])
    assert report["reports"][0]["coverage"] == coverage
    assert path.read_bytes() == before
    assert load_collection_cache(path)[1] == metadata == loaded_metadata
    assert all(np.array_equal(arrays[name], value) for name, value in loaded_arrays.items())


def test_fit_takes_max_of_policy_percentiles_not_a_pooled_percentile():
    caches = _all_caches()
    # Increase prefix motion. Max-over-policy must preserve its p99 even when the
    # training population is arbitrarily larger, not dilute it through pooling.
    for arrays, meta in caches:
        if meta["identity"]["source_manifest"]["window_policy"] == "prefix":
            arrays["particle"] *= 10
            arrays["pose"] *= 10
    artifact = fit_artifact(caches, artifact_version="fixture-fit-v1")
    for arrays, meta in caches:
        source = meta["identity"]["source_manifest"]
        if source["window_policy"] == "prefix":
            q = np.percentile(np.abs(arrays["particle"]), 99, axis=0).astype(np.float32)
            expected = [float(max(q[0], q[1])), float(q[2])]
            assert artifact["profiles"]["xy_p99_per_domain"]["particle"][source["domain"]] == expected


def test_fit_rejects_validation_partial_duplicate_inconsistent_and_zero_scale_inputs():
    with pytest.raises(ValueError, match="training episodes"):
        fit_artifact([_cache(split="val")], artifact_version="candidate")
    with pytest.raises(ValueError, match="missing domains"):
        fit_artifact([_cache()], artifact_version="candidate")
    with pytest.raises(ValueError, match="unique"):
        fit_artifact([_cache(), _cache()], artifact_version="candidate")
    caches = _all_caches()
    caches[0][0]["particle"][:] = 0
    with pytest.raises(ValueError, match="Zero/nonfinite"):
        fit_artifact(caches, artifact_version="candidate")
    with pytest.raises(ValueError, match="candidate artifact_version"):
        fit_artifact(_all_caches(), artifact_version="frozen-replacement")
    caches = _all_caches()
    caches[0][1]["identity"]["source_manifest"]["context_len"] = 4
    with pytest.raises(ValueError, match="share context"):
        fit_artifact(caches, artifact_version="candidate")


def test_audit_reports_physical_clipping_without_refitting(tmp_path):
    before = resolve_normalization()
    report = audit_caches([_cache("pushing", split="val"), _cache("mpm")])
    assert resolve_normalization() == before
    assert report["reports"][0]["split"] == "val"
    small = report["reports"][0]["profiles"]["xy_p99_per_domain"]["particle"]
    large = report["reports"][0]["profiles"]["xy_p99_global"]["particle"]
    assert small["clipped_fraction_any_channel"] > large["clipped_fraction_any_channel"]
    assert small["clipped_target_reconstruction_rmse_per_channel"][2] > 0
    assert report["reports"][1]["profiles"]["xy_p99_global"]["pose"] is None
    path = tmp_path / "report.json"
    write_json_new(path, report)
    assert json.loads(path.read_text()) == report
    with pytest.raises(FileExistsError):
        write_json_new(path, {"replacement": True})


def test_audit_records_selected_artifact_identity_without_changing_cache(tmp_path):
    caches = _all_caches()
    original = copy.deepcopy(caches)
    candidate = tmp_path / "candidate.json"
    write_json_new(candidate, fit_artifact(caches, artifact_version="audited-candidate-v1"))
    candidate_bytes = candidate.read_bytes()
    frozen = audit_caches(caches)
    fitted = json.loads(json.dumps(audit_caches(caches, candidate)))
    assert candidate.read_bytes() == candidate_bytes
    assert [row["cache_sha256"] for row in frozen["reports"]] == \
           [row["cache_sha256"] for row in fitted["reports"]]
    for profile, snapshot in fitted["normalization"].items():
        assert snapshot == resolve_normalization({"profile": profile, "artifact": str(candidate)})
        assert snapshot["artifact_version"] == "audited-candidate-v1"
        assert snapshot["sha256"] != frozen["normalization"][profile]["sha256"]
        # The report remains self-contained after JSON roundtrip.
        assert resolve_normalization(snapshot) == snapshot
    for (arrays, metadata), (old_arrays, old_metadata) in zip(caches, original):
        assert metadata == old_metadata
        assert all(np.array_equal(values, old_arrays[name]) for name, values in arrays.items())


def test_collection_index_checks_immutable_cache_membership(tmp_path):
    arrays, meta = _cache()
    save_collection_cache(tmp_path / "cache.npz", arrays, meta)
    index = dict(schema_version=1, caches=[dict(path="cache.npz", sha256=meta["sha256"])])
    index["sha256"] = json_digest(index)
    path = tmp_path / "collection.json"
    write_json_new(path, index)
    assert load_collection_index(path)[0][1] == meta
    index["caches"][0]["sha256"] = "bad"
    index["sha256"] = json_digest({k: v for k, v in index.items() if k != "sha256"})
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index/cache checksum"):
        load_collection_index(path)


def test_fresh_process_reservoir_and_collection_are_deterministic():
    script = """
import json
from tests.test_stats_fitting import _cache
from data.stats_fitting import _array_digest
arrays, meta = _cache()
print(json.dumps({'metadata': meta, 'arrays': {k: _array_digest(v) for k,v in arrays.items()}}, sort_keys=True))
"""
    outputs = [subprocess.check_output([sys.executable, "-c", script], text=True) for _ in range(2)]
    assert outputs[0] == outputs[1]


def test_single_cli_has_collect_fit_and_audit_help():
    root = Path(__file__).resolve().parents[1]
    for command in ([], ["collect"], ["fit"], ["audit"]):
        completed = subprocess.run([sys.executable, "scripts/compute_stats.py", *command, "--help"],
                                   cwd=root, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr


def test_cli_fit_and_audit_end_to_end_and_no_overwrite(tmp_path, capsys):
    from scripts.compute_stats import main
    rows = []
    for index, (arrays, metadata) in enumerate(_all_caches()):
        path = tmp_path / f"cache{index}.npz"
        save_collection_cache(path, arrays, metadata)
        rows.append(dict(path=path.name, sha256=metadata["sha256"]))
    index = dict(schema_version=1, caches=rows)
    index["sha256"] = json_digest(index)
    collection = tmp_path / "collection.json"
    write_json_new(collection, index)
    candidate, report = tmp_path / "candidate.json", tmp_path / "audit.json"
    args = ["fit", "--collection", str(collection), "--out", str(candidate), "--version", "fixture-v1"]
    main(args)
    with pytest.raises(FileExistsError):
        main(args)
    main(["audit", "--collection", str(collection), "--artifact", str(candidate), "--out", str(report)])
    assert len(json.loads(report.read_text())["reports"]) == 8
    assert "1 records / 2 episode windows; recorded shards 1/1; 0 recorded skips" in capsys.readouterr().out
