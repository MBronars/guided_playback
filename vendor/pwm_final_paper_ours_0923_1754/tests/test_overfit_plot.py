"""Plots must reject false comparisons and preserve complete structured records."""
import copy
import json

import numpy as np
from PIL import Image
import pytest

from scripts.unified.plot_overfit import load_runs, plot


def record():
    return {"steps": 201, "history": np.geomspace(1, 0.01, 201).tolist(),
            "batch_receipt": {"sha256": "fixed-batch"}, "stats_set": "xy_p99",
            "baselines": {"zero_motion": [1, 2], "hold_velocity": [2, 3]},
            "final_multi_seed_mm": {"values": [0.1, 0.2]},
            "evals": [{"step": s, "domain_updates": s, "sample_mean_mm": 4 / (s + 1),
                       "fixed_t_objective": {"0.05": [0.1, 0.01], "0.95": [0.01, 0.001]}}
                      for s in (0, 201)]}


def runs(tmp_path, second=None):
    paths = []
    for label, data in (("first", record()), ("second", second or record())):
        directory = tmp_path / label
        directory.mkdir()
        (directory / "mpm.json").write_text(json.dumps(data))
        paths.append((label, directory))
    return paths


@pytest.mark.parametrize("change,match", [
    ({"batch_receipt": {"sha256": "different"}}, "hashed batch"),
    ({"history": [0] * 201}, "non-positive"),
    ({"steps": 202}, "unfinished"),
    ({"stats_set": "different"}, "normalization differs"),
])
def test_plot_rejects_misleading_comparisons(tmp_path, change, match):
    data = record()
    data.update(change)
    with pytest.raises(ValueError, match=match):
        load_runs(runs(tmp_path, data), ["mpm"])


def test_completed_plot_contains_all_three_views_without_mutating_records(tmp_path):
    data = load_runs(runs(tmp_path), ["mpm"])
    before = copy.deepcopy(data)
    output = tmp_path / "plots"
    plot(data, ["mpm"], output, "Test fixture")
    assert data == before
    for name in ("loss", "fixed_t", "rollout"):
        with Image.open(output / f"{name}.png") as image:
            assert image.width >= 1000 and image.height >= 600
            pixels = np.asarray(image.convert("RGB"))
            assert pixels.std() > 20
    summary = json.loads((output / "summary.json").read_text())
    assert summary["first"]["mpm"]["domain_updates"] == 201


def test_physical_comparison_allows_different_coordinates_without_loss_plots(tmp_path):
    changed = record()
    changed["stats_set"] = "xy_p99_global"
    for entry in changed["evals"]:
        del entry["fixed_t_objective"]
    records = load_runs(runs(tmp_path, changed), ["mpm"], physical_only=True)
    output = tmp_path / "plots"
    plot(records, ["mpm"], output, "Physical normalization ablation", physical_only=True)
    assert (output / "rollout.png").exists()
    assert not (output / "loss.png").exists()
    assert not (output / "fixed_t.png").exists()
    (output / "loss.png").write_bytes(b"stale")
    with pytest.raises(ValueError, match="stale loss plots"):
        plot(records, ["mpm"], output, "Physical", physical_only=True)


@pytest.mark.parametrize("physical_only", [True, False])
def test_same_file_digest_does_not_make_different_selected_rows_comparable(tmp_path, physical_only):
    changed = record()
    changed["batch_receipt"]["selected_rows"] = [1]
    with pytest.raises(ValueError, match="selected fixture rows"):
        load_runs(runs(tmp_path, changed), ["mpm"], physical_only=physical_only)


@pytest.mark.parametrize("field,value,match", [
    ("normalization", {"sha256": "changed"}, "normalization snapshot"),
    ("flow_matching", {"prediction_type": "sample"}, "parameterization"),
])
def test_matching_profile_label_does_not_allow_different_loss_coordinates(tmp_path, field, value, match):
    changed = record()
    changed["provenance"] = {"resolved_config": {field: value}}
    paths = runs(tmp_path, changed)
    with pytest.raises(ValueError, match=match):
        load_runs(paths, ["mpm"])
    load_runs(paths, ["mpm"], physical_only=True)


@pytest.mark.parametrize("train_options", [
    {"clip_deltas": False}, {"noise_context": False}, {"lambda_pose_native": 2.0},
    {"aux_losses": {"pos_cum": 1.0}},
])
def test_different_objective_contracts_require_physical_only(tmp_path, train_options):
    changed = record()
    changed["provenance"] = {"resolved_config": {"train": train_options}}
    paths = runs(tmp_path, changed)
    with pytest.raises(ValueError, match="objective settings differ"):
        load_runs(paths, ["mpm"])
    load_runs(paths, ["mpm"], physical_only=True)


def test_explicit_objective_defaults_and_optimizer_ablation_keep_same_loss_units(tmp_path):
    changed = record()
    changed["provenance"] = {"resolved_config": {"train": {
        "clip_deltas": True, "noise_context": True, "lambda_pose_native": 1.0,
        "aux_losses": {}, "lr": 0.002, "warmup_steps": 100,
    }}}
    load_runs(runs(tmp_path, changed), ["mpm"])
