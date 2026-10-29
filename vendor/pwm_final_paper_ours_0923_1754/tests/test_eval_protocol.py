"""scripts/unified/eval_protocol.py: the one command that scores a checkpoint under docs/EVAL_PROTOCOLS.md."""
import json
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.unified import eval_protocol as protocol
from utils.eval_cases import load_cases

RELEASE = "/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1"


def test_every_dataset_names_a_shipped_case_list_of_its_own():
    for dataset, name in protocol.CURRENT.items():
        manifest = load_cases(protocol.MANIFESTS / name)
        assert manifest["dataset"] == dataset and not manifest.get("partial")
        assert protocol.headline_set(manifest["sets"]) in manifest["sets"]
    for dataset, folders in protocol.REPORT_FOLDERS.items():
        for folder, name in folders.items():
            assert load_cases(protocol.MANIFESTS / name)["dataset"] == f"{dataset}_{folder}"
    # The long recordings are rolled out to their last frame, the others' first rollouts already end there.
    long = {d for d, name in protocol.CURRENT.items() if "rollouts_full" in load_cases(protocol.MANIFESTS / name)["sets"]}
    assert long == {"folding_8mm", "yikai", "dex_v2_filtered2"}


def _cfg(**folders):
    return OmegaConf.create({"data": {"domains": list(folders), "roots": {
        domain: {"train": f"{RELEASE}/{folder}/train", "val": f"{RELEASE}/{folder}/test"} for domain, folder in folders.items()}}})


def test_the_validation_folder_names_the_dataset_for_a_specialist_and_for_a_unified_model():
    assert protocol.checkpoint_datasets(_cfg(pickup="pickup")) == [("pickup", "pickup", Path(f"{RELEASE}/pickup/test"))]
    unified = {domain: dataset for domain, dataset, _ in protocol.checkpoint_datasets(
        _cfg(pickup="pickup", pushing="pushing", rl_push="rl_push", mpm_fresh="mpm_fresh", mpm_drag="mpm_drag",
             folding="folding_8mm", yikai="yikai", dex="dex_v2_filtered2"))}
    assert unified == {"pickup": "pickup", "pushing": "pushing", "rl_push": "rl_push", "mpm_fresh": "mpm_fresh",
                       "mpm_drag": "mpm_drag", "folding": "folding_8mm", "yikai": "yikai", "dex": "dex_v2_filtered2"}
    assert set(unified.values()) == set(protocol.CURRENT)
    with pytest.raises(ValueError, match="validation folder"):
        protocol.checkpoint_datasets(OmegaConf.create({"data": {"domains": ["pickup"]}}))


def test_the_plan_fixes_everything_that_changes_a_number(tmp_path):
    datasets = protocol.checkpoint_datasets(_cfg(pickup="pickup", folding="folding_8mm", mpm_drag="mpm_drag"))
    entries = protocol.plan("run/checkpoints/final.pt", datasets, tmp_path, "pyramid", python="python")
    by_name = {(e["dataset"], e["case_set"]): e for e in entries}
    assert {name for name in by_name if name[0] == "folding_8mm"} == {("folding_8mm", s) for s in (
        "windows", "windows_motion", "rollouts_first", "rollouts_active", "rollouts_full")}
    assert {name for name, e in by_name.items() if e["headline"]} == {
        ("pickup", "rollouts_first"), ("mpm_drag", "rollouts_first"), ("folding_8mm", "rollouts_full")}
    for (dataset, case_set), entry in by_name.items():
        words = entry["words"]
        option = lambda flag: words[words.index(flag) + 1]
        assert (option("--seed"), option("--clips"), option("--schedule"), option("--case_set")) == ("0", "all", "pyramid", case_set)
        assert Path(option("--cases")).name == protocol.CURRENT[dataset] and option("--out") == entry["receipt"]
        assert ("--rollout" in words) == case_set.startswith("rollouts")
        if "--rollout" in words:
            assert int(option("--rollout_frames")) == entry["reference"]["horizon_max"]
        assert ("--save_predictions" in words) == entry["headline"]          # default: the whole-clip rollouts
        assert not {"--context_len", "--future_len", "--clamp_actuated", "--case_tag", "--case_stratum", "--test_folder"} & set(words)
    whole = by_name[("folding_8mm", "rollouts_full")]
    assert whole["reference"]["horizon_min"] == whole["reference"]["horizon_max"] == 902 - 4      # 902-frame clips
    # folding rolls out 50 episodes (a whole-episode rollout is 57 windows); its window sets keep all 236 test clips
    assert {s: by_name[("folding_8mm", s)]["reference"]["cases"] for s in ("windows", "rollouts_first", "rollouts_active", "rollouts_full")} == {
        "windows": 236, "rollouts_first": 50, "rollouts_active": 50, "rollouts_full": 50}
    everything = protocol.plan("final.pt", datasets, tmp_path, "full_sequence", report_folders=True, predictions="all", python="python")
    seen = [e for e in everything if e["folder"] == "test_seen"]
    assert {e["case_set"] for e in seen} == {"windows", "rollouts_first"} and all(
        f"mpm_drag={RELEASE}/mpm_drag/test_seen" in e["words"] and "/mpm_drag.test_seen/" in e["receipt"] for e in seen)
    assert all("--save_predictions" in e["words"] and "full_sequence" in e["words"] for e in everything)
    assert len(protocol.plan("final.pt", datasets, tmp_path, "pyramid", only_datasets=["pickup"], only_sets=["windows"],
                             predictions="none")) == 1
    for bad in (dict(schedule="cosine"), dict(schedule="pyramid", predictions="some")):
        with pytest.raises(ValueError, match="schedule is one of"):
            protocol.plan("final.pt", datasets, tmp_path, bad.pop("schedule"), **bad)
    with pytest.raises(ValueError, match="No current case list"):
        protocol.plan("final.pt", protocol.checkpoint_datasets(_cfg(dex="dex_v2")), tmp_path, "pyramid")


def test_the_command_takes_no_option_that_changes_a_number(monkeypatch, capsys):
    for words in (["--seed", "1"], ["--clamp_actuated"], ["--context_len", "2"], ["--schedule", "cosine"]):
        monkeypatch.setattr(sys, "argv", ["eval_protocol.py", "--checkpoint", "unused.pt", *words])
        with pytest.raises(SystemExit):
            protocol.main()
    capsys.readouterr()


def _receipt(path, domain, case_set, zero_mm, frames):
    rows = lambda mean: {"n_episodes": 2, "particle_mean_mm": mean, "particle_final_mm": 2 * mean, "particle_mse_mm2": mean ** 2,
                         "chamfer_mean_mm": mean / 2, "particle_per_frame_mm": [mean] * frames, "particle_per_frame_n": [2] * frames}
    metrics = {"n_episodes": 2, "horizon_frames_min": frames, "horizon_frames_max": frames, "pose_passive_mm": 1.5, "pose_passive_deg": 0.5,
               "rows": {"predicted": rows(4.0), "moving": rows(9.0), "still": rows(1.0)},
               "reference": {"zero": {"pose_passive_mm": 6.0, "rows": {"predicted": rows(zero_mm)}},
                             "hold": {"rows": {"predicted": rows(12.0)}}}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"df_schedule": "pyramid", "seed": 0, "checkpoint_sha256": "c" * 64, domain: metrics,
                                "cases": {"cases": 2, "manifest_sha256": "m" * 64, "case_set": case_set, "predictions": None}}))
    path.with_suffix(".cases.json").write_text("{}")


def test_the_summary_reads_every_set_next_to_its_trivial_references(tmp_path):
    entries = protocol.plan("final.pt", protocol.checkpoint_datasets(_cfg(pickup="pickup")), tmp_path, "pyramid", python="python")
    for entry in entries:
        frames = entry["reference"]["horizon_max"]
        _receipt(Path(entry["receipt"]), "pickup", entry["case_set"], entry["reference"]["zero_predicted_mm"], frames)
        assert protocol._matches(entry["receipt"], entry, "c" * 64, "m" * 64, "pyramid")
        assert not protocol._matches(entry["receipt"], entry, "c" * 64, "m" * 64, "full_sequence")     # another schedule: score again
        assert not protocol._matches(entry["receipt"], entry, "d" * 64, "m" * 64, "pyramid")           # another checkpoint
    rows = protocol.summarize(entries)
    assert [(r["case_set"], r["headline"], r["reference_matches_manifest"]) for r in rows] == [
        ("windows", False, True), ("rollouts_first", True, True)]
    headline = rows[1]
    assert (headline["model_mean_mm"], headline["zero_mean_mm"], headline["hold_mean_mm"], headline["model_chamfer_mm"]) == (
        4.0, entries[1]["reference"]["zero_predicted_mm"], 12.0, 2.0)
    assert [c["frame"] for c in headline["curve"]] == [16, 32, 36] and headline["curve"][-1]["clips"] == 2   # 36-frame clips
    summary = dict(checkpoint="final.pt", checkpoint_sha256="c" * 64, reported_checkpoint=True, code="abc1234", seed=0,
                   schedule="pyramid", checkpoint_schedule="pyramid", partial=False, report_folders=False, sets=rows)
    text = protocol.render(summary)
    assert "Whole-clip rollouts" in text and "every set matches" in text and "frame 36: 4.00" in text
    _receipt(Path(entries[0]["receipt"]), "pickup", "windows", entries[0]["reference"]["zero_predicted_mm"] + 1.0, 16)
    rows = protocol.summarize(entries)                                  # the evaluator read other clips than the frozen ones
    assert not rows[0]["reference_matches_manifest"] and "MISMATCH in pickup windows" in protocol.render({**summary, "sets": rows})
