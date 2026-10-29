"""Overnight observation tooling (scripts/dev): frozen fixtures are verified before reuse,
given-action protocols refuse checkpoints that hide actions, and reports describe only the
checkpoint their text was written for."""
import importlib
import json
import sys

import pytest
import torch
from omegaconf import OmegaConf


def test_cloth_fixtures_load_only_while_their_bytes_match_the_manifest(tmp_path):
    from scripts.dev.overnight_robustness import verified_fixture
    from utils.reproducibility import file_sha256
    path = tmp_path / "cloth_0.pt"
    torch.save({"positions": torch.zeros(2, 3)}, path)
    entry = {"key": "cloth_0_row0", "fixture": str(path), "fixture_sha256": file_sha256(path)}
    assert torch.equal(verified_fixture(entry)["positions"], torch.zeros(2, 3))
    torch.save({"positions": torch.full((2, 3), 0.001)}, path)      # the same path, moved by 1 mm
    with pytest.raises(ValueError, match="changed since"):
        verified_fixture(entry)


def _cfg(action_dropout):
    return OmegaConf.create({"train": {"action_dropout": action_dropout}, "model": {},
                             "data": {"context_len": 4, "future_len": 16}})


def test_given_action_protocols_refuse_checkpoints_that_hide_actions():
    from scripts.dev.observation_probe_common import require_given_actions
    require_given_actions(_cfg(0.0))
    require_given_actions(_cfg(0.5))            # partial dropout is still evaluated with actions given
    with pytest.raises(ValueError, match="every action hidden"):
        require_given_actions(_cfg(1.0))


@pytest.mark.parametrize("name", ["real_cloth_probe", "real_cloth_short_forecasts"])
def test_real_cloth_drivers_check_the_action_policy_before_any_work(tmp_path, monkeypatch, name):
    pytest.importorskip("scipy")                # the drivers' point-cloud geometry
    import utils.checkpoint
    driver = importlib.import_module(f"scripts.dev.{name}")
    for module in (utils.checkpoint, driver):
        monkeypatch.setattr(module, "read_checkpoint", lambda path: {}, raising=False)
        monkeypatch.setattr(module, "checkpoint_config", lambda state: _cfg(1.0), raising=False)
    monkeypatch.setattr(sys, "argv", [name, "--checkpoint", "unused.pt", "--source", str(tmp_path),
                                      "--out", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="every action hidden"):
        driver.main()


def _receipt(root, lane, sha256):
    (root / lane).mkdir(parents=True, exist_ok=True)
    (root / lane / "run.json").write_text(json.dumps({"checkpoint": f"/checkpoints/{lane}.pt",
                                                      "checkpoint_sha256": sha256}))


def test_reports_describe_only_the_checkpoint_their_text_was_written_for(tmp_path):
    from scripts.dev.observation_probe_common import REPORTED_CHECKPOINT_SHA256, reported_checkpoint
    with pytest.raises(ValueError, match="no run receipts"):
        reported_checkpoint(tmp_path)
    _receipt(tmp_path, "robustness", REPORTED_CHECKPOINT_SHA256)
    _receipt(tmp_path, "real_cloth", REPORTED_CHECKPOINT_SHA256)
    assert reported_checkpoint(tmp_path)["sha256"] == REPORTED_CHECKPOINT_SHA256
    _receipt(tmp_path, "stability", "0" * 64)
    with pytest.raises(ValueError, match="describes only"):
        reported_checkpoint(tmp_path)


@pytest.mark.parametrize("script", ["package_overnight", "overnight_report"])
def test_packaging_refuses_another_checkpoint_before_writing_labels(tmp_path, monkeypatch, script):
    module = importlib.import_module(f"scripts.dev.{script}")
    _receipt(tmp_path, "robustness", "0" * 64)
    monkeypatch.setattr(sys, "argv", [script, "--root", str(tmp_path)]
                        + (["--index-only"] if script == "overnight_report" else []))
    with pytest.raises(ValueError, match="describes only"):
        module.main()
    written = ("README.md", "FINDINGS.md", "morning_status.json", "index.html")
    assert not any((tmp_path / name).exists() for name in written)
