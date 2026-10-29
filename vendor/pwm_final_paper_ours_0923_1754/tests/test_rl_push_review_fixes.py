"""Frozen-release identity and audit thresholds must describe what is actually checked."""
import hashlib
import json

import numpy as np
import pytest
from omegaconf import OmegaConf

from scripts.dev.rl_push_manifests import exclusions, main
from utils.rl_push_release import verify_rl_push_release


@pytest.fixture
def audit(tmp_path):
    asset, n = "libero/basket/asset_0001", 7
    stage1, stage2, launches = [tmp_path / name for name in ["stage1.npz", "stage2_rows.json", "launches.json"]]
    provenance = [dict(source_run="run", source_shard="s.npz", source_episode_index=i, source_env=0) for i in range(n)]
    np.savez(stage1, object_file=np.array([f"objects/{asset}.npz"]*n),
             **{k: np.array([p[k] for p in provenance]) for k in provenance[0]},
             airborne_max_mm=np.full(n, 20.), bottom_min_mm=np.full(n, -5.))
    stage2.write_text(json.dumps([dict(file=f"objects/{asset}.npz", **p,
        max_penetration_mm=5., onset_before_contact_possible=False, push_onset_frame=12,
        unexplained_later_frames=[10], unexplained_later_max_gain_mps=.1, unexplained_settle_frames=[2])
        for p in provenance]))
    launches.write_text(json.dumps({"rows": [dict(asset_key=asset, **p, max_ratio=2.,
        frame=20, object_mps=.8, gripper_mps=.4) for p in provenance]}))
    summary, script = tmp_path / "stage2.json", tmp_path / "check_contacts.py"
    summary.write_text(json.dumps({"thresholds": {"accel_mps_per_frame": .1, "settle_frames": 10}}))
    script.write_text("ACCEL_MPS = 0.10\n")
    return [stage1, stage2, launches], dict(airborne_mm=20., sink_mm=5., penetration_mm=5.,
        gain_mps=.1, kick_from_frame=3, launch_ratio=2.), summary, script


def test_strict_audit_boundaries_and_frame_convention(audit):
    paths, rules, _, _ = audit
    assert exclusions(*paths, **rules) == ([], 7)
    s1 = dict(np.load(paths[0]))
    s1["airborne_max_mm"][1] += .0001
    s1["bottom_min_mm"][2] -= .0001
    np.savez(paths[0], **s1)
    rows = json.loads(paths[1].read_text())
    rows[3]["max_penetration_mm"] += .0001
    rows[4]["unexplained_later_max_gain_mps"] += .0001
    rows[5]["unexplained_settle_frames"] = [3]
    paths[1].write_text(json.dumps(rows))
    launches = json.loads(paths[2].read_text())
    launches["rows"][6]["max_ratio"] += .0001
    paths[2].write_text(json.dumps(launches))
    episodes, _ = exclusions(*paths, **rules)
    assert [e["source_episode_index"] for e in episodes] == list(range(1, 7))
    assert all(len(e["reasons"]) == 1 for e in episodes)


@pytest.mark.parametrize("gain", [.01, .05, .2, 1., float("nan")])
def test_row_only_receipts_refuse_both_directions_of_gain_retuning(audit, gain):
    paths, rules, summary, _ = audit
    rules["gain_mps"] = gain
    for metadata in [None, summary]:
        with pytest.raises(ValueError, match="must equal the stage-2 detector cutoff"):
            exclusions(*paths, **rules, stage2_summary=metadata)


def test_regenerated_detector_metadata_defines_its_own_frozen_cutoff(audit):
    paths, rules, summary, _ = audit
    summary.write_text(json.dumps({"thresholds": {"accel_mps_per_frame": .2}}))
    with pytest.raises(ValueError, match="detector cutoff 0.2"):
        exclusions(*paths, **rules, stage2_summary=summary)
    rules["gain_mps"] = .2
    assert exclusions(*paths, **rules, stage2_summary=summary) == ([], 7)


@pytest.mark.parametrize("source", [0, 1, 2])
@pytest.mark.parametrize("bad", ["duplicate", "nonfinite"])
def test_ambiguous_or_nonfinite_audit_rows_fail(audit, source, bad):
    paths, rules, _, _ = audit
    if source == 0:
        data = dict(np.load(paths[0]))
        if bad == "duplicate":
            data["source_episode_index"][1] = data["source_episode_index"][0]
        else:
            data["airborne_max_mm"][1] = np.nan
        np.savez(paths[0], **data)
    else:
        data = json.loads(paths[source].read_text())
        rows = data if source == 1 else data["rows"]
        if bad == "duplicate":
            rows.append(rows[0])
        else:
            rows[0]["max_penetration_mm" if source == 1 else "max_ratio"] = float("nan")
        paths[source].write_text(json.dumps(data))
    with pytest.raises(ValueError, match=bad):
        exclusions(*paths, **rules)


def test_new_manifest_hashes_threshold_metadata_and_detector_reference(audit, tmp_path):
    paths, _, summary, script = audit
    output = tmp_path / "new_exclusions.json"
    main(["exclusions", "--root", str(tmp_path), "--stage1", str(paths[0]), "--stage2", str(paths[1]),
          "--launches", str(paths[2]), "--stage2_summary", str(summary), "--stage2_script", str(script),
          "--out", str(output)])
    manifest = json.loads(output.read_text())
    for p in [*paths, summary, script]:
        assert manifest["receipts"][str(p.resolve())] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert manifest["contact_audit"]["thresholds"]["accel_mps_per_frame"] == .1
    assert "historical" in manifest["contact_audit"]["provenance_scope"]


@pytest.fixture
def frozen(tmp_path):
    root = tmp_path / "release"
    (root / "objects").mkdir(parents=True)
    data = root / "objects/a.npz"
    data.write_bytes(b"frozen contents")
    manifest = root / "RELEASE.json"
    manifest.write_text(json.dumps({"kind": "rl_push_frozen_release", "files": {
        "objects/a.npz": {"bytes": data.stat().st_size, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}}}))
    return root, manifest, data


@pytest.mark.parametrize("change", ["missing", "extra", "same_size", "size", "symlink", "digest"])
def test_release_preflight_detects_file_set_and_content_changes(frozen, change):
    root, manifest, data = frozen
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert verify_rl_push_release(root, digest) == digest
    if change == "missing":
        data.unlink()
    elif change == "extra":
        (root / "objects/extra.npz").write_bytes(b"new episode")
    elif change == "same_size":
        data.write_bytes(b"changed content")
    elif change == "size":
        data.write_bytes(b"x")
    elif change == "symlink":
        (root / "objects/link.npz").symlink_to(data)
    else:
        digest = "0"*64
    with pytest.raises(ValueError, match="release"):
        verify_rl_push_release(root, digest)


def _frozen_config(root):
    from utils.config import load_config
    return load_config("configs/unet.yaml", ["data.domains=[rl_push]",
        "normalization.artifact=stats/normalization_rl_push.json", f"data.roots.rl_push.train={root}"])


def test_checkpoint_binds_release_manifest_and_checks_actual_files(frozen):
    from utils.checkpoint import corpus_identity, validate_resume_configuration
    root, manifest, data = frozen
    cfg = _frozen_config(root)
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True), "corpus": corpus_identity(cfg)}
    for split in ["train", "val"]:
        assert str(manifest) in state["corpus"][split]["rl_push"]
    validate_resume_configuration(state, cfg)
    original = manifest.read_text()
    value = json.loads(original)
    value["note"] = "new manifest identity, same members"
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="files differ"):
        validate_resume_configuration(state, cfg)
    manifest.write_text(original)
    data.write_bytes(b"changed content")
    with pytest.raises(ValueError, match="contents differ"):
        corpus_identity(cfg)


@pytest.fixture
def split_release(tmp_path):
    """The pwm-data-v1 rl_push layout: each split's manifest lists its object files under `<split>/`;
    the dataset folder's manifest lists those, each split manifest and each split's INDEX.json."""
    folder = tmp_path / "rl_push"
    meta = lambda p: {"source": None, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    listed = {}
    for split in ("train", "test"):
        root = folder / split
        (root / "objects").mkdir(parents=True)
        (root / "objects/a.npz").write_bytes(f"frozen {split} contents".encode())
        (root / "INDEX.json").write_text(json.dumps(dict(schema="pwm-shard-index-v1", domain="rl_push", split=split)))
        (root / "RELEASE.json").write_text(json.dumps({"format": "pwm-data-v1-rl_push-split", "split": split,
            "files": {f"{split}/objects/a.npz": meta(root / "objects/a.npz")}}))
        listed.update({f"{split}/{name}": meta(root / name) for name in ("objects/a.npz", "INDEX.json", "RELEASE.json")})
    (folder / "RELEASE.json").write_text(json.dumps({"format": "pwm-data-v1-rl_push", "files": listed}))
    return folder


def test_release_preflight_reads_a_split_folder_of_the_data_release(split_release):
    for split in ("train", "test"):
        manifest = split_release / split / "RELEASE.json"
        assert verify_rl_push_release(split_release / split) == hashlib.sha256(manifest.read_bytes()).hexdigest()


@pytest.mark.parametrize("change", ["index", "object", "extra", "manifest", "folder_manifest", "moved", "disagree", "unlisted"])
def test_split_folder_preflight_detects_changes(split_release, change):
    root = split_release / "test"
    folder = split_release / "RELEASE.json"
    if change == "index":                                   # listed one level up only
        (root / "INDEX.json").write_text("{}")
    elif change == "object":
        (root / "objects/a.npz").write_bytes(b"changed test contents")
    elif change == "extra":
        (root / "objects/extra.npz").write_bytes(b"new episode")
    elif change == "manifest":                              # a rewritten split manifest no longer matches the folder's record
        value = json.loads((root / "RELEASE.json").read_text())
        value["note"] = "same members, new identity"
        (root / "RELEASE.json").write_text(json.dumps(value))
    elif change == "folder_manifest":
        folder.unlink()
    elif change == "moved":                                 # a train manifest cannot vouch for another folder
        (root / "RELEASE.json").write_text((split_release / "train" / "RELEASE.json").read_text())
    else:
        value = json.loads(folder.read_text())
        if change == "disagree":
            value["files"]["test/objects/a.npz"]["sha256"] = "0" * 64
        else:
            del value["files"]["test/objects/a.npz"]
        folder.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="[Rr]elease"):
        verify_rl_push_release(root)


def test_checkpoint_binds_both_split_manifests_of_the_data_release(split_release):
    from utils.checkpoint import corpus_identity, validate_resume_configuration
    from utils.config import load_config
    cfg = load_config("configs/unet.yaml", ["data.domains=[rl_push]", "normalization.artifact=stats/normalization_rl_push.json",
        f"data.roots.rl_push.train={split_release / 'train'}", f"data.roots.rl_push.val={split_release / 'test'}"])
    corpus = corpus_identity(cfg)
    for split, name in (("train", "train"), ("val", "test")):
        manifest = split_release / name / "RELEASE.json"
        assert corpus[split]["rl_push"][str(manifest)] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True), "corpus": corpus}
    validate_resume_configuration(state, cfg)
    (split_release / "test" / "objects/a.npz").write_bytes(b"changed test contents")
    with pytest.raises(ValueError, match="contents differ"):
        validate_resume_configuration(state, cfg)


def test_old_exclusion_only_checkpoint_has_explicit_release_migration_warning(frozen):
    from utils.checkpoint import corpus_identity, validate_resume_configuration
    root, manifest, _ = frozen
    cfg = _frozen_config(root)
    corpus = corpus_identity(cfg)
    for split in corpus.values():
        split["rl_push"].pop(str(manifest))
    state = {"cfg": OmegaConf.to_container(cfg, resolve=True), "corpus": corpus}
    with pytest.warns(UserWarning, match="historical release identity cannot be established"):
        validate_resume_configuration(state, cfg)
