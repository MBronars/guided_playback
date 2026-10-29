"""Frozen motion bounds and self-contained normalization schema contracts.

Expected values below were read from the seven per-domain and seven global
NPZ assets at 55c386c. Tests intentionally do not load those retired files.
"""
import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from data.normalization import (
    NORMALIZATION_DOMAINS, NORMALIZATION_PROFILES, NormalizationStats,
    domain_normalizers, load_normalization_artifact, resolve_normalization,
)

FROZEN_PARTICLE = {
    "mpm": [0.02682609111070633, 0.0968426764011383],
    "pickup": [0.017173290252685547, 0.028820812702178955],
    "cloth": [0.0056610107421875, 0.0050855278968811035],
    "pushing": [0.003877609968185425, 0.00016488609253428876],
}
FROZEN_POSE = {
    "pickup": [0.015384865924715996, 0.028212731704115868, 0.13007056713104248, 0.18876968324184418],
    "cloth": [0.0090862475335598, 0.0053994618356227875, 0.048915959894657135, 0.006584832910448313],
    "pushing": [0.00388200837187469, 0.0001652689097682014, 0.0026072999462485313, 0.05427192896604538],
}
FROZEN_GLOBAL = [0.02682609111070633, 0.0968426764011383,
                 0.13007056713104248, 0.18876968324184418]


def _rehash(snapshot):
    snapshot = copy.deepcopy(snapshot)
    snapshot.pop("sha256", None)
    checksum = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":"),
                                        allow_nan=False).encode()).hexdigest()
    return {**snapshot, "sha256": checksum}


@pytest.mark.parametrize("profile", NORMALIZATION_PROFILES)
@pytest.mark.parametrize("domain", NORMALIZATION_DOMAINS)
def test_frozen_npz_motion_and_position_bounds_reconstruct_bitwise(profile, domain):
    particle, position, pose = domain_normalizers(profile, domain)
    expected = FROZEN_GLOBAL[:2] if profile.endswith("global") else FROZEN_PARTICLE[domain]
    maximum = torch.tensor([expected[0], expected[0], expected[1]], dtype=torch.float32)
    assert torch.equal(particle.max_val, maximum)
    assert torch.equal(particle.min_val, -maximum)
    assert torch.equal(position.max_val, torch.full((3,), 0.8, dtype=torch.float32))
    assert torch.equal(position.min_val, torch.full((3,), -0.8, dtype=torch.float32))
    if domain == "mpm":
        assert pose is None
    else:
        expected = FROZEN_GLOBAL if profile.endswith("global") else FROZEN_POSE[domain]
        maximum = torch.tensor([expected[0], expected[0], expected[1],
                                expected[2], expected[2], expected[3]], dtype=torch.float32)
        assert torch.equal(pose.max_val, maximum)
        assert torch.equal(pose.min_val, -maximum)


@pytest.mark.parametrize("profile", NORMALIZATION_PROFILES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("clip", [False, True])
def test_normalization_arithmetic_matches_original_npz_path(profile, dtype, clip):
    gen = torch.Generator().manual_seed(701)
    for domain in NORMALIZATION_DOMAINS:
        particle, _, pose = domain_normalizers(profile, domain)
        p = FROZEN_GLOBAL[:2] if profile.endswith("global") else FROZEN_PARTICLE[domain]
        maximum = np.array([p[0], p[0], p[1]], dtype=np.float32)
        refs = [(particle, maximum)]
        if pose is not None:
            p = FROZEN_GLOBAL if profile.endswith("global") else FROZEN_POSE[domain]
            refs.append((pose, np.array([p[0], p[0], p[1], p[2], p[2], p[3]], dtype=np.float32)))
        for current, maximum in refs:
            legacy = NormalizationStats(torch.from_numpy(-maximum).float(),
                                        torch.from_numpy(maximum).float())
            x = (torch.randn(3, 4, len(maximum), generator=gen) * legacy.max_val * 2).to(dtype)
            actual, expected = current.normalize(x, clip=clip), legacy.normalize(x, clip=clip)
            assert torch.equal(actual, expected)
            assert torch.equal(current.denormalize(actual), legacy.denormalize(expected))


@pytest.mark.parametrize("profile", NORMALIZATION_PROFILES)
def test_json_checkpoint_roundtrip_requires_no_asset_or_npz(profile, monkeypatch):
    snapshot = json.loads(json.dumps(resolve_normalization(profile)))

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot loading must be self-contained")

    monkeypatch.setattr("data.normalization.load_normalization_artifact", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    assert resolve_normalization(snapshot) == snapshot
    for domain in NORMALIZATION_DOMAINS:
        particle, position, _ = domain_normalizers(snapshot, domain)
        assert particle.max_val.shape == position.max_val.shape == (3,)


@pytest.mark.parametrize("profile", NORMALIZATION_PROFILES)
@pytest.mark.parametrize("domain", NORMALIZATION_DOMAINS)
@pytest.mark.parametrize("kind", ["alias", "artifact", "snapshot"])
def test_unified_adapters_use_one_consistent_particle_position_pose_contract(
        profile, domain, kind, monkeypatch, tmp_path):
    from data import unified_dataset as unified

    def no_npz(*args, **kwargs):
        pytest.fail("Unified normalizers must never load an external NPZ")

    monkeypatch.setattr(NormalizationStats, "load_all", no_npz)
    monkeypatch.setattr(unified.glob, "glob", lambda *a, **k: [str(tmp_path / "shard.tar")])
    monkeypatch.setattr(unified, "_load_blocklists", lambda *a: {})
    snapshot = resolve_normalization(profile)
    if kind == "alias":
        spec = "unified://xy_p99_global" if profile.endswith("global") else "unified://xy_p99"
    elif kind == "artifact":
        artifact = load_normalization_artifact()
        artifact["artifact_version"] = "explicit-adapter-candidate-v1"
        path = tmp_path / "candidate.json"
        path.write_text(json.dumps(artifact))
        spec = {"artifact": str(path), "profile": profile}
        snapshot = resolve_normalization(spec)
    else:
        spec = json.loads(json.dumps(snapshot))
        monkeypatch.setattr("data.normalization.load_normalization_artifact", no_npz)

    dataset = unified.build_domain_dataset(domain, normalization=spec, context_len=4, future_len=16)
    assert dataset.normalization == snapshot
    expected = domain_normalizers(snapshot, domain)
    actual = (dataset.delta_stats, dataset.position_stats,
              None if domain == "mpm" else dataset._pose_stats())
    for current, reference in zip(actual, expected):
        if reference is None:
            assert current is None
        else:
            assert torch.equal(current.min_val, reference.min_val)
            assert torch.equal(current.max_val, reference.max_val)


@pytest.mark.parametrize("adapter", ["UnifiedMPMDataset", "UnifiedPickupDataset",
                                    "UnifiedClothDataset", "UnifiedPushingDataset"])
@pytest.mark.parametrize("option", ["stats_location", "stats_override", "stats_set"])
def test_unified_adapters_reject_retired_stats_overrides_before_io(adapter, option, monkeypatch):
    from data import unified_dataset as unified
    monkeypatch.setattr(unified.glob, "glob", lambda *a, **k: pytest.fail("Unexpected source I/O"))
    roots = "unused" if adapter == "UnifiedMPMDataset" else ["unused"]
    with pytest.raises(TypeError, match=option):
        getattr(unified, adapter)(roots, normalization=resolve_normalization("xy_p99_global"),
                                  **{option: "unrelated.npz"})


@pytest.mark.parametrize("adapter", ["UnifiedMPMDataset", "UnifiedPickupDataset",
                                    "UnifiedClothDataset", "UnifiedPushingDataset"])
def test_unified_adapters_reject_positional_npz_override(adapter):
    from data import unified_dataset as unified
    roots = "unused" if adapter == "UnifiedMPMDataset" else ["unused"]
    with pytest.raises(TypeError, match="positional"):
        getattr(unified, adapter)(roots, "unrelated.npz")


@pytest.mark.parametrize("factory", ["build_domain_dataset", "UnifiedTrainStream",
                                    "UnifiedValRoundRobin", "create_unified_dataloader"])
@pytest.mark.parametrize("option", ["stats_location", "stats_override", "stats_set"])
def test_unified_factories_reject_retired_stats_options(factory, option):
    from data import unified_dataset as unified
    args = ("mpm",) if factory == "build_domain_dataset" else ()
    with pytest.raises(TypeError, match=option):
        getattr(unified, factory)(*args, **{option: "unrelated.npz"})


@pytest.mark.parametrize("old_value", ["unrelated.npz", "unified://xy_p99"])
def test_dispatch_loader_rejects_legacy_unified_stats_location(old_value):
    from data.dataset import create_dataloader
    with pytest.raises(ValueError, match="stats_location; use normalization"):
        create_dataloader("unified://v1", stats_location=old_value, format="unified",
                          normalization=resolve_normalization("xy_p99_global"))


def test_global4_does_not_depend_on_domain_order_or_singleton_selection():
    snapshot = resolve_normalization("xy_p99_global")
    previous = None
    for domain in ("pushing", "mpm", "cloth", "pickup", "pushing"):
        particle, _, pose = domain_normalizers(snapshot, domain)
        if previous is not None:
            assert torch.equal(particle.max_val, previous)
        if pose is not None:
            assert torch.equal(particle.max_val, pose.max_val[:3])
        previous = particle.max_val
    assert snapshot == resolve_normalization("xy_p99_global")


def test_profile_aliases_are_exact_and_default_is_unambiguous():
    assert resolve_normalization() == resolve_normalization("xy_p99_per_domain")
    assert resolve_normalization("unified://xy_p99") == resolve_normalization()
    assert resolve_normalization("unified://xy_p99_global") == resolve_normalization("xy_p99_global")
    with pytest.raises(ValueError, match="NOT xy_p99_per_domain"):
        resolve_normalization("per-domain")


def test_selector_and_snapshots_do_not_share_mutable_values():
    first = resolve_normalization({"profile": "xy_p99_per_domain"})
    second = resolve_normalization(first)
    second["scales"]["particle"]["mpm"][0] = 99
    assert first == resolve_normalization()
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_normalization(second)


@pytest.mark.parametrize("field,value", [
    ("profile", "xy_p99_global"), ("artifact_version", "changed"),
    ("position_half_range_m", 0.9), ("units", {"position": "mm"}),
    ("sha256", "0" * 64), ("provenance", {"changed": True}),
])
def test_tampered_checkpoint_snapshots_fail(field, value):
    snapshot = resolve_normalization()
    snapshot[field] = value
    with pytest.raises(ValueError, match="checksum mismatch"):
        resolve_normalization(snapshot)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "0.1", 1e-100, 1e100])
def test_invalid_scale_values_fail(value, tmp_path):
    artifact = load_normalization_artifact()
    artifact["profiles"]["xy_p99_per_domain"]["particle"]["mpm"][0] = value
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="finite|positive|JSON"):
        resolve_normalization({"artifact": str(path)})


@pytest.mark.parametrize("mutate", [
    lambda a: a.update(unknown=True),
    lambda a: a.update(schema_version=True),
    lambda a: a.update(schema_version=2),
    lambda a: a.update(artifact_version=""),
    lambda a: a.update(units={"position": "mm"}),
    lambda a: a.update(position_half_range_m=1.0),
    lambda a: a.update(provenance={}),
    lambda a: a["profiles"].update(other={}),
    lambda a: a["profiles"]["xy_p99_per_domain"]["particle"].pop("mpm"),
    lambda a: a["profiles"]["xy_p99_per_domain"]["pose"].update(mpm=[1, 1, 1, 1]),
    lambda a: a["profiles"]["xy_p99_per_domain"]["particle"].update(mpm=[1, 2, 3]),
    lambda a: a["profiles"]["xy_p99_global"].update(translation=[0.1, 0.1]),
])
def test_malformed_artifacts_fail_before_dataset_access(mutate, tmp_path):
    artifact = load_normalization_artifact()
    mutate(artifact)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        resolve_normalization({"artifact": str(path)})


def test_duplicate_json_keys_fail(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 1, "schema_version": 2}')
    with pytest.raises(ValueError, match="Duplicate"):
        resolve_normalization({"artifact": str(path)})


@pytest.mark.parametrize("spec", [{"profiles": {}}, {"profile": "bogus"},
                                  {"artifact": ""}, {"artifact": None}, "", False])
def test_invalid_selectors_fail(spec):
    with pytest.raises(ValueError):
        resolve_normalization(spec)


def test_snapshot_cannot_relabel_profile_even_with_updated_checksum():
    snapshot = resolve_normalization()
    snapshot["profile"] = "xy_p99"
    with pytest.raises(ValueError, match="canonical profile"):
        resolve_normalization(_rehash(snapshot))
    snapshot["profile"] = "xy_p99_global"
    with pytest.raises(ValueError, match="global scales"):
        resolve_normalization(_rehash(snapshot))


def test_candidate_version_is_explicit_and_does_not_change_frozen_profile(tmp_path):
    frozen = resolve_normalization()
    artifact = load_normalization_artifact()
    artifact["artifact_version"] = "candidate-audited-v1"
    artifact["provenance"] = {"kind": "new_candidate", "source_sha256": "abc"}
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(artifact))
    candidate = resolve_normalization({"artifact": str(path)})
    assert candidate["scales"] == frozen["scales"]
    assert candidate["sha256"] != frozen["sha256"]
    assert resolve_normalization() == frozen


def test_unknown_domain_fails_clearly():
    with pytest.raises(ValueError, match="Unknown normalization domain"):
        domain_normalizers(None, "drag")


@pytest.mark.parametrize("backbone", ["transformer", "unet"])
@pytest.mark.parametrize("profile", NORMALIZATION_PROFILES)
@pytest.mark.parametrize("domains", ["[mpm]", "[mpm,pickup,cloth,pushing]"])
def test_fresh_config_profile_is_independent_of_backbone_and_domain_subset(backbone, profile, domains):
    from utils.config import load_config
    cfg = load_config(f"configs/{backbone}.yaml",
                      [f"normalization.profile={profile}", f"data.domains={domains}"])
    assert resolve_normalization(cfg.normalization) == resolve_normalization(profile)


def test_fresh_config_cannot_accept_retired_or_unknown_profile_fields():
    from utils.config import load_config
    with pytest.raises(ValueError, match="Retired unified stats"):
        load_config("configs/transformer.yaml", ["normalization.profile=per-domain"])
    with pytest.raises(ValueError, match="only profile and optional artifact"):
        load_config("configs/transformer.yaml", ["normalization.scale=1.0"])
