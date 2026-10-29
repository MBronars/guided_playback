"""Canonical raw calibration is deterministic, finite, masked and read-only."""
import hashlib
import io
import json
import pickle
import tarfile

import numpy as np
import pytest
import torch

from data.calibration import calibration_source_manifest, iter_calibration_batches
from data.normalization import NormalizationStats
from data.unified_dataset import UnifiedMPMDataset, UnifiedPushingDataset


def write_shard(path, records):
    with tarfile.open(path, "w") as archive:
        for key, record in records:
            for field, value in record.items():
                payload = pickle.dumps(value)
                member = tarfile.TarInfo(f"{key}.{field}.pyd")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))


def mpm_record(offset=0, batch_size=2):
    positions = np.arange(batch_size * 12 * 4 * 3, dtype=np.float32).reshape(batch_size, 12, 4, 3) / 1000 + offset
    valid = np.ones((batch_size, 12, 4), dtype=bool)
    valid[..., -1] = False
    return dict(particle_pos=positions, valid_particles=valid,
                material_properties=np.ones((batch_size, 4, 4), dtype=np.float32),
                object_ids=np.zeros((batch_size, 4), dtype=np.int64))


def make_mpm(tmp_path, monkeypatch, *, blocklist=None):
    def no_stats(*args, **kwargs):
        pytest.fail("Raw calibration must not load normalization assets")
    monkeypatch.setattr(NormalizationStats, "load_all", no_stats)
    from data import normalization
    monkeypatch.setattr(normalization, "resolve_normalization", no_stats)
    for i in range(3):
        write_shard(tmp_path / f"sample{i}.tar", [(f"ep{i}", mpm_record(i / 100))])
    return UnifiedMPMDataset(str(tmp_path), raw=True, context_len=3, future_len=4,
                             blocklist_paths=[str(blocklist)] if blocklist else None)


def collect(ds, seed=31, policy="training", max_records=10):
    return list(iter_calibration_batches("mpm", context_len=3, future_len=4, seed=seed,
        max_records=max_records, window_policy=policy, emit_sidecar=False, dataset=ds))


def test_finite_calibration_exact_replay_and_rng_isolation(tmp_path, monkeypatch):
    ds = make_mpm(tmp_path, monkeypatch)
    initial_shards = list(ds.shards)
    before = torch.random.get_rng_state().clone()
    first = collect(ds)
    assert len(first) == 3 and ds.shards == initial_shards
    assert torch.equal(before, torch.random.get_rng_state())
    torch.manual_seed(192)
    second = collect(ds)
    for (a, ma), (b, mb) in zip(first, second):
        torch.testing.assert_close(a.positions, b.positions, atol=0, rtol=0)
        assert ma["source"] == mb["source"] and ma["key"] == mb["key"]
        assert ma["window_starts"] == mb["window_starts"]
        assert ma["source_rows"] == [0, 1]
        assert a.domain == "mpm" and a.positions.shape == (2, 7, 4, 3)
        assert not ma["valid_particle_mask"][..., -1].any()
        assert ma["valid_particle_mask"][..., :3].all()
    assert len(collect(ds, max_records=1)) == 1
    assert any(a[1]["window_starts"] != b[1]["window_starts"] or a[1]["key"] != b[1]["key"]
               for a, b in zip(first, collect(ds, seed=50)))


def test_mpm_eval_seed_is_local_to_cpu_window_and_preserves_exact_draw(tmp_path, monkeypatch):
    import zlib
    from types import SimpleNamespace
    from data.dataset import MPMMultiObjectDataset

    ds = make_mpm(tmp_path, monkeypatch)
    ds.training = False
    seed = zlib.crc32(b"episode-17") & 0x7FFFFFFF
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        expected = torch.randint(10000, (1, 3)).float()
    before = torch.random.get_rng_state().clone()
    monkeypatch.setattr(torch.cuda, "manual_seed_all",
                        lambda *a: pytest.fail("CPU window selection must not reseed CUDA sampling"))
    monkeypatch.setattr(MPMMultiObjectDataset, "_process_sample",
                        lambda self, sample: SimpleNamespace(positions=torch.randint(10000, (1, 3)).float()))
    actual = ds._process_sample({"__key__": "episode-17"})
    torch.testing.assert_close(actual.positions, expected + ds._shift_t, rtol=0, atol=0)
    assert torch.equal(before, torch.random.get_rng_state())


def test_prefix_windows_and_canonical_shift_are_exact(tmp_path, monkeypatch):
    ds = make_mpm(tmp_path, monkeypatch)
    for batch, meta in collect(ds, policy="prefix"):
        assert meta["window_starts"] == [0, 0]
        index = int(meta["key"][-1])
        expected = torch.from_numpy(mpm_record(index / 100)["particle_pos"][:, :7])
        expected += torch.tensor([-0.75, -0.75, -0.10])
        torch.testing.assert_close(batch.positions, expected, atol=0, rtol=0)


def test_exclusions_preserve_original_episode_rows(tmp_path, monkeypatch):
    blocklist = tmp_path / "exclude.json"
    blocklist.write_text(json.dumps([{"shard": "sample0.tar", "key": "ep0", "elem": 0}]))
    ds = make_mpm(tmp_path, monkeypatch, blocklist=blocklist)
    rows = {meta["key"]: (batch.batch_size, meta["source_rows"]) for batch, meta in collect(ds)}
    assert rows["ep0"] == (1, [1])
    assert rows["ep1"] == (2, [0, 1])


def test_all_excluded_calibration_exhausts_without_restarting(tmp_path, monkeypatch):
    blocklist = tmp_path / "exclude.json"
    blocklist.write_text(json.dumps([{"shard": f"sample{i}.tar", "key": f"ep{i}"} for i in range(3)]))
    ds = make_mpm(tmp_path, monkeypatch, blocklist=blocklist)
    assert collect(ds) == []


def test_cache_identity_covers_exclusions_sources_windows_seed(tmp_path, monkeypatch):
    blocklist = tmp_path / "exclude.json"
    blocklist.write_text("[]")
    ds = make_mpm(tmp_path, monkeypatch, blocklist=blocklist)
    def manifest(**kwargs):
        options = dict(seed=31, window_policy="training", max_records=10)
        options.update(kwargs)
        return calibration_source_manifest(ds, **options)
    original = manifest()
    assert manifest() == original
    assert manifest(seed=9)["sha256"] != original["sha256"]
    assert manifest(window_policy="prefix")["sha256"] != original["sha256"]
    blocklist.write_text("[ ]")
    assert manifest()["sha256"] != original["sha256"]
    ds.context_len = 2
    changed_window = manifest()["sha256"]
    ds.context_len = 3
    assert manifest()["sha256"] != changed_window
    previous = manifest()["sha256"]
    write_shard(tmp_path / "sample0.tar", [("ep0", mpm_record(1)), ("extra", mpm_record(2))])
    assert manifest()["sha256"] != previous
    assert {asset["path"] for asset in original["assets"]} == {str(blocklist)}


def test_corrupt_sources_fail_instead_of_silently_changing_calibration(tmp_path, monkeypatch):
    ds = make_mpm(tmp_path, monkeypatch)
    (tmp_path / "sample0.tar").write_bytes(b"invalid archive")
    with pytest.raises(tarfile.ReadError):
        collect(ds)


def test_raw_pushing_emits_physical_pose_without_any_stats(tmp_path, monkeypatch):
    from data import normalization
    def no_stats(*args, **kwargs):
        pytest.fail("Raw rigid sidecars must not depend on stats")
    monkeypatch.setattr(normalization, "resolve_normalization", no_stats)
    monkeypatch.setattr(NormalizationStats, "load_all", no_stats)
    rng = np.random.default_rng(9)
    base = rng.normal(size=(8, 3)).astype(np.float32) / 100
    positions = base[None] + np.arange(12, dtype=np.float32)[:, None, None] * .001
    write_shard(tmp_path / "pushing.tar", [("ep", dict(particle_pos=positions,
        obj_ids=np.array([0] * 4 + [1] * 4), material_properties=np.ones((8, 4), dtype=np.float32)))])
    ds = UnifiedPushingDataset([str(tmp_path)], raw=True, context_len=3, future_len=4, split="all")
    before = set(tmp_path.iterdir())
    batch, metadata = next(iter_calibration_batches("pushing", context_len=3, future_len=4,
        seed=0, max_records=1, emit_sidecar=True, dataset=ds))
    assert set(tmp_path.iterdir()) == before
    assert batch.pose_deltas_gt.shape == (1, 4, 2, 6)
    torch.testing.assert_close(batch.pose_deltas_gt[..., :3], torch.full((1, 4, 2, 3), .001),
                               atol=1e-7, rtol=0)
    assert torch.equal(batch.pose_action_mask, torch.tensor([[False, True]]))
    assert metadata["valid_particle_mask"].all()


@pytest.mark.parametrize("options", [{"seed": -1}, {"max_records": 0}, {"seed": 1.2},
    {"window_policy": "typo"}, {"context_len": 0}, {"future_len": 0}])
def test_calibration_rejects_bad_selectors_before_io(monkeypatch, options):
    from data import calibration
    monkeypatch.setattr(calibration, "build_domain_dataset", lambda *a, **k: pytest.fail("unexpected I/O"))
    request = dict(context_len=3, future_len=4, seed=0, max_records=1)
    request.update(options)
    with pytest.raises(ValueError):
        list(iter_calibration_batches("mpm", **request))


def test_exact_cloth_holdout_preserved_independent_of_cwd(monkeypatch, tmp_path):
    from scripts.unified.materialize_cloth_wds import HOLDOUT_MANIFEST, load_holdout_files
    original = json.loads(HOLDOUT_MANIFEST.read_text())
    assert len(original["garment_ids"]) == 38 and len(original["val_files"]) == 144
    assert hashlib.sha256("\n".join(original["val_files"]).encode()).hexdigest() == \
        "93c7ac8b1d08743632e94ce0eba3b81b34d73731410ca018f25f96db8862a4c1"
    monkeypatch.chdir(tmp_path)
    assert load_holdout_files() == set(original["val_files"])


def test_materializer_never_replaces_existing_shards(tmp_path):
    from scripts.unified.materialize_cloth_wds import ShardWriter
    writer = ShardWriter(str(tmp_path))
    writer.add("episode", {"particle_pos": np.zeros((2, 4, 3))})
    writer.close()
    path = tmp_path / "shard-00000.tar"
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing"):
        ShardWriter(str(tmp_path))
    assert path.read_bytes() == before
