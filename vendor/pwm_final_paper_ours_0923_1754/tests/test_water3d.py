"""Water-3D contract: the pure-Python TFRecord reader (hand-encoded protobuf round trip), the frame
mapping, the materializer, the opt-in water domain (normalization entry optional and outside the
global aggregation, material 6, config checks, not a default domain), and UnifiedWaterDataset's
training/validation/rollout windows. Real files are exercised only when present on this machine."""
import copy
import json
import os
import struct

import numpy as np
import pytest
import torch

from data.normalization import (DEFAULT_ARTIFACT, domain_normalizers, load_normalization_artifact,
                                resolve_normalization)
from data.unified_dataset import (DOMAIN_ORDER, MAT_WATER, WATER_ROOT, build_domain_dataset, resolve_data_roots,
                                  selected_domains)
from data.water3d import (WATER3D_FILL_E, WATER3D_FILL_NU, WATER3D_FILL_RHO, UnifiedWaterDataset, from_model_frame,
                          iter_water3d, particle_subset, select_windows, to_model_frame, water_batch)
from utils.config import load_config

ROOT = "/data/group_data/katefgroup/datasets/deepmind_lts/Water-3D"
FROZEN_SNAPSHOT_SHA = "2bc31504482f745a5bfce97558e7a366e1b6800d2bb149a8853e663d797587de"
META = {"sequence_length": 40, "dim": 3, "bounds": [[0.125, 0.875]] * 3, "dt": 0.005}


# ----------------------------------------------------------------------------- protobuf encoder (test only)
def _varint(v: int) -> bytes:
    out = b""
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out += bytes([b | 0x80])
        else:
            return out + bytes([b])


def _ld(field: int, payload: bytes) -> bytes:                # length-delimited field (wire type 2)
    return _varint((field << 3) | 2) + _varint(len(payload)) + payload


def _bytes_feature(items) -> bytes:                          # Feature{bytes_list=1{value=1...}}
    return _ld(1, b"".join(_ld(1, b) for b in items))


def _int64_feature(items) -> bytes:                          # Feature{int64_list=3{value=1 packed}}
    return _ld(3, _ld(1, b"".join(_varint(int(v)) for v in items)))


def _features(mapping: dict) -> bytes:                       # Features{feature=1 map<string, Feature>}
    return b"".join(_ld(1, _ld(1, k.encode()) + _ld(2, v)) for k, v in mapping.items())


def _feature_lists(mapping: dict) -> bytes:                  # FeatureLists{feature_list=1 map<string, FeatureList>}
    return b"".join(_ld(1, _ld(1, k.encode()) + _ld(2, b"".join(_ld(1, f) for f in feats)))
                    for k, feats in mapping.items())


def _sequence_example(key: int, positions: np.ndarray, ptype: np.ndarray) -> bytes:
    context = _features({"key": _int64_feature([key]),
                         "particle_type": _bytes_feature([ptype.astype("<i8").tobytes()])})
    lists = _feature_lists({"position": [_bytes_feature([p.astype("<f4").tobytes()]) for p in positions]})
    return _ld(1, context) + _ld(2, lists)


def _write_tfrecord(path: str, records) -> None:
    with open(path, "wb") as f:
        for r in records:
            f.write(struct.pack("<Q", len(r)) + b"\0\0\0\0" + r + b"\0\0\0\0")   # CRCs are not checked


def _synthetic(T=41, N=50, seed=0):
    g = np.random.default_rng(seed)
    pos = np.empty((T, N, 3), np.float32)
    pos[0] = 0.3 + 0.4 * g.random((N, 3))
    vel = np.zeros((N, 3), np.float32)
    for t in range(1, T):
        if 10 <= t < 25:                                   # a burst of motion in the middle, falling along -y
            vel[:, 1] -= 0.002
            vel[:, 0] += 0.0005
        else:
            vel[:] = 0.0
        pos[t] = np.clip(pos[t - 1] + vel, 0.125, 0.875)
    return pos, np.full(N, 5, np.int64)


def _water_artifact(tmp_path, scale=(0.07, 0.086)):
    art = json.load(open(DEFAULT_ARTIFACT))
    art["profiles"]["xy_p99_per_domain"]["particle"]["water"] = list(scale)
    art["artifact_version"] = "test-water"
    path = str(tmp_path / "normalization_water_test.json")
    json.dump(art, open(path, "w"))
    return path


def _materialized_split(root, trajectories):
    os.makedirs(root)
    rows = []
    for key, pos in trajectories:
        np.save(os.path.join(root, f"{key:05d}.npy"), pos.astype(np.float32))
        rows.append({"key": key, "file": f"{key:05d}.npy", "frames": int(pos.shape[0]), "particles": int(pos.shape[1]),
                     "outside_bounds": 0.0})
    json.dump({"split": os.path.basename(root), "trajectories": rows}, open(os.path.join(root, "manifest.json"), "w"))
    return root


def test_reader_round_trip_frame_mapping_and_windows(tmp_path):
    pos, ptype = _synthetic()
    pos2 = pos[:, :30] * 0.5 + 0.25
    path = str(tmp_path / "valid.tfrecord")
    _write_tfrecord(path, [_sequence_example(7, pos, ptype), _sequence_example(8, pos2, ptype[:30])])
    trs = list(iter_water3d(path, META))
    assert [t.key for t in trs] == [7, 8] and trs[0].num_frames == 41 and trs[1].num_particles == 30
    assert np.array_equal(trs[0].positions, pos) and np.array_equal(trs[0].particle_type, ptype)
    assert trs[0].step_context is None and trs[0].outside_bounds == 0.0
    assert list(iter_water3d(path, META, limit=1, skip=1))[0].key == 8
    # y up -> z up is a proper rotation about x; the water shift floors the box at z = 0
    p = np.array([[0.125, 0.875, 0.5]], np.float32)
    assert np.allclose(to_model_frame(p), [[-0.375, 0.0, 0.75]], atol=1e-6)
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    assert np.isclose(np.linalg.det(R), 1.0) and np.allclose(to_model_frame(p, shift=(0, 0, 0)), p @ R.T)
    assert np.allclose(from_model_frame(to_model_frame(pos)), pos, atol=1e-6)
    # motion-ranked windows (stride 2, C2/F4 -> span 11): the burst (frames 10..24) must win
    idx = particle_subset(50, 20, seed=1)
    assert idx.shape == (20,) and np.all(np.diff(idx) > 0) and particle_subset(10, 20).tolist() == list(range(10))
    starts = select_windows(pos, 2, 2, 4, stride=2, idx=idx)
    assert len(starts) == 2 and any(8 <= s <= 16 for s in starts) and starts[1] - starts[0] >= 11


def test_materializer_writes_npy_and_never_overwrites(tmp_path):
    from scripts.unified.materialize_water3d import main as materialize
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    pos, ptype = _synthetic()
    json.dump(META, open(src / "metadata.json", "w"))
    _write_tfrecord(str(src / "train.tfrecord"), [_sequence_example(3, pos, ptype)])
    _write_tfrecord(str(src / "valid.tfrecord"), [_sequence_example(5, pos[:, :20], ptype[:20])])
    materialize(["--src", str(src), "--out", str(out), "--splits", "train", "valid"])
    m = json.load(open(out / "valid" / "manifest.json"))
    assert m["trajectories"] == [{"key": 5, "file": "00005.npy", "frames": 41, "particles": 20, "outside_bounds": 0.0}]
    assert np.array_equal(np.load(out / "train" / "00003.npy"), pos) and json.load(open(out / "metadata.json")) == META
    with pytest.raises(FileExistsError):
        materialize(["--src", str(src), "--out", str(out), "--splits", "valid"])
    _write_tfrecord(str(src / "test.tfrecord"), [_sequence_example(9, pos, np.full(50, 3, np.int64))])
    with pytest.raises(ValueError, match="water only"):
        materialize(["--src", str(src), "--out", str(out), "--splits", "test"])


def test_water_entry_is_optional_and_outside_the_global_profile(tmp_path):
    frozen = resolve_normalization("xy_p99_per_domain")
    assert frozen["sha256"] == FROZEN_SNAPSHOT_SHA and "water" not in frozen["scales"]["particle"]
    assert resolve_normalization(copy.deepcopy(frozen))["sha256"] == FROZEN_SNAPSHOT_SHA   # 4-domain snapshots still load
    with pytest.raises(ValueError, match="no 'water' entry"):
        domain_normalizers("xy_p99_per_domain", "water")
    path = _water_artifact(tmp_path)
    base, water = load_normalization_artifact(DEFAULT_ARTIFACT), load_normalization_artifact(path)
    assert water["profiles"]["xy_p99_global"] == base["profiles"]["xy_p99_global"]
    assert {d: water["profiles"]["xy_p99_per_domain"]["particle"][d] for d in ("mpm", "pickup", "cloth", "pushing")} == \
        base["profiles"]["xy_p99_per_domain"]["particle"]
    snap = resolve_normalization({"profile": "xy_p99_per_domain", "artifact": path})
    assert resolve_normalization(copy.deepcopy(snap))["sha256"] == snap["sha256"] != FROZEN_SNAPSHOT_SHA
    delta, position, pose = domain_normalizers(snap, "water")
    assert pose is None and torch.allclose(delta.max_val, torch.tensor([0.07, 0.07, 0.086]))
    assert torch.equal(domain_normalizers(snap, "mpm")[0].max_val, domain_normalizers(frozen, "mpm")[0].max_val)
    glob = domain_normalizers({"profile": "xy_p99_global", "artifact": path}, "water")[0]
    assert torch.equal(glob.max_val, domain_normalizers("xy_p99_global", "mpm")[0].max_val)   # shared reference scales


def test_water_batch_contract_and_padding():
    delta, position, _ = domain_normalizers(resolve_normalization("xy_p99_per_domain"), "mpm")
    w = [np.zeros((6, 5, 3), np.float32), np.ones((6, 3, 3), np.float32)]
    b = water_batch(w, context_len=2, delta_stats=delta, position_stats=position, particles=5)
    assert tuple(b.positions.shape) == (2, 6, 5, 3) and b.domain == "water" and b.prediction_target == "delta"
    assert b.valid_lengths.tolist() == [5, 3] and bool((b.positions[1, :, 3:] == 0).all())
    assert b.get_valid_mask()[1].tolist() == [True, True, True, False, False]
    assert bool((b.mat_type == MAT_WATER).all()) and MAT_WATER == 6
    assert bool((b.actions == 0).all()) and not bool(b.action_mask.any()) and bool((b.object_ids == 0).all())
    assert np.allclose([float(b.E[0, 0]), float(b.nu[0, 0]), float(b.rho[0, 0])], [WATER3D_FILL_E, WATER3D_FILL_NU, WATER3D_FILL_RHO])
    assert b.body_type is None and b.body_poses is None and b.pose_delta_stats is None


def test_unified_water_dataset_train_eval_and_rollout_windows(tmp_path):
    pos, _ = _synthetic(T=41, N=50)
    small = pos[:, :30] * 0.5 + 0.25
    root = _materialized_split(str(tmp_path / "valid"), [(4, pos), (6, small)])
    spec = {"profile": "xy_p99_per_domain", "artifact": _water_artifact(tmp_path)}
    ds = UnifiedWaterDataset(root, context_len=2, future_len=4, training=True, normalization=spec, stride=2, particles=40)
    ds.batch_size = 3
    torch.manual_seed(0)
    b = next(iter(ds))
    assert tuple(b.positions.shape) == (3, 6, 40, 3) and b.domain == "water" and ds._pose_stats() is None
    assert set(b.valid_lengths.tolist()) <= {40, 30} and torch.allclose(b.delta_stats.max_val, torch.tensor([0.07, 0.07, 0.086]))
    assert float(b.positions[..., 2].min()) >= -1e-6 and float(b.positions.abs().max()) <= 0.8
    torch.manual_seed(0)
    assert torch.equal(next(iter(ds)).positions, b.positions)       # seeded by torch.initial_seed()

    ev = UnifiedWaterDataset(root, context_len=2, future_len=4, training=False, normalization=spec, stride=2, particles=40)
    first, again = list(ev), list(ev)
    assert len(first) == 1 and tuple(first[0].positions.shape) == (2, 6, 40, 3)
    assert torch.equal(first[0].positions, again[0].positions)       # deterministic windows and subsets
    ev.first_window = True
    idx = particle_subset(50, 40)
    b0 = next(iter(ev))
    assert torch.allclose(b0.positions[0], torch.from_numpy(to_model_frame(pos[0:11:2][:, idx])))
    ev.first_window, ev.future_len = False, 10_000                  # the evaluator's rollout convention
    assert next(iter(ev)).positions.shape[1] == (41 - 1) // 2 + 1   # the rest of the trajectory from frame 0
    ev.future_len = -1
    assert next(iter(ev)).positions.shape[1] == 21


def test_water_is_opt_in_and_config_checks(tmp_path):
    assert selected_domains(None)[0] == DOMAIN_ORDER == ("mpm", "pickup", "cloth", "pushing")
    assert selected_domains(["water", "mpm"])[0] == ("mpm", "water")
    base = ["run_name=t", "data.domains=[water]"]
    with pytest.raises(ValueError, match="num_mat_types=7"):
        load_config("configs/transformer.yaml", base)
    with pytest.raises(ValueError, match="water entry"):
        load_config("configs/transformer.yaml", base + ["model.num_mat_types=7"])
    art = _water_artifact(tmp_path)
    cfg = load_config("configs/transformer.yaml", base + ["model.num_mat_types=7", f"normalization.artifact={art}"])
    assert "water" in cfg.normalization["scales"]["particle"]
    with pytest.raises(ValueError, match=r"missing selected-domain materials: \[6\]"):
        load_config("configs/unet.yaml", base + ["model.num_mat_types=7", f"normalization.artifact={art}"])
    load_config("configs/unet.yaml", base + ["model.num_mat_types=7", f"normalization.artifact={art}",
                                             'model.unet_l1_by_mat="0:256,1:256,2:256,3:128,4:128,5:512,6:1024"'])
    load_config("configs/transformer.yaml", ["run_name=t"])           # four-domain recipes are unaffected


def test_water_roots_override(tmp_path):
    pos, _ = _synthetic()
    train = _materialized_split(str(tmp_path / "train"), [(1, pos)])
    val = _materialized_split(str(tmp_path / "valid"), [(2, pos)])
    with pytest.raises(ValueError, match="requires both"):
        resolve_data_roots({"water": {"train": train}})
    roots = resolve_data_roots({"water": {"train": train, "val": val}})
    ds = build_domain_dataset("water", split="val", training=False, context_len=2, future_len=4,
                              normalization={"profile": "xy_p99_per_domain", "artifact": _water_artifact(tmp_path)},
                              roots=roots)
    assert isinstance(ds, UnifiedWaterDataset) and ds.root == os.path.realpath(val) and ds.items[0]["key"] == 2


@pytest.mark.skipif(not os.path.exists(os.path.join(WATER_ROOT, "valid", "manifest.json")),
                    reason="materialized Water-3D not on this machine")
def test_real_materialized_valid_split():
    ds = UnifiedWaterDataset(f"{WATER_ROOT}/valid", training=False, raw=True)
    assert len(ds.items) == 100 and all(it["frames"] == 801 for it in ds.items)
    b = next(iter(ds))
    assert tuple(b.positions.shape) == (4, 20, 4096, 3) and b.domain == "water"
    assert float(b.positions[..., 2].min()) >= -0.01 and float(b.positions.abs().max()) <= 0.8
