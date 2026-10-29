import hashlib
import io
import json
from itertools import islice
from pathlib import Path
import pickle
import tarfile

import numpy as np
import pytest
import torch

from data.indexed_episodes import INDEX_SCHEMA, IndexedEpisodeDataset, load_index, validate_index_pair
from scripts.unified.prepare_indexed_datasets import file_identity, write_indices
from scripts.unified.verify_collections import audit_batch


def mpm_source(tmp_path, episodes=4, drag=False, moving_from=None, asset_ids=None):
    path = tmp_path / "episodes.tar"
    b, t, n = episodes, 49, 8
    pos = np.zeros((b, t, n, 3), dtype=np.float32)
    # 1 mm per frame along x; with moving_from the scene rests until that frame.
    advance = np.full(t, .001) if moving_from is None else (np.arange(t) >= moving_from) * .001
    pos[..., 0] = .75 + np.concatenate([[0.], np.cumsum(advance[1:])])[None, :, None]
    pos[..., 1] = .75 + np.arange(n)[None, None, :] * .001
    pos[..., 2] = .2
    material = np.broadcast_to(np.array([1e5, .2, 1000., 0., 0.], np.float32), (b, n, 5)).copy()
    selected = np.zeros((b, n), dtype=bool)
    active = np.zeros((b, t), dtype=bool)
    velocity = np.zeros((b, t - 1, n, 3), dtype=np.float32) if drag else np.zeros((b, n, 3), np.float32)
    if drag:
        selected[:, :2] = True
        active[:, :16] = True
        velocity[:, :8, :2, 1] = .1  # 8 active movement, 8 active hold, then release
    provenance = [dict(episode_id=f"episode-{i}", task="drag" if drag else "passive",
        state_time_s=(np.arange(t) * .0417).tolist(), action_schema="mpm-drag-command-schedule-v1") for i in range(b)]
    sample = {"particle_pos.pyd": pos, "valid_particles.pyd": np.ones((b, t, n), bool),
              "material_properties.pyd": material, "object_ids.pyd": np.zeros((b, n), np.int64),
              "grab_velocity.pyd": velocity, "is_grabbed.pyd": selected, "drag_active.pyd": active,
              "provenance.json": provenance}
    with tarfile.open(path, "w") as archive:
        for field, value in sample.items():
            content = json.dumps(value).encode() if field.endswith("json") else pickle.dumps(value)
            member = tarfile.TarInfo("batch_000." + field); member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    records = [dict(file_identity(path), key="batch_000", row=i, episode_id=f"episode-{i}",
                    group=f"episode-{i}", frames=t, dt=.0417) for i in range(b)]
    if asset_ids is not None:
        for record, ids in zip(records, asset_ids):
            record["asset_ids"] = list(ids)
    return records, sample


def index(root, records, domain="mpm_fresh", split="train"):
    root.mkdir(parents=True)
    (root / "INDEX.json").write_text(json.dumps(dict(schema=INDEX_SCHEMA, domain=domain, split=split, records=records)))
    return root


def test_mpm_window_is_exact_raw_source_and_unclipped(tmp_path):
    rows, raw = mpm_source(tmp_path)
    ds = IndexedEpisodeDataset(index(tmp_path / "train", rows), domain="mpm_fresh", raw=True)
    b = ds.window(rows[1], 7)
    expected = torch.from_numpy(raw["particle_pos.pyd"][1:2, 7:27]) + torch.tensor([-.75, -.75, -.1])
    torch.testing.assert_close(b.positions, expected, atol=0, rtol=0)
    assert not b.action_mask.any() and b.pose_delta_stats is None
    assert b.domain == "mpm_fresh"
    assert audit_batch(b)["particle_roundtrip_max_mm"] < .01


def test_drag_keeps_command_motion_hold_and_release_distinct(tmp_path):
    rows, raw = mpm_source(tmp_path, drag=True)
    ds = IndexedEpisodeDataset(index(tmp_path / "train", rows, "mpm_drag"), domain="mpm_drag", raw=True)
    b = ds.window(rows[0], 0)
    dt = np.diff(np.asarray(raw["provenance.json"][0]["state_time_s"]))
    oracle = (raw["grab_velocity.pyd"][0:1, 3:19].astype(np.float64) * dt[None, 3:19, None, None]).astype(np.float32)
    torch.testing.assert_close(b.actions, torch.from_numpy(oracle), rtol=0, atol=0)
    assert b.action_active[:, 8:13, :2].all() and not b.actions[:, 8:13].any()
    assert not b.action_active[:, 13:].any() and not b.actions[:, 13:].any()
    assert audit_batch(b)["command_vs_achieved_max_mm"] > 1


@pytest.mark.parametrize("field", ["episode_id", "group"])
def test_split_pair_rejects_logical_leakage(tmp_path, field):
    rows, _ = mpm_source(tmp_path)
    other = dict(rows[1], **{field: rows[0][field]})
    a, b = index(tmp_path / "train", [rows[0]]), index(tmp_path / "val", [other], split="val")
    with pytest.raises(ValueError, match="overlap"):
        validate_index_pair(a, b, "mpm_fresh")


def test_split_pair_rejects_same_physical_episode_under_different_names(tmp_path):
    rows, _ = mpm_source(tmp_path)
    other = dict(rows[0], episode_id="other-name", group="other-group")
    a, b = index(tmp_path / "train", [rows[0]]), index(tmp_path / "val", [other], split="val")
    with pytest.raises(ValueError, match="physical source episode"):
        validate_index_pair(a, b, "mpm_fresh")


def test_duplicate_index_rows_rejected(tmp_path):
    rows, _ = mpm_source(tmp_path)
    root = index(tmp_path / "train", [rows[0], rows[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        load_index(root)


def test_freezer_rejects_official_test_source_alias(tmp_path):
    from scripts.unified.prepare_indexed_datasets import split_group
    rows, _ = mpm_source(tmp_path)
    groups = {split: next(f"group-{i}" for i in range(100)
                         if split_group(f"group-{i}", 916) == split) for split in ("train", "val")}
    train = dict(rows[0], group=groups["train"])
    val = dict(rows[1], group=groups["val"])
    test = dict(rows[0], episode_id="renamed-test", group="official-test", official_test=True)
    with pytest.raises(ValueError, match="test physical source"):
        write_indices(tmp_path / "release", "mpm_fresh", [train, val, test], {})
    assert test["official_test"]  # input records are not consumed/mutated


def test_source_mutation_detected_even_after_cached_read(tmp_path):
    rows, _ = mpm_source(tmp_path)
    ds = IndexedEpisodeDataset(index(tmp_path / "train", rows), domain="mpm_fresh", raw=True)
    ds.window(rows[0], 0)
    with open(rows[0]["file"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Source changed"):
        ds.window(rows[0], 0)


def test_evaluation_windows_do_not_consume_ambient_rng(tmp_path):
    rows, _ = mpm_source(tmp_path)
    ds = IndexedEpisodeDataset(index(tmp_path / "val", rows, split="val"), domain="mpm_fresh", raw=True, training=False)
    ds.batch_size = 2
    before = torch.random.get_rng_state().clone()
    first = list(ds)
    assert torch.equal(before, torch.random.get_rng_state())
    torch.manual_seed(875)
    for a, b in zip(first, ds):
        assert torch.equal(a.positions, b.positions)
        assert a.batch_size == 2


def test_ddp_rank_partition_keeps_episodes_disjoint(tmp_path, monkeypatch):
    rows, _ = mpm_source(tmp_path)
    ds = IndexedEpisodeDataset(index(tmp_path / "val", rows, split="val"), domain="mpm_fresh", raw=True, training=False)
    seen = []
    for rank in range(2):
        monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (rank, 2, 0, 1))
        seen.append({b.source_episode_id for b in ds})
    assert not seen[0] & seen[1]
    assert seen[0] | seen[1] == {r["episode_id"] for r in rows}


@pytest.mark.parametrize("future", [-1, 10000])
def test_full_episode_evaluation_never_crosses_terminal(tmp_path, future):
    rows, _ = mpm_source(tmp_path)
    ds = IndexedEpisodeDataset(index(tmp_path / "val", rows, split="val"), domain="mpm_fresh",
                              raw=True, training=False, future_len=future)
    assert next(iter(ds)).num_frames == 49
    with pytest.raises(ValueError, match="Window outside"):
        ds.window(rows[0], 1)


def dex_source(tmp_path):
    from utils.pose_ops import so3_exp
    t = 45
    rng = np.random.default_rng(37)
    angles = np.arange(t, dtype=np.float32) * .025
    q = np.stack([np.cos(angles / 2), np.zeros(t), np.zeros(t), np.sin(angles / 2)], -1).astype(np.float32)
    lq = np.tile(np.array([1, 0, 0, 0], np.float32), (t, 1, 2, 1)); lq[:, 0, 1] = q
    op = np.tile(np.array([.1, 0, .3], np.float32), (t, 1, 1))
    lp = np.tile(np.array([0, 0, .2], np.float32), (t, 1, 2, 1))
    pivot = np.array([.02, .01, 0], np.float32)
    offset = np.array([.025, 0, 0], np.float32)
    R = so3_exp(torch.from_numpy(np.stack([angles * 0, angles * 0, angles], -1))).numpy()
    lp[:, 0, 1] += pivot + np.einsum("tij,j->ti", R, offset)
    source = tmp_path / "dex.npz"
    np.savez(source, link_pos=lp, link_quat=lq, obj_pos=op, obj_quat=q[:, None],
             episode_length=np.array([39]), episode_shape_id=np.array([0]),
             link_names=np.array(["root", "child"]), control_dt=np.array(.025))
    geometry = tmp_path / "geometry.npz"
    np.savez(geometry, link_names=np.array(["root", "child"]), parent=np.array([-1, 0]),
             axis=np.array([[0, 0, 0], [0, 0, 1]], np.float32), pivot=np.stack([pivot * 0, pivot]),
             hand_0=rng.normal(0, .01, (9, 3)).astype(np.float32),
             hand_1=rng.normal(0, .005, (8, 3)).astype(np.float32),
             object_0=rng.normal(0, .01, (11, 3)).astype(np.float32))
    gi = file_identity(geometry)
    return dict(file_identity(source), episode_id="teacher/0", group="teacher/0", env=0, frames=20, stride=2,
                geometry_file=gi["file"], geometry_size=gi["size"], geometry_mtime_ns=gi["mtime_ns"],
                geometry_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest())


@pytest.mark.parametrize("articulated", [False, True])
def test_revolute_dex_recorded_geometry_targets_and_terminal(tmp_path, articulated):
    record = dex_source(tmp_path)
    ds = IndexedEpisodeDataset(index(tmp_path / "train", [record], "dex"), domain="dex", raw=True)
    ds.articulated = articulated
    b = ds.window(record, 0)
    assert b.positions.shape == (1, 20, 28, 3)
    assert audit_batch(b)["joint_target_roundtrip_max_mm"] < .01
    assert b.action_mask[:, 11:].all() and not b.action_mask[:, :11].any()
    if articulated:
        assert torch.allclose(b.pose_deltas_gt[0, :, 2, 3], torch.full((16,), .05), atol=2e-6)
        assert not b.pose_deltas_gt[0, :, 2, :3].any()
    with pytest.raises(ValueError, match="Window outside"):
        ds.window(record, 1)


def test_geometry_mutation_rejected_after_cached_verification(tmp_path):
    record = dex_source(tmp_path)
    ds = IndexedEpisodeDataset(index(tmp_path / "train", [record], "dex"), domain="dex", raw=True)
    ds.window(record, 0)
    with open(record["geometry_file"], "ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Auxiliary geometry changed"):
        ds.window(record, 0)


def test_new_geometry_release_does_not_reuse_old_cached_arrays(tmp_path):
    import os
    record = dex_source(tmp_path)
    ds = IndexedEpisodeDataset(index(tmp_path / "old", [record], "dex"), domain="dex", raw=True)
    old = ds.window(record, 0)
    path = Path(record["geometry_file"])
    with np.load(path, allow_pickle=False) as z:
        geometry = {key: z[key] for key in z.files}
    geometry["object_0"] *= 2
    np.savez(path, **geometry)
    stamp = record["geometry_mtime_ns"] + 10_000_000
    os.utime(path, ns=(stamp, stamp))
    changed = dict(record, geometry_size=path.stat().st_size, geometry_mtime_ns=stamp,
                   geometry_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    new_ds = IndexedEpisodeDataset(index(tmp_path / "new", [changed], "dex"), domain="dex", raw=True)
    new = new_ds.window(changed, 0)
    assert not torch.equal(new.positions[:, :, :11], old.positions[:, :, :11])
    assert torch.equal(new.positions[:, :, 11:], old.positions[:, :, 11:])


@pytest.mark.parametrize("domain,field", [("dex", "teacher"), ("folding", "action")])
def test_evaluation_prefix_covers_strata_and_rotates_groups_without_rng(domain, field):
    from data.indexed_episodes import evaluation_records
    rows = [dict(episode_id=f"{label}/{group}/{episode}", group=f"group-{group}",
                 **{field: label})
            for label in range(5) for group in range(4) for episode in range(2)]
    original = json.dumps(rows)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    ordered = evaluation_records(rows, domain)
    assert {row[field] for row in ordered[:5]} == set(range(5))
    for label in range(5):
        members = [row for row in ordered if row[field] == label]
        assert len({row["group"] for row in members[:4]}) == 4
    assert {row["episode_id"] for row in ordered} == {row["episode_id"] for row in rows}
    assert len(ordered) == len(rows)
    assert ordered == evaluation_records(list(reversed(rows)), domain)
    assert original == json.dumps(rows)
    assert torch.equal(torch_before, torch.random.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    assert np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]


def test_replicated_validation_ranks_still_partition_workers(tmp_path, monkeypatch):
    rows, _ = mpm_source(tmp_path, episodes=5)
    ds = IndexedEpisodeDataset(index(tmp_path / "val", rows, split="val"),
                              domain="mpm_fresh", raw=True, training=False)
    ds.no_shard_split = True
    for rank in range(2):
        seen = []
        for worker in range(3):
            monkeypatch.setattr("webdataset.utils.pytorch_worker_info",
                                lambda: (rank, 2, worker, 3))
            seen.extend(b.source_episode_id for b in ds)
        assert len(seen) == len(set(seen)) == len(rows)
        assert set(seen) == {row["episode_id"] for row in rows}


def test_empty_validation_worker_finishes_but_empty_training_worker_fails(tmp_path, monkeypatch):
    rows, _ = mpm_source(tmp_path, episodes=1)
    root = index(tmp_path / "val", rows, split="val")
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 2, 3))
    assert list(IndexedEpisodeDataset(root, domain="mpm_fresh", raw=True, training=False)) == []
    with pytest.raises(ValueError, match="No indexed episodes"):
        next(iter(IndexedEpisodeDataset(root, domain="mpm_fresh", raw=True, training=True)))


def test_training_serialized_order_is_unchanged(tmp_path, monkeypatch):
    from itertools import islice
    rows, _ = mpm_source(tmp_path, episodes=5)
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "train", rows), domain="mpm_fresh",
                              raw=True, training=True, shuffle=False)
    assert [b.source_episode_id for b in islice(ds, len(rows))] == [r["episode_id"] for r in rows]


def test_release_test_index_serves_as_validation_root(tmp_path):
    from data.indexed_episodes import validate_index_pair
    rows, _ = mpm_source(tmp_path, episodes=4)
    train = index(tmp_path / "train", rows[:2])
    test = index(tmp_path / "test", rows[2:], split="test")
    a, b = validate_index_pair(train, test, "mpm_fresh")
    assert (a["split"], b["split"]) == ("train", "test")
    other_train = index(tmp_path / "other", rows[2:], split="train")
    with pytest.raises(ValueError, match="differs from requested val/test"):
        validate_index_pair(train, other_train, "mpm_fresh")
    with pytest.raises(ValueError, match="overlap"):
        validate_index_pair(train, index(tmp_path / "leak", rows[1:], split="test"), "mpm_fresh")


def test_alpha_zero_reproduces_the_uniform_draws(tmp_path, monkeypatch):
    rows, _ = mpm_source(tmp_path, episodes=5)
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "train", rows), domain="mpm_fresh", raw=True)
    ds.batch_size = 3
    torch.manual_seed(7)
    drawn = []
    for batch in islice(ds, 5):
        drawn.append((batch.source_episode_id, [w["start"] for w in ds.last_windows]))
    torch.manual_seed(7)                        # the historical consumption: one randperm, then randint per episode
    order = torch.randperm(5).tolist()
    expected = [(rows[i]["episode_id"], torch.randint(49 - 20 + 1, (3,)).tolist()) for i in order]
    assert drawn == expected
    assert all(w["start_policy"] == "uniform" for w in ds.last_windows)


def test_motion_weighted_starts_prefer_moving_windows(tmp_path, monkeypatch):
    rows, _ = mpm_source(tmp_path, episodes=1, moving_from=30)
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "train", rows), domain="mpm_fresh", raw=True)
    ds.batch_size, ds.motion_alpha, ds.motion_floor_m = 8, 1.0, 1e-5
    weights = ds._start_weights(rows[0])
    assert weights.shape == (30,) and bool((weights > 0).all())
    # Future transitions of start s are [s+3, s+19); motion begins at transition 29, so starts below 11 see none.
    moving = torch.tensor([min(16, max(0, s + 19 - 29)) for s in range(30)], dtype=torch.float64) * .001 / 16
    torch.testing.assert_close(weights, moving + 1e-5, rtol=1e-4, atol=1e-12)   # float32 positions step 0.99999 mm
    torch.manual_seed(0)
    starts = [w["start"] for _ in islice(ds, 50) for w in ds.last_windows]
    assert len(starts) == 400 and sum(s >= 11 for s in starts) / 400 > .97 and np.mean(starts) > 20
    assert all(w["start_policy"] == "motion" for w in ds.last_windows)
    ds.motion_alpha = 0.0                       # uniform again: about 19 of 30 starts see motion
    torch.manual_seed(0)
    uniform = [w["start"] for _ in islice(ds, 50) for w in ds.last_windows]
    assert .5 < sum(s >= 11 for s in uniform) / 400 < .75


def test_motion_weighting_needs_a_positive_floor_and_a_profile(tmp_path, monkeypatch):
    rows, _ = mpm_source(tmp_path, episodes=1)
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "train", rows), domain="mpm_fresh", raw=True)
    ds.motion_alpha = 1.0
    with pytest.raises(ValueError, match="floor"):
        ds._start_weights(rows[0])
    record = dex_source(tmp_path)
    dex = IndexedEpisodeDataset(index(tmp_path / "dex", [record], "dex"), domain="dex", raw=True)
    dex.motion_alpha, dex.motion_floor_m = 1.0, 1e-4
    with pytest.raises(ValueError, match="no motion profile"):
        dex._start_weights(record)


def test_evaluation_starts_ignore_the_window_policy(tmp_path, monkeypatch):
    rows, _ = mpm_source(tmp_path, episodes=3, moving_from=30, asset_ids=[["a"], ["b"], ["c"]])
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "val", rows, split="val"), domain="mpm_fresh", raw=True, training=False)
    ds.batch_size, ds.motion_alpha, ds.motion_floor_m, ds.cross_episode = 2, 1.0, 1e-5, True
    for batch in ds:
        seed = int.from_bytes(hashlib.sha256(batch.source_episode_id.encode()).digest()[:8], "little")
        assert [w["start"] for w in ds.last_windows] == np.random.default_rng(seed).integers(30, size=2).tolist()
        assert batch.source_episode_ids == [batch.source_episode_id] * 2
        assert all(w["start_policy"] == "uniform" for w in ds.last_windows)


def test_cross_episode_batches_mix_distinct_scenes_of_equal_point_count(tmp_path, monkeypatch):
    ids = [["a"], ["b"], ["c"], ["d", "e"], ["f", "g"], ["h", "i"]]
    rows, _ = mpm_source(tmp_path, episodes=6, asset_ids=ids)
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "train", rows), domain="mpm_fresh", raw=True)
    ds.batch_size, ds.cross_episode = 2, True
    objects = {r["episode_id"]: len(r["asset_ids"]) for r in rows}
    torch.manual_seed(1)
    seen = []
    for batch in islice(ds, 150):               # buckets of three: one episode always waits for the next pass
        assert list(batch.positions.shape) == [2, 20, 8, 3]
        a, b = batch.source_episode_ids
        assert a != b and objects[a] == objects[b]
        assert [w["episode_id"] for w in ds.last_windows] == [a, b] and batch.source_episode_id == a
        seen += [a, b]
    counts = [seen.count(r["episode_id"]) for r in rows]             # 300 draws over 6 scenes: 50 each in expectation
    assert sum(counts) == 300 and max(counts) - min(counts) <= 10    # exchangeable scenes; measured spreads were 4 to 6


def test_window_sampling_validation_and_recipe_values(tmp_path):
    from data.unified_dataset import validate_window_sampling
    weighting, across = validate_window_sampling({"mpm_fresh": dict(alpha=1, floor_mm=2)}, ["mpm_drag", "mpm_fresh"])
    assert weighting == {"mpm_fresh": dict(alpha=1.0, floor_mm=2.0)} and across == ["mpm_drag", "mpm_fresh"]
    for bad in ({"dex": dict(alpha=1, floor_mm=1)}, {"folding": dict(alpha=1)}, {"folding": dict(alpha=-1, floor_mm=1)},
                {"folding": dict(alpha=1, floor_mm=0)}, {"folding": dict(alpha=True, floor_mm=1)}):
        with pytest.raises(ValueError):
            validate_window_sampling(bad, None)
    for bad in (["folding"], ["mpm_fresh", "mpm_fresh"]):
        with pytest.raises(ValueError, match="batch_across_episodes"):
            validate_window_sampling(None, bad)


def yikai_source(tmp_path, frames=60, table_top=0.12, name="episode"):
    """A wm_v2-shaped episode: gripper links, a manipulated cup, one distractor and a table that sits at
    z = table_top in env coordinates; the cup slides +x from frame 20 on, the distractor jitters in the
    first ten frames, the table never moves."""
    from data.native_episodes import GRIPPER_PARTS
    names = [*GRIPPER_PARTS, "cup_0001", "table", "can_0002"]
    counts = [40, 20, 20, 10, 10, 32, 16, 32]
    pid = np.repeat(np.arange(len(names)), counts).astype(np.int32)
    rng = np.random.default_rng(3)
    base = rng.normal(0, .02, (pid.size, 3)).astype(np.float32)
    base[:, 2] += table_top + .03
    table_rows = pid == names.index("table")
    base[table_rows, 2] = table_top - .01 * rng.random(int(table_rows.sum()))       # table points below its top
    pos = np.repeat(base[None], frames, axis=0).copy()
    cup = pid == names.index("cup_0001")
    pos[20:, cup, 0] += (np.arange(frames - 20, dtype=np.float32)[:, None] + 1) * .002
    can = pid == names.index("can_0002")
    pos[:10, can, 1] += rng.normal(0, .003, (10, int(can.sum()))).astype(np.float32)
    poses = np.zeros((frames, len(names), 7), np.float32); poses[..., 3] = 1
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{name}_traj0.npz"
    np.savez(path, **{"state/particles": pos, "meta/particles/part_names": np.array(names), "meta/particles/part_ids": pid,
                      "meta/particles/part_counts": np.array(counts), "state/physics_step": np.arange(frames) * 25,
                      "state/physics_dt": np.full(frames, .002, np.float32), "meta/spatial/src_name": np.array("cup_0001"),
                      "state/particle_part_poses_wxyz": poses, "state/table_top_z_env": np.full(frames, table_top, np.float32)})
    record = dict(file_identity(path), episode_id=f"yikai-{name}", group="scene-a", scene="scene-a", frames=frames,
                  source_object="cup_0001", population="gen", success=True, dt=.05, table_top_z_env=table_top)
    return record, names, pid


def test_yikai_adapter_drops_the_table_and_puts_its_top_at_zero(tmp_path, monkeypatch):
    record, names, pid = yikai_source(tmp_path)
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "train", [record], "yikai"), domain="yikai", raw=True)
    assert ds.first_start == 10
    batch = ds.window(record, 10)
    from data.native_episodes import PICKUP_LAYOUT_NAMES, shared_gripper_points
    links = shared_gripper_points()
    n_grip = sum(len(links[n]) for n in PICKUP_LAYOUT_NAMES)
    objects = np.isin(pid, [names.index("cup_0001"), names.index("can_0002")])
    assert n_grip == 671 and batch.positions.shape == (1, 20, n_grip + int(objects.sum()), 3)
    with np.load(record["file"]) as z:
        raw = z["state/particles"][10:30]
    shift = np.array([-.45, 0, -.12], np.float32)
    # the gripper is the Franka point set rl_push and pickup use, under the recorded link poses (identity at the origin
    # in this source), not the recording's own 100 gripper points ...
    expected = np.concatenate([links[n] for n in PICKUP_LAYOUT_NAMES]) + shift
    assert np.allclose(batch.positions[0, :, :n_grip].numpy(), expected[None], atol=1e-6)
    # ... and the objects are the recording's points, shifted by (-0.45, 0, -table_top): the table top is z = 0 and
    # the cup rests just above it
    kept = raw[:, objects] + shift
    assert np.allclose(np.sort(batch.positions[0, :, n_grip:, 0].numpy().ravel()), np.sort(kept[..., 0].ravel()), atol=1e-6)
    assert np.allclose(np.sort(batch.positions[0, :, n_grip:, 2].numpy().ravel()), np.sort(kept[..., 2].ravel()), atol=1e-6)
    assert batch.body_poses.shape[2] == 5 and batch.action_mask.sum() == n_grip     # cup, 3 gripper links, can: no table body; the gripper rows are actuated
    with pytest.raises(ValueError, match="outside"):
        ds.window(record, 9)
    with pytest.raises(ValueError, match="table height differs"):
        ds.window(dict(record, table_top_z_env=0.0), 10)


def test_yikai_windows_start_at_frame_ten_in_training_and_evaluation(tmp_path, monkeypatch):
    record, _, _ = yikai_source(tmp_path)
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "train", [record], "yikai"), domain="yikai", raw=True)
    ds.batch_size = 8
    torch.manual_seed(0)
    starts = [w["start"] for _ in islice(ds, 20) for w in ds.last_windows]
    assert min(starts) >= 10 and max(starts) <= 60 - 20 and min(starts) < 20 and max(starts) > 30
    assert ds.last_windows[0]["table_top_z_env"] == 0.12
    ds.first_window = True
    next(iter(ds)); assert [w["start"] for w in ds.last_windows] == [10] * 8
    ev = IndexedEpisodeDataset(index(tmp_path / "val", [record], "yikai", split="val"), domain="yikai", raw=True, training=False)
    ev.batch_size = 4
    for _ in ev:
        assert min(w["start"] for w in ev.last_windows) >= 10
    short, _, _ = yikai_source(tmp_path / "short", frames=29, name="short")
    with pytest.raises(ValueError, match="too short"):
        IndexedEpisodeDataset(index(tmp_path / "short" / "train", [short], "yikai"), domain="yikai", raw=True)


def test_first_start_is_zero_for_the_other_domains(tmp_path, monkeypatch):
    rows, _ = mpm_source(tmp_path, episodes=2)
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    ds = IndexedEpisodeDataset(index(tmp_path / "train", rows), domain="mpm_fresh", raw=True)
    assert ds.first_start == 0 and ds._draw_starts(rows[0], 1)[0] >= 0


def test_calibration_draws_starts_from_the_first_allowed_frame(tmp_path, monkeypatch):
    from scripts.unified.verify_collections import calibrate
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    # The fixtures hold too little motion for a p99 fit; only the drawn starts are under test here.
    monkeypatch.setattr("scripts.unified.verify_collections._half_ranges", lambda rows, pose=False: np.ones(2))
    record, _, _ = yikai_source(tmp_path / "source")
    index(tmp_path / "yikai" / "train", [record], "yikai")
    report = calibrate(tmp_path, "yikai", tmp_path / "fit_yikai", episodes=1, windows=16, seed=0, capacity=256)
    starts = [row["start"] for row in report["ledger"]]
    assert len(starts) == 16 and min(starts) >= 10 and max(starts) <= 60 - 20
    # Domains that start at frame 0 draw exactly what they drew before this rule existed.
    (tmp_path / "mpm").mkdir()
    rows, _ = mpm_source(tmp_path / "mpm", episodes=1)
    index(tmp_path / "mpm_fresh" / "train", rows)
    report = calibrate(tmp_path, "mpm_fresh", tmp_path / "fit_mpm", episodes=1, windows=16, seed=0, capacity=256)
    assert [row["start"] for row in report["ledger"]] == np.random.default_rng(0).integers(49 - 20 + 1, size=16).tolist()


def folding_source(tmp_path, nc=60, frames=40):
    """A retained 30 Hz folding clip as the adapter validates it: nc cloth vertices then two 1,000-point grippers."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    n = nc + 2000
    pos = np.cumsum(rng.normal(0, .002, (frames, n, 3)), axis=0).astype(np.float32) + rng.uniform(-.3, .3, (1, n, 3)).astype(np.float32)
    np.savez(tmp_path / "action-0.npz", particle_pos=pos, obj_ids=np.repeat([0, 1, 2], [nc, 1000, 1000]),
             valid_particles=np.ones(n, bool), n_cloth=np.int64(nc), dt=np.float32(1 / 30),
             state_time_s=np.arange(30, 30 + frames) / 30, source_state_index=np.arange(30, 30 + frames),
             ee_link_ids=np.repeat([0, 1, 2], [200, 400, 400]).astype(np.int32))
    np.savez(tmp_path / "gripper-template-1000.npz", link_ids=np.repeat([0, 1, 2], [200, 400, 400]).astype(np.int32),
             left_local_points=rng.normal(0, .01, (1000, 3)).astype(np.float32), right_local_points=rng.normal(0, .01, (1000, 3)).astype(np.float32))
    def identity(path, prefix=None):
        info = file_identity(path)
        info["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        return info if prefix is None else {f"{prefix}_{k}": v for k, v in info.items()}
    record = dict(identity(tmp_path / "action-0.npz"), episode_id="garment/0", group="garment", frames=frames, garment_id="garment",
                  action="0", particles=n, dt=1 / 30, **identity(tmp_path / "gripper-template-1000.npz", "template"))
    return record, pos


def folding_subset(tmp_path, record, keep, name="cloth-subset.npz"):
    keep = np.asarray(keep, dtype=np.int64)
    np.savez(tmp_path / name, indices=keep, target_mm=np.float64(8.0), cap=np.int64(10000))
    info = file_identity(tmp_path / name); info["sha256"] = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    return dict(record, particles=len(keep) + 2000, **{f"subset_{k}": v for k, v in info.items()})


def test_folding_subset_keeps_the_listed_vertices_and_every_gripper_point(tmp_path, monkeypatch):
    from data.recording_adapters import folding_motion
    monkeypatch.setattr("webdataset.utils.pytorch_worker_info", lambda: (0, 1, 0, 1))
    record, pos = folding_source(tmp_path / "g")
    keep = [1, 4, 5, 17, 30, 59]
    sub = folding_subset(tmp_path / "g", record, keep)
    ds = IndexedEpisodeDataset(index(tmp_path / "train", [sub], "folding"), domain="folding", raw=True)
    b = ds.window(sub, 3)
    assert tuple(b.positions.shape) == (1, 20, len(keep) + 2000, 3)
    expected = np.concatenate([pos[3:23][:, keep], pos[3:23][:, 60:]], axis=1)
    assert np.array_equal(b.positions[0].numpy(), expected)
    assert b.action_mask[0].sum().item() == 2000 and (b.object_ids[0][: len(keep)] == 0).all()
    # the motion profile (window weighting) uses the same vertices
    full = folding_motion(record); part = folding_motion(sub)
    assert part.shape == full.shape and np.allclose(part, np.linalg.norm(np.diff(pos[:, keep], axis=0), axis=-1).mean(axis=1), atol=1e-6)
    # a record without a subset still reads every vertex
    whole = IndexedEpisodeDataset(index(tmp_path / "whole", [record], "folding"), domain="folding", raw=True).window(record, 3)
    assert tuple(whole.positions.shape) == (1, 20, 2060, 3)
    # the training stream emits the reduced clouds
    ds.batch_size = 2
    batch = next(iter(ds))
    assert tuple(batch.positions.shape) == (2, 20, len(keep) + 2000, 3)


def test_folding_subset_is_checked_against_the_garment_and_the_index(tmp_path):
    from data.recording_adapters import _folding_subset
    record, _ = folding_source(tmp_path / "g")
    ds = IndexedEpisodeDataset(index(tmp_path / "train", [record], "folding"), domain="folding", raw=True)
    beyond = folding_subset(tmp_path / "g", record, [0, 60], name="beyond.npz")
    with pytest.raises(ValueError, match="beyond the garment"):
        ds.window(beyond, 0)
    unsorted = folding_subset(tmp_path / "g", record, [5, 2], name="unsorted.npz")
    with pytest.raises(ValueError, match="sorted unique"):
        ds.window(unsorted, 0)
    wrong_count = dict(folding_subset(tmp_path / "g", record, [0, 1, 2]), particles=99)
    with pytest.raises(ValueError, match="differs from the index record"):
        ds.window(wrong_count, 0)
    _folding_subset.cache_clear()
    tampered = folding_subset(tmp_path / "g", record, [0, 1, 2], name="tampered.npz")
    np.savez(tmp_path / "g" / "tampered.npz", indices=np.array([0, 1, 3]), target_mm=np.float64(8.0), cap=np.int64(10000))
    with pytest.raises(ValueError, match="changed|checksum"):
        ds.window(dict(tampered, subset_mtime_ns=file_identity(tmp_path / "g" / "tampered.npz")["mtime_ns"]), 0)
