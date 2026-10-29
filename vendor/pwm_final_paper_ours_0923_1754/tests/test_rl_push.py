"""rl_push contract: the low-dim release format, pickup-layout points built from poses, the rigid sidecar and
jointed fingers, the validation object list and provenance-keyed exclusions, training window starts,
evaluation order, batching and rollout windows, the optional rigid normalization entry, config checks, data.roots
and the unified loader. The real release is exercised only when it is readable on this machine."""
import copy
import json
import os
import sys

import numpy as np
import pytest
import torch

from data.native_episodes import reconstruction_error
from data.normalization import DEFAULT_ARTIFACT, domain_normalizers, load_normalization_artifact, resolve_normalization
from data.rl_push import (GRIPPER_LINKS, RL_PUSH_FINGER_Z, RL_PUSH_FIRST_START, RL_PUSH_SHIFT, UnifiedRLPushDataset,
                          load_push_object, window_points)
from data.unified_dataset import (DOMAIN_ORDER, MAT_GRIPPER, MAT_RIGID, RL_PUSH_EXCLUSIONS, RL_PUSH_ROOT,
                                  RL_PUSH_VAL_ASSETS, build_domain_dataset, create_unified_dataloader, exclusion_files,
                                  resolve_data_roots, selected_domains)
from utils.config import load_config
from utils.pose_ops import quat_wxyz_to_matrix

FROZEN_SNAPSHOT_SHA = "2bc31504482f745a5bfce97558e7a366e1b6800d2bb149a8853e663d797587de"
OBJECTS = {"Kitchen_Table/mug/asset_0001": ([30, 25, 12], False),
           "Kitchen_Table/mug/asset_0002": ([40, 26], False),
           "libero/basket/asset_0003": ([28, 33], True)}          # True: source_run stored as one value
N_GRIPPER, N_OBJECT = 671, 512


def _smooth_quats(t, phase):
    a = 0.02 * t + phase
    q = np.stack([np.cos(a), 0.3 * np.sin(a), 0.2 * np.sin(1.3 * a), 0.9 * np.sin(a)], -1)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def _write_object(root, key, lengths, scalar_run, seed):
    g = np.random.default_rng(seed)
    F, E = int(sum(lengths)), len(lengths)
    t = np.arange(F, dtype=np.float64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])[:E].astype(np.int64)
    hand_quat = _smooth_quats(t, seed)
    hand_quat[starts + 5] *= 1.3                                     # stored non-unit quaternions (seen in the release)
    joints = np.zeros((F, 9), np.float32)
    joints[:, 7] = 0.0008 * (1 + np.sin(0.3 * t))                    # left finger joint (m)
    joints[:, 8] = 0.0005 * (1 + np.cos(0.2 * t))                    # right finger joint
    arrays = dict(
        format_version=np.array("rl_push_lowdim_v1"), asset_key=np.array(key), category=np.array(key.split("/")[1]),
        scale=np.array(1.0), object_cloud_outer_surface=(0.03 * g.normal(size=(1000, 3))).astype(np.float32),
        object_cloud_env_pcd=(0.03 * g.normal(size=(1000, 3))).astype(np.float32),
        episode_start=starts, episode_len=np.asarray(lengths, np.int32), joint_pos=joints,
        hand_pos=np.stack([0.2 + 0.004 * t, 0.05 * np.sin(0.1 * t), 0.15 + 0.01 * np.cos(0.05 * t)], -1).astype(np.float32),
        hand_quat_wxyz=hand_quat.astype(np.float32),
        object_pos=np.stack([0.45 + 0.003 * t, 0.01 * np.sin(0.2 * t), np.full(F, 0.03)], -1).astype(np.float32),
        object_quat_wxyz=_smooth_quats(1.5 * t, seed + 1).astype(np.float32),
        goal_pos=np.zeros((E, 3), np.float32), success_step=np.asarray(lengths, np.int32) - 11,
        source_run=np.array("run_scalar") if scalar_run else np.array([f"run_{seed}"] * E),
        source_shard=np.array([f"shard_{i:05d}.npz" for i in range(E)]),
        source_episode_index=np.arange(100, 100 + E, dtype=np.int64), source_env=np.arange(E, dtype=np.int32) % 4,
        finger_range_mm=np.full(E, 0.8, np.float32))
    path = os.path.join(root, "objects", key + ".npz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    return path


@pytest.fixture
def release(tmp_path):
    root = str(tmp_path / "release")
    for seed, (key, (lengths, scalar)) in enumerate(OBJECTS.items()):
        _write_object(root, key, lengths, scalar, seed)
    val = tmp_path / "val_assets.json"
    json.dump({"kind": "rl_push_val_assets", "assets": ["Kitchen_Table/mug/asset_0002"]}, open(val, "w"))
    excl = tmp_path / "exclusions.json"
    json.dump({"kind": "rl_push_episode_exclusions", "episodes": [
        {"asset_key": "Kitchen_Table/mug/asset_0001", "source_run": "run_0", "source_shard": "shard_00001.npz",
         "source_episode_index": 101, "source_env": 1, "reasons": ["test"]}]}, open(excl, "w"))
    art = json.load(open(DEFAULT_ARTIFACT))
    art["profiles"]["xy_p99_per_domain"]["particle"]["rl_push"] = [0.045, 0.019]
    art["profiles"]["xy_p99_per_domain"]["pose"]["rl_push"] = [0.04, 0.018, 0.25, 0.3]
    art["artifact_version"] = "test-rl-push"
    artifact = tmp_path / "normalization_rl_push_test.json"
    json.dump(art, open(artifact, "w"))
    return {"root": root, "val": str(val), "excl": str(excl), "artifact": str(artifact),
            "spec": {"profile": "xy_p99_per_domain", "artifact": str(artifact)}}


def _dataset(release, split="all", manifests=(), **kw):
    ds = UnifiedRLPushDataset(release["root"], split=split, manifests=manifests, normalization=release["spec"],
                              context_len=4, future_len=16, **kw)
    return ds


def _homogeneous(R, t):
    H = np.tile(np.eye(4), R.shape[:-2] + (1, 1))
    H[..., :3, :3], H[..., :3, 3] = R, t
    return H


def test_points_follow_the_pickup_layout(release):
    ds = _dataset(release, training=False)
    ds.first_window, ds.batch_size = True, 8
    batch = next(iter(ds))
    assert batch.source_shard == "objects/Kitchen_Table/mug/asset_0001.npz" and batch.domain == "rl_push"
    assert tuple(batch.positions.shape) == (2, 20, N_GRIPPER + N_OBJECT, 3)
    assert batch.object_ids[0, :N_GRIPPER].eq(1).all() and batch.object_ids[0, N_GRIPPER:].eq(0).all()
    assert batch.mat_type[0, 0] == MAT_GRIPPER and batch.mat_type[0, -1] == MAT_RIGID
    assert torch.equal(batch.action_mask, batch.object_ids == 1)
    steps = batch.positions[:, 1:] - batch.positions[:, :-1]
    assert torch.equal(batch.actions, steps[:, 3:] * batch.action_mask[:, None, :, None])

    # the same window rebuilt with homogeneous transforms: world <- hand <- finger joint <- link points
    obj, s = ds.objects[0], RL_PUSH_FIRST_START
    frames = slice(int(obj.starts[0]) + s, int(obj.starts[0]) + s + 20)
    q = torch.from_numpy(obj.hand_quat[frames].astype(np.float64))
    world_hand = _homogeneous(quat_wxyz_to_matrix(q / q.norm(dim=-1, keepdim=True)).numpy(), obj.hand_pos[frames])
    links = dict(np.load(GRIPPER_LINKS))
    assert {k: len(v) for k, v in links.items()} == {"panda_hand": 241, "panda_leftfinger": 212, "panda_rightfinger": 218}
    rows = []
    for name, sign, col in (("panda_hand", 0.0, 0), ("panda_leftfinger", 1.0, 0), ("panda_rightfinger", -1.0, 1)):
        hand_finger = _homogeneous(np.tile(np.eye(3), (20, 1, 1)),
                                   np.stack([np.zeros(20), sign * obj.finger_q[frames, col],
                                             np.full(20, RL_PUSH_FINGER_Z if sign else 0.0)], -1))
        p = np.concatenate([links[name], np.ones((len(links[name]), 1))], -1)
        rows.append(np.einsum("tij,tjk,pk->tpi", world_hand, hand_finger, p)[..., :3])
    qo = torch.from_numpy(obj.object_quat[frames].astype(np.float64))
    world_obj = _homogeneous(quat_wxyz_to_matrix(qo / qo.norm(dim=-1, keepdim=True)).numpy(), obj.object_pos[frames])
    tmpl = obj.template[ds._object_rows(obj)].astype(np.float64)
    rows.append(np.einsum("tij,pj->tpi", world_obj, np.concatenate([tmpl, np.ones((N_OBJECT, 1))], -1))[..., :3])
    expected = np.concatenate(rows, axis=1) + np.asarray(RL_PUSH_SHIFT)
    assert np.allclose(batch.positions[0].numpy(), expected, atol=2e-6)


def test_sidecar_is_exact_and_fingers_are_jointed(release):
    ds = _dataset(release, training=False)
    ds.first_window, ds.batch_size, ds.emit_sidecar, ds.articulated = True, 8, True, True
    batch = next(iter(ds))
    assert reconstruction_error(batch) < 1e-5
    assert batch.body_type[0].tolist() == [1, 1, 1, 1] and batch.body_parent[0].tolist() == [-1, -1, 1, 1]
    obj = ds.objects[0]
    q = obj.finger_q[int(obj.starts[0]) + RL_PUSH_FIRST_START:][:20].astype(np.float64)
    for body, col in ((2, 0), (3, 1)):                               # a finger's joint channel is its slide change
        assert np.allclose(batch.pose_deltas_gt[0, :, body, 0].numpy(), q[4:20, col] - q[3:19, col], atol=2e-6)
    assert batch.pose_action_mask[0].tolist() == [False, True, True, True]
    assert tuple(batch.pose_delta_stats.max_val.shape) == (6,)


def test_split_exclusions_and_training_windows(release, monkeypatch):
    manifests = (release["val"], release["excl"])
    train = _dataset(release, split="train", manifests=manifests)
    val = _dataset(release, split="val", manifests=manifests, training=False)
    assert [o.asset_key for o in train.objects] == ["Kitchen_Table/mug/asset_0001", "libero/basket/asset_0003"]
    assert [o.asset_key for o in val.objects] == ["Kitchen_Table/mug/asset_0002"] and val.shards == [val.objects[0].file]
    assert train.excluded_episodes == 1 and (0, 1) not in train.episodes and len(train.episodes) == 4
    with pytest.raises(ValueError, match="validation object list"):
        _dataset(release, split="train", manifests=(release["excl"],))
    with pytest.raises(ValueError, match="not an rl_push manifest"):
        _dataset(release, manifests=(DEFAULT_ARTIFACT,))

    drawn = []
    real = window_points

    def record(obj, episode, start, length, links, rows):
        drawn.append((obj.asset_key, episode, start, length, int(obj.lengths[episode])))
        return real(obj, episode, start, length, links, rows)

    monkeypatch.setattr("data.rl_push.window_points", record)
    train.batch_size = 64
    stream = iter(train)
    torch.manual_seed(0)
    first = next(iter(train))
    torch.manual_seed(0)
    assert torch.equal(next(iter(train)).positions, first.positions)          # seeded by torch.initial_seed()
    next(stream)
    assert all(RL_PUSH_FIRST_START <= s and s + L <= n and L == 20 for _, _, s, L, n in drawn)
    assert ("Kitchen_Table/mug/asset_0001", 1) not in {(k, e) for k, e, *_ in drawn}   # excluded
    assert ("Kitchen_Table/mug/asset_0001", 2) not in {(k, e) for k, e, *_ in drawn}   # 12 frames: no window
    assert len({s for *_, s, _, _ in drawn}) > 1


def test_evaluation_order_batching_and_rollout_windows(release):
    ds = _dataset(release, training=False)
    ds.batch_size = 2
    first, again = list(ds), list(ds)
    assert [b.source_shard for b in first] == [
        "objects/Kitchen_Table/mug/asset_0001.npz", "objects/libero/basket/asset_0003.npz",
        "objects/Kitchen_Table/mug/asset_0002.npz", "objects/Kitchen_Table/mug/asset_0002.npz"][:len(first)]
    assert all(torch.equal(a.positions, b.positions) for a, b in zip(first, again))
    assert sum(b.batch_size for b in first) == 6                      # every episode with 23 frames or more
    ds.future_len = 10_000                                            # the evaluator's rollout convention
    lengths = [(b.source_shard.split("/")[-1], b.positions.shape[1], b.batch_size) for b in ds]
    assert lengths[:3] == [("asset_0001.npz", 27, 1), ("asset_0001.npz", 22, 1), ("asset_0001.npz", 9, 1)]
    assert all(L > 4 for _, L, _ in lengths)


def test_object_files_without_episodes_are_skipped(release, tmp_path):
    _write_object(release["root"], "libero/basket/asset_0004", [], False, 3)    # every episode filtered out upstream
    assert len(load_push_object(release["root"], "objects/libero/basket/asset_0004.npz").starts) == 0
    ds = _dataset(release, training=False)
    assert ds.empty_files == ["objects/libero/basket/asset_0004.npz"]
    assert ds.shards == [o.file for o in ds.objects] and len(ds.objects) == 3
    ds.batch_size = 64
    assert sum(b.batch_size for b in ds) == 6
    lone = str(tmp_path / "lone")
    _write_object(lone, "libero/basket/asset_0004", [], False, 3)
    with pytest.raises(ValueError, match="holds an episode"):
        UnifiedRLPushDataset(lone, split="all", normalization=release["spec"])


def test_optional_rigid_normalization_entry(release, tmp_path):
    frozen = resolve_normalization("xy_p99_per_domain")
    assert frozen["sha256"] == FROZEN_SNAPSHOT_SHA and "rl_push" not in frozen["scales"]["pose"]
    with pytest.raises(ValueError, match="no 'rl_push' entry"):
        domain_normalizers("xy_p99_per_domain", "rl_push")
    base, art = load_normalization_artifact(DEFAULT_ARTIFACT), load_normalization_artifact(release["artifact"])
    assert art["profiles"]["xy_p99_global"] == base["profiles"]["xy_p99_global"]
    for kind in ("particle", "pose"):
        assert {d: v for d, v in art["profiles"]["xy_p99_per_domain"][kind].items() if d != "rl_push"} == \
            base["profiles"]["xy_p99_per_domain"][kind]
    delta, _, pose = domain_normalizers(release["spec"], "rl_push")
    assert torch.allclose(delta.max_val, torch.tensor([0.045, 0.045, 0.019]))
    assert torch.allclose(pose.max_val, torch.tensor([0.04, 0.04, 0.018, 0.25, 0.25, 0.3]))
    glob = domain_normalizers({"profile": "xy_p99_global", "artifact": release["artifact"]}, "rl_push")[2]
    assert torch.equal(glob.max_val, domain_normalizers("xy_p99_global", "pickup")[2].max_val)
    for drop in ("particle", "pose"):                                   # an opt-in rigid entry needs both scales
        broken = json.load(open(release["artifact"]))
        del broken["profiles"]["xy_p99_per_domain"][drop]["rl_push"]
        path = tmp_path / f"broken_{drop}.json"
        json.dump(broken, open(path, "w"))
        with pytest.raises(ValueError, match="domains must contain exactly"):
            load_normalization_artifact(str(path))


def test_rl_push_is_opt_in_with_config_and_roots_checks(release, tmp_path):
    assert selected_domains(None)[0] == DOMAIN_ORDER and selected_domains(["rl_push", "pickup"])[0] == ("pickup", "rl_push")
    assert exclusion_files("rl_push", "train") == exclusion_files("rl_push", "val") == [RL_PUSH_EXCLUSIONS, RL_PUSH_VAL_ASSETS]
    base = ["run_name=t", "data.domains=[rl_push]"]
    with pytest.raises(ValueError, match="rl_push entry"):
        load_config("configs/unet.yaml", base)
    cfg = load_config("configs/unet.yaml", base + [f"normalization.artifact={release['artifact']}"])
    assert "rl_push" in cfg.normalization["scales"]["pose"]
    load_config("configs/transformer.yaml", ["run_name=t"])             # four-domain recipes are unaffected

    other = str(tmp_path / "other")
    os.makedirs(other)
    with pytest.raises(ValueError, match="at most one as val"):
        resolve_data_roots({"rl_push": {"train": release["root"], "val": [other, str(tmp_path)]}})
    no_list = resolve_data_roots({"rl_push": {"train": release["root"]}})
    with pytest.raises(ValueError, match="validation object list"):
        build_domain_dataset("rl_push", split="val", training=False, context_len=4, future_len=16,
                             normalization=release["spec"], roots=no_list)
    named = resolve_data_roots({"rl_push": {"train": release["root"], "exclusions": {
        "train": [release["val"], release["excl"]], "val": [release["val"], release["excl"]]}}})
    ds = build_domain_dataset("rl_push", split="val", training=False, context_len=4, future_len=16,
                              normalization=release["spec"], roots=named)
    assert isinstance(ds, UnifiedRLPushDataset) and ds.roots_override and [o.asset_key for o in ds.objects] == \
        ["Kitchen_Table/mug/asset_0002"] and ds.blocklist_paths == (release["val"], release["excl"])


def test_unified_loader_yields_rl_push_batches(release):
    roots = resolve_data_roots({"rl_push": {"train": release["root"], "exclusions": {
        "train": [release["val"], release["excl"]], "val": [release["val"], release["excl"]]}}})
    loader = create_unified_dataloader(domains=["rl_push"], split="train", batch_size=2, num_workers=0,
                                       emit_sidecar=True, articulated=True, future_len=16, context_len=4,
                                       normalization=release["spec"], roots=roots)
    batch = next(iter(loader))
    assert batch.domain == "rl_push" and tuple(batch.positions.shape) == (2, 20, N_GRIPPER + N_OBJECT, 3)
    assert tuple(batch.pose_deltas_gt.shape) == (2, 16, 4, 6) and reconstruction_error(batch) < 1e-5


def test_evaluation_receipts_name_the_release_manifests_and_object_files(release):
    from scripts.unified.eval_unified import _input_provenance
    from utils.evaluation import iter_clips, read_provenance
    ds = _dataset(release, split="val", manifests=(release["val"], release["excl"]), training=False)
    inputs = _input_provenance(ds, resolve_normalization(release["spec"]))
    assert inputs["roots"] == [release["root"]] and set(inputs["blocklists_sha256"]) == {release["val"], release["excl"]}
    sources = {}
    for batch, source in iter_clips(ds, None):
        sources[source] = sources.get(source, 0) + batch.batch_size
    assert read_provenance(sources, ds.shards) == {"shards_declared": 1, "shards_read": 1,
                                                   "clips_per_shard_read": {"objects/Kitchen_Table/mug/asset_0002.npz": 2}}


def test_exclusion_rules_cover_settle_kicks_and_contact_launches(tmp_path):
    from scripts.dev.rl_push_manifests import exclusions
    asset, provenance = "libero/basket/asset_0001", [("run_a", "shard_00000.npz", i, i % 2) for i in range(4)]
    np.savez(tmp_path / "stage1.npz", object_file=np.array([f"objects/{asset}.npz"] * 4),
             **{name: np.array([p[i] for p in provenance]) for i, name in
                enumerate(("source_run", "source_shard", "source_episode_index", "source_env"))},
             airborne_max_mm=np.zeros(4), bottom_min_mm=np.zeros(4))
    json.dump([{"file": f"objects/{asset}.npz", "source_run": p[0], "source_shard": p[1], "source_episode_index": p[2],
                "source_env": p[3], "max_penetration_mm": 0.5, "onset_before_contact_possible": False,
                "push_onset_frame": 12, "unexplained_later_frames": [], "unexplained_later_max_gain_mps": 0.0,
                "unexplained_settle_frames": settle} for p, settle in zip(provenance, ([2], [3], [], []))],
              open(tmp_path / "stage2.json", "w"))
    launches = {"rows": [{"asset_key": asset, "source_run": p[0], "source_shard": p[1], "source_episode_index": p[2],
                          "source_env": p[3], "max_ratio": ratio, "frame": 20, "object_mps": 0.8, "gripper_mps": 0.3}
                         for p, ratio in zip(provenance, (1.0, 1.0, 2.5, 1.9))]}
    json.dump(launches, open(tmp_path / "launches.json", "w"))
    paths = [str(tmp_path / name) for name in ("stage1.npz", "stage2.json", "launches.json")]
    rules = dict(airborne_mm=20.0, sink_mm=5.0, penetration_mm=5.0, gain_mps=0.1, kick_from_frame=3, launch_ratio=2.0)
    episodes, scanned = exclusions(*paths, **rules)
    assert scanned == 4 and [e["source_episode_index"] for e in episodes] == [1, 2]   # frame-2 kick and 1.9x stay
    assert episodes[0]["values"]["settle_kick_frames"] == [3] and "frame 3 or later" in episodes[0]["reasons"][0]
    assert episodes[1]["values"]["launch_ratio"] == 2.5 and "faster than the gripper" in episodes[1]["reasons"][0]
    launches["rows"].pop()
    json.dump(launches, open(tmp_path / "launches.json", "w"))
    with pytest.raises(ValueError, match="launch check"):
        exclusions(*paths, **rules)


@pytest.mark.skipif(not os.path.isfile(os.path.join(RL_PUSH_ROOT, "reconstruct.py")),
                    reason="the RL push release is not readable on this machine")
def test_real_release_matches_the_owners_reconstruction():
    sys.path.insert(0, RL_PUSH_ROOT)
    bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True      # never write into the release folder
    try:
        import reconstruct as owner
    finally:
        sys.path.remove(RL_PUSH_ROOT)
        sys.dont_write_bytecode = bytecode
    rel = next(r for r in sorted(os.path.relpath(p, RL_PUSH_ROOT) for p in
                                 __import__("glob").glob(f"{RL_PUSH_ROOT}/objects/**/*.npz", recursive=True))
               if len(np.load(os.path.join(RL_PUSH_ROOT, r))["episode_start"]))
    obj = load_push_object(RL_PUSH_ROOT, rel)
    e = int(np.argmax(obj.lengths))
    links = {k: v.astype(np.float64) for k, v in dict(np.load(GRIPPER_LINKS)).items()}
    rows = np.arange(N_OBJECT)
    points, _ = window_points(obj, e, RL_PUSH_FIRST_START, 20, links, rows)
    raw = owner.load_object(os.path.join(RL_PUSH_ROOT, rel))
    ep = owner.episode(raw, e)
    frames = slice(RL_PUSH_FIRST_START, RL_PUSH_FIRST_START + 20)
    grip = owner.gripper_cloud(ep["hand_pos"][frames], ep["hand_quat_wxyz"][frames], ep["joint_pos"][frames, 7:9],
                               links=links)
    body = owner.object_cloud(raw["object_cloud_outer_surface"][rows], ep["object_pos"][frames], ep["object_quat_wxyz"][frames])
    expected = np.concatenate([grip, body], axis=1) + np.asarray(RL_PUSH_SHIFT, np.float32)
    assert np.allclose(points, expected, atol=1e-5)
