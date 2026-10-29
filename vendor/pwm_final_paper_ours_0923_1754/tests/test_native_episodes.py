"""Native production episodes -> pose-native TrajectoryBatch windows.

A synthetic episode in the native NPZ schema (parts stored non-contiguously,
rigid motion generated from known poses) checks the layout contract, the
template@pose reconstruction and the window round trip. The real sample episodes
are checked when present.
"""

import os

import numpy as np
import pytest
import torch

from data.native_episodes import (GRIPPER_PARTS, RIGID_FILL_E, load_native_episode,
                                  native_window_batch, reconstruction_error, select_windows)
from data.unified_dataset import MAT_GRIPPER, MAT_RIGID
from utils.pose_ops import so3_exp

SAMPLES = "/data/user_data/robertre/yikai_native_0910"


def _synthetic_episode(path, T=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    names = list(GRIPPER_PARTS) + ["mug_0001", "table", "can_0002"]
    counts = [40, 24, 24, 8, 8, 64, 64, 64]
    parts, pids, motion = [], [], {}
    for k, (name, P) in enumerate(zip(names, counts)):
        tmpl = torch.rand(P, 3, generator=g) * 0.05
        base = torch.tensor([0.4, 0.0, 0.2]) + 0.15 * k * torch.tensor([1.0, 0.3, 0.0])
        if name == "table":
            w, v = torch.zeros(3), torch.zeros(3)                     # static
        elif name == "can_0002":
            w, v = torch.zeros(3), torch.tensor([0.0, 0.0, 0.0])      # untouched distractor
        elif name.endswith("finger_tip"):                             # a region of its finger: same rigid motion
            w, v, base = motion[name.replace("_tip", "")]
            tmpl = tmpl + torch.tensor([0.0, 0.0, -0.03])             # offset inside the finger frame
        else:
            w, v = 0.02 * torch.randn(3, generator=g), 0.004 * torch.randn(3, generator=g)
        motion[name] = (w, v, base)
        frames = [(so3_exp(w * f) @ tmpl.T).T + base + v * f for f in range(T)]
        parts.append(torch.stack(frames)); pids += [k] * P
    pos = torch.cat(parts, dim=1).numpy().astype(np.float32)            # [T, N, 3]
    pid = np.array(pids, dtype=np.int32)
    # store points in a scrambled order, as the native files do
    order = np.random.RandomState(seed).permutation(pos.shape[1])
    np.savez(path, **{"state/particles": pos[:, order], "meta/particles/part_ids": pid[order],
                      "meta/particles/part_names": np.array(names), "meta/particles/part_counts": np.array(counts),
                      "meta/spatial/src_name": np.array("mug_0001")})
    return pos, pid, order, names


def test_synthetic_layout_reconstruction_and_window(tmp_path):
    path = str(tmp_path / "trajectory.npz")
    pos, pid, order, names = _synthetic_episode(path)
    ep = load_native_episode(path, shift=(0.0, 0.0, 0.0))
    # contiguous by part, permutation recorded, gripper/object ids as pickup
    assert torch.equal(ep.part_index, torch.from_numpy(np.sort(pid)).long())
    assert np.array_equal(pid[order][ep.permutation], np.sort(pid))          # permutation indexes the stored order
    assert torch.equal(ep.positions, torch.from_numpy(pos[:, order][:, ep.permutation]))
    o = ep.object_ids
    assert int((o[1:] != o[:-1]).sum()) + 1 == len(o.unique())            # one run per id
    assert set(o[ep.gripper].tolist()) == {1} and set(o[ep.part_mask(ep.src_part)].tolist()) == {0}
    assert ep.part_names[ep.src_part] == "mug_0001"
    assert ep.part_names.index("table") not in ep.moving_parts and ep.part_names.index("can_0002") not in ep.moving_parts

    starts = select_windows(ep, 2, 3, 6)
    assert len(starts) == 2 and all(abs(a - b) >= 9 for a in starts for b in starts if a != b)
    b = native_window_batch(ep, starts, context_len=3, future_len=6)
    B, L, N = 2, 9, ep.num_particles
    assert b.positions.shape == (B, L, N, 3) and b.actions.shape == (B, 6, N, 3)
    assert torch.equal(b.positions[0], ep.positions[starts[0]:starts[0] + L])
    assert bool((b.mat_type[0][b.action_mask[0]] == MAT_GRIPPER).all()) and bool((b.mat_type[0][~b.action_mask[0]] == MAT_RIGID).all())
    assert float(b.E.min()) == RIGID_FILL_E and bool((b.valid_lengths == N).all())
    assert reconstruction_error(b) < 1e-5
    # actions are the achieved gripper motion, zero on passive rows; pose actions only on gripper bodies
    d = b.positions[:, 1:] - b.positions[:, :-1]
    torch.testing.assert_close(b.actions[:, :, b.action_mask[0]], d[:, 2:, b.action_mask[0]])
    assert torch.count_nonzero(b.actions[:, :, ~b.action_mask[0]]) == 0
    gripper_bodies = [k for k, n in enumerate(ep.part_names) if n in GRIPPER_PARTS]
    assert b.pose_action_mask[0].nonzero().flatten().tolist() == gripper_bodies
    assert bool((b.body_type == 1).all()) and b.body_poses.shape == (B, L, len(names), 4, 4)
    # a static body has zero pose deltas
    table = ep.part_names.index("table")
    assert float(b.pose_deltas_gt[:, :, table].abs().max()) < 1e-6

    dropped = load_native_episode(path, shift=(0.0, 0.0, 0.0), drop_static=True)
    assert "table" not in dropped.part_names and "can_0002" not in dropped.part_names and "mug_0001" in dropped.part_names
    assert dropped.num_particles == N - 128


def test_pickup_layout_is_exactly_the_pickup_corpus_layout(tmp_path):
    path = str(tmp_path / "trajectory.npz")
    pos, pid, order, names = _synthetic_episode(path)
    ep = load_native_episode(path, shift=(0.0, 0.0, 0.0), pickup_layout=True)
    # four bodies: object 0, hand 1, left finger 2 (finger + tip), right finger 3 (finger + tip)
    assert ep.part_names == ["mug_0001", "panda_hand", "panda_leftfinger", "panda_rightfinger"]
    assert ep.num_particles == 64 + 40 + (24 + 8) + (24 + 8)
    assert [int((ep.part_index == k).sum()) for k in range(4)] == [64, 40, 32, 32]
    # gripper first (object id 1), object last (object id 0), one run each, as UnifiedPickupDataset
    o = ep.object_ids
    assert o[:104].tolist() == [1] * 104 and o[104:].tolist() == [0] * 64
    assert ep.gripper.sum() == 104 and ep.src_part == 0
    b = native_window_batch(ep, select_windows(ep, 1, 3, 6), context_len=3, future_len=6)
    assert b.body_type.shape == (1, 4) and bool((b.body_type == 1).all())
    assert b.pose_action_mask[0].tolist() == [False, True, True, True]
    assert reconstruction_error(b) < 1e-5
    # the kept points are the stored points of those parts, by name
    stored = pos[:, order]; stored_pid = pid[order]
    src = names.index("mug_0001")
    assert torch.equal(ep.positions[:, ep.part_index == 0], torch.from_numpy(stored[:, stored_pid == src]))


def test_scene_layout_keeps_every_object_and_feeds_fixtures_as_zero_motion_input(tmp_path):
    path = str(tmp_path / "trajectory.npz")
    pos, pid, order, names = _synthetic_episode(path)
    ep = load_native_episode(path, shift=(0.0, 0.0, 0.0), scene_layout=True)
    # body 0 = manipulated object, 1..3 = the pickup gripper links, then the distractor, fixtures last
    assert ep.part_names == ["mug_0001", "panda_hand", "panda_leftfinger", "panda_rightfinger", "can_0002", "table"]
    assert ep.num_particles == 64 + 40 + 32 + 32 + 64 + 64
    assert ep.gripper.sum() == 104 and ep.kinematic.sum() == 64 and ep.src_part == 0
    assert bool((ep.part_index[ep.kinematic] == 5).all())
    b = native_window_batch(ep, select_windows(ep, 1, 3, 6), context_len=3, future_len=6)
    assert b.body_type.shape == (1, 6) and bool((b.body_type == 1).all())
    # gripper and table are input (actuated); the objects are predicted
    assert b.pose_action_mask[0].tolist() == [False, True, True, True, False, True]
    assert bool((b.action_mask[0] == (ep.gripper | ep.kinematic)).all())
    assert float(b.pose_actions[0, :, 5].abs().max()) == 0.0 and float(b.actions[0][:, ep.kinematic].abs().max()) == 0.0
    # material: gripper links only are MAT_GRIPPER; table and objects are MAT_RIGID (the pushing rod convention)
    assert bool((b.mat_type[0][ep.gripper] == MAT_GRIPPER).all()) and bool((b.mat_type[0][~ep.gripper] == MAT_RIGID).all())
    # object ids: gripper links share 1, every other body has its own (object 0, can 2, table 3)
    assert sorted(set(b.object_ids[0].tolist())) == [0, 1, 2, 3] and reconstruction_error(b) < 1e-5
    with pytest.raises(ValueError):
        load_native_episode(path, pickup_layout=True, scene_layout=True)


def test_scene_layout_can_drop_the_fixtures(tmp_path):
    """keep_fixtures=False: every object stays, the table's points are gone, nothing is kinematic."""
    path = tmp_path / "ep.npz"
    _synthetic_episode(path)
    full = load_native_episode(str(path), scene_layout=True)
    ep = load_native_episode(str(path), scene_layout=True, keep_fixtures=False)
    assert "table" in full.part_names and "table" not in ep.part_names
    assert set(ep.part_names) == set(full.part_names) - {"table"}
    assert ep.num_particles == full.num_particles - int((full.part_index == full.part_names.index("table")).sum())
    assert not bool(ep.kinematic.any())
    b = native_window_batch(ep, [0], context_len=4, future_len=8)
    assert reconstruction_error(b) < 1e-5


@pytest.mark.skipif(not os.path.isdir(SAMPLES), reason="native sample episodes not on this machine")
def test_real_samples_reconstruct():
    for name in ("native_episode_02", "native_object_episode", "native_shelf_episode"):
        ep = load_native_episode(os.path.join(SAMPLES, name, "trajectory.npz"))
        b = native_window_batch(ep, select_windows(ep, 2, 4, 38))
        assert reconstruction_error(b) < 1e-5, name
        assert b.positions.shape[1] == 42 and bool((b.valid_lengths == ep.num_particles).all())
        pk = load_native_episode(os.path.join(SAMPLES, name, "trajectory.npz"), pickup_layout=True)
        assert pk.num_particles == 512 + 237 + (159 + 56) * 2 and len(pk.part_names) == 4, name
        pb = native_window_batch(pk, select_windows(pk, 2, 4, 38))
        assert reconstruction_error(pb) < 1e-5, name
        sc = load_native_episode(os.path.join(SAMPLES, name, "trajectory.npz"), scene_layout=True)
        assert sc.num_particles == ep.num_particles and len(sc.part_names) == len(ep.part_names) - 2, name
        sb = native_window_batch(sc, select_windows(sc, 2, 4, 16), future_len=16)
        assert reconstruction_error(sb) < 1e-5 and float(sb.pose_actions[0][:, sc.part_index[sc.kinematic].unique()].abs().max()) == 0.0, name


def test_clamp_actuated_pose_materializes_given_bodies_exactly(tmp_path):
    """With the clamp, actuated bodies (gripper links, zero-motion fixtures) come out
    exactly as their input; passive bodies keep the network's prediction."""
    from utils.pose_native import clamp_actuated_pose, materialize_pose_native
    path = str(tmp_path / "trajectory.npz")
    _synthetic_episode(path)
    ep = load_native_episode(path, shift=(0.0, 0.0, 0.0), scene_layout=True)
    b = native_window_batch(ep, select_windows(ep, 2, 3, 6), context_len=3, future_len=6)
    ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(0)
    x0pose = torch.randn(b.batch_size, 6, len(ridx), 6, generator=g)          # a "prediction" in normalized units
    clamped = clamp_actuated_pose(b, x0pose)
    act = b.pose_action_mask[0, ridx]
    assert torch.equal(clamped[:, :, ~act], x0pose[:, :, ~act])                # passive untouched
    assert not torch.equal(clamped[:, :, act], x0pose[:, :, act])
    x0p = torch.zeros_like(b.future_deltas) if hasattr(b, "future_deltas") and b.future_deltas is not None else None
    pred_pos, R, T, rid = materialize_pose_native(b, x0p, clamped)
    err = (pred_pos - b.future).norm(dim=-1)                                   # [B, F, N]
    given = (b.action_mask[0]).nonzero(as_tuple=True)[0]
    assert float(err[:, :, given].max()) < 1e-4, float(err[:, :, given].max())  # given bodies exact (m)
    fixture = ep.kinematic.nonzero(as_tuple=True)[0]
    assert float((pred_pos[:, :, fixture] - b.positions[:, b.context_len - 1:b.context_len, fixture]).abs().max()) < 1e-4


@pytest.mark.skipif(not os.path.isdir(SAMPLES), reason="native sample episodes not on this machine")
def test_real_samples_articulate_the_fingers_on_their_joints():
    """Scene layout with articulated=True: fingers are prismatic children of the hand with axes from
    the recorded hand pose, and every rigid point rebuilds from the pose channels."""
    from utils.pose_native import JOINT_PRISMATIC, materialize_pose_native, pose_channel_mask
    for name in ("native_episode_02", "native_object_episode", "native_shelf_episode"):
        sc = load_native_episode(os.path.join(SAMPLES, name, "trajectory.npz"), scene_layout=True)
        b = native_window_batch(sc, select_windows(sc, 2, 4, 16), future_len=16, articulated=True)
        assert b.body_parent[0, 2:4].tolist() == [1, 1] and b.joint_type[0, 2:4].tolist() == [JOINT_PRISMATIC] * 2
        ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
        mask = pose_channel_mask(b, ridx)
        assert float((b.pose_deltas_gt[:, :, ridx] * (1 - mask)).abs().max()) == 0.0, name
        pred, *_ = materialize_pose_native(b, None, b.pose_delta_stats.normalize(b.pose_deltas_gt[:, :, ridx], clip=False))
        assert float((pred - b.future).norm(dim=-1).max()) < 5e-5, name


def _rotvec_to_wxyz(w):
    angle = float(w.norm())
    axis = w / angle if angle > 0 else torch.tensor([1.0, 0.0, 0.0])
    return torch.cat([torch.tensor([np.cos(angle / 2)], dtype=torch.float32), axis * float(np.sin(angle / 2))])


def test_shared_gripper_swaps_the_recording_rows_for_the_pickup_link_points(tmp_path):
    """One Franka gripper, one point set: with shared_gripper the three gripper bodies are the pickup corpus's
    241 + 212 + 218 link points under the RECORDED link poses; every object keeps the recording's own points."""
    from data.native_episodes import PICKUP_LAYOUT_NAMES, shared_gripper_points
    path = str(tmp_path / "trajectory.npz")
    pos, pid, order, names = _synthetic_episode(path)
    T = pos.shape[0]
    g = torch.Generator().manual_seed(5)
    poses = torch.zeros(T, len(names), 7); poses[..., 3] = 1
    expected = {}
    for link in PICKUP_LAYOUT_NAMES:
        w, v, base = 0.03 * torch.randn(3, generator=g), 0.005 * torch.randn(3, generator=g), torch.rand(3, generator=g)
        for f in range(T):
            poses[f, names.index(link), :3] = base + v * f
            poses[f, names.index(link), 3:] = 2.5 * _rotvec_to_wxyz(w * f)          # not unit length: normalized on read
        local = torch.from_numpy(shared_gripper_points()[link])
        expected[link] = torch.stack([(so3_exp(w * f) @ local.T).T + base + v * f for f in range(T)])
    with np.load(path) as z:
        stored = {k: z[k] for k in z.files}
    np.savez(path, **stored, **{"state/particle_part_poses_wxyz": poses.numpy()})

    assert [len(p) for p in shared_gripper_points().values()] == [241, 212, 218]
    plain = load_native_episode(path, shift=(0.0, 0.0, 0.0), scene_layout=True)
    ep = load_native_episode(path, shift=(0.1, 0.0, -0.2), scene_layout=True, shared_gripper=True)
    assert ep.part_names == plain.part_names and int(ep.gripper.sum()) == 671
    assert [int((ep.part_index == k).sum()) for k in (1, 2, 3)] == [241, 212, 218]
    assert ep.part_index[:671].tolist() == [1] * 241 + [2] * 212 + [3] * 218 and set(ep.object_ids[:671].tolist()) == {1}
    shift = torch.tensor([0.1, 0.0, -0.2])
    for k, link in enumerate(PICKUP_LAYOUT_NAMES, start=1):
        assert torch.allclose(ep.positions[:, ep.part_index == k], expected[link] + shift, atol=2e-6)
    # the objects and the fixture: the recording's own rows, untouched; the swapped rows point at no stored point
    assert torch.allclose(ep.positions[:, ~ep.gripper], plain.positions[:, ~plain.gripper] + shift, atol=1e-7)
    assert bool((ep.permutation[:671] == -1).all()) and np.array_equal(ep.permutation[671:], plain.permutation[int(plain.gripper.sum()):])
    b = native_window_batch(ep, [2], context_len=3, future_len=6, articulated=False)
    assert int(b.action_mask[0].sum()) == 671 + int(ep.kinematic.sum()) and reconstruction_error(b) < 1e-5
    assert torch.allclose(b.actions[0][:, :671], ep.positions[5:11, :671] - ep.positions[4:10, :671], atol=1e-7)
    pick = load_native_episode(path, shift=(0.0, 0.0, 0.0), pickup_layout=True, shared_gripper=True)
    assert pick.num_particles == 671 + 64 and torch.allclose(pick.positions[:, :671], ep.positions[:, :671] - shift, atol=1e-7)

    with pytest.raises(ValueError, match="pickup or the scene layout"):
        load_native_episode(path, shared_gripper=True)
    bare = str(tmp_path / "bare.npz")
    np.savez(bare, **stored)
    with pytest.raises(ValueError, match="recorded poses"):
        load_native_episode(bare, scene_layout=True, shared_gripper=True)
