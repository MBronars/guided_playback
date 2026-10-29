"""Physical-unit windows from the September 2026 indexed collections.

No simulator imports. The index chooses an episode, and ``start`` chooses a
window in that episode's declared saved clock. Targets always difference the
recorded positions; only MPM drag uses commands instead of achieved actions.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import pickle
import tarfile

import numpy as np
import torch

from data.batch import TrajectoryBatch
from data.normalization import NormalizationStats, domain_normalizers
from data.npz_arrays import npz_array
from data.unified_dataset import (
    CANONICAL_TRANSLATION, CLOTH_FILL_E, CLOTH_FILL_NU, CLOTH_FILL_RHO,
    MAT_CLOTH, MAT_GRIPPER, MAT_RIGID, RIGID_FILL_E, RIGID_FILL_NU,
    _articulate_fingers, _augment_pose_channels, _rigid_sidecar_from_particles,
    unified_position_stats,
)


def normalizers(domain, normalization=None, raw=False):
    if not raw:
        return domain_normalizers(normalization, domain)
    from data.normalization import PASSIVE_DOMAINS
    identity = lambda n: NormalizationStats(-torch.ones(n), torch.ones(n))
    return identity(3), unified_position_stats(), None if domain in PASSIVE_DOMAINS else identity(6)


def _tensor(array, dtype=None):
    # A memmap slice can be read-only; never hand a writable torch view of it to
    # augmentation or collation code.
    return torch.tensor(np.asarray(array), dtype=dtype)


@lru_cache(maxsize=2)
def _tar_sample(path, key, size, mtime_ns):
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (size, mtime_ns):
        raise ValueError(f"Source shard changed since indexing: {path}")
    sample = {}
    required = {"particle_pos.pyd", "valid_particles.pyd", "material_properties.pyd",
                "object_ids.pyd", "grab_velocity.pyd", "is_grabbed.pyd", "drag_active.pyd", "provenance.json"}
    # These are trusted local WebDataset pickle members, the same serialization
    # used by the existing MPM loader. Ignore velocities and unrelated payloads.
    with tarfile.open(path) as archive:
        for member in archive:
            if member.name.startswith(key + "."):
                suffix = member.name[len(key) + 1:]
                if suffix in required:
                    with archive.extractfile(member) as stream:
                        sample[suffix] = json.load(stream) if suffix.endswith(".json") else pickle.load(stream)
    if set(sample) != required:
        raise ValueError(f"Incomplete indexed MPM sample {path}:{key}: {required - set(sample)}")
    return sample


def mpm_window(record, start, context_len, future_len, stats, domain, articulated=False):
    sample = _tar_sample(record["file"], record["key"], record["size"], record["mtime_ns"])
    row, length = record["row"], context_len + future_len
    provenance = sample["provenance.json"][row]
    if provenance["episode_id"] != record["episode_id"]:
        raise ValueError("MPM provenance episode identity differs from the index")
    times = np.asarray(provenance["state_time_s"])
    if len(times) != record["frames"]:
        raise ValueError("MPM episode length changed since indexing")
    if not np.allclose(np.diff(times), .0417, rtol=0, atol=1e-12):
        raise ValueError("Fresh MPM requires its recorded 0.0417 s clock")
    pos = _tensor(sample["particle_pos.pyd"][row:row + 1, start:start + length], torch.float32)
    valid = _tensor(sample["valid_particles.pyd"][row:row + 1, start:start + length], torch.bool)
    if not valid.all():
        raise ValueError("Indexed fresh MPM requires reviewed, unpadded persistent particles")
    pos += torch.tensor(CANONICAL_TRANSLATION["mpm"])
    material = _tensor(sample["material_properties.pyd"][row:row + 1], torch.float32)
    ids = _tensor(sample["object_ids.pyd"][row:row + 1], torch.long)
    n = pos.shape[2]
    gates = {}
    if domain == "mpm_drag":
        from data.mpm_drag import commands
        delta, active, selected = commands(sample)
        selected = selected[row:row + 1]
        actions = delta[row:row + 1, start + context_len - 1:start + length - 1].clone()
        gates = dict(action_active=active[row:row + 1, start + context_len - 1:start + length - 1].clone(),
                     context_action_active=active[row:row + 1, start:start + context_len - 1].clone())
    else:
        from collection.mpm.contracts import require_passive_consumer_input
        require_passive_consumer_input(sample)
        selected = torch.zeros((1, n), dtype=torch.bool)
        actions = torch.zeros((1, future_len, n, 3))
    return TrajectoryBatch(pos, actions, selected, ids, torch.tensor([n]),
        material[..., 0], material[..., 1], material[..., 2], mat_type=material[..., 3].long(),
        context_len=context_len, delta_stats=stats[0], position_stats=stats[1], domain=domain, **gates)


def mpm_motion(record):
    """Mean per-particle step per transition [T-1] in metres; every MPM particle is a prediction target."""
    sample = _tar_sample(record["file"], record["key"], record["size"], record["mtime_ns"])
    pos = np.asarray(sample["particle_pos.pyd"][record["row"]], dtype=np.float32)
    return np.linalg.norm(np.diff(pos, axis=0), axis=-1).mean(axis=1)


def folding_motion(record):
    """Mean cloth-vertex step per transition [T-1] in metres; gripper points are inputs, not targets."""
    arrays = _folding_arrays(record["file"], record["mtime_ns"])
    nc = int(arrays["n_cloth"])
    cloth = arrays["particle_pos"][:, :nc]
    keep = _folding_cloth_ids(record, nc)
    if keep is not None:
        cloth = cloth[:, keep]
    return np.linalg.norm(np.diff(cloth, axis=0), axis=-1).mean(axis=1)


@lru_cache(maxsize=8)
def _folding_subset(path, mtime_ns):
    """Retained cloth vertex ids of a garment (a release's `subset_file`, sorted, one file per garment shared
    by all its clips). The loader checks the file's identity against the index on every read."""
    if Path(path).stat().st_mtime_ns != mtime_ns:
        raise ValueError(f"Folding subset changed: {path}")
    with np.load(path, allow_pickle=False) as z:
        idx = np.asarray(z["indices"], dtype=np.int64)
    if idx.ndim != 1 or len(idx) == 0 or idx[0] < 0 or np.any(np.diff(idx) <= 0):
        raise ValueError(f"Folding subset must be sorted unique non-negative vertex ids: {path}")
    return idx


def _folding_cloth_ids(record, n_cloth):
    """The cloth vertices a record keeps: None for every vertex, else the garment's retained subset
    (pwm-data-v1/folding_8mm: farthest-point prefix to an 8 mm radius, at most 10,000, decision 2026-09-18)."""
    if "subset_file" not in record:
        return None
    idx = _folding_subset(record["subset_file"], record["subset_mtime_ns"])
    if idx[-1] >= n_cloth:
        raise ValueError("Folding subset names a vertex beyond the garment")
    return idx


@lru_cache(maxsize=1)
def _folding_arrays(path, mtime_ns):
    if Path(path).stat().st_mtime_ns != mtime_ns:
        raise ValueError(f"Folding source changed: {path}")
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    nc = int(arrays["n_cloth"])
    p = arrays["particle_pos"]
    if (p.dtype != np.float32 or p.ndim != 3 or p.shape[1:] != (nc + 2000, 3)
            or not arrays["valid_particles"].all()
            or not np.array_equal(arrays["obj_ids"], np.repeat([0, 1, 2], [nc, 1000, 1000]))
            or not np.array_equal(arrays["ee_link_ids"], np.repeat([0, 1, 2], [200, 400, 400]))
            or not np.isclose(float(arrays["dt"]), 1 / 30)
            or not np.array_equal(arrays["source_state_index"], np.arange(30, 30 + len(p)))
            or not np.allclose(arrays["state_time_s"], arrays["source_state_index"] / 30, rtol=0, atol=1e-8)):
        raise ValueError(f"Invalid SIM1 30 Hz candidate export: {path}")
    return arrays


def folding_window(record, start, context_len, future_len, stats, domain, articulated=False):
    arrays = _folding_arrays(record["file"], record["mtime_ns"])
    if len(arrays["particle_pos"]) != record["frames"]:
        raise ValueError("Folding episode length changed since indexing")
    nc = int(arrays["n_cloth"])
    p = arrays["particle_pos"][start:start + context_len + future_len]
    keep = _folding_cloth_ids(record, nc)
    if keep is not None:                                   # retained cloth vertices, then every gripper point
        p = np.concatenate([p[:, keep], p[:, nc:]], axis=1)
        nc = len(keep)
    pos = _tensor(p[None], torch.float32)
    n = pos.shape[2]
    if "particles" in record and n != int(record["particles"]):
        raise ValueError("Folding point count differs from the index record")
    ids = torch.from_numpy(np.repeat([0, 1, 2], [nc, 1000, 1000]))[None]
    act = ids > 0
    pb = torch.cat([torch.zeros(nc, dtype=torch.long),
                    _tensor(arrays["ee_link_ids"], torch.long) + 1,
                    _tensor(arrays["ee_link_ids"], torch.long) + 4])[None]
    sc = _rigid_sidecar_from_particles(pos, pb, 7, rigid_bodies=range(1, 7))
    sc["particle_body"] = pb
    if articulated:
        # Points in the retained per-arm templates are in the Panda link frames.
        # Kabsch maps the hand-local opening axes into our frame-0-centred body.
        from utils.rigid_ops import kabsch_batched
        with np.load(record["template_file"], allow_pickle=False) as template:
            if not np.array_equal(template["link_ids"], arrays["ee_link_ids"]):
                raise ValueError("Folding template link membership mismatch")
            axes = []
            for side, hand in (("left", 1), ("right", 4)):
                local = _tensor(template[f"{side}_local_points"][:200], torch.float32)[None]
                R, _ = kabsch_batched(local, pos[:, 0, pb[0] == hand])
                axes.append(torch.einsum("bij,fj->bfi", R, torch.tensor([[0., 1., 0.], [0., -1., 0.]])))
        sc = _articulate_fingers(sc, [(1, 2), (1, 3), (4, 5), (4, 6)], torch.cat(axes, 1))
    sc = _augment_pose_channels(sc, act, context_len, stats[2])
    return TrajectoryBatch(pos, (pos[:, context_len:] - pos[:, context_len - 1:-1]) * act[:, None, :, None],
        act, ids, torch.tensor([n]), torch.where(act, RIGID_FILL_E, CLOTH_FILL_E),
        torch.full((1, n), CLOTH_FILL_NU), torch.full((1, n), CLOTH_FILL_RHO),
        mat_type=torch.where(act, MAT_GRIPPER, MAT_CLOTH), context_len=context_len,
        delta_stats=stats[0], position_stats=stats[1], domain=domain, **sc)


# Yikai episodes start with distractors still settling (fastest step in the first ten transitions: median
# 0.66 mm, p95 7 mm; about zero afterwards), so training and evaluation windows begin at frame 10 or later.
# Every other indexed domain starts at frame 0. Decision 2026-09-17.
FIRST_START = {"yikai": 10}
# Yikai's robot is the Franka gripper of rl_push and pickup, so its gripper rows are the same 671 link points
# (data/embodiments/panda_gripper_pickup.npz: hand 241, left finger 212, right finger 218) under the recorded link
# poses, not the recording's own 667-point sampling of the same links. Measured 2026-09-19 on the release: equal
# link frames and boxes (under 1 mm), recorded fingers at (0, +-q, 58.4 mm) in the hand frame within 0.003 mm, the
# two samplings differ by sampling noise only (two-sided Chamfer 6 mm hand, 2 mm fingers). User decision 2026-09-19:
# "make sure the gripper template is the same". Every object, the shift and the window policy are unchanged.
YIKAI_SHARED_GRIPPER = True


def yikai_window(record, start, context_len, future_len, stats, domain, articulated=False):
    from data.native_episodes import (_scene_layout_episode, _with_hand_rotation, native_window_batch,
                                      pose_gripper_links)
    path, end = record["file"], start + context_len + future_len
    read = lambda key: npz_array(path, key)
    source_positions = read("state/particles")
    if len(source_positions) != record["frames"]:
        raise ValueError("Yikai episode length differs from frozen audit/index")
    pos = np.asarray(source_positions[start:end], dtype=np.float32)
    names = [str(n) for n in read("meta/particles/part_names")]
    pid = np.asarray(read("meta/particles/part_ids"), dtype=np.int64)
    if not np.array_equal(np.bincount(pid, minlength=len(names)), read("meta/particles/part_counts")):
        raise ValueError("Yikai part counts differ from particle membership")
    clock = np.asarray(read("state/physics_step")[start:end])
    dt = np.asarray(read("state/physics_dt")[start:end])
    if not np.all(np.diff(clock) == 25) or not np.allclose(dt, .002, rtol=0, atol=1e-12):
        raise ValueError("Yikai adapter requires native 20 Hz contiguous saved states")
    # The table top height varies per episode (-0.05 to 0.18 m in env coordinates), so on top of the domain's
    # fixed xy shift each episode is lowered by its own table height: the table top sits at z = 0 as in every
    # other dataset. The table's points are dropped (keep_fixtures=False), the other bodies all stay. This is
    # the one departure from "one fixed shift per domain"; decisions of 2026-09-17.
    table_top = np.asarray(read("state/table_top_z_env"), dtype=np.float64)
    if table_top.shape != (record["frames"],) or not np.isfinite(table_top).all() or np.ptp(table_top) > 1e-6:
        raise ValueError("Yikai adapter requires one finite table height per episode")
    if "table_top_z_env" in record and abs(float(record["table_top_z_env"]) - float(table_top[0])) > 1e-6:
        raise ValueError("Yikai table height differs from the index record")
    shift = (*CANONICAL_TRANSLATION[domain][:2], float(CANONICAL_TRANSLATION[domain][2]) - float(table_top[0]))
    # The shared native reader builds a torch view. Copy this small pose window
    # so the tensor never refers to the read-only mapped source recording.
    part_poses = np.array(read("state/particle_part_poses_wxyz")[start:end], copy=True)
    links = pose_gripper_links(part_poses, names) if YIKAI_SHARED_GRIPPER else None
    ep = _scene_layout_episode(path, pos, pid, names, str(read("meta/spatial/src_name")), shift, 1e-6,
                               keep_fixtures=False, gripper_links=links)
    ep = _with_hand_rotation(ep, part_poses, names)
    return native_window_batch(ep, [0], context_len=context_len, future_len=future_len,
                               normalization_domain=domain, articulated=articulated, normalizers=stats)


@lru_cache(maxsize=5)
def _dex_geometry(path, size, mtime_ns):
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (size, mtime_ns):
        raise ValueError(f"Dex geometry changed: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def dex_window(record, start, context_len, future_len, stats, domain, articulated=False):
    from utils.pose_ops import quat_wxyz_to_matrix
    from utils.pose_native import JOINT_FREE, JOINT_REVOLUTE
    path, env, stride = record["file"], record["env"], record["stride"]
    geo = _dex_geometry(record["geometry_file"], record["geometry_size"], record["geometry_mtime_ns"])
    names = [str(n) for n in npz_array(path, "link_names")]
    if names != list(geo["link_names"]):
        raise ValueError("Dex geometry and shard link order differ")
    length = int(npz_array(path, "episode_length")[env])
    if stride != 2 or record["frames"] != (length + 1) // 2:
        raise ValueError("Dex requires indexed first-episode length and native stride 2")
    idx = (start + np.arange(context_len + future_len)) * stride
    if idx[-1] >= length:
        raise ValueError("Dex window crosses the first episode termination")
    if float(npz_array(path, "control_dt")) != .025:
        raise ValueError("Dex source clock changed")
    lp = _tensor(npz_array(path, "link_pos")[idx, env], torch.float32)
    lq = _tensor(npz_array(path, "link_quat")[idx, env], torch.float32)
    op = _tensor(npz_array(path, "obj_pos")[idx, env], torch.float32)
    oq = _tensor(npz_array(path, "obj_quat")[idx, env], torch.float32)
    quats = torch.cat([oq[:, None], lq], 1)
    if not torch.isfinite(quats).all() or ((quats.norm(dim=-1) - 1).abs() > 1e-4).any():
        raise ValueError("Dex requires finite unit wxyz quaternions")
    rotations = quat_wxyz_to_matrix(quats)
    origins = torch.cat([op[:, None], lp], 1)
    shape = int(npz_array(path, "episode_shape_id")[env])
    local = [_tensor(geo[f"object_{shape}"], torch.float32)]
    local += [_tensor(geo[f"hand_{k}"], torch.float32) for k in range(len(names))]
    points = [torch.einsum("tij,pj->tpi", rotations[:, k], pts) + origins[:, k, None]
              for k, pts in enumerate(local)]
    pos = torch.cat(points, 1)[None]
    pb = torch.cat([torch.full((len(pts),), k, dtype=torch.long) for k, pts in enumerate(local)])[None]
    n, k = pos.shape[2], len(local)
    act = pb > 0
    sc = _rigid_sidecar_from_particles(pos, pb, k)
    sc["particle_body"] = pb
    if articulated:
        parent = torch.full((1, k), -1, dtype=torch.long)
        kind = torch.full((1, k), JOINT_FREE, dtype=torch.long)
        axis, pivot = torch.zeros(1, k, 3), torch.zeros(1, k, 3)
        for child, p in enumerate(geo["parent"]):
            if p < 0:
                continue
            body, par = child + 1, int(p) + 1
            parent[:, body], kind[:, body] = par, JOINT_REVOLUTE
            # Geometry stores joint axis/origin in the recorded parent's frame;
            # the model's centred body frame has its window-0 world orientation.
            Rp = rotations[0, par]
            axis[0, body] = Rp @ _tensor(geo["axis"][child], torch.float32)
            pivot[0, body] = origins[0, par] + Rp @ _tensor(geo["pivot"][child], torch.float32) - sc["body_poses"][0, 0, par, :3, 3]
        sc.update(body_parent=parent, joint_type=kind, joint_axis=axis, joint_pivot=pivot)
    sc = _augment_pose_channels(sc, act, context_len, stats[2])
    return TrajectoryBatch(pos, (pos[:, context_len:] - pos[:, context_len - 1:-1]) * act[:, None, :, None],
        act, act.long(), torch.tensor([n]), torch.full((1, n), RIGID_FILL_E),
        torch.full((1, n), RIGID_FILL_NU), torch.full((1, n), 750.),
        mat_type=torch.where(act, MAT_GRIPPER, MAT_RIGID), context_len=context_len,
        delta_stats=stats[0], position_stats=stats[1], domain=domain, **sc)


ADAPTERS = {"mpm_fresh": mpm_window, "mpm_drag": mpm_window, "folding": folding_window,
            "yikai": yikai_window, "dex": dex_window}
# Domains whose training windows may be drawn by motion (data.window_motion); the others stay uniform.
MOTION = {"mpm_fresh": mpm_motion, "mpm_drag": mpm_motion, "folding": folding_motion}
