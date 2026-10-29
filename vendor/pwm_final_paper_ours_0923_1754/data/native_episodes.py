"""Native production episodes (schema ``3dpcwm_surface``) -> pose-native TrajectoryBatch windows.

One ``trajectory.npz`` holds a whole episode: ``state/particles`` [T, N, 3] metres in
the environment frame, ``meta/particles/part_ids`` [N] (a rigid part per point:
five Panda links, the manipulated object, distractors, table or shelf), part names
and counts. Points are not stored contiguously by part, so the adapter reorders
them (stable, by part id) and keeps the permutation; the unified backbones need one
contiguous run per object id.

The adapter reuses the pickup adapter's conventions exactly: the pickup canonical
shift, MAT_GRIPPER / MAT_RIGID materials with the rigid fill physics, achieved
per-particle actions on the gripper points, and the rigid sidecar computed by the
same Kabsch path from the particles (frame-0-centred templates), then the pose
channels via ``_augment_pose_channels``. Decisions that are not forced by the data
(kept parts, shift, window, normalization scales) are arguments with the defaults
stated here, never hidden.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from data.batch import TrajectoryBatch
from data.unified_dataset import (CANONICAL_TRANSLATION, MAT_GRIPPER, MAT_RIGID,
                                  RIGID_FILL_E, RIGID_FILL_NU,
                                  _augment_pose_channels, _rigid_sidecar_from_particles)

GRIPPER_PARTS = ("panda_hand", "panda_leftfinger", "panda_rightfinger",
                 "panda_leftfinger_tip", "panda_rightfinger_tip")
NATIVE_DT = 0.05                     # 20 fps (physics_dt 0.002 x control_decimation 25); pickup's dt
RIGID_FILL_RHO = 750.0               # the pushing adapter's fill for rigid bodies without a recorded density
GRIPPER_OBJECT_ID = 1                # pickup convention: gripper links are one object with id 1 ...
SRC_OBJECT_ID = 0                    # ... and the manipulated object is object 0


@dataclass
class NativeEpisode:
    path: str
    positions: torch.Tensor          # [T, N, 3] metres, part-contiguous, shifted
    part_index: torch.Tensor         # [N] 0..K-1 in part order (== body index)
    part_names: list                 # K names
    object_ids: torch.Tensor         # [N] pickup-style opaque ids: gripper 1, src 0, others 2..
    gripper: torch.Tensor            # [N] bool
    src_part: int                    # part index of the manipulated object
    moving_parts: list               # part indices whose points ever move
    permutation: np.ndarray          # original point index of each reordered point
    shift: tuple
    kinematic: torch.Tensor = None   # [N] bool: fixture points (table, shelf) fed as zero-motion actuator input
    dt: float = NATIVE_DT
    hand_rotation: torch.Tensor = None  # [T, 3, 3] recorded panda_hand rotation (finger joint axes)

    def __post_init__(self):
        if self.kinematic is None:
            self.kinematic = torch.zeros(self.num_particles, dtype=torch.bool)

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.positions.shape[1])

    def part_mask(self, part: int) -> torch.Tensor:
        return self.part_index == part


PICKUP_LAYOUT_NAMES = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
FIXTURE_PART_PREFIXES = ("table", "shelf")   # static scene fixtures: bodies given as input with zero motion, never predicted
# One Franka gripper, one point set. The pickup corpus stores 241 + 212 + 218 link points (identical in every episode,
# data/embodiments/panda_gripper_pickup.json) and rl_push builds its gripper from the same file; native recordings carry
# their own sampling of the same links (237 hand, 159 + 56 per finger with its tip: same link frames, boxes equal to
# under 1 mm, two-sided Chamfer 6 / 2 mm = sampling noise; scratchpad/launch_pwm_data_v1/feedback_0919b, 2026-09-19).
# ``shared_gripper`` swaps the recording's gripper rows for the shared points under the recorded link poses.
SHARED_GRIPPER_LINKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embodiments", "panda_gripper_pickup.npz")


@lru_cache(maxsize=1)
def shared_gripper_points() -> dict:
    """{link name: [n, 3] float32 points in the link frame, metres} of the gripper every Franka dataset shares."""
    with np.load(SHARED_GRIPPER_LINKS, allow_pickle=False) as z:
        return {name: np.asarray(z[name], dtype=np.float32) for name in PICKUP_LAYOUT_NAMES}


def pose_gripper_links(part_poses, names) -> dict:
    """The shared link points under recorded link poses: {link: [T, n, 3] float32, recording frame}.
    ``part_poses`` [T, parts, 7] holds xyz + wxyz per part (quaternions are normalized here); the fingers take their
    own recorded poses, which sit at (0, +-q, 58.4 mm) in the hand frame without rotation like the pickup corpus."""
    missing = [n for n in PICKUP_LAYOUT_NAMES if n not in names]
    if part_poses is None or missing:
        raise ValueError(f"the shared gripper needs recorded poses of {PICKUP_LAYOUT_NAMES}; missing {missing or 'the pose array'}")
    from utils.pose_ops import quat_wxyz_to_matrix
    poses = torch.as_tensor(np.asarray(part_poses), dtype=torch.float32)
    if poses.ndim != 3 or poses.shape[-1] != 7 or not bool(torch.isfinite(poses).all()):
        raise ValueError("recorded part poses must be finite [T, parts, xyz + wxyz]")
    links = {}
    for name, points in shared_gripper_points().items():
        pose = poses[:, names.index(name)]
        quat = pose[:, 3:7] / pose[:, 3:7].norm(dim=-1, keepdim=True)
        rows = torch.einsum("tij,nj->tni", quat_wxyz_to_matrix(quat), torch.from_numpy(points)) + pose[:, None, :3]
        links[name] = rows.numpy()
    return links


def load_native_episode(npz_path: str, *, shift=CANONICAL_TRANSLATION["pickup"],
                        drop_static: bool = False, static_tol_m: float = 1e-6,
                        pickup_layout: bool = False, scene_layout: bool = False,
                        keep_fixtures: bool = True, shared_gripper: bool = False) -> NativeEpisode:
    """Read one episode. ``shift`` is the fixed rigid translation applied to all
    positions (default: the pickup canonical shift, since units and frame are the
    robot table frame like pickup). ``drop_static`` removes parts whose points never
    move by more than ``static_tol_m`` (table, shelf, untouched distractors).
    ``pickup_layout`` reproduces the pickup corpus exactly: four bodies and nothing
    else, body 0 = the manipulated object (512 points), bodies 1..3 = hand, left
    finger, right finger (the tip regions merged into their fingers), particles
    stored gripper first then the object, as UnifiedPickupDataset lays them out;
    table, shelf and every distractor are dropped.
    ``scene_layout`` keeps the whole scene with the pickup gripper convention: the
    same three gripper links (tips merged) actuated, body 0 = the manipulated
    object, then every other object as a passive predicted rigid body, and the
    fixtures (``FIXTURE_PART_PREFIXES``: table, shelf) as rigid bodies that are
    INPUT with zero motion (actuated, ``MAT_RIGID`` like the pushing rod), so
    they are in the scene but never predicted. ``keep_fixtures=False`` drops the fixtures'
    points instead, as the pickup corpus has no table at all.
    ``shared_gripper`` (pickup and scene layouts) replaces the recording's own gripper points by the
    pickup corpus's link points under the recorded link poses (``pose_gripper_links``): 671 rows in the
    pickup order, the point set rl_push and pickup use. Default False = the recording's points, as before."""
    with np.load(npz_path, allow_pickle=True) as d:
        pos = np.asarray(d["state/particles"], dtype=np.float32)          # [T, N, 3]
        pid = np.asarray(d["meta/particles/part_ids"]).astype(np.int64)     # [N]
        names = [str(n) for n in d["meta/particles/part_names"]]
        counts = np.asarray(d["meta/particles/part_counts"]).astype(np.int64)
        src_name = str(d["meta/spatial/src_name"]) if "meta/spatial/src_name" in d.files else ""
        part_poses = (np.asarray(d["state/particle_part_poses_wxyz"], dtype=np.float32)
                      if "state/particle_part_poses_wxyz" in d.files else None)    # [T, parts, xyz + wxyz]
    T, N, _ = pos.shape
    if pid.min() != 0 or pid.max() != len(names) - 1:
        raise ValueError(f"part ids {pid.min()}..{pid.max()} do not index {len(names)} part names")
    for k, c in enumerate(counts):
        if int((pid == k).sum()) != int(c):
            raise ValueError(f"part {k} ({names[k]}): {int((pid == k).sum())} points, meta says {int(c)}")

    if pickup_layout and scene_layout:
        raise ValueError("pickup_layout and scene_layout are exclusive")
    if shared_gripper and not (pickup_layout or scene_layout):
        raise ValueError("shared_gripper needs the pickup or the scene layout (three gripper links, tips merged)")
    links = pose_gripper_links(part_poses, names) if shared_gripper else None
    if pickup_layout:
        return _with_hand_rotation(_pickup_layout_episode(npz_path, pos, pid, names, src_name, shift, static_tol_m,
                                                          gripper_links=links), part_poses, names)
    if scene_layout:
        return _with_hand_rotation(_scene_layout_episode(npz_path, pos, pid, names, src_name, shift, static_tol_m,
                                                         keep_fixtures=keep_fixtures, gripper_links=links),
                                   part_poses, names)
    perm = np.argsort(pid, kind="stable")
    pos, pid = pos[:, perm], pid[perm]
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=-1) if T > 1 else np.zeros((1, N))
    moving = sorted({int(k) for k in np.unique(pid) if step[:, pid == k].max() > static_tol_m})
    gripper_parts = [k for k, n in enumerate(names) if n in GRIPPER_PARTS]
    if not gripper_parts:
        raise ValueError(f"no gripper parts among {names}")
    src_candidates = [k for k, n in enumerate(names) if n == src_name]
    if len(src_candidates) != 1:
        raise ValueError(f"manipulated object {src_name!r} not found once in {names}")
    src_part = src_candidates[0]

    keep_parts = list(range(len(names)))
    if drop_static:
        keep_parts = [k for k in keep_parts if k in moving or k in gripper_parts or k == src_part]
    keep_mask = np.isin(pid, keep_parts)
    pos, pid, perm = pos[:, keep_mask], pid[keep_mask], perm[keep_mask]
    # re-index parts densely in part order so body index == part index
    remap = {old: new for new, old in enumerate(keep_parts)}
    part_index = np.vectorize(remap.get)(pid)
    names_kept = [names[k] for k in keep_parts]
    gripper = np.isin(pid, gripper_parts)
    object_ids = np.empty(pos.shape[1], dtype=np.int64)
    next_id = 2
    for old in keep_parts:
        sel = pid == old
        if old in gripper_parts:
            object_ids[sel] = GRIPPER_OBJECT_ID
        elif old == src_part:
            object_ids[sel] = SRC_OBJECT_ID
        else:
            object_ids[sel] = next_id
            next_id += 1
    moving_kept = [remap[k] for k in moving if k in remap]
    shift_t = torch.tensor(shift, dtype=torch.float32)
    return _with_hand_rotation(NativeEpisode(
        path=npz_path, positions=torch.from_numpy(pos) + shift_t,
        part_index=torch.from_numpy(part_index.astype(np.int64)), part_names=names_kept,
        object_ids=torch.from_numpy(object_ids), gripper=torch.from_numpy(gripper),
        src_part=remap[src_part], moving_parts=moving_kept, permutation=perm, shift=tuple(shift)),
        part_poses, names)


def _with_hand_rotation(ep: NativeEpisode, part_poses, names) -> NativeEpisode:
    """Attach the recorded panda_hand rotation per frame (articulated batches take the finger joint
    axes from it, as the pickup loader does from its link poses)."""
    if part_poses is not None and "panda_hand" in names:
        from utils.pose_ops import quat_wxyz_to_matrix
        ep.hand_rotation = quat_wxyz_to_matrix(torch.from_numpy(part_poses[:, names.index("panda_hand"), 3:7]))
    return ep


def _assemble_segments(pos, pid, by_name, segments, gripper_links=None):
    """Rows of a layout, segment by segment: (positions [T, N', 3], permutation [N'], body index [N']).
    A segment takes the recording's own points of its member parts, in stored order. With ``gripper_links``
    ({link: [T, n, 3]} from pose_gripper_links) bodies 1..3 take the shared link points instead; those rows have
    no counterpart in the recording, so their permutation entry is -1."""
    blocks, perm, body = [], [], []
    for new_id, members in segments:
        if gripper_links is not None and 1 <= new_id <= 3:
            rows = np.asarray(gripper_links[PICKUP_LAYOUT_NAMES[new_id - 1]], dtype=pos.dtype)
            if rows.ndim != 3 or rows.shape[0] != pos.shape[0] or rows.shape[2] != 3:
                raise ValueError("shared gripper rows must be [T, n, 3] over the frames of the episode")
            blocks.append(rows); perm.append(np.full(rows.shape[1], -1, dtype=np.int64)); body += [new_id] * rows.shape[1]
            continue
        for m in members:
            idx = np.nonzero(pid == by_name[m])[0]
            blocks.append(pos[:, idx]); perm.append(idx); body += [new_id] * len(idx)
    return np.concatenate(blocks, axis=1), np.concatenate(perm), np.asarray(body, dtype=np.int64)


def _pickup_layout_episode(npz_path, pos, pid, names, src_name, shift, static_tol_m, gripper_links=None):
    """Exactly the pickup corpus layout (see load_native_episode)."""
    by_name = {n: k for k, n in enumerate(names)}
    missing = [n for n in GRIPPER_PARTS if n not in by_name]
    if missing or src_name not in by_name:
        raise ValueError(f"pickup layout needs {GRIPPER_PARTS} and {src_name!r}; have {names}")
    # particle order: hand, left finger (+ its tip), right finger (+ its tip), object
    segments = [(1, ["panda_hand"]), (2, ["panda_leftfinger", "panda_leftfinger_tip"]),
                (3, ["panda_rightfinger", "panda_rightfinger_tip"]), (0, [src_name])]
    pos, perm, part_index = _assemble_segments(pos, pid, by_name, segments, gripper_links)
    gripper = part_index != 0
    object_ids = np.where(gripper, GRIPPER_OBJECT_ID, SRC_OBJECT_ID).astype(np.int64)
    T, N = pos.shape[:2]
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=-1) if T > 1 else np.zeros((1, N))
    moving = sorted({int(k) for k in np.unique(part_index) if step[:, part_index == k].max() > static_tol_m})
    shift_t = torch.tensor(shift, dtype=torch.float32)
    return NativeEpisode(
        path=npz_path, positions=torch.from_numpy(pos) + shift_t,
        part_index=torch.from_numpy(part_index), part_names=[src_name, *PICKUP_LAYOUT_NAMES],
        object_ids=torch.from_numpy(object_ids), gripper=torch.from_numpy(gripper),
        src_part=0, moving_parts=moving, permutation=perm, shift=tuple(shift))


def _scene_layout_episode(npz_path, pos, pid, names, src_name, shift, static_tol_m, keep_fixtures=True,
                          gripper_links=None):
    """The whole scene under the pickup gripper convention (see load_native_episode).
    Particle order: hand, left finger (+tip), right finger (+tip), manipulated
    object, the other objects in file order, then the fixtures. Body index:
    0 = manipulated object, 1..3 = gripper links, 4.. = other objects, fixtures last.
    ``gripper_links``: the shared link points in the recording frame (see _assemble_segments)."""
    by_name = {n: k for k, n in enumerate(names)}
    missing = [n for n in GRIPPER_PARTS if n not in by_name]
    if missing or src_name not in by_name:
        raise ValueError(f"scene layout needs {GRIPPER_PARTS} and {src_name!r}; have {names}")
    is_fixture = lambda n: n.startswith(FIXTURE_PART_PREFIXES)
    others = [n for n in names if n not in GRIPPER_PARTS and n != src_name and not is_fixture(n)]
    fixtures = [n for n in names if is_fixture(n)] if keep_fixtures else []   # dropped: not in the scene at all
    segments = [(1, ["panda_hand"]), (2, ["panda_leftfinger", "panda_leftfinger_tip"]),
                (3, ["panda_rightfinger", "panda_rightfinger_tip"]), (0, [src_name])]
    segments += [(4 + i, [n]) for i, n in enumerate(others)]
    segments += [(4 + len(others) + i, [n]) for i, n in enumerate(fixtures)]
    pos, perm, part_index = _assemble_segments(pos, pid, by_name, segments, gripper_links)
    gripper = (part_index >= 1) & (part_index <= 3)
    kinematic = part_index >= 4 + len(others)
    object_ids = np.where(gripper, GRIPPER_OBJECT_ID, np.where(part_index == 0, SRC_OBJECT_ID, part_index - 2)).astype(np.int64)
    T, N = pos.shape[:2]
    step = np.linalg.norm(pos[1:] - pos[:-1], axis=-1) if T > 1 else np.zeros((1, N))
    if kinematic.any() and step[:, kinematic].max() > static_tol_m:
        raise ValueError(f"fixture points move by {step[:, kinematic].max() * 1000:.3f} mm; fixtures must be static")
    moving = sorted({int(k) for k in np.unique(part_index) if step[:, part_index == k].max() > static_tol_m})
    shift_t = torch.tensor(shift, dtype=torch.float32)
    return NativeEpisode(
        path=npz_path, positions=torch.from_numpy(pos) + shift_t,
        part_index=torch.from_numpy(part_index), part_names=[src_name, *PICKUP_LAYOUT_NAMES, *others, *fixtures],
        object_ids=torch.from_numpy(object_ids), gripper=torch.from_numpy(gripper),
        src_part=0, moving_parts=moving, permutation=perm, shift=tuple(shift),
        kinematic=torch.from_numpy(kinematic))


def window_motion(ep: NativeEpisode, start: int, length: int) -> float:
    """Mean per-frame displacement (m) of the manipulated object over a window."""
    sel = ep.part_mask(ep.src_part)
    seg = ep.positions[start:start + length, sel]
    return float((seg[1:] - seg[:-1]).norm(dim=-1).mean()) if length > 1 else 0.0


def select_windows(ep: NativeEpisode, k: int, context_len: int, future_len: int,
                   stride: int = 1) -> list:
    """The ``k`` non-overlapping window starts with the most manipulated-object
    motion (greedy, descending)."""
    L = context_len + future_len
    T = ep.num_frames
    if T < L:
        raise ValueError(f"episode has {T} frames, window needs {L}")
    starts = list(range(0, T - L + 1, stride))
    ranked = sorted(starts, key=lambda s: window_motion(ep, s, L), reverse=True)
    chosen = []
    for s in ranked:
        if all(abs(s - c) >= L for c in chosen):
            chosen.append(s)
        if len(chosen) == k:
            break
    return sorted(chosen)


def native_window_batch(ep: NativeEpisode, starts, *, context_len: int = 4, future_len: int = 38,
                        normalization=None, normalization_domain: str = "pickup",
                        articulated: bool = False, normalizers=None) -> TrajectoryBatch:
    """Windows ``[start, start + C + F)`` of one episode as a pose-native batch.

    All windows share the episode's layout (the backbones require that). Scales
    come from ``normalization_domain`` of the resolved profile (default: the pickup
    per-domain scales; units and frame rate match). Nothing is refit."""
    from data.normalization import domain_normalizers
    delta_stats, position_stats, pose_stats = (domain_normalizers(normalization, normalization_domain)
                                              if normalizers is None else normalizers)
    L = context_len + future_len
    starts = [int(s) for s in starts]
    for s in starts:
        if s < 0 or s + L > ep.num_frames:
            raise ValueError(f"window {s}..{s + L} outside the {ep.num_frames}-frame episode")
    positions = torch.stack([ep.positions[s:s + L] for s in starts])          # [B, L, N, 3]
    B, _, N, _ = positions.shape
    actuated = (ep.gripper | ep.kinematic)[None].expand(B, N).clone()
    d = positions[:, 1:] - positions[:, :-1]
    actions = d[:, context_len - 1:] * actuated[:, None, :, None].to(d.dtype)   # achieved motion (zero for fixtures)
    mat_type = torch.where(ep.gripper[None].expand(B, N), MAT_GRIPPER, MAT_RIGID).long()
    particle_body = ep.part_index[None].expand(B, N).clone()
    n_bodies = len(ep.part_names)
    sidecar = _rigid_sidecar_from_particles(positions, particle_body, n_bodies=n_bodies)
    sidecar["particle_body"] = particle_body
    if articulated:                       # fingers as prismatic children of the hand (model.articulated_bodies)
        if tuple(ep.part_names[1:4]) != PICKUP_LAYOUT_NAMES or ep.hand_rotation is None:
            raise ValueError("articulated native batches need the pickup or scene layout and the recorded hand pose")
        from data.unified_dataset import UnifiedPickupDataset, _articulate_fingers
        axes = torch.einsum("bij,fj->bfi", ep.hand_rotation[starts],            # each window's first frame
                            torch.tensor(UnifiedPickupDataset.PANDA_FINGER_AXES))
        sidecar = _articulate_fingers(sidecar, [(1, 2), (1, 3)], axes)
    sidecar = _augment_pose_channels(sidecar, actuated, context_len, pose_stats)
    return TrajectoryBatch(
        positions=positions, actions=actions, action_mask=actuated,
        object_ids=ep.object_ids[None].expand(B, N).clone(),
        valid_lengths=torch.full((B,), N, dtype=torch.long),
        E=torch.full((B, N), RIGID_FILL_E), nu=torch.full((B, N), RIGID_FILL_NU),
        rho=torch.full((B, N), RIGID_FILL_RHO), mat_type=mat_type,
        context_len=context_len, delta_stats=delta_stats, position_stats=position_stats,
        prediction_target="delta", domain=normalization_domain, **sidecar)


def reconstruction_error(batch: TrajectoryBatch) -> float:
    """Max |template @ pose - positions| over all rigid bodies and frames, metres."""
    worst = 0.0
    for k in range(batch.body_type.shape[1]):
        if int(batch.body_type[0, k]) != 1:
            continue
        n_t = int(batch.body_template_mask[0, k].sum())
        tmpl = batch.body_templates[:, k, :n_t]                                    # [B, P, 3]
        R = batch.body_poses[:, :, k, :3, :3]; t = batch.body_poses[:, :, k, :3, 3]
        pts = torch.einsum("btij,bpj->btpi", R, tmpl) + t[:, :, None]              # [B, T, P, 3]
        rows = (batch.particle_body[0] == k).nonzero(as_tuple=True)[0]
        worst = max(worst, float((pts - batch.positions[:, :, rows]).norm(dim=-1).max()))
    return worst


class NativeEpisodeDataset:
    """Iterable of native-episode windows for the production evaluator
    (`scripts/unified/eval_unified.eval_domain` / `rollout_domain` with an explicit
    `dataset=`). Items are (episode, window start) pairs, one window per batch, the
    starts ranked by manipulated-object motion over a fixed-window length; the
    batch spans `context_len + min(future_len, frames left)` frames, so the
    evaluator's `future_len = 10_000` convention yields the rest of the episode from
    that start for autoregression. Layout: "pickup" (exactly the pickup corpus
    layout), "scene" (pickup gripper + every object predicted + fixtures as
    zero-motion input), "scene_objects" (the same without the fixtures, like the pickup
    corpus without its table), "all_parts" or "moving_only"."""

    domain = "pickup"

    def __init__(self, paths, *, context_len: int = 4, future_len: int = 38, windows_per_episode: int = 3,
                 layout: str = "pickup", normalization=None, rank_future_len: int = 38):
        from data.normalization import domain_normalizers, resolve_normalization
        if layout not in ("pickup", "scene", "scene_objects", "all_parts", "moving_only"):
            raise ValueError(f"unknown layout {layout!r}")
        self.paths = [str(p) for p in paths]
        self.context_len, self.future_len, self.layout = int(context_len), int(future_len), layout
        self.windows_per_episode, self.rank_future_len = int(windows_per_episode), int(rank_future_len)
        self.normalization = resolve_normalization(normalization)
        self.emit_sidecar = True
        self.articulated = False                  # set by the consumer from model.articulated_bodies
        self._pose = domain_normalizers(self.normalization, self.domain)[2]
        self.episodes = [load_native_episode(p, drop_static=(layout == "moving_only"), pickup_layout=(layout == "pickup"),
                                             scene_layout=(layout in ("scene", "scene_objects")),
                                             keep_fixtures=(layout != "scene_objects"))
                         for p in self.paths]
        self.items = [(i, s) for i, ep in enumerate(self.episodes)
                      for s in select_windows(ep, self.windows_per_episode, self.context_len, self.rank_future_len)]

    def _pose_stats(self):
        return self._pose

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        for i, s in self.items:
            ep = self.episodes[i]
            L = min(self.future_len, ep.num_frames - s - self.context_len)
            batch = native_window_batch(ep, [s], context_len=self.context_len, future_len=L,
                                        normalization=self.normalization, normalization_domain=self.domain,
                                        articulated=self.articulated)
            batch.domain = self.domain
            yield batch
