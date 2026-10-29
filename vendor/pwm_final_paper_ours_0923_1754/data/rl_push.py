"""Low-dim RL push release as the opt-in ``rl_push`` domain.

A Franka Panda gripper driven by an RL policy pushes one rigid object toward a goal on a table (Isaac
Sim 4.5, 20 Hz). The release (format ``rl_push_lowdim_v1``, documented in its README.md) stores no
particles. One ``.npz`` per object holds, for every episode, the panda_hand pose, the nine joint
positions (columns 7 and 8 are the finger joints in metres) and the object pose, in the robot base frame
(metres, z up, table top at z = 0, quaternions wxyz), plus point templates in the object's own frame and
per-episode collection provenance.

Batches are built at load time in exactly the pick-and-place layout of UnifiedPickupDataset:
- gripper points first: the pickup corpus's own link point sets (data/embodiments/panda_gripper_pickup.npz:
  panda_hand 241, panda_leftfinger 212, panda_rightfinger 218 points, identical in every pickup episode),
  posed by the recorded hand pose, with the fingers at (0, +q_left, 58.4 mm) and (0, -q_right, 58.4 mm) in
  the hand frame and no rotation of their own, as in the pickup corpus;
- then RL_PUSH_OBJECT_POINTS object points from ``object_cloud_outer_surface``: a fresh random subset per
  training window, one seeded subset per object for validation and evaluation;
- object id 1 for the gripper and 0 for the object, MAT_GRIPPER / MAT_RIGID with the rigid fill physics,
  achieved per-particle actions on the gripper, and the rigid sidecar, finger articulation and pose
  channels computed by the same functions as pickup.

Decisions are module constants, listed in docs/RL_PUSH.md: windows start at frame RL_PUSH_FIRST_START or
later (objects spawn about 5 mm above the table and a few get a sideways kick at frame 2); validation
objects and outlier episodes come from JSON manifests named by the split's exclusion files (outliers are
keyed by collection provenance, which survives re-exports of an object file); scales come from a
normalization artifact with an ``rl_push`` entry.
"""
from __future__ import annotations

import glob
import json
import os
import zlib
from dataclasses import dataclass

import numpy as np
import torch

from data.batch import TrajectoryBatch
from data.blocklists import manifest_path
from data.dataset import NormalizationStats
from data.unified_dataset import (CANONICAL_TRANSLATION, MAT_GRIPPER, MAT_RIGID, RIGID_FILL_E, RIGID_FILL_NU,
                                  RL_PUSH_VAL_ASSETS, UnifiedPickupDataset, _articulate_fingers,
                                  _augment_pose_channels, _identity_delta_stats, _rigid_sidecar_from_particles,
                                  _validate_window, unified_position_stats)

# ----------------------------------------------------------------------------- decisions
RL_PUSH_FORMAT = "rl_push_lowdim_v1"
RL_PUSH_FIRST_START = 3          # first allowed window frame: skips the spawn settle and the frame-2 kick
RL_PUSH_OBJECT_POINTS = 512      # the pickup corpus's object point count
RL_PUSH_SUBSET_SEED = 0
RL_PUSH_FINGER_Z = 0.0584        # finger joint origins along the hand's z axis (m)
RL_PUSH_FILL_RHO = 750.0         # the pushing and native adapters' density fill for rigid bodies
RL_PUSH_MAX_FRAMES = 100         # the release ends every episode by 5 s; a longer window asks for the rest
RL_PUSH_SHIFT = CANONICAL_TRANSLATION["rl_push"]
GRIPPER_LINKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embodiments", "panda_gripper_pickup.npz")
GRIPPER_LINK_NAMES = ("panda_hand", "panda_leftfinger", "panda_rightfinger")
PROVENANCE_KEYS = ("source_run", "source_shard", "source_episode_index", "source_env")
VAL_ASSETS_KIND = "rl_push_val_assets"
EXCLUSIONS_KIND = "rl_push_episode_exclusions"


def _rotation(q):
    """wxyz quaternions [..., 4] (normalized here) -> rotation matrices [..., 3, 3], float64."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def asset_key_of(rel: str) -> str:
    """Object files mirror their asset key: objects/<root>/<category>/<asset>.npz."""
    return rel[len("objects/"):-len(".npz")]


@dataclass
class PushObject:
    """One object file: every episode's low-dim state, the object template and per-episode provenance."""
    file: str                    # path below the release root
    asset_key: str
    category: str
    starts: np.ndarray           # [E] first frame of each episode
    lengths: np.ndarray          # [E]
    hand_pos: np.ndarray         # [F, 3]
    hand_quat: np.ndarray        # [F, 4] wxyz, unit length
    finger_q: np.ndarray         # [F, 2] left, right finger joint (m)
    object_pos: np.ndarray       # [F, 3]
    object_quat: np.ndarray      # [F, 4] wxyz, unit length
    template: np.ndarray         # [P, 3] object frame
    provenance: list             # E tuples in PROVENANCE_KEYS order

    def key(self, episode: int) -> tuple:
        """An episode's exclusion key: its asset and collection provenance."""
        return (self.asset_key, *self.provenance[episode])


def load_push_object(root: str, rel: str) -> PushObject:
    """Read one object file. Quaternions are normalized: some stored hand quaternions are not unit length."""
    unit = lambda q: (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)
    with np.load(os.path.join(root, rel), allow_pickle=False) as z:
        if str(z["format_version"]) != RL_PUSH_FORMAT:
            raise ValueError(f"{rel}: format {str(z['format_version'])!r}, expected {RL_PUSH_FORMAT!r}")
        starts, lengths = z["episode_start"].astype(np.int64), z["episode_len"].astype(np.int64)
        E = len(starts)
        cols = {k: np.full(E, z[k].item()) if z[k].ndim == 0 else z[k] for k in PROVENANCE_KEYS}
        obj = PushObject(
            file=rel, asset_key=str(z["asset_key"]), category=str(z["category"]), starts=starts, lengths=lengths,
            hand_pos=z["hand_pos"].astype(np.float32), hand_quat=unit(z["hand_quat_wxyz"].astype(np.float64)),
            finger_q=z["joint_pos"][:, 7:9].astype(np.float32), object_pos=z["object_pos"].astype(np.float32),
            object_quat=unit(z["object_quat_wxyz"].astype(np.float64)),
            template=z["object_cloud_outer_surface"].astype(np.float32),
            provenance=[(str(cols["source_run"][i]), str(cols["source_shard"][i]),
                         int(cols["source_episode_index"][i]), int(cols["source_env"][i])) for i in range(E)])
    frames = len(obj.hand_pos)
    tiling = np.concatenate([[0], np.cumsum(lengths)[:-1]])[:E]             # a file may hold no episodes
    if int(lengths.sum()) != frames or not np.array_equal(starts, tiling):
        raise ValueError(f"{rel}: episode ranges do not tile its {frames} frames")
    return obj


def read_manifests(paths):
    """(validation asset keys or None, excluded episode keys) from rl_push JSON manifests."""
    val_assets, excluded = None, set()
    for path in paths:
        with open(manifest_path(path)) as f:
            spec = json.load(f)
        kind = spec.get("kind")
        if kind == VAL_ASSETS_KIND:
            if val_assets is not None:
                raise ValueError("rl_push takes a single validation object list")
            val_assets = frozenset(spec["assets"])
        elif kind == EXCLUSIONS_KIND:
            excluded |= {(e["asset_key"], e["source_run"], e["source_shard"], int(e["source_episode_index"]),
                          int(e["source_env"])) for e in spec["episodes"]}
        else:
            raise ValueError(f"{path}: not an rl_push manifest (kind {kind!r})")
    return val_assets, excluded


def window_points(obj: PushObject, episode: int, start: int, length: int, links: dict, object_rows):
    """One window in the model frame: points [length, N, 3] float32 (hand, left finger, right finger, then
    object rows) and the hand rotation at the window's first frame [3, 3] (it sets the finger joint axes)."""
    first = int(obj.starts[episode]) + int(start)
    if start < 0 or length < 1 or start + length > int(obj.lengths[episode]):
        raise ValueError(f"window {start}..{start + length} outside the {int(obj.lengths[episode])}-frame episode")
    frames = slice(first, first + int(length))
    R_hand, t_hand = _rotation(obj.hand_quat[frames]), obj.hand_pos[frames].astype(np.float64)
    q = obj.finger_q[frames].astype(np.float64)
    T = len(q)
    zero, lift = np.zeros(T), np.full(T, RL_PUSH_FINGER_Z)
    left = links["panda_leftfinger"][None] + np.stack([zero, q[:, 0], lift], -1)[:, None]
    right = links["panda_rightfinger"][None] + np.stack([zero, -q[:, 1], lift], -1)[:, None]
    hand = np.broadcast_to(links["panda_hand"][None], (T,) + links["panda_hand"].shape)
    gripper = np.einsum("tij,tnj->tni", R_hand, np.concatenate([hand, left, right], axis=1)) + t_hand[:, None]
    body = (np.einsum("tij,nj->tni", _rotation(obj.object_quat[frames]), obj.template[object_rows].astype(np.float64))
            + obj.object_pos[frames].astype(np.float64)[:, None])
    points = np.concatenate([gripper, body], axis=1) + np.asarray(RL_PUSH_SHIFT)
    return points.astype(np.float32), R_hand[0]


def push_batch(windows, *, context_len: int, delta_stats, position_stats, pose_stats, link_counts,
               emit_sidecar: bool = False, articulated: bool = False) -> TrajectoryBatch:
    """Equal-length windows from window_points -> one ``rl_push`` batch in the pickup layout."""
    if len({w[0].shape for w in windows}) != 1:
        raise ValueError("rl_push windows in one batch must share frames and points")
    positions = torch.from_numpy(np.stack([w[0] for w in windows]))          # [B, L, N, 3]
    B, L, N, _ = positions.shape
    if L <= context_len:
        raise ValueError(f"a {L}-frame window leaves no future frames after {context_len} context frames")
    n_hand, n_left, n_right = link_counts
    n_gripper = n_hand + n_left + n_right
    object_ids = torch.zeros(B, N, dtype=torch.long)
    object_ids[:, :n_gripper] = 1
    actuated = object_ids == 1
    steps = positions[:, 1:] - positions[:, :-1]
    actions = steps[:, context_len - 1:] * actuated[:, None, :, None].to(steps.dtype)
    sidecar = {}
    if emit_sidecar:
        body = torch.zeros(N, dtype=torch.long)                                 # 0 object, 1 hand, 2 left, 3 right
        body[:n_hand], body[n_hand:n_hand + n_left], body[n_hand + n_left:n_gripper] = 1, 2, 3
        particle_body = body[None].expand(B, N)
        sidecar = _rigid_sidecar_from_particles(positions, particle_body, n_bodies=4)
        sidecar["particle_body"] = particle_body.clone()
        if articulated:
            R0 = torch.from_numpy(np.stack([w[1] for w in windows])).float()
            axes = torch.einsum("bij,fj->bfi", R0, torch.tensor(UnifiedPickupDataset.PANDA_FINGER_AXES))
            sidecar = _articulate_fingers(sidecar, [(1, 2), (1, 3)], axes)
        sidecar = _augment_pose_channels(sidecar, actuated, context_len, pose_stats)
    return TrajectoryBatch(
        positions=positions, actions=actions, action_mask=actuated, object_ids=object_ids,
        valid_lengths=torch.full((B,), N, dtype=torch.long),
        E=torch.full((B, N), RIGID_FILL_E), nu=torch.full((B, N), RIGID_FILL_NU),
        rho=torch.full((B, N), RL_PUSH_FILL_RHO), mat_type=torch.where(actuated, MAT_GRIPPER, MAT_RIGID).long(),
        context_len=context_len, delta_stats=delta_stats, position_stats=position_stats,
        prediction_target="delta", domain="rl_push", **sidecar)


def _rank_world():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class UnifiedRLPushDataset(torch.utils.data.IterableDataset):
    """RL push release -> ``rl_push`` TrajectoryBatches (batch_size windows each).

    ``split``: "train" (objects off the validation list), "val" (objects on it) or "all" (a whole release,
    e.g. a separate validation folder). ``manifests``: rl_push JSON manifests, at most one validation object
    list and any number of episode exclusion lists (normally the split's exclusion_files). An episode is
    usable when it holds ``first_start`` plus a whole window.

    Training: an endless stream. Each window draws a usable episode uniformly, a start between the first
    allowed frame and the last start, and a fresh object point subset, from a generator seeded by
    (torch.initial_seed(), rank, worker). ``first_window`` starts every window at the first allowed frame.
    Validation and evaluation: finite and deterministic. Objects are interleaved across the release's roots,
    then categories, so any prefix spans as many of both as it can; one window per episode, its start drawn
    from the episode's identity (the first allowed frame with ``first_window``); each object's episodes come
    in batches of equal window length tagged with the object file (``source_shard``). A window longer than
    any episode (the evaluator's ``future_len = 10_000``, or -1) is the rest of each episode from the first
    allowed frame. Objects are split across ranks unless ``no_shard_split`` and across DataLoader workers.
    The unified loaders set batch_size, first_window, no_shard_split, shuffle, bucket_max_items, emit_sidecar
    and articulated after construction (shuffle and bucket_max_items are no-ops here).
    """

    domain = "rl_push"

    def __init__(self, root, *, split: str = "train", context_len: int = 4, future_len: int = 16,
                 training: bool = True, shuffle: bool = True, manifests=(), normalization=None, raw: bool = False,
                 first_start: int = RL_PUSH_FIRST_START, object_points: int = RL_PUSH_OBJECT_POINTS,
                 subset_seed: int = RL_PUSH_SUBSET_SEED, gripper_links: str = GRIPPER_LINKS):
        super().__init__()
        _validate_window(context_len, future_len)
        if split not in ("train", "val", "all"):
            raise ValueError(f"rl_push split must be train, val or all, got {split!r}")
        self.root = os.fspath(root).rstrip("/")
        self.roots, self.split = [self.root], split
        self.context_len, self.future_len = int(context_len), future_len
        self.training, self.shuffle, self.raw = bool(training), bool(shuffle), bool(raw)
        self.first_start, self.object_points, self.subset_seed = int(first_start), int(object_points), int(subset_seed)
        self.normalization = None
        if raw:
            self.delta_stats, self.position_stats = _identity_delta_stats(), unified_position_stats()
            self._pose = NormalizationStats(-torch.ones(6), torch.ones(6))
        else:
            from data.normalization import domain_normalizers, resolve_normalization
            self.normalization = resolve_normalization(normalization)
            self.delta_stats, self.position_stats, self._pose = domain_normalizers(self.normalization, "rl_push")
        self.blocklist_paths = tuple(os.fspath(p) for p in (manifests or ()))
        self.val_assets, self.excluded = read_manifests(self.blocklist_paths)
        if split != "all" and self.val_assets is None:
            raise ValueError(f"the rl_push {split} split needs a validation object list among its manifests "
                             f"(for example {RL_PUSH_VAL_ASSETS})")
        files = sorted(os.path.relpath(p, self.root)
                       for p in glob.glob(os.path.join(self.root, "objects", "**", "*.npz"), recursive=True))
        if not files:
            raise ValueError(f"no object files under {self.root}/objects")
        self.shards = [rel for rel in files
                       if split == "all" or (split == "val") == (asset_key_of(rel) in self.val_assets)]
        if not self.shards:
            raise ValueError(f"the rl_push {split} split of {self.root} has no objects")
        with np.load(gripper_links) as links:
            self.links = {k: links[k].astype(np.float64) for k in GRIPPER_LINK_NAMES}
        self.link_counts = tuple(len(self.links[k]) for k in GRIPPER_LINK_NAMES)
        loaded = [load_push_object(self.root, rel) for rel in self.shards]
        self.empty_files = [obj.file for obj in loaded if not len(obj.starts)]   # the owner's filters can empty a file
        self.objects = [obj for obj in loaded if len(obj.starts)]
        self.shards = [obj.file for obj in self.objects]
        if not self.objects:
            raise ValueError(f"no object file of the rl_push {split} split of {self.root} holds an episode")
        for obj in self.objects:
            if obj.asset_key != asset_key_of(obj.file):
                raise ValueError(f"{obj.file}: asset_key {obj.asset_key!r} does not match its path")
            if len(obj.template) < self.object_points:
                raise ValueError(f"{obj.file}: {len(obj.template)} template points, need {self.object_points}")
        stored = [(i, e) for i, obj in enumerate(self.objects) for e in range(len(obj.starts))]
        self.episodes = [(i, e) for i, e in stored if self.objects[i].key(e) not in self.excluded]
        self.excluded_episodes = len(stored) - len(self.episodes)
        self.batch_size, self.first_window, self.no_shard_split = 4, False, False
        self.bucket_max_items, self.emit_sidecar, self.articulated = 0, False, False
        print(f"[Unified/rl_push] split={split}: {len(self.objects)} objects, {len(self.episodes)} episodes "
              f"({self.excluded_episodes} excluded; {len(self.empty_files)} object files without episodes skipped) "
              f"under {self.root}")

    def _pose_stats(self):
        return self._pose

    def _future(self):
        """The window's future frames, or None for the rest of each episode."""
        f = self.future_len
        return None if f is None or f <= 0 or self.context_len + f > RL_PUSH_MAX_FRAMES else int(f)

    def _object_rows(self, obj: PushObject, rng=None):
        if rng is None:                                   # one subset per object for validation and evaluation
            rng = np.random.default_rng([self.subset_seed, zlib.crc32(obj.asset_key.encode())])
        return np.sort(rng.choice(len(obj.template), self.object_points, replace=False))

    def _batch(self, windows) -> TrajectoryBatch:
        return push_batch(windows, context_len=self.context_len, delta_stats=self.delta_stats,
                          position_stats=self.position_stats, pose_stats=self._pose, link_counts=self.link_counts,
                          emit_sidecar=self.emit_sidecar, articulated=self.articulated)

    def __iter__(self):
        rank, world = _rank_world()
        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        if self.training:
            yield from self._train(np.random.default_rng([torch.initial_seed() % (2 ** 63), rank, worker]))
            return
        order = self._eval_order()
        if not self.no_shard_split:
            order = order[rank::world]
        yield from self._eval(order[worker::workers])

    def _train(self, rng):
        C, F = self.context_len, self._future()
        if F is None:
            raise ValueError("rl_push training needs a positive future_len that fits an episode")
        L = C + F
        usable = [(i, e) for i, e in self.episodes if int(self.objects[i].lengths[e]) >= self.first_start + L]
        if not usable:
            raise ValueError(f"no rl_push episode holds the {self.first_start + L} frames a window needs")
        while True:
            windows = []
            for _ in range(self.batch_size):
                i, e = usable[int(rng.integers(len(usable)))]
                obj = self.objects[i]
                start = (self.first_start if self.first_window else
                         int(rng.integers(self.first_start, int(obj.lengths[e]) - L + 1)))
                windows.append(window_points(obj, e, start, L, self.links, self._object_rows(obj, rng)))
            yield self._batch(windows)

    def _eval_order(self):
        """Object indices interleaved across roots, then categories within each root."""
        from data.dataset import _round_robin
        groups = {}
        for i, obj in enumerate(self.objects):
            groups.setdefault(obj.asset_key.split("/")[0], {}).setdefault(obj.category, []).append(i)
        return list(_round_robin([_round_robin([cats[c] for c in sorted(cats)]) for _, cats in sorted(groups.items())]))

    def _eval_windows(self, i, episodes):
        """[(frames, episode, start)] of object i's evaluation clips in batch order (longest first, then by
        episode): one per episode that holds a window, its start drawn from the episode's identity (the first
        allowed frame with first_window); without a fixed future, the rest of each episode from that frame.
        scripts/unified/freeze_eval_cases.py freezes this selection."""
        C, F = self.context_len, self._future()
        obj, items = self.objects[i], []
        for e in episodes:
            n = int(obj.lengths[e])
            if F is None:
                start, L = self.first_start, n - self.first_start
                if L <= C:
                    continue
            else:
                if n < self.first_start + C + F:
                    continue
                start, L = self.first_start, C + F
                if not self.first_window:
                    start = int(np.random.default_rng([self.subset_seed, zlib.crc32(obj.asset_key.encode()), e])
                                .integers(self.first_start, n - L + 1))
            items.append((L, e, start))
        return sorted(items, key=lambda item: (-item[0], item[1]))

    def _eval(self, order):
        by_object = {}
        for i, e in self.episodes:
            by_object.setdefault(i, []).append(e)
        for i in order:
            obj = self.objects[i]
            rows, batch = self._object_rows(obj), []
            for L, e, start in self._eval_windows(i, by_object.get(i, ())):
                if batch and (len(batch) == self.batch_size or batch[0][0].shape[0] != L):
                    yield self._tagged(batch, obj)
                    batch = []
                batch.append(window_points(obj, e, start, L, self.links, rows))
            if batch:
                yield self._tagged(batch, obj)

    def _tagged(self, windows, obj: PushObject) -> TrajectoryBatch:
        batch = self._batch(windows)
        batch.source_shard = obj.file
        return batch
