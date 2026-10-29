"""DeepMind Learning-to-Simulate Water-3D (Sanchez-Gonzalez et al., ICML 2020,
arXiv 2002.09405) as the opt-in ``water`` domain.

Files (``https://storage.googleapis.com/learning-to-simulate-complex-physics/Datasets/Water-3D/``):
``metadata.json`` (bounds, sequence_length, dim, dt, connectivity radius, velocity
and acceleration statistics) and ``{train,valid,test}.tfrecord``. Each record is a
``tf.train.SequenceExample``: context ``key`` (int64) and ``particle_type`` (one
bytes entry holding an int64 array [N]); feature list ``position`` with one bytes
entry per frame holding a float32 array [N, dim] (``sequence_length + 1`` frames);
Water-3D also carries a placeholder ``step_context`` ([T, 1] = [nan, 0, 0, ...]).

The reader is pure Python (no TensorFlow): a length-prefixed TFRecord walker whose
CRC32C masks are NOT verified, and a hand decoder of the protobuf wire format for the
four messages involved. ``scripts/unified/materialize_water3d.py`` turns the records
into one ``.npy`` per trajectory; ``UnifiedWaterDataset`` reads those for training,
validation and evaluation. Water-3D is SPH water under gravity in a closed unit box
with implicit walls (no boundary particles), no actuator and no rigid bodies: a
passive domain like MPM, with its own material id, frame and normalization entry.
Design constants are module constants or adapter arguments; docs/WATER3D.md lists them.
"""
from __future__ import annotations

import json
import os
import struct
import zlib
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch

from data.batch import TrajectoryBatch
from data.unified_dataset import CANONICAL_TRANSLATION, DOMAIN_DT, MAT_WATER

# ----------------------------------------------------------------------------- decisions
WATER3D_DT_FILE = 0.005          # metadata dt, seconds per file step (200 Hz)
# Every 8th file frame: 25 fps, next to MPM's 24 (DOMAIN_DT["water"] = 0.04 s).
WATER3D_STRIDE = round(DOMAIN_DT["water"] / WATER3D_DT_FILE)
# Particles per window. Training draws a fresh random subset per sample; validation and
# evaluation use one seeded subset per trajectory. Trajectories hold 4,096 to ~13k particles.
WATER3D_PARTICLES = 4096
WATER3D_SUBSET_SEED = 0
# Hand-scaled physics features (log10 E, nu, rho / 1000). Density is water's; stiffness and
# Poisson ratio carry no information in a single-material liquid domain and are placeholders.
WATER3D_FILL_E, WATER3D_FILL_NU, WATER3D_FILL_RHO = 1.0e4, 0.40, 1000.0
WATER_PARTICLE_TYPE = 5          # the Learning-to-Simulate water type id
WATER3D_SHIFT = CANONICAL_TRANSLATION["water"]


# ----------------------------------------------------------------------------- tfrecord + protobuf
def iter_tfrecord(path: str) -> Iterator[bytes]:
    """Yield the raw bytes of each record: uint64 LE length, uint32 masked CRC32C of
    the length, data, uint32 masked CRC32C of the data. CRCs are skipped, not checked."""
    with open(path, "rb") as f:
        while True:
            head = f.read(12)
            if len(head) < 12:
                return
            (n,) = struct.unpack("<Q", head[:8])
            data = f.read(n)
            if len(data) < n:
                raise ValueError(f"truncated record in {path}")
            f.read(4)
            yield data


def _varint(buf: bytes, i: int):
    shift, out = 0, 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7


def _fields(buf: bytes):
    """Top-level (field number, wire type, value) triples of one message. Wire type 0
    yields the varint, 2 the byte slice, 1 and 5 the raw fixed-width bytes."""
    i, n, out = 0, len(buf), []
    while i < n:
        tag, i = _varint(buf, i)
        num, wt = tag >> 3, tag & 7
        if wt == 0:
            val, i = _varint(buf, i)
        elif wt == 2:
            ln, i = _varint(buf, i)
            val = buf[i:i + ln]
            i += ln
        elif wt == 1:
            val = buf[i:i + 8]
            i += 8
        elif wt == 5:
            val = buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wt}")
        out.append((num, wt, val))
    return out


def _packed_varints(buf: bytes):
    i, out = 0, []
    while i < len(buf):
        v, i = _varint(buf, i)
        out.append(v)
    return out


def _feature(buf: bytes) -> dict:
    """tf.train.Feature -> {"bytes": [...]} | {"floats": ndarray} | {"ints": ndarray}."""
    for num, _, inner in _fields(buf):
        items = _fields(inner)
        if num == 1:
            return {"bytes": [v for f, wt, v in items if f == 1]}
        if num == 2:
            vals = []
            for f, wt, v in items:
                if f == 1:
                    vals.extend(np.frombuffer(v, dtype="<f4").tolist() if wt == 2 else [struct.unpack("<f", v)[0]])
            return {"floats": np.asarray(vals, dtype=np.float32)}
        if num == 3:
            vals = []
            for f, wt, v in items:
                if f == 1:
                    vals.extend(_packed_varints(v) if wt == 2 else [v])
            return {"ints": np.asarray(vals, dtype=np.int64)}
    return {}


def _feature_map(buf: bytes) -> dict:
    """tf.train.Features -> {name: Feature dict}."""
    out = {}
    for num, _, entry in _fields(buf):
        if num != 1:
            continue
        key, val = None, None
        for f, _, v in _fields(entry):
            if f == 1:
                key = v.decode("utf-8")
            elif f == 2:
                val = _feature(v)
        out[key] = val
    return out


def decode_sequence_example(buf: bytes):
    """tf.train.SequenceExample -> (context {name: Feature}, feature_lists {name: [Feature]})."""
    context, lists = {}, {}
    for num, _, v in _fields(buf):
        if num == 1:
            context = _feature_map(v)
        elif num == 2:
            for f, _, entry in _fields(v):
                if f != 1:
                    continue
                key, feats = None, []
                for ff, _, vv in _fields(entry):
                    if ff == 1:
                        key = vv.decode("utf-8")
                    elif ff == 2:
                        feats = [_feature(x) for f3, _, x in _fields(vv) if f3 == 1]
                lists[key] = feats
    return context, lists


# ----------------------------------------------------------------------------- trajectories
@dataclass
class Water3DTrajectory:
    key: int
    positions: np.ndarray            # [T, N, dim] float32, the file's frame (y up, unit box)
    particle_type: np.ndarray        # [N] int64 (5 = water)
    step_context: Optional[np.ndarray] = None
    outside_bounds: float = float("nan")   # fraction of (frame, particle) rows outside the metadata bounds

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_particles(self) -> int:
        return int(self.positions.shape[1])


def load_metadata(root: str) -> dict:
    return json.load(open(os.path.join(root, "metadata.json")))


def iter_water3d(tfrecord: str, metadata: Optional[dict] = None, limit: Optional[int] = None,
                 skip: int = 0) -> Iterator[Water3DTrajectory]:
    """Decode trajectories in file order; validates the frame count against the
    metadata (``sequence_length + 1``) and the box bounds when metadata is given."""
    dim = int(metadata["dim"]) if metadata else 3
    n = 0
    for k, rec in enumerate(iter_tfrecord(tfrecord)):
        if k < skip:
            continue
        if limit is not None and n >= limit:
            return
        context, lists = decode_sequence_example(rec)
        key = int(context["key"]["ints"][0]) if "key" in context and "ints" in context["key"] else k
        ptype = np.frombuffer(context["particle_type"]["bytes"][0], dtype="<i8").copy()
        frames = [np.frombuffer(f["bytes"][0], dtype="<f4").reshape(-1, dim) for f in lists["position"]]
        pos = np.stack(frames).astype(np.float32)
        ctx = None
        if "step_context" in lists:
            ctx = np.stack([np.frombuffer(f["bytes"][0], dtype="<f4") for f in lists["step_context"]])
        if pos.shape[1] != ptype.shape[0]:
            raise ValueError(f"record {k}: {pos.shape[1]} positions vs {ptype.shape[0]} particle types")
        if metadata:
            want = int(metadata["sequence_length"]) + 1
            if pos.shape[0] != want:
                raise ValueError(f"record {k}: {pos.shape[0]} frames, metadata says {want}")
            lo = np.asarray(metadata["bounds"])[:, 0]
            hi = np.asarray(metadata["bounds"])[:, 1]
            outside = float(((pos < lo) | (pos > hi)).any(-1).mean())
        else:
            outside = float("nan")
        yield Water3DTrajectory(key=key, positions=pos, particle_type=ptype, step_context=ctx, outside_bounds=outside)
        n += 1


def summarize_split(tfrecord: str, metadata: dict) -> dict:
    """Trajectory count, particle-count range and per-step velocity std (file steps)
    of one split; one pass, positions kept one trajectory at a time."""
    counts, vstd, n = [], [], 0
    for tr in iter_water3d(tfrecord, metadata):
        counts.append(tr.num_particles)
        v = tr.positions[1:] - tr.positions[:-1]
        vstd.append(v.reshape(-1, v.shape[-1]).std(axis=0))
        n += 1
    return {"trajectories": n, "particles_min": int(min(counts)), "particles_max": int(max(counts)),
            "particles_mean": float(np.mean(counts)), "frames": int(metadata["sequence_length"]) + 1,
            "vel_std_per_step": np.mean(vstd, axis=0).tolist(), "metadata_vel_std": metadata["vel_std"]}


# ----------------------------------------------------------------------------- model frame
def to_model_frame(pos: np.ndarray, shift=WATER3D_SHIFT) -> np.ndarray:
    """File frame (y up) -> MPM convention (z up): rotate about x by +90 degrees,
    (x, y, z) -> (x, -z, y), then the fixed shift. Any leading dims."""
    out = np.stack([pos[..., 0], -pos[..., 2], pos[..., 1]], axis=-1)
    return out + np.asarray(shift, dtype=pos.dtype)


def from_model_frame(pos: np.ndarray, shift=WATER3D_SHIFT) -> np.ndarray:
    """Inverse of ``to_model_frame``: MPM convention (z up) -> the file's frame (y up),
    raw box units. Any leading dims."""
    q = pos - np.asarray(shift, dtype=pos.dtype)
    return np.stack([q[..., 0], q[..., 2], -q[..., 1]], axis=-1)


def particle_subset(num_particles: int, k: int = WATER3D_PARTICLES, seed: int = WATER3D_SUBSET_SEED) -> np.ndarray:
    """Sorted indices of a fixed random subset (all particles when N <= k)."""
    if num_particles <= k:
        return np.arange(num_particles)
    return np.sort(np.random.default_rng(seed).choice(num_particles, size=k, replace=False))


def window_motion(positions: np.ndarray, start: int, length: int, stride: int, idx: np.ndarray) -> float:
    """Mean per-frame displacement (file units) of the subset over one strided window of
    ``positions`` [T, N, 3] (an array or a memory map)."""
    seg = np.asarray(positions[start:start + length * stride:stride][:, idx])
    return float(np.linalg.norm(seg[1:] - seg[:-1], axis=-1).mean()) if length > 1 else 0.0


def select_windows(positions: np.ndarray, k: int, context_len: int, future_len: int, stride: int,
                   idx: np.ndarray, search_stride: Optional[int] = None) -> list:
    """The ``k`` non-overlapping strided window starts with the most motion (greedy,
    descending); starts are searched every ``search_stride`` file frames (default: ``stride``)."""
    L = context_len + future_len
    span = (L - 1) * stride + 1
    T = int(positions.shape[0])
    if T < span:
        raise ValueError(f"trajectory has {T} frames, a strided window needs {span}")
    starts = list(range(0, T - span + 1, search_stride or stride))
    ranked = sorted(starts, key=lambda s: window_motion(positions, s, L, stride, idx), reverse=True)
    chosen = []
    for s in ranked:
        if all(abs(s - c) >= span for c in chosen):
            chosen.append(s)
        if len(chosen) == k:
            break
    return sorted(chosen)


def water_batch(windows, *, context_len: int, delta_stats, position_stats, particles: Optional[int] = None,
                E: float = WATER3D_FILL_E, nu: float = WATER3D_FILL_NU, rho: float = WATER3D_FILL_RHO) -> TrajectoryBatch:
    """Model-frame windows (a list of [L, n_i, 3] arrays, equal L) -> one ``water`` batch.
    Windows with fewer than ``particles`` rows (default: the largest n_i) are zero-padded
    at the end and masked by ``valid_lengths``. Passive domain: zero actions, no actuated
    particles, one object, no rigid sidecar."""
    L = int(windows[0].shape[0])
    if any(int(w.shape[0]) != L for w in windows):
        raise ValueError("water windows in one batch must have the same number of frames")
    N = int(particles or max(int(w.shape[1]) for w in windows))
    B, F = len(windows), L - context_len
    if F <= 0:
        raise ValueError(f"a {L}-frame window leaves no future frames after {context_len} context frames")
    positions = torch.zeros(B, L, N, 3)
    lengths = torch.zeros(B, dtype=torch.long)
    for b, w in enumerate(windows):
        n = int(w.shape[1])
        if n > N:
            raise ValueError(f"window has {n} particles, the batch holds {N}")
        positions[b, :, :n] = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
        lengths[b] = n
    return TrajectoryBatch(
        positions=positions,
        actions=torch.zeros(B, F, N, 3),
        action_mask=torch.zeros(B, N, dtype=torch.bool),
        object_ids=torch.zeros(B, N, dtype=torch.long),
        valid_lengths=lengths,
        E=torch.full((B, N), float(E)), nu=torch.full((B, N), float(nu)), rho=torch.full((B, N), float(rho)),
        mat_type=torch.full((B, N), MAT_WATER, dtype=torch.long),
        context_len=context_len, delta_stats=delta_stats, position_stats=position_stats,
        prediction_target="delta", domain="water")


def _rank_world():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


class UnifiedWaterDataset(torch.utils.data.IterableDataset):
    """Materialized Water-3D split -> ``water`` TrajectoryBatches (batch_size windows each).

    ``root`` is one split directory of scripts/unified/materialize_water3d.py (manifest.json
    + one float32 [T, N, 3] .npy per trajectory, file frame). Windows are ``context_len +
    future_len`` frames at ``stride`` file frames apart, mapped to the model frame.

    Training (``training=True``): an endless stream; each window draws a trajectory, a start
    file frame whose window fits, and a fresh random subset of ``particles`` from a generator
    seeded by (torch.initial_seed(), rank, worker), so ranks and DataLoader workers differ.
    Validation/evaluation: finite and deterministic; one window per trajectory in manifest
    order, its start drawn from the trajectory key (start 0 with ``first_window``, or when the
    requested future exceeds the frames left, which yields the rest of the trajectory: the
    evaluator's ``future_len = 10_000`` rollout convention; ``future_len = -1`` likewise), one
    seeded particle subset per trajectory. Trajectories are split across ranks unless
    ``no_shard_split`` and across DataLoader workers. ``future_len`` is read at iteration
    time. The unified loaders set batch_size, first_window, no_shard_split, shuffle,
    bucket_max_items and emit_sidecar after construction (the last two are no-ops here).
    """

    domain = "water"

    def __init__(self, root, *, context_len: int = 4, future_len: int = 16, training: bool = True,
                 shuffle: bool = True, normalization=None, raw: bool = False, stride: int = WATER3D_STRIDE,
                 particles: int = WATER3D_PARTICLES, subset_seed: int = WATER3D_SUBSET_SEED):
        super().__init__()
        from data.unified_dataset import _identity_delta_stats, _validate_window, unified_position_stats
        _validate_window(context_len, future_len)
        self.root = os.fspath(root).rstrip("/")
        with open(os.path.join(self.root, "manifest.json")) as f:
            self.items = json.load(f)["trajectories"]
        if not self.items:
            raise ValueError(f"no trajectories under {self.root}")
        self.context_len, self.future_len = int(context_len), future_len
        self.stride, self.particles, self.subset_seed = int(stride), int(particles), int(subset_seed)
        self.training, self.shuffle, self.raw = bool(training), bool(shuffle), bool(raw)
        self.normalization = None
        if raw:
            self.delta_stats, self.position_stats = _identity_delta_stats(), unified_position_stats()
        else:
            from data.normalization import domain_normalizers, resolve_normalization
            self.normalization = resolve_normalization(normalization)
            self.delta_stats, self.position_stats, _ = domain_normalizers(self.normalization, "water")
        self.batch_size, self.first_window, self.no_shard_split = 4, False, False
        self.bucket_max_items, self.emit_sidecar = 0, False
        print(f"[Unified/water] {len(self.items)} trajectories under {self.root}")

    def _pose_stats(self):
        return None                                   # passive domain: no rigid bodies

    def _future(self) -> Optional[int]:
        f = self.future_len
        return None if f is None or f <= 0 else int(f)

    def _load(self, item, frames, idx) -> np.ndarray:
        array = np.load(os.path.join(self.root, item["file"]), mmap_mode="r")
        return to_model_frame(np.asarray(array[frames][:, idx], dtype=np.float32))

    def _batch(self, windows) -> TrajectoryBatch:
        return water_batch(windows, context_len=self.context_len, delta_stats=self.delta_stats,
                           position_stats=self.position_stats, particles=self.particles)

    def __iter__(self):
        rank, world = _rank_world()
        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info is not None else (0, 1)
        if self.training:
            yield from self._train(np.random.default_rng([torch.initial_seed() % (2 ** 63), rank, worker]))
        else:
            items = list(enumerate(self.items))
            if not self.no_shard_split:
                items = items[rank::world]
            yield from self._eval(items[worker::workers])

    def _train(self, rng):
        C, F = self.context_len, self._future()
        if F is None:
            raise ValueError("water training needs a positive future_len")
        L = C + F
        span = (L - 1) * self.stride + 1
        items = [it for it in self.items if int(it["frames"]) >= span]
        if not items:
            raise ValueError(f"no trajectory has the {span} file frames a {L}-frame window needs")
        while True:
            windows = []
            for _ in range(self.batch_size):
                it = items[int(rng.integers(len(items)))]
                start = int(rng.integers(0, int(it["frames"]) - span + 1))
                n = int(it["particles"])
                idx = np.arange(n) if n <= self.particles else np.sort(rng.choice(n, self.particles, replace=False))
                windows.append(self._load(it, start + self.stride * np.arange(L), idx))
            yield self._batch(windows)

    def _eval(self, items):
        C, F = self.context_len, self._future()
        windows = []
        for _, it in items:
            T = int(it["frames"])
            left = (T - 1) // self.stride + 1                          # strided frames from start 0
            if left <= C:
                raise ValueError(f"trajectory {it['key']} has only {left} strided frames")
            if F is None or C + F >= left or self.first_window:
                start = 0
            else:
                last = T - ((C + F - 1) * self.stride + 1)
                start = int(np.random.default_rng(zlib.crc32(str(it["key"]).encode())).integers(0, last + 1))
            n_frames = (T - 1 - start) // self.stride + 1
            L = n_frames if F is None else min(C + F, n_frames)
            idx = particle_subset(int(it["particles"]), self.particles, self.subset_seed)
            windows.append(self._load(it, start + self.stride * np.arange(L), idx))
            if len(windows) == self.batch_size:
                yield self._batch(windows)
                windows = []
        if windows:
            yield self._batch(windows)
