"""Frozen episode indices for opt-in collections with heterogeneous layouts.

One training batch contains independently drawn windows of one episode. This
keeps articulated topology and point identities uniform within a batch, bounds
memory, and amortizes compressed-episode I/O. Episodes are shuffled uniformly;
long clips do not silently get more weight. DDP ranks/workers own disjoint rows.
"""
from __future__ import annotations

from functools import lru_cache
from collections import deque
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
import webdataset as wds

from data.batch import TrajectoryBatch

INDEX_SCHEMA = "pwm-indexed-episodes-v1"
DOMAINS = ("mpm_fresh", "mpm_drag", "folding", "yikai", "dex")


@lru_cache(maxsize=24)
def _read_index(path, size, mtime_ns):
    content = Path(path).read_bytes()
    value = json.loads(content)
    if (value.get("schema") != INDEX_SCHEMA or value.get("domain") not in DOMAINS
            or value.get("split") not in ("train", "val", "test")
            or not isinstance(value.get("records"), list) or not value["records"]):
        raise ValueError(f"Invalid/nonempty indexed episode release required: {path}")
    ids, sources = set(), set()
    for record in value["records"]:
        if not isinstance(record, dict):
            raise ValueError("Episode records must be mappings")
        for key in ("episode_id", "group", "file"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Episode record requires {key}")
        for key in ("frames", "size", "mtime_ns"):
            if type(record.get(key)) is not int or record[key] <= 0:
                raise ValueError(f"Episode record requires positive integer {key}")
        if not Path(record["file"]).is_absolute():
            raise ValueError("Indexed source paths must be absolute")
        source = source_identity(record)
        if record["episode_id"] in ids or source in sources:
            raise ValueError(f"Duplicate episode/source in {path}: {record['episode_id']}")
        ids.add(record["episode_id"]); sources.add(source)
    value["index_sha256"] = hashlib.sha256(content).hexdigest()
    return value


def load_index(root, domain=None, split=None):
    """Read one split's INDEX.json; `split` is one accepted name or a tuple of them."""
    path = Path(root).resolve() / "INDEX.json"
    stat = path.stat()
    value = _read_index(str(path), stat.st_size, stat.st_mtime_ns)
    if domain is not None and value["domain"] != domain:
        raise ValueError(f"Index domain {value['domain']} differs from requested {domain}")
    accepted = (split,) if isinstance(split, str) else tuple(split or ())
    if accepted and value["split"] not in accepted:
        raise ValueError(f"Index split {value['split']} differs from requested {'/'.join(accepted)}")
    return value


def source_identity(record):
    physical = ((record["device"], record["inode"]) if "device" in record and "inode" in record
                else _canonical_source(record["file"]))
    return (physical, record.get("key"), record.get("row"), record.get("env"))


@lru_cache(maxsize=65536)
def _canonical_source(path):
    return str(Path(path).resolve())


@lru_cache(maxsize=2048)
def _verify_auxiliary(path, size, mtime_ns, expected):
    stat = Path(path).stat()
    if (stat.st_size, stat.st_mtime_ns) != (size, mtime_ns):
        raise ValueError(f"Auxiliary geometry changed: {path}")
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Auxiliary geometry checksum mismatch: {path}")


def validate_index_pair(train, val, domain):
    """The validation root holds a `val` index or, as in pwm-data-v1, the release's `test` index.

    Since 2026-09-17 the test sets double as validation sets: curves and checkpoint selection
    read them, and reported numbers say so. Either way the two roots must be disjoint."""
    a, b = load_index(train, domain, "train"), load_index(val, domain, ("val", "test"))
    for field in ("group", "episode_id"):
        if {r[field] for r in a["records"]} & {r[field] for r in b["records"]}:
            raise ValueError(f"{domain}: train/val {field} overlap")
    if {source_identity(r) for r in a["records"]} & {source_identity(r) for r in b["records"]}:
        raise ValueError(f"{domain}: physical source episode overlaps train/val")
    return a, b


def evaluation_stratum(record, domain):
    """Collection categories that a short validation prefix should cover."""
    field = {"dex": "teacher", "folding": "action"}.get(domain)
    return str(record.get(field, "all")) if field is not None else "all"


def _round_robin(groups):
    pending = deque(iter(group) for group in groups)
    while pending:
        stream = pending.popleft()
        try:
            value = next(stream)
        except StopIteration:
            continue
        yield value
        pending.append(stream)


def evaluation_records(records, domain):
    """Stable teacher/action coverage, then rotate scenes/garments within strata.

    Every episode appears exactly once. Hash ordering avoids a serialized-prefix
    bias without consuming sampler/training RNG or changing training membership.
    """
    buckets = {}
    for record in records:
        label = evaluation_stratum(record, domain)
        buckets.setdefault(label, {}).setdefault(record["group"], []).append(record)
    strata = []
    for label, groups in sorted(buckets.items()):
        ordered_groups = sorted(groups, key=lambda group: hashlib.sha256(
            f"eval:{label}:{group}".encode()).digest())
        strata.append(_round_robin([
            sorted(groups[group], key=lambda row: hashlib.sha256(row["episode_id"].encode()).digest())
            for group in ordered_groups]))
    return list(_round_robin(strata))


class IndexedEpisodeDataset(torch.utils.data.IterableDataset):
    emit_sidecar = True
    articulated = False
    batch_size = 0
    first_window = False
    no_shard_split = False
    # Training window policy, set per domain by the loader from data.window_motion and
    # data.batch_across_episodes (recipe decisions of 2026-09-17). alpha 0 is the historical
    # uniform draw; evaluation windows never depend on either setting.
    motion_alpha = 0.0
    motion_floor_m = 0.0
    cross_episode = False

    def __init__(self, root, *, domain, context_len=4, future_len=16,
                 training=True, shuffle=True, normalization=None, raw=False):
        super().__init__()
        from data.recording_adapters import normalizers
        from data.normalization import resolve_normalization
        from data.unified_dataset import _validate_window
        _validate_window(context_len, future_len)
        if training and future_len <= 0:
            raise ValueError("Indexed episode training requires an explicit positive future_len")
        self.root, self.domain = str(Path(root).resolve()), domain
        self.index = load_index(root, domain)
        self.records = self.index["records"]
        self.source_split = self.index["split"]
        self.training, self.shuffle, self.raw = bool(training), bool(shuffle), bool(raw)
        self.context_len, self.future_len = context_len, future_len
        self.normalization = None if raw else resolve_normalization(normalization)
        self.delta_stats, self.position_stats, self._pose_stats_cache = normalizers(domain, self.normalization, raw)
        self.roots, self.shards = [self.root], sorted({r["file"] for r in self.records})
        self.num_collate = 1
        self._motion_cache = {}
        from data.recording_adapters import FIRST_START
        self.first_start = FIRST_START.get(domain, 0)        # yikai skips its settling frames
        # An index may carry a per-record first_start (yikai_small: the settle frames of the SOURCE recording
        # a segment was cut from, max(0, 10 - source_start)); a record without one uses the domain's.
        if any("first_start" in r and (type(r["first_start"]) is not int or r["first_start"] < 0) for r in self.records):
            raise ValueError("Index record first_start must be a nonnegative integer")
        if any(r["frames"] <= self._first(r) + context_len for r in self.records):
            raise ValueError("Index contains an episode with no future after context")
        if training and any(r["frames"] < self._first(r) + context_len + future_len for r in self.records):
            raise ValueError("Index includes episodes too short for the requested C+F; rebuild its selection")

    def _first(self, record):
        """First allowed window frame: the record's own first_start if its index carries one."""
        return record.get("first_start", self.first_start)

    def _future(self, record):
        available = record["frames"] - self._first(record) - self.context_len
        return available if self.future_len == -1 else min(self.future_len, available)

    def _pose_stats(self):
        return self._pose_stats_cache

    def window(self, record, start):
        from data.recording_adapters import ADAPTERS
        future_len = self._future(record)
        if type(start) is not int or start < self._first(record) or start + self.context_len + future_len > record["frames"]:
            raise ValueError("Window outside indexed episode")
        stat = Path(record["file"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError(f"Source changed since indexing: {record['file']}")
        for prefix in ("geometry", "template", "subset"):
            if f"{prefix}_file" in record:
                auxiliary_stat = Path(record[f"{prefix}_file"]).stat()
                if (auxiliary_stat.st_size, auxiliary_stat.st_mtime_ns) != (record[f"{prefix}_size"], record[f"{prefix}_mtime_ns"]):
                    raise ValueError(f"Auxiliary geometry changed: {record[f'{prefix}_file']}")
                _verify_auxiliary(*(record[f"{prefix}_{key}"] for key in ("file", "size", "mtime_ns", "sha256")))
        stats = self.delta_stats, self.position_stats, self._pose_stats_cache
        batch = ADAPTERS[self.domain](record, start, self.context_len, future_len,
                                     stats, self.domain, self.articulated)
        if not torch.isfinite(batch.positions).all() or batch.num_frames != self.context_len + future_len:
            raise ValueError(f"Invalid positions/window in indexed episode {record['episode_id']}")
        return batch

    def _motion(self, record):
        """Mean step of the predicted particles per transition [T-1], metres, cached per episode."""
        from data.recording_adapters import MOTION
        key = record["episode_id"]
        if key not in self._motion_cache:
            if self.domain not in MOTION:
                raise ValueError(f"{self.domain} has no motion profile; motion-weighted starts need one")
            motion = np.asarray(MOTION[self.domain](record), dtype=np.float64)
            if motion.shape != (record["frames"] - 1,) or not np.isfinite(motion).all() or (motion < 0).any():
                raise ValueError(f"Invalid motion profile for {record['episode_id']}")
            self._motion_cache[key] = motion
        return self._motion_cache[key]

    def _start_weights(self, record):
        """p(start) ∝ (mean future step of the predicted particles + floor)^alpha; None means uniform."""
        if self.motion_alpha <= 0:
            return None
        if not self.motion_floor_m > 0:
            raise ValueError("motion_floor_m must be positive so every window stays reachable")
        C, F = self.context_len, self._future(record)
        total = np.concatenate([[0.0], np.cumsum(self._motion(record))])
        starts = np.arange(self._first(record), record["frames"] - C - F + 1)
        score = (total[starts + C + F - 1] - total[starts + C - 1]) / F     # the window's F future transitions
        return torch.tensor((score + self.motion_floor_m) ** self.motion_alpha, dtype=torch.float64)

    def _draw_starts(self, record, count):
        low, high = self._first(record), record["frames"] - self.context_len - self._future(record) + 1
        if self.first_window:
            return [low] * count
        if not self.training:
            # Evaluation is independent of ambient training/sampler RNG and of the window policy.
            seed = int.from_bytes(hashlib.sha256(record["episode_id"].encode()).digest()[:8], "little")
            return np.random.default_rng(seed).integers(low, high, size=count).tolist()
        weights = self._start_weights(record)
        if weights is None:                       # alpha 0: the historical uniform draw, bit for bit
            return (low + torch.randint(high - low, (count,))).tolist()
        return (low + torch.multinomial(weights / weights.sum(), count, replacement=True)).tolist()

    def _emit(self, draws):
        """Collate (record, start) pairs into one batch and keep their provenance."""
        batch = TrajectoryBatch.collate([self.window(record, start) for record, start in draws])
        batch.source_shard = draws[0][0]["file"]
        batch.source_episode_id = draws[0][0]["episode_id"]
        batch.source_episode_ids = [record["episode_id"] for record, _ in draws]
        policy = "motion" if self.training and self.motion_alpha > 0 and not self.first_window else "uniform"
        self.last_windows = [{"episode_id": record["episode_id"], "start": start, "file": record["file"],
                              "index_sha256": self.index["index_sha256"], "start_policy": policy,
                              **({"table_top_z_env": record["table_top_z_env"]} if "table_top_z_env" in record else {})}
                             for record, start in draws]
        return batch

    @staticmethod
    def _bucket(record):
        # Only equal point counts collate; an MPM scene carries 1,024 particles per listed asset.
        return len(record["asset_ids"]) if "asset_ids" in record else 0

    def _iter_across_episodes(self, selected, batch_size):
        """One window from each of B different episodes with equal point count. Passes visit every
        episode; a bucket's leftovers wait for the next pass rather than form a short batch, and an
        episode still waiting is skipped when the next pass reaches it, so a batch never repeats a scene."""
        pending = {}
        while True:
            order = torch.randperm(len(selected)).tolist() if self.shuffle else range(len(selected))
            for i in order:
                record = selected[i]
                bucket = pending.setdefault(self._bucket(record), [])
                if any(row["episode_id"] == record["episode_id"] for row in bucket):
                    continue
                bucket.append(record)
                if len(bucket) == batch_size:
                    pending[self._bucket(record)] = []
                    yield self._emit([(row, self._draw_starts(row, 1)[0]) for row in bucket])

    def __iter__(self):
        rank, world, worker, workers = wds.utils.pytorch_worker_info()
        records = self.records if self.training else evaluation_records(self.records, self.domain)
        # Validation may replicate ranks so collective metric reductions align;
        # workers must still partition rows to avoid scoring episodes twice.
        selected = (records[worker::workers] if self.no_shard_split else
                    records[rank * workers + worker::world * workers])
        if not selected:
            if self.training:
                raise ValueError("No indexed episodes assigned to this rank/worker; reduce workers")
            return
        batch_size = max(1, int(self.batch_size))
        if self.training and self.cross_episode:
            yield from self._iter_across_episodes(selected, batch_size)
            return
        while True:
            order = torch.randperm(len(selected)).tolist() if self.training and self.shuffle else range(len(selected))
            for i in order:
                record = selected[i]
                yield self._emit([(record, start) for start in self._draw_starts(record, batch_size)])
            if not self.training:
                return
