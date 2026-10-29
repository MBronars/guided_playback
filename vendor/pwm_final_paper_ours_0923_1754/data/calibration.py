"""Finite, reproducible calibration traversal of the canonical raw adapters.

No normalization assets are opened. This module selects physical trajectories;
fitting/aggregation and bounded reservoir sampling belong to the stats fitter.
The returned ledger identifies source rows and exact native-view frame windows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data.blocklists import canonical_shard, manifest_path
from data.dataset import SkipSample
from data.unified_dataset import (CANONICAL_TRANSLATION, DOMAIN_DT,
                                  _validate_window, build_domain_dataset)


def calibration_source_manifest(dataset, *, seed, window_policy, max_records):
    """Cache identity covering sources, exclusions, sidecars and window policy.

Large corpus tarballs use real path + size + nanosecond mtime (NOT a claim of
full content hashing). Small membership and pose-link assets are byte-hashed.
Changing a source inventory, exclusions, C/F or sampling policy changes sha256.
"""
    _validate_request(dataset.context_len, dataset.future_len, seed, window_policy, max_records)
    sources = []
    for name in sorted(dataset.shards):
        path = Path(canonical_shard(name))
        stat = path.stat()
        sources.append(dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    assets = []
    paths = list(getattr(dataset, "blocklist_paths", ()))
    if dataset.domain == "cloth" and getattr(dataset, "emit_sidecar", False):
        paths.append(dataset.SIDECAR_IDS)
    for name in paths:
        path = manifest_path(name).resolve()
        assets.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    implementation = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                      for name in ("calibration.py", "dataset.py", "unified_dataset.py", "blocklists.py")}
    result = dict(schema_version=1, adapter_version="unified-raw-v1", domain=dataset.domain,
                  split=getattr(dataset, "source_split", "unspecified"),
                  implementation=implementation,
                  roots=[canonical_shard(root) for root in getattr(dataset, "roots", ())],
                  sources=sources, assets=assets, context_len=dataset.context_len,
                  future_len=dataset.future_len, seed=int(seed), window_policy=window_policy,
                  max_records=int(max_records), emit_sidecar=bool(getattr(dataset, "emit_sidecar", False)),
                  position_translation_m=list(CANONICAL_TRANSLATION[dataset.domain]),
                  native_frame_stride=2 if dataset.domain == "cloth" else 1,
                  dt_s=DOMAIN_DT[dataset.domain],
                  motion_alpha=float(getattr(dataset, "motion_alpha", 0)),
                  motion_floor_m=float(getattr(dataset, "motion_floor_m", 0)),
                  shard_order="seeded_permutation_without_replacement",
                  window_seed="sha256(seed,canonical_shard,key); first 63 bits")
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return result


def _validate_request(context_len, future_len, seed, window_policy, max_records):
    _validate_window(context_len, future_len)
    if window_policy not in ("training", "prefix", "uniform"):
        raise ValueError("window_policy must be training, prefix, or uniform")
    for name, value in (("seed", seed), ("max_records", max_records)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    if seed < 0 or max_records < 1:
        raise ValueError("seed must be nonnegative and max_records must be positive")


def _window_valid_mask(sample, batch, starts):
    """Retain the actual source mask, including temporal changes and padding."""
    valid = sample.get("valid_particles.pyd")
    if valid is None:
        return torch.ones(1, 1, batch.num_particles, dtype=torch.bool).expand(
            batch.batch_size, batch.num_frames, batch.num_particles)
    valid = torch.as_tensor(valid, dtype=torch.bool)
    if valid.ndim == 1:
        valid = valid[None, None, :].expand(batch.batch_size, batch.num_frames, -1)
        return valid[:, :, :batch.num_particles]
    if valid.ndim == 2:
        # Batched static mask [B,N], rather than the usual [B,T,N].
        valid = valid[:, None, :].expand(-1, batch.num_frames, -1)
        return valid[:, :, :batch.num_particles]
    return torch.stack([valid[i, start:start + batch.num_frames, :batch.num_particles]
                        for i, start in enumerate(starts)])


def iter_calibration_batches(domain, *, context_len, future_len, seed, max_records,
                             window_policy="training", emit_sidecar=True, split="train",
                             dataset=None):
    """Yield (raw TrajectoryBatch, metadata) from a deterministic finite pass.

Each source record is visited at most once; max_records caps emitted records,
which can contain multiple episodes. Per-record RNG isolation makes window
selection independent of skipped predecessors and preserves the caller's RNG.
The source is finite even when every record is explicitly excluded.

Use split='train' for fitting. split='val' is available ONLY for separately
labelled diagnostics, never for fitting a selected normalization candidate.
"""
    _validate_request(context_len, future_len, seed, window_policy, max_records)
    if split not in ("train", "val"):
        raise ValueError("Calibration split must be train or val")
    ds = dataset if dataset is not None else build_domain_dataset(
        domain, split, context_len=context_len, future_len=future_len, raw=True)
    if not getattr(ds, "raw", False):
        raise ValueError("Calibration requires a raw=True dataset; old stats must not be loaded")
    if ds.domain != domain or ds.context_len != context_len or ds.future_len != future_len:
        raise ValueError("Calibration dataset domain/C/F do not match the requested contract")
    if getattr(ds, "source_split", split) != split:
        raise ValueError("Calibration dataset split does not match the requested contract")
    changed = ("emit_sidecar", "shuffle", "no_shard_split", "strict_read", "record_source_rows",
               "training", "first_window", "motion_alpha")
    original_attributes = {name: ds.__dict__.get(name) for name in changed}
    originally_set = set(ds.__dict__) & set(changed)
    ds.emit_sidecar = emit_sidecar
    ds.shuffle, ds.no_shard_split, ds.strict_read, ds.record_source_rows = False, True, True, True
    ds.training, ds.first_window = True, window_policy == "prefix"
    if window_policy == "uniform" and hasattr(ds, "motion_alpha"):
        ds.motion_alpha = 0.0
    shards = list(ds.shards)
    ds.shards = [shards[i] for i in np.random.default_rng(seed).permutation(len(shards))]
    count = 0
    skipped = []
    try:
        for sample in ds._create_pipeline():
            source, key = canonical_shard(sample.get("__url__", "")), str(sample.get("__key__", ""))
            encoded = json.dumps([int(seed), source, key], separators=(",", ":")).encode()
            window_seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
            try:
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(window_seed)
                    batch = ds._process_sample(sample)
            except SkipSample as error:
                skipped.append(dict(source=source, key=key, reason=str(error)))
                continue
            except Exception as error:
                raise RuntimeError(f"Calibration failed for {source}:{key}: {error}") from error
            if not torch.isfinite(batch.positions).all():
                raise ValueError(f"Nonfinite calibration positions: {source}:{key}")
            batch.domain = domain
            starts = list(getattr(ds, "_window_starts", [0] * batch.batch_size))
            rows = np.asarray(sample.get("calibration_rows.pyd", np.arange(batch.batch_size))).tolist()
            metadata = dict(domain=domain, split=split, source=source, key=key, source_rows=rows,
                            context_len=context_len, future_len=batch.num_frames - context_len,
                            window_starts=starts, window_seed=window_seed, seed=int(seed),
                            window_policy=window_policy, skipped_before=skipped,
                            valid_particle_mask=_window_valid_mask(sample, batch, starts))
            skipped = []
            yield batch, metadata
            count += 1
            if count >= max_records:
                break
    finally:
        ds.shards = shards
        for name in changed:
            if name in originally_set:
                setattr(ds, name, original_attributes[name])
            else:
                ds.__dict__.pop(name, None)
