"""Bounded source-selection parity and warm multi-shard loader probe.

Only named project staging receives new copies. Existing datasets, exclusions,
model source and normalization remain unchanged. CPU-only; no speed guarantee.
"""
import copy
from dataclasses import fields
import hashlib
import os
from pathlib import Path
import time

from collection.common import canonical, digest, fsync_dir, rebase_exclusions, resolve_release, write_once



def copy_shard(source, destination):
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with source.open('rb') as src, destination.open('xb') as dst:
        h = hashlib.sha256()
        for block in iter(lambda: src.read(8 << 20), b''):
            h.update(block); dst.write(block)
        dst.flush(); os.fsync(dst.fileno())
    duration = time.monotonic() - started
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or digest(destination) != h.hexdigest():
        raise ValueError('Source/copy mismatch')
    fsync_dir(destination.parent)
    return dict(source=str(source), destination=str(destination), bytes=before.st_size,
                sha256=h.hexdigest(), read_write_fsync_seconds=duration)


def batch_digest(batch):
    import torch
    h = hashlib.sha256()
    for field in fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor):
            a = value.detach().cpu().numpy()
            h.update(canonical([field.name, a.dtype.str, a.shape]).encode() + b'\0')
            h.update(a.tobytes())
        elif field.name.endswith('_stats') and value is not None:
            for key in ('min_val', 'max_val'):
                a = getattr(value, key).cpu().numpy()
                h.update((field.name + '.' + key).encode() + b'\0'); h.update(a.tobytes())
        else:
            h.update(canonical([field.name, value]).encode() + b'\0')
    return h.hexdigest()


def measure(dataset, release, workers, *, rebase=True):
    from torch.utils.data import DataLoader
    ds = copy.deepcopy(dataset)
    ds.shards = list(map(str, release['shards']))
    ds.roots = [str(release['root'])]; ds.wds_location = str(release['root'])
    ds.shuffle = False; ds.first_window = True; ds.strict_read = True
    ds.batch_size = 4; ds.bucket_max_items = 8; ds.no_shard_split = False
    ds.emit_sidecar = False
    if rebase: ds.blocklist = rebase_exclusions(ds.blocklist, release['canonical_root'], release['root'])
    started = time.monotonic()
    iterator = iter(DataLoader(ds, batch_size=None, num_workers=workers, persistent_workers=False))
    startup = time.monotonic() - started
    durations, hashes, sizes = [], [], []
    try:
        while True:
            began = time.monotonic()
            try: batch = next(iterator)
            except StopIteration: break
            durations.append(time.monotonic() - began)
            hashes.append(batch_digest(batch)); sizes.append(int(batch.positions.shape[0]))
    finally:
        if hasattr(iterator, '_shutdown_workers'): iterator._shutdown_workers()
    if not hashes: raise ValueError('Empty selected stream')
    return dict(workers=workers, samples=sum(sizes), batches=len(hashes), batch_hashes=hashes,
                startup_s=startup, first_batch_wait_s=durations[0], sum_batch_wait_s=sum(durations),
                wall_s=time.monotonic()-started, rebase=rebase,
                timing_scope='Warm/uncontrolled CPU loading; hashing/prefetch overlap; no model, sidecars or GPU transfer')
