"""Frozen object-level cohort assignments and exact historical source banks.

These are candidate banks, not dataset promotion or a physical-volume claim.
All variants of a UID inherit its role; historical train UIDs never enter eval.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from collection.common import digest, write_once

ROLES = ('train', 'validation', 'test')
POOL_FILES = ('objaverse_uid_train.json', 'objaverse_uid_holdout.json')


def pools(root):
    root = Path(root)
    values = [json.loads((root / name).read_text()) for name in POOL_FILES]
    for ids in values:
        if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError('Unique nonempty string pool UIDs required')
    if set(values[0]) & set(values[1]):
        raise ValueError('Asset pools overlap')
    return values


def assignment(pool, namespace):
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError('Explicit nonempty split namespace required')
    train, heldout = pools(pool)
    ranked = sorted(heldout, key=lambda uid: (hashlib.sha256((namespace + ':' + uid).encode()).hexdigest(), uid))
    if len(ranked) < 14:
        raise ValueError('At least fourteen heldout UIDs for two seven-object cohorts')
    return dict(schema='mpm-object-cohorts-v1', namespace=namespace,
        policy='Preserve historical train; hash-ranked heldout halves; all UID variants inherit role',
        source_pool_sha256={name: digest(Path(pool) / name) for name in POOL_FILES},
        cohorts=dict(train=sorted(train), validation=sorted(ranked[:len(ranked)//2]), test=sorted(ranked[len(ranked)//2:])))


def verify_assignment(path, pool):
    value = json.loads(Path(path).read_text())
    if value != assignment(pool, value.get('namespace')):
        raise ValueError('Object cohort assignment changed or does not match pinned pools/policy')
    return value


def allowed_uids(metadata, pool, *, role='train', split_manifest=None):
    if role not in ROLES:
        raise ValueError('Unknown bank role')
    train, heldout = pools(pool)
    if split_manifest is None:
        if role != 'train' or metadata.get('bank_role', 'train') != 'train' or 'split_manifest_sha256' in metadata:
            raise ValueError('Explicit split manifest required for a role-aware bank')
        return set(train)
    value = verify_assignment(split_manifest, pool)
    if metadata.get('bank_role') != role or metadata.get('split_manifest_sha256') != digest(split_manifest):
        raise ValueError('Bank role or split manifest hash mismatch')
    return set(value['cohorts'][role])


def occupied_weights(x, n_grid, grid_lim):
    # Archived dx^3 / number-of-points-in-cell estimator. Copy raw weights,
    # never replace them with these verification values or call them solid volume.
    if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError('Finite original float32 Nx3 positions required')
    if n_grid < 4 or not np.isfinite(grid_lim) or grid_lim <= 0:
        raise ValueError('Invalid source grid')
    dx = np.float32(grid_lim / n_grid)
    cells = np.floor(x / dx).astype(np.int64)
    if cells.min() < 0 or cells.max() >= n_grid:
        raise ValueError('Source outside grid')
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    return dx * dx * dx / counts[inverse].astype(np.float32)


def build_bank(root, *, pool, split_manifest, role, raw_root, count=28, scan_limit=1024):
    import h5py
    from .recipes import load_bank
    if role not in ROLES or type(count) is not int or not 7 <= count <= 10000 or type(scan_limit) is not int or scan_limit < 1:
        raise ValueError('Explicit bounded bank size/scan and valid role required')
    assignment_value = verify_assignment(split_manifest, pool)
    allowed = set(assignment_value['cohorts'][role])
    if count > len(allowed):
        raise ValueError('Requested bank exceeds assigned cohort')
    source_hashes = {str(Path(pool) / name): digest(Path(pool) / name) for name in POOL_FILES}
    source_hashes[str(Path(split_manifest))] = digest(split_manifest)
    root, raw_root = Path(root), Path(raw_root).resolve(strict=True)
    candidates = sorted(raw_root.rglob('*.h5'), key=lambda path: (
        hashlib.sha256(('mpm-bank-v1:' + str(path.relative_to(raw_root))).encode()).hexdigest(), str(path)))[:scan_limit]
    root.mkdir(parents=True, exist_ok=False)
    write_once(root / 'PLAN.json', dict(role=role, count=count, scan_limit=scan_limit, raw_root=str(raw_root),
        candidates=[str(p.relative_to(raw_root)) for p in candidates], source_sha256=source_hashes,
        selection='Deterministic path order, cohort membership only; no simulation outcome filtering',
        code_sha256=digest(__file__), accepted_for_training=False))
    rows, seen, inspected = [], set(), []
    for path in candidates:
        with h5py.File(path, 'r') as f:
            uids = [u.decode() if isinstance(u, bytes) else str(u) for u in f['object_uids'][()]]
            eligible = [(i, uid) for i, uid in enumerate(uids) if uid in allowed and uid not in seen]
            if not eligible:
                continue
            if int(f['drag_enabled'][()]) != 0:
                raise ValueError('Passive initial source required')
            before = digest(path)
            x, ids, volume, velocity = f['x'][0], f['object_ids'][()], f['vol'][()], f['v'][0]
            if volume.dtype != np.float32 or volume.shape != (len(x),) or not np.isfinite(volume).all() or np.any(volume <= 0):
                raise ValueError('Original positive finite float32 weights required')
            if ids.shape != (len(x),) or ids.dtype.kind not in 'iu' or not np.array_equal(np.unique(ids), np.arange(len(uids))):
                raise ValueError('Source object identities invalid')
            recomputed = occupied_weights(x, int(f['n_grid'][()]), float(f['grid_lim'][()]))
            ulp = np.abs(recomputed.view(np.int32).astype(np.int64) - volume.view(np.int32).astype(np.int64))
            if ulp.max() > 1:
                raise ValueError('Original occupied-cell weight rule changed')
            inspected.append(dict(path=str(path), sha256=before))
            for oid, uid in eligible:
                if uid in seen:
                    continue
                ix = np.flatnonzero(ids == oid).astype(np.int32)
                if len(ix) != 1024:
                    raise ValueError('Exactly 1024 original particles per object required')
                target = root / f'object-{len(rows):04d}.npz'
                with target.open('xb') as stream:
                    np.savez_compressed(stream, source_x=x[ix], source_v=velocity[ix], particle_volume=volume[ix], source_particle_index=ix)
                rows.append(dict(index=len(rows), asset_uid=uid, file=target.name, sha256=digest(target),
                    source_raw=str(path), source_raw_sha256=before, source_object_id=oid, points=1024,
                    object_size_m=float(f['object_size'][()]), initial_extent_m=np.ptp(x[ix], axis=0).tolist(),
                    estimator_crosscheck_max_float32_ulp=int(ulp[ix].max())))
                seen.add(uid)
                if len(rows) == count:
                    break
        if digest(path) != before:
            raise ValueError('Raw source changed during extraction')
        if len(rows) == count:
            break
    if len(rows) != count:
        raise ValueError(f'Only {len(rows)} unique assigned objects within declared scan; no automatic top-up')
    if any(digest(path) != sha for path, sha in source_hashes.items()):
        raise ValueError('Pool/cohort source changed during extraction')
    write_once(root / 'MANIFEST.json', dict(objects=rows, inspected=inspected, bank_role=role,
        split_manifest_sha256=digest(split_manifest), train_pool_sha256=digest(Path(pool)/POOL_FILES[0]),
        heldout_pool_sha256=digest(Path(pool)/POOL_FILES[1]),
        volume_policy='frozen-original-raw-occupied-cell-weights-v1', accepted_for_training=False))
    load_bank(root, pool, role=role, split_manifest=split_manifest)
    return dict(role=role, objects=len(rows), inspected_raws=len(inspected), manifest_sha256=digest(root/'MANIFEST.json'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    freeze = sub.add_parser('freeze-splits'); freeze.add_argument('--pool', type=Path, required=True)
    freeze.add_argument('--out', type=Path, required=True); freeze.add_argument('--namespace', required=True)
    build = sub.add_parser('build'); build.add_argument('--root', type=Path, required=True)
    build.add_argument('--pool', type=Path, required=True); build.add_argument('--split-manifest', type=Path, required=True)
    build.add_argument('--role', choices=ROLES, required=True); build.add_argument('--raw-root', type=Path, required=True)
    build.add_argument('--count', type=int, default=28); build.add_argument('--scan-limit', type=int, default=1024)
    a = vars(p.parse_args()); command = a.pop('command')
    if command == 'freeze-splits':
        value = assignment(a['pool'], a['namespace']); write_once(a['out'], value)
        print(json.dumps({k: len(v) for k, v in value['cohorts'].items()}))
    else:
        print(json.dumps(build_bank(**a)))


if __name__ == '__main__':
    main()
