"""CPU-only adapter: verified unit shapes to split-aware candidate MPM banks.

Occupied-cell weights are a compatibility proxy, not physical solid quadrature.
Canonical 16 cm initialization is only an adapter reference; fresh-v1 recipes
resample full extent per scene and freeze weights BEFORE rigid orientation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from collection.common import digest, relative_path, write_once
from .banks import assignment, occupied_weights, verify_assignment

VOLUME_POLICY = 'fresh-canonical-occupied-cell-proxy-v1'


def physical(points_unit, full_extent_m):
    p = np.asarray(points_unit)
    if (p.dtype != np.float32 or p.shape != (1024, 3) or not np.isfinite(p).all()
            or np.max(np.abs(p)) > .50001 or not .12 <= full_extent_m <= .20):
        raise ValueError('Invalid unit-shape or full-extent contract')
    # Fixed cell phase, no yaw/translation-dependent recomputation of mass.
    x = (p.astype(float) * full_extent_m + .75).astype(np.float32)
    weights = occupied_weights(x, 100, 100 / 64)
    return x, weights


def _one(task):
    row, role, shapes, root = task
    shapes, root = Path(shapes), Path(root)
    uid = row['uid']
    result = dict(uid=uid, asset_uid=row['asset_uid'], role=role, accepted_for_training=False)
    try:
        if role == 'train':
            source = relative_path(shapes, row['file'])
            if digest(source) != row['sha256']:
                raise ValueError('Training point source changed')
            with np.load(source, allow_pickle=False) as f:
                arrays = {k: f[k].copy() for k in f.files}
            geometry_sha = row['geometry_sha256']
            lineage = dict(point_source=str(source), point_source_sha256=row['sha256'])
        else:
            import trimesh
            from .assets import geometry, POLICY
            source = shapes / 'meshes' / (row['sha256'] + '.glb')
            if digest(source) != row['sha256']:
                raise ValueError('Protected mesh changed')
            if source.stat().st_size > POLICY['maximum_mesh_bytes']:
                raise ValueError('mesh_byte_limit')
            mesh = trimesh.load(source, force='mesh', process=False, skip_materials=True)
            arrays, quality = geometry(mesh.vertices, mesh.faces, uid)
            if not quality['eligible']:
                return dict(result, status='excluded', reasons=quality['reasons'], quality=quality)
            geometry_sha = quality['geometry_sha256']
            lineage = dict(mesh_source=str(source), mesh_source_sha256=row['sha256'], quality=quality)
        x, volume = physical(arrays['points_unit'], .16)
        folder = root / role / 'objects' / uid
        folder.mkdir(parents=True, exist_ok=False)
        payload = folder / 'points.npz'
        with payload.open('xb') as stream:
            np.savez_compressed(stream, **arrays, source_x=x, source_v=np.zeros_like(x),
                                particle_volume=volume, source_particle_index=np.arange(1024, dtype=np.int32))
            stream.flush(); os.fsync(stream.fileno())
        return dict(result, status='prepared', file=str(payload.relative_to(root / role)),
                    sha256=digest(payload), object_size_m=.08, maximum_full_extent_m=.16,
                    geometry_sha256=geometry_sha, lineage=lineage,
                    reference_proxy_volume_m3=float(volume.astype(float).sum()))
    except Exception as exc:
        # Errors are never silently recategorized as successful qualification.
        return dict(result, status='error', reasons=[f'{type(exc).__name__}:{exc}'])


def build(root, shapes, historical_splits, workers=4):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or not 1 <= workers <= 4:
        raise ValueError('Explicit CPU-only preparation with 1..4 workers required')
    root, shapes = Path(root).resolve(), Path(shapes).resolve(strict=True)
    m = json.loads((shapes / 'MANIFEST.json').read_text())
    plan = json.loads((shapes / 'PLAN.json').read_text())
    if digest(shapes / 'PLAN.json') != m['plan_sha256'] or len(m['objects']) != 10000:
        raise ValueError('Verified 10,000-shape bank required')
    original = verify_assignment(historical_splits, shapes / 'inputs')
    inputs = {str(shapes / name): digest(shapes / name) for name in ('MANIFEST.json', 'PLAN.json')}
    inputs[str(Path(historical_splits).resolve())] = digest(historical_splits)
    root.mkdir(parents=True, exist_ok=False)
    pool = root / 'pool'
    write_once(pool / 'objaverse_uid_train.json', sorted(r['asset_uid'] for r in m['objects']))
    write_once(pool / 'objaverse_uid_holdout.json', sorted(r['asset_uid'] for r in plan['protected']))
    splits = assignment(pool, original['namespace'])
    for role in ('validation', 'test'):
        if splits['cohorts'][role] != original['cohorts'][role]:
            raise ValueError('Historical held-out role changed')
    write_once(root / 'SPLITS.json', splits)
    tasks = [(r, 'train', str(shapes), str(root)) for r in m['objects']]
    for r in plan['protected']:
        role = 'validation' if r['asset_uid'] in splits['cohorts']['validation'] else 'test'
        tasks.append((r, role, str(shapes), str(root)))
    write_once(root / 'PLAN.json', dict(schema='fresh-physical-banks-v1', inputs_sha256=inputs,
               volume_policy=VOLUME_POLICY, source_sha256={str(Path(__file__)): digest(__file__)},
               tasks=len(tasks), accepted_for_training=False))
    results = []
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        for i, result in enumerate(executor.map(_one, tasks, chunksize=8)):
            write_once(root / 'receipts' / (result['uid'] + '.json'), result)
            results.append(result)
            if (i + 1) % 500 == 0:
                print(json.dumps(dict(prepared_or_excluded=i + 1, total=len(tasks))), flush=True)
    if any(r['status'] != 'prepared' for r in results if r['role'] == 'train'):
        raise ValueError('Training adapter failure; full 10k membership mandatory, inspect receipts')
    # Geometry/resource exclusions are allowed for heldouts without reassignment.
    # Unexpected I/O/source/parser failures block completion, not silently drop data.
    allowed_errors = ('ValueError:mesh_byte_limit', 'ValueError:invalid_or_excessive_faces')
    if any(r['status'] == 'error' and tuple(r['reasons']) not in [(s,) for s in allowed_errors] for r in results):
        raise ValueError('Unexpected held-out preparation error; inspect receipts')
    cohorts = {}
    for role in ('train', 'validation', 'test'):
        rows = [r for r in results if r['role'] == role and r['status'] == 'prepared']
        if len(rows) < 7:
            raise ValueError('Insufficient qualified role members')
        cohorts[role] = len(rows)
        write_once(root / role / 'MANIFEST.json', dict(schema='fresh-candidate-bank-v1', bank_role=role,
                   objects=rows, volume_policy=VOLUME_POLICY,
                   train_pool_sha256=digest(pool / 'objaverse_uid_train.json'),
                   heldout_pool_sha256=digest(pool / 'objaverse_uid_holdout.json'),
                   split_manifest_sha256=digest(root / 'SPLITS.json'),
                   accepted_for_training=False, proxy_volume_qualified=False))
    if any(digest(p) != h for p, h in inputs.items()):
        raise ValueError('Sources changed during preparation')
    from .recipes import load_bank
    for role in cohorts:
        load_bank(root / role, pool, role=role, split_manifest=root / 'SPLITS.json')
    complete = dict(schema='fresh-physical-banks-v1', cohorts=cohorts,
                    exclusions=[r for r in results if r['status'] != 'prepared'],
                    artifacts={str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()},
                    ready_for_pilot_preparation=True, accepted_for_training=False, proxy_volume_qualified=False)
    write_once(root / 'COMPLETE.json', complete)
    return dict(cohorts=cohorts, excluded=len(complete['exclusions']), root=str(root))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shapes', type=Path, required=True)
    p.add_argument('--historical-splits', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    print(json.dumps(build(a.root, a.shapes, a.historical_splits, a.workers)), flush=True)


if __name__ == '__main__':
    main()
