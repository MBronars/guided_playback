"""Sand-only wall acceptance using conservative native-step clearance bounds.

Legacy native event logs have no material attribution. For the pinned guarded
backend, saved endpoints plus its enforced speed cap can rule out non-sand
particles reaching wall grid support BETWEEN observations. Uncertain != sand.
This does not measure impulses or rule out indirect forces transmitted by sand.
"""
from pathlib import Path

import numpy as np

from collection.common import digest
from .walls import NAMES, POINTS, NORMALS, validate_evidence

STRICT = 'strict-wall-free-v1'
SAND_ONLY = 'sand-only-guarded-clearance-v2'
POLICY = dict(version=SAND_ONLY, sand_mat_type=2, speed_bound_m_s=8.0001,
              grouped_source_sha256='6476bf57d72ed3e9a64d95d9168941df070093cca80923ea72e050d949f75041',
              position_roundoff_margin_m=1e-6, support_radius_cells=2,
              ambiguous='exclude', floor='allowed for all materials',
              scope='Allow sand at the five observed side/ceiling planes; no contact-amount threshold',
              limitation='Direct non-sand stencil participation excluded; indirect sand-mediated effects not excluded')

# Explicitly audited implementation, not whatever source happens to be current.
# CollectionBackend checks <=8 m/s before/after each native step, and the FP64
# master updates x by checked v*dt. 8.0001 bounds float32 norm-rounding error.
# Changes to these implementations need a new reviewed policy/version.
SUPPORTED_SOURCES = {
    'physics.py': '82b611058dfa4d572e7cae93c023a68d0c4a8c4e2f9dcc45eb56505b288502a7',
    'runtime.py': 'adea110eb1c330545a54faa2d79cfdb4d25f8bc7cd7ce0521fbb90e1348bd9d9',
    'observer.py': '401d656be70493e9946f0c7b98bb7c1d082e008b8f44fa872388d305104ed356',
    'kernels.py': '4b371a178226814be0063b3f20fd6422d250a8f2c9794b1b8e5fd6c85da18712',
    'solver.py': '0fe764faac85d889247f181a20dd121caf39c683097fb18109b4ba77cf6e50c4',
    'state.py': '931a6e8bdcd80f3c29cdc6ebacc429912260e545def336cbf8bf3eeb84fedca6',
    'material_control.py': '6cc0ad075ab4e66cdd91c295bd9e3af2b0baf98efdbf01b8185445dea4929cd5',
}


def interval_lower_bound(distance, times):
    """Lipschitz lower envelope from BOTH endpoints, not linear interpolation.

With speed <=V over dt, min distance >= (d0+d1-V*dt)/2. Also include
endpoints explicitly; subtract a float32-position roundoff margin. This is a
sufficient clearance certificate, not a reconstructed trajectory.
"""
    distance, times = np.asarray(distance, float), np.asarray(times, float)
    if (distance.ndim != 2 or times.shape != (len(distance),) or len(times) < 2
            or not np.isfinite(distance).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid clearance endpoints/clock')
    envelope = (distance[:-1] + distance[1:] - POLICY['speed_bound_m_s'] * np.diff(times)[:, None]) / 2
    return np.minimum(np.minimum(distance[:-1], distance[1:]), envelope) - POLICY['position_roundoff_margin_m']


def check_guard_provenance(manifest):
    config = manifest.get('solver_config', {})
    if (config.get('precision') != 'float64-master' or config.get('n_grid') != 100
            or config.get('grid_lim_m') != 100 / 64
            or 'benchmark' in manifest
            or manifest.get('geometry_provenance', {}).get('recipe_version') != 'fresh-shapes-v1'):
        raise ValueError('Unsupported guarded position-update contract')
    sources = manifest.get('source_sha256', {})
    group = manifest.get('group_execution')
    if group is not None:
        if (set(group) != {'version','worlds','block','fused_walls','source_sha256','guards_per_substep'}
                or group['version'] != 'grouped-native-guards-v1'
                or type(group['worlds']) is not int or not 1 <= group['worlds'] <= 32
                or group['block'] not in (64,256) or type(group['fused_walls']) is not bool
                or group['guards_per_substep'] != 2
                or group['source_sha256'] != POLICY['grouped_source_sha256']):
            raise ValueError('Unsupported grouped guard contract')
        matches = [(path,sha) for path,sha in sources.items()
                   if Path(path).parts[-3:] == ('collection','mpm','grouped.py')]
        if (len(matches) != 1 or matches[0][1] != POLICY['grouped_source_sha256']
                or digest(matches[0][0]) != POLICY['grouped_source_sha256']):
            raise ValueError('Unverified grouped guard implementation')
    for name, expected in SUPPORTED_SOURCES.items():
        matches = [(path, sha) for path, sha in sources.items()
                   if Path(path).parts[-3:] == ('collection', 'mpm', name)]
        if len(matches) != 1 or matches[0][1] != expected or digest(matches[0][0]) != expected:
            raise ValueError('Unverified native speed-guard source: ' + name)


def classify(manifest, arrays):
    evidence = validate_evidence(manifest, arrays)
    result = dict(policy=POLICY, wall_evidence=evidence, status='unknown', eligible=False,
                  reasons=[], per_wall=[], inference='guarded endpoint clearance; no per-material native event counts')
    if evidence['wall_free']:
        return dict(result, status='wall_free', eligible=True)
    x = np.asarray(arrays['x'], float)
    ids, materials = np.asarray(arrays['object_ids']), np.asarray(arrays['mat_type'])
    if (x.ndim != 3 or x.shape[-1] != 3 or ids.shape != (x.shape[1],) or ids.dtype.kind not in 'iu'
            or ids.size == 0
            or not np.isfinite(x).all() or materials.ndim != 1 or not np.isin(materials, [0, 1, 2]).all()
            or ids.min() < 0 or ids.max() >= len(materials)):
        raise ValueError('Invalid wall-material input')
    non_sand = materials[ids] != 2
    if non_sand.all():
        return dict(result, status='non_sand_contact', reasons=['wall_contact_without_sand'])
    if not non_sand.any():
        return dict(result, status='sand_only', eligible=True, inference='all particles are sand')
    try:
        check_guard_provenance(manifest)
    except (ValueError, OSError) as exc:
        return dict(result, reasons=['wall_material_attribution_unknown:' + str(exc)])
    speed = np.linalg.norm(np.asarray(arrays['v'], float), axis=-1)
    if not np.isfinite(speed).all() or speed.max() > POLICY['speed_bound_m_s']:
        return dict(result, reasons=['saved_velocity_exceeds_certified_bound'])
    p = x[:, non_sand]
    padding = POLICY['support_radius_cells'] / 64
    ambiguous, contact = False, False
    for wall, point, normal in zip(NAMES, POINTS, NORMALS):
        axis = int(np.flatnonzero(normal)[0])
        distances = (p[..., axis] - float(point[axis])) * normal[axis]
        lower = interval_lower_bound(distances, arrays['state_time_s'])
        saved_contact = bool(np.any(distances <= 0))
        possible_support = bool(np.any(lower <= padding))
        ambiguous |= possible_support; contact |= saved_contact
        result['per_wall'].append(dict(wall=wall, minimum_saved_non_sand_clearance_m=float(distances.min()),
                                      minimum_certified_non_sand_clearance_m=float(lower.min()),
                                      saved_non_sand_contact=saved_contact,
                                      possible_non_sand_grid_support=possible_support))
    if contact:
        return dict(result, status='non_sand_contact', reasons=['non_sand_saved_wall_contact'])
    if ambiguous:
        return dict(result, status='unknown', reasons=['non_sand_between_frame_grid_contact_not_excluded'])
    return dict(result, status='sand_only', eligible=True)


def reselect(pilot, output):
    """New immutable selection from completed non-test pilot raws; no reruns."""
    import json
    from collections import Counter
    from .export import export_candidates
    from .review import check_model
    from .runtime import ROOT, sources
    from collection.common import check_hashes, write_once
    pilot, output = Path(pilot).resolve(strict=True), Path(output).resolve()
    original_complete = digest(pilot / 'COMPLETE.json')
    original = json.loads((pilot / 'COMPLETE.json').read_text())
    check_hashes(pilot, original['artifacts'])
    executed = sources()
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / 'REQUEST.json', dict(pilot=str(pilot), original_complete_sha256=original_complete,
               policy=POLICY, source_sha256=executed, accepted_for_training=False))
    for name, sha in executed.items():
        write_once(output / 'source' / (name + '.json'), dict(path=name, sha256=sha, text=(ROOT / name).read_text()))
    groups = []
    for report in sorted((pilot / 'review').glob('*/RESULT.json')):
        if report.parent.name.startswith('test-'):
            raise ValueError('Test-set reselection is not part of this pilot operation')
        review = json.loads(report.read_text())
        candidates = [r for r in review['rows'] if r.get('kind') == 'coverage']
        raws = []
        for row in candidates:
            if row['status'] == 'complete':
                raw = Path(row['raw_path']).resolve(strict=True)
                if not raw.is_relative_to(pilot) or digest(raw) != row['raw_sha256']:
                    raise ValueError('Original raw identity changed')
                raws.append(raw)
        group = output / report.parent.name
        complete = export_candidates(raws, group / 'filtered', wall_policy=SAND_ONLY)
        filtered = json.loads((group / 'filtered/FILTER.json').read_text())
        kept = {row['episode_id']: Path(row['raw_path']) for row in filtered['episodes'] if row['eligible_for_export']}
        model = check_model(group, kept) if kept else None
        row = dict(group=report.parent.name, attempts=len(candidates), complete_raws=len(raws),
                   old_retained=review['retained_coverage'], retained=complete['retained'],
                   recovered=complete['retained'] - review['retained_coverage'],
                   wall_classification=dict(Counter(r.get('wall_materials', {}).get('status', 'unknown')
                                                   for r in filtered['episodes'])),
                   rejection_reasons=dict(Counter(reason for r in filtered['episodes']
                                           if not r['eligible_for_export'] for reason in set(r['reasons']))),
                   model=model, accepted_for_training=False)
        write_once(group / 'RESULT.json', row); groups.append(row)
        print(json.dumps({k: row[k] for k in ('group', 'attempts', 'retained', 'recovered', 'wall_classification')}), flush=True)
    if len(groups) != 5:
        raise ValueError('Expected five completed train/validation pilot groups')
    if digest(pilot / 'COMPLETE.json') != original_complete or sources() != executed:
        raise ValueError('Source changed during reselection')
    check_hashes(pilot, original['artifacts'])
    complete = dict(schema='mpm-pilot-sand-only-reselection-v1', policy=POLICY, groups=groups,
                    attempts=sum(g['attempts'] for g in groups), retained=sum(g['retained'] for g in groups),
                    recovered=sum(g['recovered'] for g in groups), source_sha256=executed,
                    original_complete_sha256=original_complete, accepted_for_training=False,
                    artifacts={str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_once(output / 'COMPLETE.json', complete)
    return {k: complete[k] for k in ('attempts', 'retained', 'recovered')}


if __name__ == '__main__':
    import argparse
    import json
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Reselection and consumer review must hide CUDA')
    print(json.dumps(reselect(args.pilot, args.out)), flush=True)
