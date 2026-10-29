"""Passive atomic collection and wall-free *post-filtered diagnostic* exports.

Does not promote training data. Legacy raws missing substep evidence are unknown.
Use the outer FILTER.json + COMPLETE.json receipt with the nested numeric TARs;
the TAR batch format, surface points, clocks and normalization stay unchanged.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .walls import FIELDS, validate_evidence
from .contracts import Clock, canonical_json, finite_array, identity
from .validate import Policy, read_raw, validate_raw, export_diagnostic, validate_export_cohorts, _fsync_dir
from collection.common import write_once as write_json_once
from collection.common import digest as sha256
from .recording import atomic_raw
from .wall_materials import STRICT, SAND_ONLY

POLICY = Policy(floor_tolerance_m=.012, name='historical-passive-floor-compatibility-review-only-v1')


def collect_passive(recording, observed, manifest, static, output):
    """Record state and persistent wall evidence in ONE atomic raw completion.

    `recording` may wrap `observed` for durable trace/pacing. The clock and
    coverage checks fail closed if these accidentally refer to different runs.
    """
    manifest = json.loads(canonical_json(manifest))
    clock = Clock(**manifest['clock'])
    eid, seed = identity(manifest['case'])
    if manifest['task'] != 'passive' or manifest['case']['task'] != 'passive':
        raise ValueError('Passive only; active drag needs a separate observed-settling contract')
    if manifest['episode_id'] != eid or clock != observed.clock or observed.steps:
        raise ValueError('Collector identity, clock or initial state mismatch')
    if set(static) != set(observed.backend.static) or any(
            not np.array_equal(static[k], observed.backend.static[k]) for k in static):
        raise ValueError('Static arrays differ from the simulation')
    n = len(static['object_ids'])
    selected = np.zeros(n, np.int32)
    records = {key: [] for key in ('x', 'v', 'F_trial', 'C', *FIELDS)}
    manifest['wall_observation'] = observed.wall_metadata
    sources = dict(manifest['source_sha256'], **observed.source_sha256)
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    manifest['source_sha256'] = sources
    if any(sha256(p) != digest for p, digest in sources.items()):
        raise ValueError('Collection source changed before recording')
    for state in range(clock.transitions + 1):
        if state:
            for _ in range(clock.substeps_per_transition):
                recording.step(clock.substep_dt_s, None, selected)
        value = recording.read()
        actual = observed.read()
        if any(not np.array_equal(value[key], actual[key]) for key in actual):
            raise ValueError('Recording and observed backend state differ')
        if observed.steps != state * clock.substeps_per_transition:
            raise ValueError('Recording and wall evidence clocks differ')
        value = dict(value, **observed.wall_snapshot())
        for key in records:
            shape = (n, 3) if key in ('x', 'v') else (n, 3, 3)
            records[key].append(value[key].copy() if key in FIELDS else finite_array(value[key], shape, key).copy())
    arrays = {key: np.stack(value).astype(np.int64 if key in FIELDS else np.float32) for key, value in records.items()}
    arrays.update(static)
    arrays.update(state_time_s=clock.state_time_s, selected=selected,
                  command_velocity=np.zeros((n, 3), np.float32),
                  transition_active=np.zeros(clock.transitions, np.int32),
                  active_substeps=np.zeros(clock.transitions, np.int32))
    manifest.update(seed_uint64=seed, command_semantics='passive; no velocity overrides')
    evidence = validate_evidence(manifest, arrays)
    if any(sha256(p) != digest for p, digest in sources.items()):
        raise ValueError('Collection source changed during recording')
    # Wall contact is an exclusion later, not an excuse to truncate a safe run.
    atomic_raw(output, manifest, arrays)
    return dict(episode_id=eid, output=str(output), states=len(arrays['x']),
                wall_evidence=evidence, accepted_for_training=False)


def validate_wall_free(path, policy=POLICY):
    integrity = validate_raw(path, policy)
    result = dict(raw_path=str(Path(path).resolve()), raw_sha256=integrity['raw_sha256'],
                  episode_id=integrity['episode_id'], integrity_pass=integrity['integrity_pass'],
                  wall_status='unknown', eligible_for_wall_free_export=False,
                  accepted_for_training=False, reasons=[], strata=integrity['strata'])
    if 'collapse' in integrity['metrics']:
        result['collapse'] = integrity['metrics']['collapse']
    manifest, arrays = {}, {}
    try:
        manifest, arrays = read_raw(path, policy)
        result['strata'].update(scenario=manifest.get('scenario', 'unspecified'),
                                split=manifest.get('split', 'unspecified'))
        if integrity['integrity_pass']:
            point_step = np.linalg.norm(np.diff(arrays['x'].astype(np.float64), axis=0), axis=-1).mean(axis=1)
            result['motion'] = dict(mean_point_step_m=float(point_step.mean()),
                                   fraction_transitions_below_1mm=float(np.mean(point_step < .001)))
        evidence = validate_evidence(manifest, arrays)
        if sha256(path) != result['raw_sha256']:
            raise ValueError('raw_changed_during_wall_validation')
        result['wall_evidence'] = evidence
        result['wall_status'] = 'wall_free' if evidence['wall_free'] else 'wall_contact'
        if not evidence['wall_free']:
            result['reasons'].append('side_wall_or_ceiling_contact')
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        result['reasons'].append(str(exc))
    if not integrity['integrity_pass']:
        result['reasons'].extend(integrity['reasons'])
    result['eligible_for_wall_free_export'] = integrity['integrity_pass'] and result['wall_status'] == 'wall_free'
    if integrity['integrity_pass'] and manifest.get('geometry_provenance', {}).get('recipe_version') == 'fresh-shapes-v1':
        from .interactions import analyze
        try:
            result['interaction'] = analyze(manifest, arrays)
            result['eligible_for_wall_free_export'] &= result['interaction']['semantic_eligible']
            result['reasons'].extend(result['interaction']['reasons'])
        except (ValueError, KeyError, TypeError) as exc:
            result['eligible_for_wall_free_export'] = False
            result['reasons'].append('interaction:' + str(exc))
    return result


def validate_candidate(path, policy=POLICY, *, wall_policy=STRICT):
    """Explicit alternate selection; never label accepted sand contact wall-free."""
    if wall_policy not in (STRICT, SAND_ONLY):
        raise ValueError('Unknown wall selection policy')
    result = validate_wall_free(path, policy)
    if wall_policy == STRICT:
        return result
    result.update(selection_policy=wall_policy, eligible_for_export=False)
    try:
        from .wall_materials import classify
        manifest, arrays = read_raw(path, policy)
        attribution = classify(manifest, arrays)
        if sha256(path) != result['raw_sha256']:
            raise ValueError('Raw changed during material-aware selection')
        result['wall_materials'] = attribution
        result['reasons'] = [r for r in result['reasons'] if r != 'side_wall_or_ceiling_contact']
        result['reasons'].extend(attribution['reasons'])
        result['eligible_for_export'] = bool(result['integrity_pass'] and attribution['eligible']
            and not result['reasons'] and result.get('interaction', {}).get('semantic_eligible', True))
    except (ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        result['reasons'].append('material_aware_wall_filter:' + str(exc))
    return result


def eligible(row):
    return row.get('eligible_for_export', row['eligible_for_wall_free_export'])


def retention(rows):
    groups = defaultdict(lambda: dict(proposed_raws=0, retained=0, wall_contact=0, unknown=0, integrity_failed=0))
    for row in rows:
        labels = ['all', 'scenario:' + str(row['strata'].get('scenario', 'unspecified')),
                  'count:' + str(row['strata'].get('n_objects')),
                  'split:' + str(row['strata'].get('split', 'unspecified'))]
        labels += ['contains_material:' + m for m in set(row['strata'].get('material_names') or [])]
        for label in labels:
            g = groups[label]; g['proposed_raws'] += 1
            g['retained'] += int(eligible(row))
            g['wall_contact'] += int(row['wall_status'] == 'wall_contact')
            g['unknown'] += int(row['wall_status'] == 'unknown')
            g['integrity_failed'] += int(not row['integrity_pass'])
    return dict(sorted(groups.items()))


def export_wall_free(paths, output, *, policy=POLICY, batch_size=4):
    return export_candidates(paths, output, policy=policy, batch_size=batch_size, wall_policy=STRICT)


def export_candidates(paths, output, *, policy=POLICY, batch_size=4, wall_policy=STRICT):
    """Immutable outer transaction binds the filter receipt to unchanged TARs.

    Zero survivors is a valid measured result (receipt, no TAR). Failed attempts
    with no complete raw must be reported by the collection orchestrator too.
    """
    if wall_policy not in (STRICT, SAND_ONLY):
        raise ValueError('Unknown wall selection policy')
    paths = [Path(p) for p in paths]
    if any(p.is_symlink() for p in paths):
        raise ValueError('Raw symlink')
    paths = sorted(p.resolve() for p in paths)
    if not paths or len(paths) != len(set(paths)) or batch_size < 1:
        raise ValueError('Nonempty unique raw paths and positive batch size required')
    if any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError('Raw missing or symlink')
    validate_export_cohorts(paths)
    sources = {str(p): sha256(p) for p in (Path(__file__).resolve(), Path(__file__).with_name('walls.py').resolve(),
                                          Path(__file__).with_name('validate.py'), Path(__file__).with_name('contracts.py'))}
    request = dict(raws=[dict(path=str(p), sha256=sha256(p)) for p in paths],
                   policy=asdict(policy), batch_size=batch_size, source_sha256=sources,
                   schema='wall-free-diagnostic-export-v1')
    if wall_policy != STRICT:
        from .wall_materials import POLICY as material_policy
        source = Path(__file__).with_name('wall_materials.py').resolve()
        sources[str(source)] = sha256(source)
        request.update(schema='material-aware-wall-diagnostic-export-v1', wall_selection_policy=material_policy)
    digest = hashlib.sha256(canonical_json(request).encode()).hexdigest()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        complete = json.loads((output / 'COMPLETE.json').read_text())
        if complete['request_sha256'] != digest:
            raise ValueError('Existing export request differs')
        for artifact in complete['artifacts']:
            if sha256(output / artifact['path']) != artifact['sha256']:
                raise ValueError('Existing wall-free export artifact changed')
        return dict(complete, resumed=True)
    lock = output.with_name(output.name + '.lock')
    write_json_once(lock, request)
    stage = Path(tempfile.mkdtemp(prefix='.' + output.name + '.partial-', dir=output.parent))
    rows = [validate_candidate(p, policy, wall_policy=wall_policy) for p in paths]
    if any(r['raw_sha256'] != req['sha256'] for r, req in zip(rows, request['raws'])):
        raise ValueError('Raw changed during filter validation')
    eids = [r['episode_id'] for r in rows if r['episode_id'] is not None]
    if len(eids) != len(set(eids)):
        raise ValueError('Duplicate episode identity')
    selected = [Path(r['raw_path']) for r in rows if eligible(r)]
    receipt = dict(request=request, request_sha256=digest, episodes=rows, retention=retention(rows),
                   accepted_for_training=False,
                   policy_scope=('wall-free' if wall_policy == STRICT else wall_policy) + ' selection plus existing integrity gate, not production promotion')
    write_json_once(stage / 'FILTER.json', receipt)
    if selected:
        kwargs = {} if wall_policy == STRICT else dict(selection_policy=request['wall_selection_policy'])
        inner = export_diagnostic(selected, stage / 'diagnostic-export', policy=policy, batch_size=batch_size, **kwargs)
        if {r['episode_id'] for r in inner['episodes']} != {r['episode_id'] for r in rows if eligible(r)}:
            raise ValueError('Filtered export membership mismatch')
    if any(sha256(r['path']) != r['sha256'] for r in request['raws']):
        raise ValueError('Raw changed during export')
    if any(sha256(p) != h for p, h in sources.items()):
        raise ValueError('Export source changed')
    artifacts = [dict(path=str(p.relative_to(stage)), sha256=sha256(p)) for p in sorted(stage.rglob('*')) if p.is_file()]
    complete = dict(schema=request['schema'], request_sha256=digest, retained=len(selected),
                    excluded=len(paths) - len(selected), artifacts=artifacts, accepted_for_training=False)
    write_json_once(stage / 'COMPLETE.json', complete)
    _fsync_dir(stage)
    if output.exists():
        raise FileExistsError(output)
    os.rename(stage, output)
    _fsync_dir(output.parent)
    return complete
