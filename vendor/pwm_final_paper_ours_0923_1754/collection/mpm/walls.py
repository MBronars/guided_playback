"""Versioned, fail-closed wall evidence; no simulator or training dependencies."""
from pathlib import Path

import numpy as np

from .contracts import Clock

SCHEMA = 'mpm-substep-wall-observation-v1'
NAMES = ('ceiling', 'x_low', 'x_high', 'y_low', 'y_high')
POINTS = np.array([[0, 0, 1.47], [.03, 0, 0], [1.47, 0, 0],
                   [0, .03, 0], [0, 1.47, 0]], np.float32)
NORMALS = np.array([[0, 0, -1], [1, 0, 0], [-1, 0, 0],
                    [0, 1, 0], [0, -1, 0]], np.float32)
FIELDS = ('wall_first_grid_step', 'wall_first_particle_step',
          'wall_covered_substeps', 'wall_simulation_steps')


def metadata(observer_sha256):
    return dict(schema=SCHEMA, wall_names=list(NAMES), points_m=POINTS.tolist(),
                inward_normals=NORMALS.tolist(), surface_type='separate', friction=0.,
                physical_box_lim_m=1.5, floor_excluded=True,
                grid_rule='occupied node outside plane with inward normal velocity before that collider',
                particle_rule='point on or outside plane at initial state and before/after every substep',
                step_rule='initial=0; first completed substep=1; absent=-1; cumulative',
                observer_sha256=observer_sha256)


def validate_evidence(manifest, arrays):
    """Raise on absent/inconsistent evidence; return contact, not physics approval.

    Checking saved positions can *disprove* a wall-free claim. It cannot prove
    one, so complete substep evidence is mandatory even for quiet old raws.
    """
    meta = manifest.get('wall_observation')
    if not isinstance(meta, dict):
        raise ValueError('wall_observation_missing')
    digest = meta.get('observer_sha256')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('wall_observer_hash_invalid')
    if meta != metadata(digest) or digest not in manifest['source_sha256'].values():
        raise ValueError('wall_policy_or_source_mismatch')
    if manifest['task'] != 'passive' and not (
            manifest['task'] == 'drag' and manifest.get('action_schema') in (
                'mpm-drag-command-schedule-v1','mpm-drag-command-sequence-v2','mpm-drag-command-sequence-v3','mpm-drag-command-sequence-v4')):
        raise ValueError('wall_observation_passive_only')
    clock = Clock(**manifest['clock'])
    t = clock.transitions + 1
    steps = np.arange(t, dtype=np.int64) * clock.substeps_per_transition
    for key in FIELDS:
        value = arrays.get(key)
        shape = (t,) if key == 'wall_simulation_steps' else (t, 5)
        if value is None or value.shape != shape or value.dtype != np.dtype('int64'):
            raise ValueError('wall_evidence_shape_or_dtype:' + key)
    if not np.array_equal(arrays['wall_simulation_steps'], steps):
        raise ValueError('wall_clock_mismatch')
    if not np.array_equal(arrays['wall_covered_substeps'], np.repeat(steps[:, None], 5, axis=1)):
        raise ValueError('wall_substep_coverage_incomplete')
    touched = set()
    first = {}
    for key, minimum in [('wall_first_grid_step', 1), ('wall_first_particle_step', 0)]:
        values = arrays[key]
        final = values[-1]
        if np.any((final != -1) & ((final < minimum) | (final > steps[-1]))):
            raise ValueError('wall_first_step_invalid:' + key)
        expected = np.where((final[None, :] >= 0) & (steps[:, None] >= final[None, :]), final[None, :], -1)
        if not np.array_equal(values, expected):
            raise ValueError('wall_evidence_not_cumulative:' + key)
        touched.update(name for name, step in zip(NAMES, final) if step >= 0)
        first[key] = {name: int(step) for name, step in zip(NAMES, final) if step >= 0}
    # Independent countercheck against actual saved states, with the solver's
    # float32 plane convention. No extra geometric margin is introduced.
    x = arrays['x']
    for wall, (point, normal) in enumerate(zip(POINTS, NORMALS)):
        axis = int(np.flatnonzero(normal)[0])
        contacts = np.any((x[:, :, axis] - point[axis]) * normal[axis] <= 0, axis=1)
        if np.any(contacts & (arrays['wall_first_particle_step'][:, wall] < 0)):
            raise ValueError('saved_contact_missing_from_substep_evidence:' + NAMES[wall])
    return dict(wall_free=not touched, touched_walls=sorted(touched),
                first_events=first, covered_substeps=int(steps[-1]), evidence_schema=SCHEMA)
