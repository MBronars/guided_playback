"""Review command replay integrity and measured motion, without declaring a fold.

Dense substeps are not saved: their online trace cannot be independently rebuilt.
All saved frame endpoints and state chunks ARE independently cross-checked.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from collection.common import check_hashes, digest, write_once
from .contracts import geometry_diagnostics
from .recording import StateRecorder
from .replay import SCHEMA
from .review import read_arrays
from .settling import SettlingGate


def review(root):
    root = Path(root).resolve(strict=True)
    receipt = json.loads((root / 'RESULT.json').read_text())
    p = json.loads((root / 'PROTOCOL.json').read_text())
    n = p['frames']
    if (receipt['status'] != 'complete_diagnostic' or receipt['accepted_for_training'] is not False
            or p['schema'] != SCHEMA or type(n) is not int or not 1 <= n <= 2400
            or p['fps'] != 60 or p['substeps'] != 24 or p['dt_s'] != 1 / 1440):
        raise ValueError('Complete bounded replay on the declared native clock required')
    required = {'PROTOCOL.json', 'SOURCES.json', 'executed_sources.tar.gz', 'commands.npz',
                'finalized_model.npz', 'actual_states.npz', 'collision_flags.npz', 'SUBSTEPS.json'}
    if not required.issubset(receipt['artifacts']):
        raise ValueError('Missing required receipt hashes')
    check_hashes(root, receipt['artifacts'])
    if receipt['frames'] != n + 1 or receipt['substeps'] != n * 24:
        raise ValueError('Completion count mismatch')
    if digest(root / 'commands.npz') != p['command_plan_sha256']:
        raise ValueError('Command plan mismatch')
    if digest(root / 'finalized_model.npz') != p['finalized_sha256']:
        raise ValueError('Finalized model mismatch')
    if json.loads((root / 'SOURCES.json').read_text()) != p['source_hashes']:
        raise ValueError('Source inventory mismatch')
    a, m, c = (read_arrays(root / name) for name in
               ('actual_states.npz', 'finalized_model.npz', 'commands.npz'))
    budget = p.get('recorder_max_bytes', 1024**3)
    if type(budget) is not int or not 0 < budget <= 2*1024**3:
        raise ValueError('Unsupported recorder RAM budget')
    recorder = StateRecorder(60, len(m['particle_q']), a['body_q'].shape[1], len(m['home']), max_states=n + 1, max_bytes=budget)
    def state(i):
        return dict(particle_q=a['particle_q'][i], particle_qd=a['particle_qd'][i],
                    body_q=a['body_q'][i], joint_q=a['actual_joint_q'][i])
    recorder.append_initial(**state(0))
    np.testing.assert_array_equal(a['particle_q'][0], m['particle_q'])
    np.testing.assert_array_equal(a['particle_qd'][0], 0)
    np.testing.assert_array_equal(a['actual_joint_q'][0], m['home'])
    for key in ('command_joint_q', 'command_openness'):
        np.testing.assert_array_equal(a[key], c[key])
    for i in range(n):
        recorder.append_transition(i, command_joint_q=a['command_joint_q'][i],
                                   command_openness=a['command_openness'][i], **state(i + 1))
    rebuilt = recorder.arrays(n)
    if set(a) != set(rebuilt): raise ValueError('Unexpected actual-state fields')
    for key in a:
        if a[key].dtype != rebuilt[key].dtype: raise ValueError('State dtype mismatch: ' + key)
        np.testing.assert_array_equal(a[key], rebuilt[key])
    del recorder, rebuilt
    cursor, chunks = 0, sorted(root.glob('frames-*.npz'))
    for chunk in chunks:
        proof = json.loads(chunk.with_suffix('.json').read_text())
        if (chunk.name not in receipt['artifacts'] or chunk.with_suffix('.json').name not in receipt['artifacts']
                or proof['sha256'] != digest(chunk) or proof['first_state'] != cursor):
            raise ValueError('Unverified or nonconsecutive state chunks')
        stop = proof['last_state']
        if not cursor < stop <= n: raise ValueError('Invalid chunk bounds')
        z = read_arrays(chunk)
        if set(z) != set(a): raise ValueError('Chunk field mismatch')
        for key in a:
            value = a[key] if key == 'fps' else a[key][cursor:stop + (
                0 if key.startswith('command_') or key == 'transition_index' else 1)]
            np.testing.assert_array_equal(z[key], value)
        cursor = stop
    if cursor != n: raise ValueError('Incomplete state chunks')
    flags = read_arrays(root / 'collision_flags.npz')['shape_flags']
    baseline = np.asarray(p['gripper_original_flags'], dtype=np.int32)
    if flags.shape != (n, len(baseline)): raise ValueError('Invalid dynamic collision flags')
    for i, op in enumerate(a['command_openness']):
        expected = baseline.copy()
        for side, name in enumerate(('left_shapes', 'right_shapes')):
            if op[side] > .6: expected[p['gripper_filter'][name]] &= ~(1 << 2)
        np.testing.assert_array_equal(flags[i], expected)
    rows = json.loads((root / 'SUBSTEPS.json').read_text())
    if len(rows) != n * 24: raise ValueError('Missing online substep diagnostics')
    for i, row in enumerate(rows):
        if (row['step'] != i + 1 or row['time_s'] != (i + 1) / 1440
                and row['time_s'] != (i + 1) * p['dt_s']):
            raise ValueError('Nonconsecutive physical trace')
        if row['guard'] or not row['finite'] or row['overflow']:
            raise ValueError('Physical numerical guard failed')
        if not all(np.isfinite(row[k]) for k in ('max_speed_m_s', 'rms_speed_m_s', 'min_height_m')):
            raise ValueError('Nonfinite online trace metrics')
    endpoint_fields = ('finite', 'rms_speed_m_s', 'max_speed_m_s', 'vertices_outside_table', 'min_height_m', 'guard')
    for i in range(n):
        measured = SettlingGate(p['dt_s']).observe(1, a['particle_q'][i + 1], a['particle_qd'][i + 1])
        for key in endpoint_fields:
            if rows[(i + 1) * 24 - 1][key] != measured[key]:
                raise ValueError('Frame endpoint disagrees with trace: ' + key)
    geometry = geometry_diagnostics(a['particle_q'], a['state_time_s'], table_top_z=.8,
        table_xy_bounds=[ [.15, 1.15], [-1., 1.] ], faces=m['tri_indices'], inverse_material_rest=m['tri_poses'])
    q, times = a['particle_q'], a['state_time_s']
    ee = a['body_q'][:, p['end_effector_body_ids'], :3]
    events = []
    for side in range(2):
        closed = a['command_openness'][:, side] < .1
        for i in np.flatnonzero(np.diff(closed.astype(int), prepend=0) == 1):
            releases = np.flatnonzero(a['command_openness'][i:, side] > .8)
            end = int(i + releases[0]) if len(releases) else n
            events.append(dict(side=('left', 'right')[side], close_command_time_s=float(times[i]),
                reopen_command_time_s=float(times[end]) if end < n else None,
                wrist_distance_to_cloth_at_close_m=float(np.linalg.norm(q[i] - ee[i, side], axis=1).min()),
                cloth_p95_height_gain_from_close_m=float(np.quantile(q[i:end + 1, :, 2], .95, axis=1).max()
                                                        - np.quantile(q[i, :, 2], .95)),
                scope='command-delimited motion diagnostic, not confirmed attachment'))
    return dict(status='replay_integrity_passed', frames=n + 1, seconds=n / 60,
        raw_result_sha256=digest(root / 'RESULT.json'), chunks_verified=len(chunks),
        all_frame_endpoints_verified=True, dynamic_collision_flags_verified=True,
        dense_substep_states_independently_verified=False,
        max_online_speed_m_s=max(r['max_speed_m_s'] for r in rows),
        maximum_cloth_height_m=float(q[:, :, 2].max()),
        final_cloth_p95_height_m=float(np.quantile(q[-1, :, 2], .95)),
        actual_wrist_max_displacement_m=np.linalg.norm(ee - ee[0], axis=2).max(axis=0).tolist(),
        command_delimited_events=events, geometry=geometry,
        grasp_verified=False, fold_verified=False, accepted_for_training=False,
        pending=['inspect actual manipulation GIF', 'phase-aware grasp/release/fold scoring',
                 'self/robot intersection review if manipulation warrants scaling'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = review(args.root)
    write_once(Path(args.out), result)
    print(json.dumps(result))
