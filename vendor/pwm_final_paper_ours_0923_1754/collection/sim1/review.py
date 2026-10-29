"""Independently review saved matched-home diagnostics, without physics approval.

Checks every physical substep against its recorded motion/overflow trace and each
frame endpoint against actual_states. Geometry is measured at every saved frame;
self-intersection, robot intersection and manipulation remain separate gates.
No simulator import or GPU is required.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from collection.common import check_hashes, digest, write_once
from .contracts import geometry_diagnostics, principal_stretches
from .recording import StateRecorder
from .settling import SettlingGate, SettlingAndHold


def read_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {name: arrays[name] for name in arrays.files}


def review(root):
    root = Path(root).resolve()
    result = json.loads((root / 'RESULT.json').read_text())
    if result['status'] != 'complete_diagnostic' or result['accepted_for_training'] is not False:
        raise ValueError('Only complete, unpromoted execution diagnostics are supported')
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    if protocol['numpy_version'] != np.__version__:
        raise ValueError('Exact trace replay requires the recorded NumPy version')
    frames, substeps = protocol['frames'], protocol['substeps']
    if (protocol['schema'] not in ('sim1-matched-home-execution-v1', 'sim1-matched-home-execution-v2') or
            type(frames) is not int or not 1 <= frames <= (120 if protocol['schema'].endswith('v1') else 720) or substeps != 24 or
            protocol['fps'] != 60 or protocol['dt_s'] != 1 / 1440):
        raise ValueError('Unsupported diagnostic protocol or clock')
    required = {'PROTOCOL.json', 'SOURCES.json', 'executed_sources.tar.gz',
                'finalized_model.npz', 'actual_states.npz', 'SUBSTEPS.json'}
    required.update(f'substeps-{i:04d}.npz' for i in range(frames))
    if not required.issubset(result['artifacts']):
        raise ValueError('Required artifacts omitted from completion receipt')
    if result['frames'] != frames + 1 or result['substeps'] != frames * substeps:
        raise ValueError('Completion counts disagree with protocol')
    check_hashes(root, result['artifacts'])
    model = read_arrays(root / 'finalized_model.npz')
    states = read_arrays(root / 'actual_states.npz')
    rows = json.loads((root / 'SUBSTEPS.json').read_text())
    if len(rows) != frames * substeps:
        raise ValueError('Incomplete physical-substep trace')
    # Replay the actual-state schema without substituting commanded joint values.
    recorder = StateRecorder(60, len(model['particle_q']), states['body_q'].shape[1],
                             len(model['home']), max_states=frames + 1)
    keys = ('particle_q', 'particle_qd', 'body_q', 'actual_joint_q')
    def state(index):
        return {('joint_q' if key == 'actual_joint_q' else key): states[key][index] for key in keys}
    recorder.append_initial(**state(0))
    np.testing.assert_array_equal(states['particle_q'][0], model['particle_q'])
    np.testing.assert_array_equal(states['particle_qd'][0], 0)
    np.testing.assert_array_equal(states['actual_joint_q'][0], model['home'])
    for i in range(frames):
        np.testing.assert_array_equal(states['command_joint_q'][i], model['home'])
        np.testing.assert_array_equal(states['command_openness'][i], model['openness'])
        recorder.append_transition(i, command_joint_q=states['command_joint_q'][i],
            command_openness=states['command_openness'][i], **state(i + 1))
    reconstructed = recorder.arrays(frames)
    if set(reconstructed) != set(states):
        raise ValueError('Unexpected or missing actual-state fields')
    for key in reconstructed:
        if reconstructed[key].dtype != states[key].dtype:
            raise ValueError('Wrong actual-state dtype: ' + key)
        np.testing.assert_array_equal(reconstructed[key], states[key], err_msg=key)

    gate = SettlingGate(protocol['dt_s'])
    table_penetration = 0.
    first_low = None
    for i in range(frames):
        arrays = read_arrays(root / f'substeps-{i:04d}.npz')
        for key in keys:
            expected = (substeps, *states[key].shape[1:])
            if arrays[key].shape != expected or not np.isfinite(arrays[key]).all():
                raise ValueError('Invalid substep state: ' + key)
            np.testing.assert_array_equal(arrays[key][-1], states[key][i + 1], err_msg=key)
        overflows = []
        for prefix in ('vertex_colliding_triangles', 'edge_colliding_edges'):
            count, capacity = (arrays[prefix + suffix] for suffix in ('_count', '_buffer_sizes'))
            if (count.shape != capacity.shape or count.ndim != 2 or count.shape[0] != substeps
                    or count.dtype.kind not in 'iu' or capacity.dtype.kind not in 'iu'
                    or np.any(count < 0) or np.any(capacity < 0)):
                raise ValueError('Invalid self-contact count/capacity buffers')
            overflows.append(np.maximum(count.astype(np.int64) - capacity.astype(np.int64), 0).max(axis=1, initial=0))
        for s in range(substeps):
            index = i * substeps + s
            q = arrays['particle_q'][s]
            for vertex in protocol.get('probe_vertex_ids', []):
                pairs = arrays[f'probe_vertex_{vertex}_pairs'][s]
                count = int(arrays['vertex_colliding_triangles_count'][s, vertex])
                capacity = int(arrays['vertex_colliding_triangles_buffer_sizes'][s, vertex])
                if pairs.shape != (capacity, 2) or pairs.dtype.kind not in 'iu':
                    raise ValueError('Invalid recorded contact-pair buffer')
                active = pairs[:min(count,capacity)]
                if (np.any(active[:,0] != vertex) or np.any(active[:,1] < 0) or
                        np.any(active[:,1] >= len(model['tri_indices']))):
                    raise ValueError('Invalid active contact pair identity')
            actual = gate.observe(index + 1, q, arrays['particle_qd'][s],
                                  int(max(v[s] for v in overflows)))
            for key, value in actual.items():
                if rows[index].get(key) != value:
                    raise ValueError(f'Substep trace disagrees at {index + 1}: {key}')
            if actual['guard']:
                raise ValueError('Completed diagnostic contains a numerical guard failure')
            if first_low is None and actual['low_motion']:
                first_low = actual['time_s']
            over = (q[:, 0] >= .15) & (q[:, 0] <= 1.15) & (np.abs(q[:, 1]) <= 1.)
            table_penetration = max(table_penetration, float(np.maximum(.8 - q[over, 2], 0).max(initial=0)))
    geometry = geometry_diagnostics(states['particle_q'], states['state_time_s'],
        table_top_z=.8, table_xy_bounds=np.array([[.15, 1.15], [-1., 1.]]),
        faces=model['tri_indices'], inverse_material_rest=model['tri_poses'])
    strain = []
    for q in states['particle_q']:
        stretches = principal_stretches(q, model['tri_indices'], model['tri_poses'])
        strain.append(dict(max_abs=float(np.abs(stretches - 1).max()),
                           p99_abs=float(np.percentile(np.abs(stretches - 1), 99))))
    hold_result = None
    if frames == 720:
        hold = SettlingAndHold(protocol['dt_s'], 10., 2.)
        for row in rows:
            hold_result = hold.observe(row)
            if hold_result['stopping']: break
    return dict(root=str(root), result_sha256=digest(root / 'RESULT.json'),
        review_numpy_version=np.__version__, reviewer_sha256=digest(__file__),
        integrity='all_saved_substeps_trace_and_frame_endpoints_match',
        states=frames + 1, substeps=len(rows), duration_s=frames / 60,
        maximum_substep_speed_m_s=max(r['max_speed_m_s'] for r in rows),
        maximum_substep_table_penetration_m=table_penetration,
        maximum_outside_table_vertices=max(r['vertices_outside_table'] for r in rows),
        self_contact_overflow_observed=gate.ever_overflow,
        first_low_motion_time_s=first_low,
        maximum_continuous_quiet_seconds=max(r['quiet_duration_s'] for r in rows),
        motion_gate_pass_observed=gate.passed_step is not None,
        settling_10s_hold_2s=hold_result,
        saved_frame_geometry=geometry, saved_frame_strain=strain,
        physics_qualified=False, accepted_for_training=False,
        limitations=['fixed-home diagnostic; full 10s deadline/2s hold evaluated only for 720-frame runs',
                     'no self-intersection or robot-intersection certification',
                     'no achieved grasp, release or fold',
                     'source-metric strain measured, no post-hoc acceptance threshold'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.root)
    write_once(args.out, result)
    print(json.dumps({key: result[key] for key in ('integrity', 'states', 'substeps', 'motion_gate_pass_observed')}))
