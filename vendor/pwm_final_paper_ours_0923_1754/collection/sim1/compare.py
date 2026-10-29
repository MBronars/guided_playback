"""Compare matched execution diagnostics without treating drift as a physics pass."""
import argparse
import json
from pathlib import Path
import numpy as np
from collection.common import check_hashes, digest, write_once


MATCH_FIELDS = ('schema', 'frames', 'substeps', 'fps', 'dt_s', 'input_sha256', 'home_sha256',
    'source_hashes', 'numpy_version', 'warp_version', 'device', 'uuid', 'robot_included',
    'gripper_filter', 'initial_policy', 'physical_scope', 'contact_policy', 'data_scope',
    'rigid_update_data_interval', 'actual_cloth_solver_file', 'self_contact_distance_m', 'probe_vertex_ids')


def match_protocols(left, right):
    optional_v1 = {'self_contact_distance_m', 'probe_vertex_ids'}
    for protocol in (left, right):
        schema = protocol.get('schema')
        if schema not in ('sim1-matched-home-execution-v1', 'sim1-matched-home-execution-v2'):
            raise ValueError('Unsupported matched execution schema')
        required = set(MATCH_FIELDS) - (optional_v1 if schema.endswith('v1') else set())
        if not required.issubset(protocol) or protocol.get('mode') not in ('eager', 'captured'):
            raise ValueError('Incomplete matched execution protocol')
    for key in MATCH_FIELDS:
        if left.get(key) != right.get(key):
            raise ValueError('Unmatched protocol setting: ' + key)


def compare_arrays(left, right):
    if set(left) != set(right): raise ValueError('Diagnostic array fields differ')
    rows = {}
    for name in sorted(left):
        a, b = np.asarray(left[name]), np.asarray(right[name])
        if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Mismatched/nonfinite array: '+name)
        difference = a.astype(np.float64)-b.astype(np.float64)
        rows[name] = dict(exact=np.array_equal(a,b), max_abs=float(np.abs(difference).max(initial=0)),
                          rms=float(np.sqrt(np.mean(difference**2))))
        if name in ('particle_q', 'particle_qd'):
            norm = np.linalg.norm(difference, axis=-1)
            rows[name].update(max_vector_difference=float(norm.max(initial=0)),
                              rms_vector_difference=float(np.sqrt(np.mean(norm**2))))
    return rows


def compare(left, right):
    roots = [Path(left).resolve(), Path(right).resolve()]
    protocols = []
    for root in roots:
        result = json.loads((root/'RESULT.json').read_text())
        if result['status'] != 'complete_diagnostic' or result['accepted_for_training'] is not False:
            raise ValueError('Only complete, unpromoted matched diagnostics can be compared')
        check_hashes(root, result['artifacts'])
        protocols.append(json.loads((root/'PROTOCOL.json').read_text()))
    match_protocols(*protocols)
    with np.load(roots[0]/'finalized_model.npz',allow_pickle=False) as a, np.load(roots[1]/'finalized_model.npz',allow_pickle=False) as b:
        initial = compare_arrays(a,b)
        if not all(v['exact'] for v in initial.values()): raise ValueError('Finalized models differ')
    with np.load(roots[0]/'actual_states.npz',allow_pickle=False) as a, np.load(roots[1]/'actual_states.npz',allow_pickle=False) as b:
        states = compare_arrays(a,b)
        for name in ('state_time_s','state_index','transition_index','fps','command_joint_q','command_openness'):
            if not states[name]['exact']: raise ValueError('Different clock or commands: '+name)
        for name in ('particle_q','particle_qd','body_q','actual_joint_q'):
            np.testing.assert_array_equal(a[name][0],b[name][0],err_msg='Different initial '+name)
    substeps = []
    for index in range(protocols[0]['frames']):
        with np.load(roots[0]/f'substeps-{index:04d}.npz',allow_pickle=False) as a, np.load(roots[1]/f'substeps-{index:04d}.npz',allow_pickle=False) as b:
            rows = compare_arrays(a,b)
            substeps.append(dict(frame=index, arrays=rows))
    exact = all(v['exact'] for v in states.values()) and all(v['exact'] for r in substeps for v in r['arrays'].values())
    return dict(left=str(roots[0]),right=str(roots[1]),modes=[p['mode'] for p in protocols],
        result_sha256=[digest(p/'RESULT.json') for p in roots],finalized_model_exact=True,
        state_comparison=states,substep_comparisons=substeps,
        maximum_substep_position_difference_mm=1000*max(r['arrays']['particle_q']['max_vector_difference'] for r in substeps),
        verdict='bitwise_same' if exact else 'differences_measured_requires_same_mode_repeat_controls',
        limitation='Matching this short fixed-home diagnostic does not prove settling or manipulation correctness',
        accepted_for_training=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('left','right','out'): p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args(); result=compare(args.left,args.right); write_once(args.out,result)
    print(json.dumps({k:result[k] for k in ('verdict','maximum_substep_position_difference_mm')}))
