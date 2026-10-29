"""Export actual cloth/robot states with explicit integer-clock correspondence.

Templates use each arm's own visible meshes and local shape transforms. Native
commands remain on their original clock; downsampling never pretends a sampled
command was held throughout a coarse transition. This is not a physics gate.
"""
from __future__ import annotations

import numpy as np

from .contracts import (finite_array, integer_state_times, points_from_recorded_links,
                        require_link_membership, unit_quaternions)
from .export import check_export, outgoing_state_deltas


def transform_points(points, pose):
    pose = finite_array(pose, 'shape pose', (7,)).astype(np.float64)
    q = unit_quaternions(pose[3:])
    v = finite_array(points, 'mesh vertices', (None, 3)).astype(np.float64)
    cross = 2 * np.cross(q[:3], v)
    return v + q[3] * cross + np.cross(q[:3], cross) + pose[:3]


def surface_samples(vertices, triangles, count, rng):
    v = finite_array(vertices, 'vertices', (None, 3))
    t = finite_array(triangles, 'triangles', (None, 3))
    if type(count) is not int or count < 1:
        raise ValueError('Positive integer sample count required')
    if t.dtype.kind not in 'iu' or not len(t) or t.min() < 0 or t.max() >= len(v):
        raise ValueError('Nonempty in-range integer mesh topology required')
    tri = v[t].astype(np.float64)
    areas = .5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.isfinite(areas).all() or not np.isfinite(areas.sum()) or areas.sum() <= 0:
        raise ValueError('Positive finite surface area required')
    ids = rng.choice(len(tri), size=count, p=areas / areas.sum())
    r = rng.random((count, 2)); s = np.sqrt(r[:, 0])
    bary = np.stack((1 - s, s * (1 - r[:, 1]), s * r[:, 1]), axis=1)
    return np.sum(tri[ids] * bary[:, :, None], axis=1), ids, bary


def make_templates(model, count=1000, seed=42):
    """Persistent link-local identities, including triangle/barycentric lineage."""
    if type(count) is not int or count < 5 or type(seed) is not int or seed < 0:
        raise ValueError('At least five particles and nonnegative integer seed required')
    names = list(model.body_key)
    shape_body, flags = model.shape_body.numpy(), model.shape_flags.numpy()
    scales, poses = model.shape_scale.numpy(), model.shape_transform.numpy()
    wrist = max(count // 5, 1)
    budget = [wrist, (count - wrist) // 2, count - wrist - (count - wrist) // 2]
    link_ids = np.repeat(np.arange(3, dtype=np.int32), budget)
    require_link_membership(link_ids, count)
    arrays = {'link_ids': link_ids}
    evidence = dict(seed=seed, points_per_link=budget, sides={})
    for side_index, (side, links) in enumerate((
            ('left', ['left_link16', 'left_link17', 'left_link18']),
            ('right', ['right_link26', 'right_link27', 'right_link28']))):
        body_ids, points, triangle_ids, barycentrics, rows = [], [], [], [], []
        for index, name in enumerate(links):
            if names.count(name) != 1:
                raise ValueError('Missing or ambiguous actual robot link: ' + name)
            body = names.index(name)
            shapes = np.flatnonzero((shape_body == body) & ((flags & 1) != 0))
            if not len(shapes):
                raise ValueError('No visible actual mesh on ' + name)
            vertices, triangles, offset = [], [], 0
            for shape in shapes:
                src = model.shape_source[int(shape)]
                if not hasattr(src, 'vertices') or not hasattr(src, 'indices'):
                    raise ValueError('Explicitly meshed gripper geometry required')
                v = transform_points(np.asarray(src.vertices) * scales[shape], poses[shape])
                t = np.asarray(src.indices, dtype=np.int64).reshape(-1, 3)
                vertices.append(v); triangles.append(t + offset); offset += len(v)
            v, t = np.concatenate(vertices), np.concatenate(triangles)
            p, ids, bary = surface_samples(v, t, budget[index],
                np.random.default_rng(seed + side_index * 1000 + index))
            points.append(p); triangle_ids.append(ids); barycentrics.append(bary); body_ids.append(body)
            rows.append(dict(body_name=name, body_id=body, shape_ids=shapes.tolist(),
                             triangle_count=len(t), vertex_count=len(v)))
        arrays[side + '_local_points'] = np.concatenate(points).astype(np.float32)
        arrays[side + '_triangle_ids'] = np.concatenate(triangle_ids)
        arrays[side + '_barycentric'] = np.concatenate(barycentrics)
        arrays[side + '_body_ids'] = np.asarray(body_ids, dtype=np.int32)
        evidence['sides'][side] = rows
    return arrays, evidence


def export_actual(raw, template, *, stride=1):
    """Select complete integer intervals; preserve terminal state and ALL commands.

    source_state_index also defines native command interval boundaries: the
    commands for exported transition i are [index[i]:index[i+1]]. Actual joint
    and end-effector deltas are recomputed on the retained state sequence.
    """
    if type(stride) is not int or stride < 1:
        raise ValueError('Positive integer cadence stride required')
    q = finite_array(raw['particle_q'], 'particle_q', (None, None, 3))
    S = len(q)
    if S < 2 or not q.shape[1] or (S - 1) % stride:
        raise ValueError('At least two states and exact terminal-preserving stride required')
    b = finite_array(raw['body_q'], 'body_q', (S, None, 7))
    j = finite_array(raw['actual_joint_q'], 'actual_joint_q', (S, None))
    if not b.shape[1] or not j.shape[1]:
        raise ValueError('Nonempty actual robot states required')
    unit_quaternions(b[:, :, 3:])
    command = finite_array(raw['command_joint_q'], 'command_joint_q', (S - 1, j.shape[1]))
    openness = finite_array(raw['command_openness'], 'command_openness', (S - 1, 2))
    if np.any((openness < 0) | (openness > 1)):
        raise ValueError('Native openness commands outside [0,1]')
    fps = float(finite_array(raw['fps'], 'fps', ()))
    indices = finite_array(raw['state_index'], 'state_index', (S,))
    times = integer_state_times(indices, fps)
    transitions = finite_array(raw['transition_index'], 'transition_index', (S - 1,))
    if transitions.dtype.kind not in 'iu' or not np.array_equal(transitions, np.arange(S - 1)):
        raise ValueError('Complete native outgoing command indices required')
    if not np.array_equal(finite_array(raw['state_time_s'], 'state_time_s', (S,)), times):
        raise ValueError('Raw times disagree with the integer physical clock')
    selected = np.arange(0, S, stride)
    link_ids = np.asarray(template['link_ids'])
    require_link_membership(link_ids, len(link_ids))
    output = dict(cloth_points=q[selected].copy(), joint_q=j[selected].copy(),
        openness=np.concatenate((openness, openness[-1:]))[selected].copy(),
        ee_link_ids=link_ids.copy(), fps=np.array(fps / stride), table_top_z=np.array(.8),
        state_time_s=times[selected], source_state_index=indices[selected].copy(),
        native_command_joint_q=command.copy(), native_command_openness=openness.copy(),
        native_transition_index=transitions.copy(), native_fps=np.array(fps),
        command_interval_offsets=selected.copy(),
        terminal_openness_observed=np.array(False))
    all_bodies = []
    for side in ('left', 'right'):
        ids = finite_array(template[side + '_body_ids'], side + '_body_ids', (3,))
        if ids.dtype.kind not in 'iu' or len(set(ids.tolist())) != 3 or ids.min() < 0 or ids.max() >= b.shape[1]:
            raise ValueError('Three distinct actual body IDs per gripper required')
        all_bodies.extend(ids.tolist())
        poses = b[selected][:, ids]
        output[side + '_ee_points'] = points_from_recorded_links(
            template[side + '_local_points'], link_ids, poses)
        output[side + '_ee_pose'] = poses[:, 0].astype(np.float64)
    if len(set(all_bodies)) != 6:
        raise ValueError('Left and right grippers must use distinct actual links')
    output.update(outgoing_state_deltas(output))
    check_export(output, expected_ee_points=len(link_ids))
    return output


def export_run(root, out, *, stride=2):
    """CPU-only immutable export of one completed, hash-verified diagnostic.

    Reconstruct visible meshes using the pinned external code/assets, and require
    exact finalized cloth, robot limits, home and collision flags before using
    their templates. Export approval never substitutes for task qualification.
    """
    import json
    import os
    from pathlib import Path
    from collection.common import check_hashes, digest, write_once
    from .environment import build_environment
    from .provenance import snapshot_sources
    from .qualify import save_arrays
    from .robot import read_home, gripper_filter

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicitly hide CUDA for CPU export construction')
    root, out = Path(root).resolve(strict=True), Path(out)
    result = json.loads((root / 'RESULT.json').read_text())
    if result['status'] != 'complete_diagnostic' or result['accepted_for_training'] is not False:
        raise ValueError('Complete unpromoted raw diagnostic required')
    required = {'PROTOCOL.json', 'SOURCES.json', 'actual_states.npz', 'finalized_model.npz'}
    if not required.issubset(result['artifacts']):
        raise ValueError('Raw completion omits required artifact hashes')
    check_hashes(root, result['artifacts'])
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    if protocol['schema'] not in ('sim1-matched-home-execution-v1', 'sim1-matched-home-execution-v2',
                                  'sim1-command-replay-v1'):
        raise ValueError('Unsupported source protocol')
    sources = json.loads((root / 'SOURCES.json').read_text())
    if sources != protocol['source_hashes']:
        raise ValueError('Protocol and source inventory disagree')
    def input_path(key):
        matches = [Path(p) for p, sha in sources.items() if sha == protocol[key]]
        if len(matches) != 1:
            raise ValueError('Missing or ambiguous recorded input: ' + key)
        return matches[0]
    # Collection evolves separately; executed external simulator/assets must not.
    external = {p: sha for p, sha in sources.items() if 'collection' not in Path(p).parts}
    package = Path(__file__).resolve().parents[1]
    current = {str(p.resolve()): digest(p) for p in
               [*package.glob('*.py'), *Path(__file__).resolve().parent.rglob('*.py')]}
    current.update(external)
    for p, sha in current.items():
        if digest(p) != sha:
            raise ValueError('Pinned export dependency changed: ' + p)
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='initializing', accepted_for_training=False)
    try:
        import warp as wp
        wp.config.enable_cuda = False
        wp.config.kernel_cache_dir = str(out / 'kernel-cache')
        if wp.__version__ != protocol['warp_version']:
            raise ValueError('Use the recorded Warp version for model reconstruction')
        env, model, _ = build_environment(input_path('input_sha256'), device='cpu',
            sim1_source=protocol['simulator_source'], assets_root=protocol['assets_root'])
        home, openness = read_home(input_path('home_sha256'), env, model)
        gripper_filter(env, model, openness)
        with np.load(root / 'finalized_model.npz', allow_pickle=False) as finalized:
            for key in finalized.files:
                actual = (home if key == 'home' else openness if key == 'openness' else
                          getattr(model, key).numpy() if hasattr(getattr(model, key), 'numpy')
                          else np.asarray(getattr(model, key)))
                np.testing.assert_array_equal(actual, finalized[key], err_msg=key)
        template, evidence = make_templates(model)
        save_arrays(out / 'templates.npz', template)
        write_once(out / 'TEMPLATES.json', evidence)
        with np.load(root / 'actual_states.npz', allow_pickle=False) as raw:
            exported = export_actual(raw, template, stride=stride)
        save_arrays(out / 'episode.npz', exported)
        archive = snapshot_sources(current, out / 'export_sources.tar.gz')
        for p, sha in current.items():
            if digest(p) != sha:
                raise ValueError('Source changed during export: ' + p)
        receipt.update(status='complete_diagnostic_export', raw_root=str(root),
            raw_result_sha256=digest(root / 'RESULT.json'), stride=stride,
            frames=len(exported['cloth_points']), fps=float(exported['fps']),
            existing_cloth_loader_clock_compatible=float(exported['fps']) == 30.,
            finalized_model_exact=True, template_lineage=evidence,
            source_archive=archive, source_sha256=current,
            command_contract='all native outgoing commands retained; command_interval_offsets delimit coarse intervals',
            openness_contract='outgoing command at retained state; last value held only for terminal padding',
            achieved_state_contract='cloth, joints and each arm own links from recorded physical states',
            pending=['physics/settling/intersection qualification', 'achieved manipulation', 'training-loader pilot'],
            artifacts={name: digest(out / name) for name in
                       ('episode.npz', 'templates.npz', 'TEMPLATES.json', 'export_sources.tar.gz')})
    except Exception as error:
        receipt.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        raise
    finally:
        write_once(out / 'RESULT.json', receipt)
    return receipt


def training_sample(exported):
    """Materialize a 30 Hz candidate once, without the legacy extra stride.

    Returns numeric WDS fields, not a training-release decision. No corpus
    split, exclusion decision, temporal resampling or padding is implicit here.
    The unchanged loader already handles native ragged particle counts.
    """
    check_export(exported)
    if float(exported['fps']) != 30. or 'state_time_s' not in exported:
        raise ValueError('Existing unified cloth loader requires explicit actual 30 Hz states')
    cloth, left, right = (np.asarray(exported[key], dtype=np.float32) for key in
                          ('cloth_points', 'left_ee_points', 'right_ee_points'))
    n_cloth, n_left, n_right = cloth.shape[1], left.shape[1], right.shape[1]
    positions = np.concatenate((cloth, left, right), axis=1)
    positions += np.array([-.65, 0., -.8], dtype=np.float32)
    if not np.isfinite(positions).all():
        raise ValueError('Canonical float32 positions overflowed')
    ids = np.repeat(np.arange(3, dtype=np.int64), [n_cloth, n_left, n_right])
    return dict(particle_pos=positions, obj_ids=ids, valid_particles=np.ones(len(ids), dtype=bool),
                n_cloth=np.int64(n_cloth), dt=np.float32(1 / 30),
                state_time_s=np.asarray(exported['state_time_s']).copy(),
                source_state_index=np.asarray(exported['source_state_index']).copy(),
                ee_link_ids=np.asarray(exported['ee_link_ids']).copy())


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--stride', type=int, default=2,
                        help='Native 60 Hz divided by stride; default 2 matches the existing 30 Hz cloth loader')
    args = parser.parse_args()
    receipt = export_run(args.root, args.out, stride=args.stride)
    print(json.dumps({key: receipt[key] for key in ('status', 'frames', 'fps', 'accepted_for_training')}))
