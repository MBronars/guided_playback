"""Export a completed objective comparison as interactive, synchronized point clouds.

Requires rerun-sdk==0.29.1. Reads saved results only; does not run the model or replay.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
import torch


LABELS = {
    'goal': 'Goal only',
    'geometry': 'Goal + geometry/joints',
    'contact': 'Goal + geometry/joints + contact',
}
BODY_NAMES = ('object', 'hand', 'left_finger', 'right_finger')
COLORS = ([223, 165, 44], [116, 123, 132], [41, 140, 200], [38, 161, 131])


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Source changed since its audit: {path}')
    return path


def load_comparison(root, name):
    plan, queue = read(root/'plan.json'), read(root/'queue.json')
    scene = next(s for s in plan['scenes'] if s['name'] == name)
    selected = queue['selected'][name]
    arms = [selected['arm'], *LABELS]
    for arm in LABELS:
        task = next(t for t in queue['tasks'] if t['scene']['name'] == name and t['arm'] == arm and t['phase'] == 3)
        if task['state'] != 'completed':
            raise ValueError(f'{name}/{arm} is {task["state"]}; export requires the complete four-way comparison')
    directory = Path(scene['directory'])
    sources = {str(checked(directory/k, v)): v for k, v in scene['hashes'].items()}
    batch = torch.load(directory/'input.pt', map_location='cpu', weights_only=False)
    geometry = torch.load(directory/'collision_geometry.pt', map_location='cpu', weights_only=False)
    calibration = torch.load(directory/'calibration.pt', map_location='cpu', weights_only=False)['calibration']
    body_ids = batch.particle_body[0].numpy()
    keys = sorted(geometry['bodies'])
    if keys != list(range(4)) or set(body_ids.tolist()) != set(keys):
        raise ValueError('This exporter expects the pickup object, hand and two fingers')
    templates = {k: batch.body_templates[0, k, batch.body_template_mask[0, k]].numpy() for k in keys}
    start = batch.context_len - 1
    tracks, results, original = {}, {}, None
    for arm in arms:
        artifact = root/'artifacts'/name/arm
        visual = read(artifact/'visualization.json')
        protocol = read(artifact/'plan/protocol.json')
        audit = read(artifact/'audit.json')
        model_path = root/'runs'/name/arm/f'seed_{scene["seed"]}.pt'
        replay_path = artifact/'dt00025/execution.npz'
        original_path = Path(visual['original_sample_source'])
        for path, expected in [(model_path, visual['source_sha256']),
                               (replay_path, visual['execution_sha256']),
                               (original_path, visual['original_sample_sha256'])]:
            sources[str(checked(path, expected))] = expected
        for path in [artifact/'audit.json', artifact/'visualization.json', artifact/'plan/protocol.json']:
            sources[str(path)] = sha(path)
        saved = torch.load(model_path, map_location='cpu', weights_only=False)
        baseline = torch.load(original_path, map_location='cpu', weights_only=False)['initial']
        if read(original_path.parent/'protocol.json')['actions'] != 'hidden':
            raise ValueError('The original sample must be unconditional on future actions')
        np.testing.assert_array_equal(saved['initial']['positions'].numpy(), baseline['positions'].numpy())
        if original is None:
            original = baseline
            tracks['original'] = dict(points=baseline['positions'][0, start:].numpy(),
                                      T=baseline['T'][0, start:].numpy())
        else:
            np.testing.assert_array_equal(original['positions'].numpy(), baseline['positions'].numpy())
        rollout = saved['result']['rollout']
        points = rollout['positions'][0, start:].numpy()
        R, T = rollout['R'][0, start:].numpy(), rollout['T'][0, start:].numpy()
        with np.load(replay_path) as replay:
            np.testing.assert_array_equal(replay['desired_R'], R)
            np.testing.assert_array_equal(replay['desired_T'], T)
            replay_points = np.empty(points.shape, dtype=np.float32)
            for k in keys:
                reconstructed = np.einsum('fij,nj->fni', R[:, k], templates[k]) + T[:, k, None]
                np.testing.assert_allclose(reconstructed, points[:, body_ids == k], rtol=0, atol=1e-5)
                replay_points[:, body_ids == k] = (
                    np.einsum('fij,nj->fni', replay['model_R'][:, k], templates[k]) + replay['model_T'][:, k, None])
            tracks[f'{arm}/guided'] = dict(points=points, T=T)
            tracks[f'{arm}/replay'] = dict(points=replay_points, T=replay['model_T'].copy())
        np.testing.assert_allclose(protocol['target'][0], scene['target'], rtol=0, atol=1e-7)
        if (protocol['arguments']['iterations'] != selected['iterations'] or
                protocol['arguments']['horizon'] != scene['horizon']):
            raise ValueError(f'Mismatched comparison budget: {arm}')
        results[arm] = dict(label=LABELS.get(arm, 'Full objective'), audit=audit, protocol=protocol,
                            replay_finger_limit_peak_mm=visual['replay_finger_limit_peak_mm'])
    for track in tracks.values():
        if track['points'].shape != (scene['horizon']+1, len(body_ids), 3) or not np.isfinite(track['points']).all():
            raise ValueError('Unexpected point count, frame count or nonfinite trajectory')
        np.testing.assert_allclose(track['points'][0], tracks['original']['points'][0], rtol=0, atol=1e-5)
    for relative in ['guidance/pick_objective.py', 'guidance/physical_objective.py', 'guidance/mesh_physical.py',
                     'guidance/place_objective.py', 'guidance/collision_distance.py', 'guidance/unconditional.py']:
        path = Path(plan['source'])/relative
        sources[str(path)] = sha(path)
    return dict(scene=scene, selected=selected, arms=arms, tracks=tracks, results=results, calibration=calibration,
                body_ids=body_ids, support_z=float(geometry['metadata']['support_z']), sources=sources)


def term_explanations(data):
    """Explain the frozen study's mesh objectives, with this run's actual settings."""
    protocol = data['results'][data['selected']['arm']]['protocol']
    args, weights = protocol['arguments'], protocol['interaction_weights']
    if args['collision_backend'] != 'fcl' or protocol['obstacle_spec'] is not None:
        raise ValueError('Term explanations currently cover the obstacle-free FCL objective comparison')
    goal_weight, physical_weight = args['goal_weight'], args['physical_weight']
    contact_mm, penetration_mm = args['mesh_contact_tolerance_mm'], args['mesh_penetration_tolerance_mm']
    support_mm = protocol['support_tolerance_m']*1000
    hold = args['hold_frames']
    limits = data['calibration']['limits'].numpy()
    dt = float(data['calibration']['dt'])
    pick = data['scene']['task'] == 'pick'
    reduction = args['goal_reduction']
    weight = lambda name: weights[name]*physical_weight
    lines = ['# Terms explained', '',
        '**Every variant includes the goal.** The complete label is **Goal + geometry/joints + contact**.', '',
        'These definitions describe the frozen implementation that generated this recording. '
        'Distances are calculated in metres; millimetres below make the tolerances easier to read. '
        '`positive(x) = max(x, 0)` and `clamp01(x)` clips to [0, 1]. Means include all indicated future frames.', '',
        '## What each variant includes', '',
        '- **Goal only:** goal + noise-change regularizer; noise RMS cap stays active.',
        '- **Goal + geometry/joints:** add object/gripper penetration, object/table intrusion, '
        'gripper/table intrusion, finger travel limits, and finger speed limits.',
        '- **Goal + geometry/joints + contact:** retain all preceding terms and add finger proximity.',
        '- **Full objective:** retain all preceding terms and add unsupported upward motion and relative slip.'+
        (' Placement additionally uses rest, release and settling.' if not pick else ''), '',
        '## Goal and geometry', '',
        f'- **Goal — weight {goal_weight:g}.** At each of the final {hold} future frames, calculate the Euclidean '
        f'distance from the object body origin to the requested target. Take the **{reduction}**, then multiply by '
        f'{goal_weight:g}. This is distance, not squared distance. The reported held error instead uses the worst '
        'distance across these frames. The 20 mm ring is an evaluation tolerance; the goal loss continues to '
        'encourage smaller distances inside it.', '',
        '- **Collision distance measurement.** Place each body\'s supplied convex collision components at its '
        'predicted pose. FCL returns a signed distance for every object/component pair with the hand and each '
        'finger; take the minimum component-pair distance for each body. Positive means separation, negative '
        'means overlap. These are full collision-geometry queries, even though the viewer shows only points. '
        'Overlapping convex components do not define an exact union penetration distance.', '',
        f'- **Object–gripper penetration — weight {weight("penetration"):g}.** For each future frame and each '
        f'of the hand/two fingers, calculate `positive(-signed_distance - {penetration_mm:g} mm)`. '
        'Take the worst body at each frame, average those values over time, and add the worst value over the '
        'entire trajectory. This penalizes average intrusion and isolated bad frames; up to the stated '
        'tolerance contributes zero loss.', '',
        f'- **Object/table intrusion — weight {weight("support"):g}.** Transform the object\'s collision vertices '
        'into world coordinates; its bottom is the minimum vertex height. Calculate '
        f'`positive(table_height - object_bottom - {support_mm:g} mm)` at each future frame. '
        'The term is mean excess plus maximum excess. It discourages going below the table; '
        'it does not by itself require the object to rest on the table.', '',
        f'- **Gripper/table intrusion — weight {weight("gripper_support"):g}.** Repeat the bottom-height '
        f'calculation for the hand and both fingers, with the same {support_mm:g} mm tolerance. '
        'Take the mean over frames and bodies plus the maximum over frames and bodies.', '',
        '## Finger travel and speed', '',
        '- **Finger coordinate used by both terms.** Express the finger origin relative to the hand, '
        'in the hand coordinate frame. Subtract its calibrated anchor and project onto its calibrated '
        'slide axis: `q = q_anchor + dot(finger_local - finger_anchor, slide_axis)`. '
        'The following terms constrain that scalar slide coordinate.', '',
        f'- **Finger travel limits — weight {weight("joint"):g}.** For each finger and future frame, calculate '
        '`max(q_min - q, q - q_max, 0)`. Add the mean and maximum violation over all frames/fingers. '
        'These are per-finger coordinates, not the total opening width. Calibrated ranges: '+
        '; '.join(f'finger {i+1}: {lo*1000:g}–{hi*1000:g} mm' for i, (lo, hi, _) in enumerate(limits))+'.', '',
        f'- **Finger speed — weight {weight("joint_speed"):g}.** Include the observed anchor coordinate, '
        'then calculate `positive(abs(q[t] - q[t-1]) - speed_limit * dt)` for every transition/finger. '
        'Add mean and maximum excess. The loss is excess travel per transition, not squared speed. '
        f'Here `dt = {dt:g} s`; allowed travel is '+
        '; '.join(f'finger {i+1}: {speed*dt*1000:g} mm/step ({speed:g} m/s)' for i, (_, _, speed) in enumerate(limits))+'.', '',
        '## Contact and motion consistency', '',
        '- **Shared contact quantities.** Clamp each finger\'s signed object distance to zero from below, '
        'then take the larger finger gap: `g[t] = max(positive(d_left[t]), positive(d_right[t]))`. '
        'This asks both fingers to be near the object. The lift gate is '
        f'`lift[t] = clamp01((object_bottom[t] - table_height - {support_mm:g} mm) / 20 mm)`. '
        f'It ramps from zero at {support_mm:g} mm above the table to one at {support_mm+20:g} mm.', '',
        f'- **Finger proximity / contact — weight {weight("contact"):g}.** First calculate '
        f'`excess[t] = positive(g[t] - {contact_mm:g} mm)`. Average `lift[t] * excess[t]` over all future frames.'+
        (' For this pick, also add the average excess over the final four frames, encouraging a held grasp '
         'at the end. That final contact interval is four frames; the goal interval is six.' if pick else
         ' Placement has no final held-contact add-on, allowing release at the end.')+
        ' This measures proximity; it does not calculate gripping force or force closure.', '',
        '- **Missing-contact gate.** For each transition use the previous frame\'s farther-finger gap: '
        f'`missing[t] = clamp01((g[t-1] - {contact_mm:g} mm) / 10 mm)`. '
        f'It is zero at or below {contact_mm:g} mm and one at or above {contact_mm+10:g} mm.', '',
        f'- **Unsupported upward motion — weight {weight("unsupported_rise"):g}.** Calculate '
        '`positive(object_z[t] - object_z[t-1]) * missing[t]` and average over transitions. '
        'This discourages upward motion when a finger was too far away at the preceding frame. '
        'It is a motion/contact proxy, not a Newtonian force-balance calculation.', '',
        f'- **Relative slip — weight {weight("slip"):g}.** Compute the object body origin in the hand frame: '
        '`local[t] = transpose(R_hand[t]) * (object_origin[t] - hand_origin[t])`. '
        'Calculate `norm(local[t] - local[t-1]) * (1 - missing[t]) * lift[t]`, then average. '
        'It discourages object translation relative to the hand while lifted and apparently held. '
        'This term does not directly penalize relative rotation; the independent audit checks it separately.', '',
        '## Noise optimization', '',
        f'- **Noise-change regularizer — weight {args["prior_weight"]:g}, in every variant.** Let '
        '`delta_noise` be the change from the original sampled pose noise. Average its squared entries '
        'over all valid optimized pose channels and all rollout windows. Add the weighted result to '
        'the trajectory objective. This regularizes the noise change, not the absolute sampled noise.', '',
        f'- **Noise RMS cap — {args["max_noise_rms"]:g}, in every variant.** After each Adam update, calculate '
        '`sqrt(mean(delta_noise**2))` separately for each window. If it exceeds the cap, scale that '
        'window\'s update down to the cap. This is a projection step, not an additional loss term.', '',
        f'- **Total loss.** `{goal_weight:g} * goal + sum(enabled physical weight * term) + '
        f'{args["prior_weight"]:g} * mean(delta_noise**2)`. The physical-weight multiplier is '
        f'{physical_weight:g}; the weights above include it. The model weights stay frozen; Adam changes '
        'the initial pose noise and differentiates through the generated rollout. No recorded future path '
        'is used as the target.', '',
        '## Reading the results', '',
        '- **Loss and audit have different jobs.** An average loss can be small despite a bad frame. '
        'The reported checks also examine worst-case errors, relative rotation, contact and simulator replay. '
        'Checks remain active even when the corresponding optimization term is removed.', '',
        '- **Inactive in this pick:** orientation targets, placement rest/release/settling '
        'and obstacle avoidance.' if pick else
        '- **Placement terms** are listed below; obstacle avoidance is inactive in this comparison.', '',
        '- **Implementation references:** `guidance/mesh_physical.py`, `guidance/physical_objective.py`, '
        '`guidance/pick_objective.py`, `guidance/collision_distance.py`, `guidance/unconditional.py`. '
        'The export receipt records hashes of the frozen run source.', '']
    if not pick:
        lines += ['## Placement-only additions', '',
            f'- **Orientation goal — weight {args["rotation_weight"]:g}.** Average the geodesic rotation error '
            f'in radians to the target orientation over the final {hold} frames. All variants retain this '
            'task goal when a target orientation is supplied.', '',
            f'- **Rest — weight {5*physical_weight:g}, full only.** Average the absolute distance of the object '
            f'bottom from the table over the final {hold} frames.', '',
            f'- **Release — weight {2*physical_weight:g}, full only.** Over the final {hold} frames, average '
            '`positive(5 mm - min(signed_distance_left, signed_distance_right))`. This requests clearance '
            'from both fingers at the end.', '',
            f'- **Settling — weight {2*physical_weight:g}, full only.** Average the norm of object displacement '
            f'over the final {max(1,hold-1)} transitions. This reduces translational motion, not angular velocity.', '']
    return '\n'.join(lines)


def summary(data):
    scene, selected = data['scene'], data['selected']
    lines = [f'# {scene["name"]}', '',
             f'{selected["iterations"]} updates; goal weight {selected["goal_weight"]:g}; seed {scene["seed"]}.', '',
             'Use the tabs to compare all guided plans, all simulator replays, or one objective against its original sample.', '',
             'All views share the **time** timeline. Press play or scrub; frame 0 is the last observed frame. '
             'The recording contains the anchor plus 68 future frames at 20 Hz.' if scene['horizon'] == 68 else
             f'All views share the **time** timeline: anchor plus {scene["horizon"]} future frames at 20 Hz.', '',
             '**Colors:** orange object; gray hand; blue/green fingers; red cross is the goal. Red rings mark 20 mm.', '',
             '| Objective | Held error (mm) | Overlap (mm) | Model | Replay |',
             '|---|---:|---:|---|---|']
    for result in data['results'].values():
        audit = result['audit']; metric = audit['model_metrics']
        lines.append(f'| {result["label"]} | {metric["held_worst_goal_mm"]:.1f} | {metric["peak_overlap_mm"]:.2f} | '+
                     ('Pass' if metric['geometry_pass'] else 'Fail')+' | '+('Pass' if audit['qualified_task_pass'] else 'Fail')+' |')
    lines += ['', 'Held error is the worst position error over the final six frames.', '',
              'The original is the actual saved unconditional draw, identical across all four objectives. '
              'No ground-truth future trajectory is displayed.', '',
              'Replay executes the generated gripper in an independent, approximate simulator; it is not '
              'source-simulator or real-robot validation. Finger positions are not clamped for display.', '',
              '**All four variants include the goal.** Scroll the visible **Terms explained** pane for bullet-point '
              'calculations, weights, tolerances and the noise regularizer.', '',
              '**Optimization terms**', '',
              '- Goal only: destination (and orientation for placement).',
              '- Geometry/joints: add penetration, table, finger-limit and speed costs.',
              '- Goal + geometry/joints + contact: add finger proximity; the goal remains active.',
              '- Full: add unsupported rise and slip (plus rest/release/settling for placement).', '',
              'Every objective keeps the noise regularizer and RMS cap. This is one scene and one seed.', '']
    for result in data['results'].values():
        residuals = result['audit']['model_metrics']['validation']['residuals_m']
        failures = ', '.join(k for k, v in residuals.items() if v > 0) or 'none'
        lines += [f'**{result["label"]} failed model checks:** {failures}. '
                  f'Replay peak finger overtravel: {result["replay_finger_limit_peak_mm"]:.3f} mm.', '']
    return '\n'.join(lines)


def export(data, output):
    import rerun as rr
    import rerun.blueprint as rrb

    terms_path = output.with_suffix('.terms.md')
    if output.exists() or output.with_suffix('.json').exists() or terms_path.exists():
        raise FileExistsError(f'Choose a new export path; existing recordings are preserved: {output}')
    explanations = term_explanations(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    goal = np.asarray(data['scene']['target'], dtype=np.float32)
    clouds = [t['points'].reshape(-1, 3) for t in data['tracks'].values()]
    cloud = np.concatenate([*clouds, goal[None]])
    lo, hi = cloud.min(0), cloud.max(0)
    center = (lo+hi)/2; span = max(float((hi-lo).max())*1.15, .3)
    eye = rrb.EyeControls3D(position=center+span*np.array([.55, -1.15, .7]), look_target=center, eye_up=[0, 0, 1])

    def view(key, label):
        return rrb.Spatial3DView(origin=f'/world/{key}', name=label, eye_controls=eye,
                                 background=[251, 250, 247], line_grid=False)

    tabs = [rrb.Grid(*(view(f'{a}/guided', data['results'][a]['label']) for a in data['arms']),
                     grid_columns=2, name='Guided comparison'),
            rrb.Grid(*(view(f'{a}/replay', data['results'][a]['label']+' — replay') for a in data['arms']),
                     grid_columns=2, name='Replay comparison'),
            rrb.TextDocumentView(origin='/readme', name='Results and legend')]
    for arm in data['arms']:
        tabs.append(rrb.Horizontal(view('original', 'Original unconditional sample'),
                                   view(f'{arm}/guided', 'Guided prediction'),
                                   view(f'{arm}/replay', 'Simulator replay'), name=data['results'][arm]['label']))
    blueprint = rrb.Blueprint(rrb.Horizontal(rrb.TextDocumentView(origin='/terms', name='Terms explained'),
                                            rrb.Tabs(*tabs, active_tab=0), column_shares=[.37, .63]),
                              rrb.TimePanel(timeline='time', playback_speed=1, play_state='paused'),
                              rrb.SelectionPanel(expanded=False), rrb.BlueprintPanel(expanded=False),
                              auto_layout=False, auto_views=False)
    # Default-only blueprints leave previously saved viewer layouts active.
    application_id = f'guidance_ablation_terms_visible_v3_{data["scene"]["name"]}'
    record = rr.RecordingStream(application_id, recording_id=uuid.uuid4())
    record.save(output)
    record.send_blueprint(blueprint, make_active=True, make_default=True)
    record.log('/readme', rr.TextDocument(summary(data), media_type='text/markdown'), static=True)
    record.log('/terms', rr.TextDocument(explanations, media_type='text/markdown'), static=True)
    record.log('/world', rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    theta = np.linspace(0, 2*np.pi, 65)
    cross, rings = [], []
    for axis in range(3):
        delta = np.eye(3)[axis]*.012
        cross.append([goal-delta, goal+delta])
        circle = np.zeros((len(theta), 3)); others = [i for i in range(3) if i != axis]
        circle[:, others[0]] = .02*np.cos(theta); circle[:, others[1]] = .02*np.sin(theta)
        rings.append(circle+goal)
    edge = np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5],[-.5,-.5]])*span+center[:2]
    floor = np.column_stack([edge, np.full(5, data['support_z'])])
    for key in data['tracks']:
        path = '/world/'+key
        record.log(path+'/goal', rr.LineStrips3D(cross, colors=[191,23,77], radii=rr.Radius.ui_points(2)), static=True)
        record.log(path+'/goal_tolerance', rr.LineStrips3D(rings, colors=[191,23,77,110], radii=rr.Radius.ui_points(.6)), static=True)
        record.log(path+'/table', rr.LineStrips3D([floor], colors=[199,195,188], radii=rr.Radius.ui_points(.7)), static=True)
    for frame in range(data['scene']['horizon']+1):
        record.set_time('frame', sequence=frame)
        record.set_time('time', duration=frame*.05)
        for key, track in data['tracks'].items():
            for body, (name, color) in enumerate(zip(BODY_NAMES, COLORS)):
                record.log(f'/world/{key}/{name}', rr.Points3D(track['points'][frame, data['body_ids'] == body],
                                                              colors=color, radii=rr.Radius.ui_points(2)))
            record.log(f'/metrics/{key}/goal_distance_mm', rr.Scalars(float(np.linalg.norm(track['T'][frame, 0]-goal)*1000)))
    record.flush(); record.disconnect()
    terms_path.write_text(explanations)
    receipt = dict(created_utc=datetime.now(timezone.utc).isoformat(), scene=data['scene']['name'],
                   rerun_version=rr.__version__, recording=str(output), recording_sha256=sha(output),
                   size_bytes=output.stat().st_size, frames=data['scene']['horizon']+1, timestep_s=.05,
                   point_count=len(data['body_ids']), tracks=list(data['tracks']), selected=data['selected'],
                   source_sha256=data['sources'], exporter_sha256=sha(Path(__file__)),
                   terms_document=str(terms_path), terms_sha256=sha(terms_path),
                   application_id=application_id, blueprint_make_active=True,
                   terms_view='Persistent left pane, visible alongside all trajectory tabs',
                   data='All saved particle points; replay uses the same templates at saved simulator poses; no joint clamping')
    output.with_suffix('.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k: receipt[k] for k in ['recording', 'size_bytes', 'frames', 'point_count', 'tracks']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    export(load_comparison(args.root.resolve(), args.scene), args.out.resolve())


if __name__ == '__main__':
    main()
