"""Compare fixed-noise predictions under different goals; write Markdown and movies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


COLORS = dict(left='#0072b2', right='#d55e00', near='#009e73', far='#cc0077')
LABELS = dict(left='Left (-x)', right='Right (+x)', near='Near (-y)', far='Far (+y)')


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(text)
    temporary.replace(path)


def read_runs(root, manifest, scene):
    completed = []
    first = None
    for goal in scene['goals']:
        directory = root / goal['run']
        result = directory / f'seed_{manifest["seed"]}.json'
        # results.json is written only after all arrays and the per-seed record.
        if not (directory / 'results.json').exists():
            continue
        record = json.loads(result.read_text())
        protocol = json.loads((directory / 'protocol.json').read_text())
        with np.load(directory / f'seed_{manifest["seed"]}.npz') as archive:
            data = {key: archive[key] for key in archive.files}
        expected = np.asarray(goal['target'], dtype=data['target'].dtype)
        np.testing.assert_array_equal(data['target'][0], expected)
        assert protocol['checkpoint_sha256'] == manifest['checkpoint_sha256']
        assert protocol['arguments']['seeds'] == str(manifest['seed'])
        assert protocol['arguments']['input'] == scene['input']
        assert protocol['arguments']['objective'] == 'physical'
        assert protocol['arguments']['iterations'] == manifest['iterations']
        assert protocol['goal_kind'] == 'point'
        assert protocol['arguments']['hold_frames'] == 6
        assert protocol['target_rotation'] == scene['target_rotation']
        if scene['task'] == 'pick':
            assert protocol['arguments']['pick_goal_hold']
        if first is None:
            first = data
        else:
            for key in ['initial', 'initial_T', 'particle_body', 'context_len']:
                np.testing.assert_array_equal(data[key], first[key])
        C = int(data['context_len'])
        original = data['initial_T'][0, C-1:, 0]
        guided = data['T'][0, C-1:, 0]
        distance = np.linalg.norm(guided - expected, axis=-1) * 1000
        assert abs(distance[-1] - record['selected']['target_distance_mm']) < .001
        final_distance = float(distance[-1])
        hold_distance = float(distance[-6:].max())
        violations = {k: v for k, v in record['validation']['residuals_m'].items() if v > 0}
        metrics = dict(direction=goal['name'], run=goal['run'],
                       initial_goal_mm=float(np.linalg.norm(original[-1]-expected)*1000),
                       final_goal_mm=final_distance, held_max_goal_mm=hold_distance,
                       moved_from_ordinary_mm=float(np.linalg.norm(guided[-1]-original[-1])*1000),
                       goal_held_pass=hold_distance <= manifest['goal_distance_tolerance_mm'],
                       mesh_peak_overlap_mm=record['selected']['mesh_penetration_max_mm'],
                       full_geometry_pass=record['validation']['accepted'],
                       violations=violations, selected_update=record['selected_iteration'],
                       seconds=record['seconds'], source_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
        completed.append(dict(goal=goal, data=data, record=record, metrics=metrics))
    return completed


def trajectory_plot(scene, completed, destination):
    previous = Path(scene['previous_run']) / 'seed_8311.npz'
    with np.load(previous) as old:
        C = int(old['context_len'])
        ordinary = old['initial_T'][0, C-1:, 0].copy()
    if completed:
        data = completed[0]['data']
        ordinary = data['initial_T'][0, int(data['context_len'])-1:, 0]
    anchor = np.asarray(scene['anchor'])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, coordinates, xlabel, ylabel in zip(axes, [(0, 1), (0, 2)],
                                               ['x relative to start (cm)']*2,
                                               ['y relative to start (cm)', 'z relative to start (cm)']):
        a, b = coordinates
        base = (ordinary-anchor)*100
        axis.plot(base[:, a], base[:, b], '--', c='#666666', lw=2, label='Ordinary sample')
        axis.scatter(base[-1, a], base[-1, b], c='#666666', s=55, marker='s')
        axis.scatter([0], [0], c='black', marker='o', s=30, label='Observed start')
        for goal in scene['goals']:
            target = (np.asarray(goal['target'])-anchor)*100
            axis.scatter(target[a], target[b], marker='X', s=125, color=COLORS[goal['name']],
                         label=LABELS[goal['name']]+' target')
            if coordinates == (0, 1):
                axis.annotate(LABELS[goal['name']], (target[a], target[b]),
                              xytext=(5, 8), textcoords='offset points', fontsize=9)
        for item in completed:
            data = item['data']; C = int(data['context_len'])
            track = (data['T'][0, C-1:, 0]-anchor)*100
            color = COLORS[item['goal']['name']]
            axis.plot(track[:, a], track[:, b], c=color, lw=2)
            axis.scatter(track[-1, a], track[-1, b], color=color, s=50, marker='o')
        axis.set(xlabel=xlabel, ylabel=ylabel, aspect='equal')
        axis.margins(.2); axis.grid(alpha=.2)
    axes[0].set_title('Top view: does the object move toward a different target?')
    axes[1].set_title('Side view: height and lateral motion')
    axes[1].legend(fontsize=8, loc='best')
    fig.suptitle(f'{scene["name"]} · same noise seed 8311 · {len(completed)}/4 goals completed\n'
                 'X = requested destination; circle = guided endpoint; square = ordinary endpoint. No GT trajectory.')
    fig.tight_layout(); fig.savefig(destination, dpi=150); plt.close(fig)


def movie(scene, completed, destination):
    first = completed[0]['data']; C = int(first['context_len'])
    body = first['particle_body'][0]
    subset = np.concatenate([np.where(body == k)[0][::max(1, int((body == k).sum())//180)]
                             for k in np.unique(body)])
    point_colors = np.array(['#e69f00', '#777777', '#0072b2', '#009e73'])[body[subset]]
    anchor = np.asarray(scene['anchor'])
    # Basic indexing first avoids NumPy advanced-axis reordering.
    ordinary = (first['initial'][0, C-1:][:, subset]-anchor)*100
    base_t = (first['initial_T'][0, C-1:, 0]-anchor)*100
    items = {item['goal']['name']: item for item in completed}
    panels = [dict(label='Ordinary sample', points=ordinary, center=base_t, goal=None)]
    for goal in scene['goals']:
        item = items.get(goal['name'])
        panels.append(dict(label=LABELS[goal['name']], goal=(np.asarray(goal['target'])-anchor)*100,
                           points=None if item is None else (item['data']['positions'][0, C-1:][:, subset]-anchor)*100,
                           center=None if item is None else (item['data']['T'][0, C-1:, 0]-anchor)*100))
    all_points = np.concatenate([p['points'].reshape(-1, 3) for p in panels if p['points'] is not None]
                                + [np.asarray([p['goal'] for p in panels if p['goal'] is not None])])
    low, high = all_points.min(0), all_points.max(0)
    center = (low+high)/2; span = max(np.ptp(all_points, axis=0).max()*1.15, 40.)
    fig = plt.figure(figsize=(17, 7.4)); artists = []; texts = []
    for index, panel in enumerate(panels):
        for row in [0, 1]:
            axis = fig.add_subplot(2, 5, row*5+index+1, projection='3d' if row == 0 else None)
            if row == 0:
                axis.set(xlim=(center[0]-span/2, center[0]+span/2), ylim=(center[1]-span/2, center[1]+span/2),
                         zlim=(center[2]-span/2, center[2]+span/2), title=panel['label'],
                         xlabel='x (cm)', ylabel='y (cm)', zlabel='z (cm)')
                axis.set_box_aspect((1, 1, 1)); axis.view_init(24, -65)
                if panel['goal'] is not None:
                    axis.scatter(*panel['goal'], marker='X', c='crimson', s=90)
                axis.plot(*base_t.T, '--', c='#aaaaaa', lw=1)
            else:
                axis.set(xlim=(center[0]-span/2, center[0]+span/2), ylim=(center[1]-span/2, center[1]+span/2),
                         xlabel='x (cm)', ylabel='y (cm)', aspect='equal')
                axis.plot(base_t[:, 0], base_t[:, 1], '--', c='#aaaaaa', lw=1)
                if panel['goal'] is not None:
                    axis.scatter(*panel['goal'][:2], marker='X', c='crimson', s=90)
                axis.grid(alpha=.15)
            if panel['points'] is None:
                if row == 0: axis.text2D(.25, .45, 'Not completed', transform=axis.transAxes)
                else: axis.text(.2, .5, 'Not completed', transform=axis.transAxes)
                continue
            points = panel['points'][0]
            artist = axis.scatter(*points.T if row == 0 else points[:, :2].T, c=point_colors, s=5, alpha=.8)
            artists.append((artist, panel, row))
            if row == 1:
                texts.append((axis.text(.02, 1.04, '', transform=axis.transAxes, fontsize=10), panel))
    title = fig.suptitle('')
    fig.subplots_adjust(left=.035, right=.99, bottom=.09, top=.84, wspace=.22, hspace=.30)
    def update(frame):
        for artist, panel, row in artists:
            points = panel['points'][frame]
            if row == 0: artist._offsets3d = tuple(points.T)
            else: artist.set_offsets(points[:, :2])
        for label, panel in texts:
            if panel['goal'] is None: label.set_text('Same ordinary sample for all goals')
            else:
                error = np.linalg.norm(panel['center'][frame]-panel['goal'])*10
                change = np.linalg.norm(panel['center'][frame]-base_t[frame])*10
                label.set_text(f'Goal: {error:.0f} mm | shift: {change:.0f} mm')
        title.set_text(f'{scene["name"]} · {frame*.05:.2f} s · four distinct goals, fixed seed 8311\n'
                       'Top: 3D. Bottom: top view. Orange: object. Red X: goal. Gray dashed line: ordinary object path.\n'
                       'Model predictions only; physical execution has not been tested for these goals.')
        return [a for a, _, _ in artists]+[t for t, _ in texts]+[title]
    FuncAnimation(fig, update, frames=len(ordinary), interval=100).save(destination, writer=PillowWriter(fps=10))
    update(len(ordinary)-1); fig.savefig(destination.with_suffix('.png'), dpi=140); plt.close(fig)


def render(root):
    manifest = json.loads((root/'manifest.json').read_text())
    output = root/'figures'; output.mkdir(exist_ok=True)
    results = []; sections = []; signature = []
    for scene in manifest['scenes']:
        completed = read_runs(root, manifest, scene)
        signature.append([scene['name'], [r['metrics']['source_sha256'] for r in completed]])
        tag = f'{scene["name"]}_{len(completed):02d}_complete'
        plot = output/f'{tag}_paths.png'
        if not plot.exists(): trajectory_plot(scene, completed, plot)
        text = f'## {scene["name"]}\n\n'
        text += ('Goal: lift 15 cm and move 15 cm in the requested horizontal direction; hold the requested point for the final six frames.\n\n'
                 if scene['task'] == 'pick' else
                 'Goal: place 15 cm left/right/near/far of the original requested position, preserving its requested orientation, then settle and release.\n\n')
        text += f'![Requested destinations and generated paths](figures/{plot.name})\n\n'
        if completed:
            mov = output/f'{tag}.gif'
            if not mov.exists(): movie(scene, completed, mov)
            text += f'![Ordinary sample and all completed guided predictions](figures/{mov.name})\n\n'
        text += '| Goal | Ordinary error | Guided error | Shift from ordinary | Worst held error | Peak overlap | Full geometry |\n'
        text += '|---|---:|---:|---:|---:|---:|---|\n'
        by_name = {r['goal']['name']: r for r in completed}
        for goal in scene['goals']:
            item = by_name.get(goal['name'])
            if item is None:
                text += f'| {LABELS[goal["name"]]} | {goal["ordinary_endpoint_distance_mm"]:.1f} mm* | Pending | Pending | Pending | Pending | Pending |\n'
                continue
            m = item['metrics']
            text += (f'| {LABELS[goal["name"]]} | {m["initial_goal_mm"]:.1f} mm | {m["final_goal_mm"]:.1f} mm | '
                     f'{m["moved_from_ordinary_mm"]:.1f} mm | {m["held_max_goal_mm"]:.1f} mm | '
                     f'{m["mesh_peak_overlap_mm"]:.2f} mm | {"PASS" if m["full_geometry_pass"] else "FAIL: "+", ".join(m["violations"])} |\n')
        text += '\n*Pending rows use the existing saved ordinary sample for target preparation. Completed rows use their newly generated ordinary sample.\n\n'
        separations = {}
        for a, b in [('left', 'right'), ('near', 'far')]:
            if a in by_name and b in by_name:
                A = by_name[a]['data']['T'][0, -1, 0]; B = by_name[b]['data']['T'][0, -1, 0]
                separations[f'{a}_vs_{b}_mm'] = float(np.linalg.norm(A-B)*1000)
                text += f'**{a.title()} versus {b}: guided endpoints are {separations[f"{a}_vs_{b}_mm"]:.1f} mm apart.** Their targets are 300 mm apart.\n\n'
        text += ('Completed goals share bitwise-identical initial particle trajectories and pose translations.\n\n'
                 if len(completed) > 1 else 'Initial-draw identity across goals will be checked as their results complete.\n\n')
        sections.append(text)
        results.append(dict(scene=scene['name'], completed=len(completed), initial_pairing_checked=len(completed)>1,
                            endpoint_separations=separations, goals=[i['metrics'] for i in completed]))
    statuses = {p.stem: json.loads(p.read_text()) for p in root.glob('packet_gpu*_status.json')}
    total = sum(r['completed'] for r in results)
    payload = dict(updated_utc=datetime.now(timezone.utc).isoformat(), completed=total, expected=8,
                   results=results, workers=statuses, signature=signature,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_text(root/'summary.json', json.dumps(payload, indent=2)+'\n')
    header = ('# Same ordinary sample, different guidance goals\n\n'
              f'**{total}/8 goal experiments complete.** Two fixed scenes, four declared destinations each, '
              'the same seed within a scene, 24 initial-noise updates, 36 predicted frames.\n\n'
              'All targets start more than 12 cm from the previously saved ordinary endpoint. Opposite targets '
              'are 30 cm apart. A different target does not guarantee a successful generated motion; every result stays visible.\n\n'
              'Left/right mean world −x/+x; near/far mean −y/+y in the top view. '
              'A circle marks a guided endpoint, a square the ordinary endpoint, and an X a requested destination. '
              'The plots compare generated predictions directly; they do not show GT as another guidance result.\n\n'
              '**Read three things separately:** distance to the requested goal, movement away from the ordinary '
              'sample, and physical plausibility. The held goal tolerance is 20 mm in 3D. Passing geometric '
              'checks does not establish an executable grasp. These new goals have not been replayed in physics.\n\n'
              'Physical losses and checkpoint are frozen across directions. Future actions are hidden. '
              'The bottle uses a held point goal; placing also requests the original orientation and release. '
              'This is an exploratory steering diagnostic on previously inspected scenes, not a new holdout benchmark.\n\n'
              '[Exact targets and protocol](manifest.json) · [Live measurements and worker status](summary.json)\n\n')
    atomic_text(root/'RESULTS.md', header+''.join(sections))
    terminal = len(statuses) == 2 and all(v['state'] == 'completed' or v['state'].startswith('stopped:') for v in statuses.values())
    print(json.dumps(dict(updated_utc=payload['updated_utc'], completed=total, terminal=terminal)), flush=True)
    return terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', default=45., type=float)
    args = parser.parse_args()
    root = args.root.resolve()
    while True:
        terminal = render(root)
        if not args.watch or terminal: break
        deadline = max(datetime.fromisoformat(json.loads(p.read_text())['deadline_utc']).timestamp()
                       for p in root.glob('packet_gpu[01].json'))
        if time.time() > deadline+300: break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
