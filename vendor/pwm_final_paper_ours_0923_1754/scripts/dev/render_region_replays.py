"""Render declared region goals against both independent physics integration steps."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'region_replay_report_v1'
    out.mkdir(exist_ok=True)
    scenes = {s['name']: s for s in json.loads((root / 'scenes_v2.json').read_text())['scenes']}
    audits = json.loads((root / 'region_replay_summary_v1.json').read_text())['results']
    sections = []
    for audit in audits:
        if audit['variant'] != 'selected':
            continue
        run, seed = audit['run'], audit['seed']
        directory = root / 'generated_replay_v1' / f'{run}_seed{seed}_selected'
        scene = Path(scenes[audit['scene']]['directory'])
        batch = torch.load(scene / 'input.pt', weights_only=False, map_location='cpu')
        geometry = torch.load(scene / 'collision_geometry.pt', weights_only=False, map_location='cpu')
        arrays = [np.load(directory / step / 'execution.npz') for step in ('dt0005', 'dt00025')]
        poses = [(arrays[0]['desired_R'], arrays[0]['desired_T'])] + [(a['model_R'], a['model_T']) for a in arrays]
        body = batch.particle_body[0].numpy()
        select = np.concatenate([np.where(body == k)[0][::max(1, int((body == k).sum()) // 100)] for k in range(4)])
        colors = np.array(['#e69f00', '#777777', '#0072b2', '#009e73'])[body[select]]
        tracks = []
        for rotation, translation in poses:
            points = np.empty((len(translation), len(body), 3))
            for k in range(4):
                template = batch.body_templates[0, k, batch.body_template_mask[0, k]].numpy()
                points[:, body == k] = np.einsum('tij,nj->tni', rotation[:, k], template) + translation[:, k, None]
            tracks.append(points[:, select])
        lower, upper = [np.array([float(v) for v in audit['goal_region_bounds'][k].split(',')]) for k in ('lower', 'upper')]
        # This report visualizes the declared pickup workspace: bounded XY, minimum Z.
        assert np.isfinite(lower).all() and np.isfinite(upper[:2]).all() and np.isposinf(upper[2])
        rectangle = np.array([[lower[0], lower[1], lower[2]], [upper[0], lower[1], lower[2]],
                              [upper[0], upper[1], lower[2]], [lower[0], upper[1], lower[2]],
                              [lower[0], lower[1], lower[2]]])
        all_points = np.concatenate([t.reshape(-1, 3) for t in tracks] + [rectangle])
        low, high = all_points.min(0), all_points.max(0)
        center, span = (low + high) / 2, max((high - low).max() * 1.12, .25)
        tag = f'{run}_seed{seed}'
        movie, still, chart = [out / (tag + suffix) for suffix in ('.gif', '_final.png', '_traces.png')]
        titles = ['Model prediction', 'Physics replay · 0.5 ms', 'Physics replay · 0.25 ms']
        origin_z = poses[0][1][0, 0, 2]
        if not movie.exists():
            fig = plt.figure(figsize=(13, 5.3)); artists, texts = [], []
            for index, track in enumerate(tracks):
                ax = fig.add_subplot(1, 3, index + 1, projection='3d')
                artists.append(ax.scatter(*track[0].T, c=colors, s=9))
                ax.plot(*rectangle.T, color='#cc0077', lw=2)
                for corner in rectangle[:4]:
                    ax.plot([corner[0]] * 2, [corner[1]] * 2, [corner[2], center[2] + span / 2], ':', color='#cc0077', alpha=.4)
                ax.set(xlim=(center[0]-span/2, center[0]+span/2), ylim=(center[1]-span/2, center[1]+span/2),
                       zlim=(center[2]-span/2, center[2]+span/2), xticks=[], yticks=[], zlabel='height (m)', title=titles[index])
                ax.set_box_aspect((1, 1, 1)); ax.view_init(22, -65)
                xx, yy = np.meshgrid(center[0]+np.array([-.5, .5])*span, center[1]+np.array([-.5, .5])*span)
                ax.plot_surface(xx, yy, np.full_like(xx, geometry['metadata']['support_z']), color='#999999', alpha=.12)
                texts.append(ax.text2D(.08, -.02, '', transform=ax.transAxes, fontsize=10))
            title = fig.suptitle(''); fig.subplots_adjust(left=.01, right=.97, top=.77, bottom=.12, wspace=.08)
            def update(frame):
                for artist, track, text, (_, translation) in zip(artists, tracks, texts, poses):
                    artist._offsets3d = tuple(track[frame].T)
                    text.set_text(f'Object lift: {(translation[frame, 0, 2]-origin_z)*1000:.1f} mm')
                title.set_text(f'{audit["scene"]} · seed {seed} · {frame*.05:.2f}s\nGoal: lift ≥100 mm and stay inside the pink XY workspace. Orange: object. Blue/green: fingers.')
                return artists + texts + [title]
            FuncAnimation(fig, update, frames=len(tracks[0]), interval=85).save(movie, writer=PillowWriter(fps=12))
            update(len(tracks[0])-1); fig.savefig(still, dpi=130); plt.close(fig)
        if not chart.exists():
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.7)); time = np.arange(len(tracks[0])) * .05
            for (_, translation), label, color in zip(poses, titles, ['#222222', '#0072b2', '#d55e00']):
                object_t = translation[:, 0]
                axes[0].plot(time, (object_t[:, 2]-origin_z)*1000, label=label, color=color)
                signed = np.maximum(lower-object_t, object_t-upper).max(-1)*1000
                axes[1].plot(time, signed, color=color)
                axes[2].plot(time, np.linalg.norm(object_t-poses[0][1][:, 0], axis=-1)*1000, color=color)
            axes[0].axhline(100, color='#cc0077', ls='--'); axes[0].legend(fontsize=8)
            axes[1].axhline(0, color='#cc0077', ls='--')
            hold = audit['goal_region_bounds']['hold_frames']
            axes[1].axvspan(time[-hold], time[-1], color='#009e73', alpha=.12)
            for ax, title in zip(axes, ['Object height change', 'Region violation (≤0 inside)', 'Divergence from model prediction']):
                ax.set(title=title, xlabel='seconds', ylabel='mm'); ax.grid(alpha=.2)
            fig.tight_layout(); fig.savefig(chart, dpi=130); plt.close(fig)
        model = json.loads((root / run / f'seed_{seed}.json').read_text())
        passed = model['validation']['accepted']
        rows = ''.join(f'<tr><td>{v["step"]}</td><td>{v["result"]["object_lift_mm"]:.1f}</td><td>{v["region_held_signed_residual_mm"]:.1f}</td><td>{"PASS" if v["validation"]["accepted"] else "FAIL"}</td></tr>' for v in audit['replays'])
        sections.append(f'''<details {"open" if not sections else ""}><summary>{html.escape(audit['scene'])} · {html.escape(audit['arm'])}</summary>
<p>Model region + geometry screen: <b>{"PASS" if passed else "FAIL"}</b>. Qualified physics replay: <b>{"PASS" if audit['qualified_task_pass'] else "FAIL"}</b>.
Recorded control qualification: {audit['control_qualified']}; timestep agreement: {audit['timestep_stable']} ({audit['timestep_position_max_mm']:.1f} mm / {audit['timestep_rotation_max_deg']:.1f}° maximum difference).</p>
<img loading="lazy" src="{movie.name}"><table><tr><th>Physics step</th><th>Final lift (mm)</th><th>Worst held region violation (mm; ≤0 inside)</th><th>Region + geometry screen</th></tr>{rows}</table>
<img loading="lazy" src="{chart.name}"></details>''')
    header = '''<!doctype html><meta charset="utf-8"><title>Lift workspace: model versus physics</title><style>body{font:16px system-ui;margin:30px;max-width:1450px;line-height:1.5}img{max-width:100%}summary{cursor:pointer;font-weight:600;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style>
<h1>Does a plausible-looking lift actually move the object?</h1><p>The model predicts every body's motion on the left. The other panels command only the gripper; an independent physics engine moves the object. Compare the orange object across the panels.</p>
<p>This exploratory task asks for at least 100 mm lift inside an XY workspace extending 100 mm from the observed object origin. It does not ask the object to stop exactly 100 mm higher. The original precise-target experiment remains separate. The last six frames must satisfy the region and all existing contact, overlap, joint and support checks.</p>
<p>The physics diagnostic uses a floating hand, approximate collision geometry/inertia and fixed contact preload. It is not source-PhysX or robot execution. A replay pass requires recorded-control qualification, hand tracking, task checks at both integration steps and agreement between those steps. An attractive model prediction alone is insufficient.</p>
<p><a href="../region_replay_summary_v1.json">All region replay measurements</a> · <a href="../index.html">Main report</a></p>'''
    (out / 'index.html').write_text(header + ''.join(sections))
    print(f'Rendered {len(sections)} region replay comparisons')


if __name__ == '__main__':
    main()
