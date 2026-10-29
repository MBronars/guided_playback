"""Plot saved optimizer prefixes against the campaign's unchanged task screen."""
import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def measurements(audit, final_goal):
    residuals = audit['validation']['residuals_m']
    values = dict(final_goal_mm=final_goal,
                  hold_goal_mm=(residuals['goal'] + .020) * 1000,
                  mesh_overlap_mm=audit['penetration_max_mm'],
                  gripper_table_mm=max(0., residuals['gripper_support'] + .003) * 1000,
                  accepted=audit['validation']['accepted'])
    if 'goal_rotation' in residuals:
        values['hold_orientation_deg'] = np.rad2deg(residuals['goal_rotation'] / .04) + 15
    # Catch unit/shape mistakes in this report using an independent relationship.
    if values['hold_goal_mm'] + .01 < final_goal:
        raise ValueError('Worst held goal distance cannot be smaller than final distance')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'convergence_report_v1'
    out.mkdir(exist_ok=True)
    source = root / 'prefix_mesh_summary_v3.json'
    prefixes = json.loads(source.read_text())['results']
    rows, sections = [], []
    long = [p for p in prefixes if p['arm'] in ('goal100_h36', 'physical100_h36')]
    for scene in sorted({p['scene'] for p in long}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        plot_specs = [('final_goal_mm', 'Last-frame distance to goal', 20),
                      ('hold_goal_mm', 'Worst distance over the last six frames', 20),
                      ('mesh_overlap_mm', 'Worst gripper–object overlap', 2),
                      ('gripper_table_mm', 'Worst gripper intrusion below the table', 3)]
        scene_rows = []
        for arm, label, color in [('goal100_h36', 'Goal objective', '#d55e00'),
                                  ('physical100_h36', 'Physical objective', '#0072b2')]:
            selected = sorted([p for p in long if p['scene'] == scene and p['arm'] == arm],
                              key=lambda p: p['updates'])
            if not selected:
                continue
            run, seed = selected[0]['run'], selected[0]['seed']
            directory = root / run
            audit_path = directory / f'mesh_audit_v2_{seed}.json'
            result_path = directory / f'seed_{seed}.json'
            series = []
            if audit_path.exists() and result_path.exists():
                audit = json.loads(audit_path.read_text())['initial']
                result = json.loads(result_path.read_text())
                series.append(dict(scene=scene, arm=arm, run=run, seed=seed, updates=0,
                                   selected_iteration=0,
                                   **measurements(audit, result['initial']['target_distance_mm'])))
            for p in selected:
                series.append(dict(scene=scene, arm=arm, run=run, seed=seed,
                                   updates=p['updates'], selected_iteration=p['selected_iteration'],
                                   **measurements(p, p['goal_error_mm'])))
            for ax, (key, _, _) in zip(axes, plot_specs):
                ax.plot([p['updates'] for p in series], [p[key] for p in series],
                        marker='o', color=color, label=label)
            scene_rows.extend(series)
        for ax, (key, title, threshold) in zip(axes, plot_specs):
            ax.axhline(threshold, color='#555555', ls='--', lw=1,
                       label=f'Criterion: ≤{threshold} mm')
            ax.set(title=title, xlabel='Optimization updates evaluated', ylabel='mm',
                   xticks=[0, 6, 24, 48, 100], ylim=(0, None))
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f'{scene} · seed8311 · H36 (1.8 seconds)\n'
                     'Each point is the candidate selected by its run’s declared objective.', fontsize=12)
        fig.tight_layout(rect=(0, .03, 1, .93))
        fig.text(.5, .015, 'Passing one panel does not pass the complete task screen. '
                 'These are geometric measurements, not executed success.', ha='center', fontsize=9)
        image = scene + '.png'
        fig.savefig(out / image, dpi=150)
        plt.close(fig)
        table = '<table><tr><th>Objective / updates / selected update</th><th>Final goal mm</th><th>Held goal mm</th><th>Overlap mm</th><th>Gripper below table mm</th><th>Held orientation ° (place)</th><th>Complete mesh screen</th></tr>'
        for p in scene_rows:
            orientation = f"{p['hold_orientation_deg']:.1f}" if 'hold_orientation_deg' in p else '—'
            table += (f'<tr><td>{html.escape(p["arm"])} / {p["updates"]} / {p["selected_iteration"]}</td>'
                      f'<td>{p["final_goal_mm"]:.2f}</td><td>{p["hold_goal_mm"]:.2f}</td>'
                      f'<td>{p["mesh_overlap_mm"]:.2f}</td><td>{p["gripper_table_mm"]:.2f}</td>'
                      f'<td>{orientation}</td><td>{"pass" if p["accepted"] else "fail"}</td></tr>')
        sections.append(f'<h2>{html.escape(scene)}</h2><img src="{image}">{table}</table>')
        rows.extend(scene_rows)
    header = '''<!doctype html><meta charset="utf-8"><title>More optimization: goal accuracy and physical checks</title>
<style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1400px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style>
<h1>Does more optimization improve the whole motion?</h1>
<p>The top left panel asks whether the object reaches the requested position in the final frame. The top right asks whether it stays near that position throughout the final six frames. The bottom panels check whether the gripper moves through the object or below the support plane. Lower is better; dashed lines show the unchanged screening thresholds.</p>
<p>These are the first development pick and place scenes, one fixed seed, the same existing EMA checkpoint and sampler, and up to 100 updates. Missing points are unfinished work. Six/24/48-update points are saved best-so-far candidates from each 100-update run; they are not independent restarts. Goal and physical objectives differ, so neither must improve every plotted measure monotonically. The pick physical cost targets the final frame; the place physical cost already averages its goal over six frames. Neither plot implies independent physics or robot success.</p>
<p><a href="../index.html">All trials and paired movies</a> · <a href="../generated_replay_report_v2/index.html">Independent physics replays</a> · <a href="summary.json">Exact measurements</a></p>'''
    payload = dict(results=rows, prefix_audit_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   checkpoint_sha256='99ec6cc6102a0d654c375c9f7b9cc9c86edd0964acf9de1acb42d4fb578c9fba',
                   protocol='Shared v2 mesh criteria, H36, seed8311, first development pick/place; no exclusions or changed thresholds.')
    (out / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n')
    temporary = out / 'index.pending'
    temporary.write_text(header + ''.join(sections))
    temporary.replace(out / 'index.html')
    print(f'Rendered {len(rows)} prefix measurements across {len(sections)} scenes')


if __name__ == '__main__':
    main()
