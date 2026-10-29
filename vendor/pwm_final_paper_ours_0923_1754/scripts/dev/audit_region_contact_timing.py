"""Compare predicted and executed proximity without changing a saved rollout."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from guidance.collision_distance import ConvexCollisionDistance


def contact_summary(gaps, translations, tolerance):
    any_near = np.flatnonzero((gaps <= tolerance).any(-1))
    both_near = np.flatnonzero((gaps[:, 1:] <= tolerance).all(-1))
    first = int(any_near[0]) if len(any_near) else len(gaps)
    lift = (translations[:, 0, 2] - translations[0, 0, 2]) * 1000
    return dict(first_any_link_near_frame=int(any_near[0]) if len(any_near) else None,
                first_both_fingers_near_frame=int(both_near[0]) if len(both_near) else None,
                both_fingers_near_frames=both_near.tolist(),
                max_rise_before_any_link_near_mm=float(max(0, lift[:max(first, 1)].max())),
                lift_at_first_both_near_mm=float(lift[both_near[0]]) if len(both_near) else None,
                final_lift_mm=float(lift[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(exist_ok=False)
    scenes = {s['name']: Path(s['directory']) for s in json.loads((root/'scenes_v2.json').read_text())['scenes']}
    audits = json.loads((root/'region_replay_summary_v1.json').read_text())['results']
    rows, sections = [], []
    for audit in audits:
        if audit['variant'] != 'selected':
            continue
        run, seed = audit['run'], audit['seed']
        protocol_path = root/run/'protocol.json'
        options = json.loads(protocol_path.read_text())['arguments']
        tolerance = options['mesh_contact_tolerance_mm']/1000
        directory = scenes[audit['scene']]
        geometry = torch.load(directory/'collision_geometry.pt', weights_only=False, map_location='cpu')
        query = ConvexCollisionDistance(geometry, vendor_paths=[root/'vendor_fcl', root.parent/'overnight_2026-09-14/vendor'])
        replay_paths = [root/'generated_replay_v1'/f'{run}_seed{seed}_selected'/step/'execution.npz'
                        for step in ('dt0005', 'dt00025')]
        coarse, fine = [np.load(p) for p in replay_paths]
        for key in ('desired_R', 'desired_T', 'requested_q', 'commanded_q', 'planned_contact'):
            np.testing.assert_array_equal(coarse[key], fine[key])
        tracks = [(coarse['desired_R'], coarse['desired_T']),
                  (coarse['model_R'], coarse['model_T']), (fine['model_R'], fine['model_T'])]
        gaps = [np.array([query.query(R, T)[0] for R, T in zip(rr, tt)]) for rr, tt in tracks]
        # Independently reproduce the controller's declared proximity gate.
        np.testing.assert_array_equal((gaps[0][:, 1:] <= tolerance).all(-1), coarse['planned_contact'])
        records = [contact_summary(gap, T, tolerance) for gap, (_, T) in zip(gaps, tracks)]
        np.savez_compressed(out/(run+'.npz'), gaps=np.stack(gaps),
                            translations=np.stack([T for _, T in tracks]),
                            requested_q=coarse['requested_q'], commanded_q=coarse['commanded_q'],
                            achieved_q=np.stack([coarse['joint_q'], fine['joint_q']]))
        labels = ['Model prediction', 'Execution · 0.5 ms', 'Execution · 0.25 ms']
        colors = ['#333333', '#0072b2', '#d55e00']
        times = np.arange(len(gaps[0])) * float(coarse['dt'])
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for (R, T), gap, label, color in zip(tracks, gaps, labels, colors):
            axes[0, 0].plot(times, (T[:, 0, 2]-T[0, 0, 2])*1000, color=color, label=label)
            axes[0, 1].plot(times, gap[:, 1:].max(-1)*1000, color=color, label=label)
            axes[1, 0].plot(times, gap.min(-1)*1000, color=color, label=label)
        axes[0, 0].axhline(100, color='#cc0077', ls=':', label='Required lift')
        axes[0, 0].set(title='Does the object follow the predicted lift?', ylabel='Height change (mm)')
        axes[0, 1].set(title='Farther of the two finger surfaces', ylabel='Signed mesh gap (mm)')
        axes[1, 0].set(title='Nearest hand or finger surface', ylabel='Signed mesh gap (mm)')
        for ax in (axes[0, 1], axes[1, 0]):
            ax.axhline(tolerance*1000, color='#cc0077', ls=':', label='Declared proximity tolerance')
            ax.axhline(0, color='#777777', lw=.7)
        for finger, style in enumerate(['-', '--']):
            axes[1, 1].plot(times, coarse['requested_q'][:, finger]*1000, style, color='#333333', label=f'Predicted q{finger}')
            axes[1, 1].plot(times, coarse['commanded_q'][:, finger]*1000, style, color='#999999', label=f'Motor target q{finger}')
            axes[1, 1].plot(times, coarse['joint_q'][:, finger]*1000, style, color='#0072b2', label=f'Achieved q{finger} · 0.5 ms')
        axes[1, 1].set(title='Prediction, preload command and achieved opening', ylabel='Joint position (mm)')
        for ax in axes.flat:
            ax.set_xlabel('Rollout time (s)'); ax.grid(alpha=.2); ax.legend(fontsize=7)
        model = json.loads((root/run/f'seed_{seed}.json').read_text())
        fig.suptitle(f'{audit["scene"]} · seed {seed} · selected region24 plan\n'
                     f'Model task screen: {model["validation"]["accepted"]}; '
                     f'qualified execution: {audit["qualified_task_pass"]}. Proximity is not contact force.')
        fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(out/(run+'.png'), dpi=140); plt.close(fig)
        row = dict(run=run, scene=audit['scene'], seed=seed, model_task_pass=model['validation']['accepted'],
                   qualified_execution_pass=audit['qualified_task_pass'], control_qualified=audit['control_qualified'],
                   timestep_stable=audit['timestep_stable'], proximity_tolerance_mm=tolerance*1000,
                   predicted=records[0], executed=records[1:], controller_gate_bitwise=True,
                   sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            [protocol_path, directory/'collision_geometry.pt', *replay_paths]})
        rows.append(row)
        table = '<table><tr><th>Trajectory</th><th>First any-link proximity</th><th>First two-finger proximity</th><th>Final lift</th></tr>'
        def frame_text(value):
            return 'never in saved frames' if value is None else f'{value} ({value*float(coarse["dt"]):.2f} s)'
        for label, record in zip(labels, records):
            table += (f'<tr><td>{label}</td><td>{frame_text(record["first_any_link_near_frame"])}</td>'
                      f'<td>{frame_text(record["first_both_fingers_near_frame"])}</td>'
                      f'<td>{record["final_lift_mm"]:.2f} mm</td></tr>')
        sections.append(f'<section><h2>{html.escape(audit["scene"])}</h2><p>Model task screen: '
                        f'{row["model_task_pass"]}; controller qualified: {row["control_qualified"]}; '
                        f'timestep stable: {row["timestep_stable"]}; complete execution pass: '
                        f'{row["qualified_execution_pass"]}.</p><img src="{run}.png">{table}</table></section>')
    report = dict(updated_utc=datetime.now(timezone.utc).isoformat(), results=rows,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='Read-only audit of every completed selected region plan. Both physics steps use bitwise identical requested commands. Sampled output-frame proximity does not measure forces or exclude contacts between frames. One-finger or palm contact may move an object; absence of two-finger proximity alone does not prove impossible motion.')
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    page = '''<!doctype html><meta charset="utf-8"><title>Predicted and executed contact timing</title>
<style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1300px}img{max-width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}section{margin-top:40px}</style>
<h1>Are the fingers close to the object in prediction and execution?</h1>
<p>The same gripper plan can approach the predicted object and miss the actual object. Compare the height and surface-gap curves together. The joint plot separates the model's achieved-opening prediction, the fixed preload motor command, and the resulting joint motion.</p>
<p>Each case uses a selected lift-workspace plan, seed8311, the same pinned unconditional checkpoint and both existing replay steps. The controller's proximity gate is independently reproduced bitwise. These are saved 50 ms output frames: proximity is not force, and contact between frames is not excluded. A single finger or the palm can move an object, so lack of two-finger proximity alone does not prove impossible motion.</p>
<p><a href="../region_replay_report_v1/index.html">Model versus execution movies</a> · <a href="report.json">All measurements and provenance</a> · <a href="../index.html">Full report</a></p>'''
    (out/'index.html').write_text(page+''.join(sections))
    print('Audited contact timing:', len(rows), 'selected region plans')


if __name__ == '__main__':
    main()
