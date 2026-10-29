"""Compare each committed prediction with the segment that actually executed."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); out = root/'receding_prediction_report_v1'; out.mkdir(exist_ok=True)
    audits = json.loads((root/'feedback_execution_summary_v1.json').read_text())['results']
    rows, sections = [], []
    for audit in audits:
        if audit['kind'] != 'experiment' or not audit['replan_schedule'] or audit['context_source'] != 'observed':
            continue
        run = root/audit['run']; report = json.loads((run/'report.json').read_text())
        saved = torch.load(run/'replans.pt', weights_only=False, map_location='cpu')
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.5)); stages = []
        for index, (variant, label) in enumerate([('feedback_sample', 'Fresh sample at each replan'),
                                                 ('feedback_noise_opt', f'{audit["iterations"]} updates at each replan')]):
            execution = np.load(run/f'{variant}_step0.npz')
            T = execution['T']; R = execution['R']; origin = T[0, 0, 2]
            times = np.arange(len(T))*.05
            axes[index, 0].plot(times, (T[:, 0, 2]-origin)*1000, color='#0072b2', label='Executed object')
            for stage, schedule in enumerate(audit['replan_schedule']):
                entry = saved[f'{variant}_{stage}']; batch = entry['batch']; C = batch.context_len
                roll = entry['result']['rollout'] if entry['result'] is not None else entry['initial']
                frame, commit = schedule['frame'], schedule['commit']
                pred_T = roll['T'][0, C-1:C+commit].double().numpy()
                pred_R = roll['R'][0, C-1:C+commit].double().numpy()
                actual_T, actual_R = T[frame:frame+commit+1], R[frame:frame+commit+1]
                # The observed anchor must be the real current state, allowing
                # only the explicit FP32 conversion made by feedback construction.
                np.testing.assert_allclose(pred_T[0], actual_T[0], rtol=0, atol=1e-7)
                np.testing.assert_allclose(pred_R[0], actual_R[0], rtol=0, atol=1e-7)
                delta = np.linalg.norm(pred_T[:, 0]-actual_T[:, 0], axis=-1)*1000
                relative = pred_R[:, 0] @ actual_R[:, 0].swapaxes(-1, -2)
                angle = np.rad2deg(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))
                stage_times = times[frame:frame+commit+1]
                axes[index, 0].plot(stage_times, (pred_T[:, 0, 2]-origin)*1000, '--', color='#d55e00',
                                    label='Predicted committed segment' if stage == 0 else None)
                axes[index, 0].scatter(stage_times[0], (actual_T[0, 0, 2]-origin)*1000, color='#333333', s=20)
                axes[index, 1].plot(stage_times, delta, color='#d55e00')
                stages.append(dict(variant=variant, frame=frame, commit=commit,
                    selected_iteration=entry['result']['iteration'] if entry['result'] else None,
                    predicted_committed_lift_mm=float((pred_T[-1, 0, 2]-pred_T[0, 0, 2])*1000),
                    executed_committed_lift_mm=float((actual_T[-1, 0, 2]-actual_T[0, 0, 2])*1000),
                    object_position_error_max_mm=float(delta.max()),
                    object_position_error_end_mm=float(delta[-1]),
                    object_rotation_error_end_deg=float(angle[-1])))
            for ax in axes[index]:
                for stage in audit['replan_schedule']:
                    ax.axvline(stage['frame']*.05, color='#777777', ls=':', alpha=.5)
                ax.axvspan(0, audit['prefix_frames']*.05, color='#777777', alpha=.1)
                ax.set_xlabel('Rollout time (s)'); ax.grid(alpha=.2)
            axes[index, 0].set(title=label, ylabel='Object height change (mm)'); axes[index, 0].legend(fontsize=8)
            axes[index, 1].set(title='Prediction error during the next executed segment', ylabel='Object origin distance (mm)')
        fig.suptitle(f'{audit["scene"]} · seed {audit["seed"]} · observed feedback at four stages\n'
                     'Each dashed prediction is compared only with the commands executed before the next replan.')
        fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(out/(run.name+'.png'), dpi=140); plt.close(fig)
        record = dict(run=run.name, scene=audit['scene'], iterations=audit['iterations'], seed=audit['seed'],
                      physics_step_ms=report['results'][1]['replays'][0]['step']*1000,
                      stages=stages, complete_qualified_passes=sum(v['qualified_full_task_pass'] for v in audit['results']),
                      report_sha256=hashlib.sha256((run/'report.json').read_bytes()).hexdigest(),
                      plans_sha256=hashlib.sha256((run/'replans.pt').read_bytes()).hexdigest())
        rows.append(record)
        table = '<table><tr><th>Arm</th><th>Replan frame</th><th>Predicted / executed height change</th><th>Next-segment endpoint error</th><th>Retained update</th></tr>'
        for row in stages:
            table += (f'<tr><td>{row["variant"]}</td><td>{row["frame"]}</td>'
                      f'<td>{row["predicted_committed_lift_mm"]:.1f} / {row["executed_committed_lift_mm"]:.1f} mm</td>'
                      f'<td>{row["object_position_error_end_mm"]:.1f} mm / {row["object_rotation_error_end_deg"]:.1f}°</td>'
                      f'<td>{row["selected_iteration"]}</td></tr>')
        sections.append(f'<section><h2>{html.escape(audit["scene"])} · {audit["iterations"]} updates per replan</h2>'
                        f'<p>Complete qualified passes among original, fresh and optimized execution: {record["complete_qualified_passes"]}/3. '
                        f'Physics step for this prediction check: {record["physics_step_ms"]:g} ms. '
                        f'<a href="../{run.name}/report.json">Experiment receipt</a></p>'
                        f'<img src="{run.name}.png">{table}</table></section>')
    payload = dict(updated_utc=datetime.now(timezone.utc).isoformat(), results=rows,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   note='Only committed segments are compared with their actual execution. Later unexecuted predictions are excluded. Fresh and optimized arms can observe different histories after different controls. The independent floating-hand controller and approximate physics remain limitations.')
    (out/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    page = '''<!doctype html><meta charset="utf-8"><title>Prediction error between replans</title>
<style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1350px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}section{margin-top:40px}</style>
<h1>After seeing what happened, does the model predict the next executed segment?</h1>
<p>Blue is the object in independent physics. Orange dashed lines are the next eight frames of each newly generated plan. Black dots mark the observed states used to replan. Later parts of a plan that were replaced by another replan are not compared against execution.</p>
<p>The right panels measure disagreement during those same committed segments. Each starts at the observed state, verified within the feedback interface's FP32 conversion tolerance. Replanning resets the prediction error at that instant; it does not establish that the subsequent prediction is accurate.</p>
<p>These plots use the 0.5 ms engine that supplied the observations. The separate 0.25 ms execution is a replay of the same realized commands, not an independently replanned policy. The gripper controller, geometry and inertia are approximations; this does not isolate world-model error from controller or engine mismatch.</p>
<p><a href="../feedback_replay_report_v1/index.html">Full executions at both physics steps</a> · <a href="summary.json">All segment measurements</a> · <a href="../index.html">Full report</a></p>'''
    (out/'index.html').write_text(page+''.join(sections))
    print('Rendered committed prediction diagnostics:', len(rows), 'experiments')


if __name__ == '__main__':
    main()
