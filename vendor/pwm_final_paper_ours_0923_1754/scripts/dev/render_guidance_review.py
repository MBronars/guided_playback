"""A short entry page backed by the campaign's detailed scientific reports."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', required=True, type=Path)
    root = parser.parse_args().root.resolve(); out = root/'review_11am'; out.mkdir(exist_ok=True)
    primary = json.loads((root/'primary_overview_v1/summary.json').read_text())
    region = [r for r in json.loads((root/'region_replay_summary_v1.json').read_text())['results'] if r['variant'] == 'selected']
    feedback = [r for r in json.loads((root/'feedback_execution_summary_v1.json').read_text())['results'] if r['kind'] == 'experiment']
    banks = [r for r in json.loads((root/'feedback_candidates_report_v1/summary.json').read_text())['results'] if r['kind'] == 'experiment']
    pairing = json.loads((root/'feedback_pairing_v1.json').read_text())
    trials = json.loads((root/'summary.json').read_text())['trials']
    done = [r for r in primary['rows'] if r['status'] == 'completed']
    region_model = sum(json.loads((root/r['run']/f'seed_{r["seed"]}.json').read_text())['validation']['accepted'] for r in region)
    groups = []
    for task in ('pick', 'place'):
        for method in ('Initial sample', 'Goal · 6', 'Physical · 6', 'Goal · 24', 'Physical · 24'):
            subset = [r for r in done if r['task'] == task and r['method'] == method]
            groups.append(dict(task=task, method=method, scenes=len(subset),
                               goal_mm=float(np.median([r['goal_mm'] for r in subset])) if subset else None,
                               penetration_mm=float(np.median([r['penetration_mm'] for r in subset])) if subset else None))
    table = '<table><tr><th>Task</th><th>Method</th><th>Scenes</th><th>Median final point error</th><th>Median peak mesh overlap</th></tr>'
    for row in groups:
        table += (f'<tr><td>{row["task"]}</td><td>{row["method"]}</td><td>{row["scenes"]}</td>'
                  f'<td>{row["goal_mm"]:.2f} mm</td><td>{row["penetration_mm"]:.2f} mm</td></tr>')
    runtime = []
    runtime_table = '<table><tr><th>Method</th><th>Standalone runs</th><th>Median seconds</th><th>Model calls</th><th>Peak allocated GPU GB, range</th></tr>'
    for arm in ('goal6_h36', 'physical6_h36', 'goal24_h36', 'physical24_h36'):
        subset = [r for r in trials if r['arm'] == arm and r['seed'] == primary['seed']]
        if not subset:
            continue
        row = dict(arm=arm, runs=len(subset), median_seconds=float(np.median([r['seconds'] for r in subset])),
                   model_calls=sorted(set(r['calls'] for r in subset)),
                   peak_allocated_gpu_gb_range=[min(r['peak_gpu_gb'] for r in subset), max(r['peak_gpu_gb'] for r in subset)])
        runtime.append(row)
        low, high = row['peak_allocated_gpu_gb_range']
        runtime_table += (f'<tr><td>{arm}</td><td>{len(subset)}</td><td>{row["median_seconds"]:.1f}</td>'
                          f'<td>{", ".join(map(str, row["model_calls"]))}</td><td>{low:.3f}–{high:.3f}</td></tr>')
    feedback_table = '<table><tr><th>Scene</th><th>Correction</th><th>Optimized/selected executed outcome, both steps</th><th>Complete qualified task</th></tr>'
    for row in sorted(feedback, key=lambda r: (r['scene'], r['prefix_frames'], r['iterations'], r['context_source'])):
        selected = row['results'][-1]
        if row['method'] == 'best_of_k':
            label = f'Best of {row["k"]}, frame{row["prefix_frames"]}'
        elif row['replan_schedule']:
            label = f'{row["iterations"]} updates at each of four replans'
        else:
            label = f'{row["iterations"]} updates, frame{row["prefix_frames"]}'
        key = 'object_lift_mm' if row['task'] == 'pick' else 'final_goal_mm'
        outcome = ' / '.join(f'{v[key]:.1f}' for v in selected['replays'])
        outcome += ' mm ' + ('final lift' if row['task'] == 'pick' else 'final point error')
        feedback_table += (f'<tr><td>{html.escape(row["scene"])}</td><td>{label}<br>{row["context_source"]} context</td>'
                           f'<td>{outcome}</td><td>{"PASS" if selected["qualified_full_task_pass"] else "FAIL"}</td></tr>')
    payload = dict(updated_utc=datetime.now(timezone.utc).isoformat(), checkpoint_sha256=primary['checkpoint_sha256'],
        primary_seed=primary['seed'], primary_cells_completed=len(done), primary_cells_expected=len(primary['rows']),
        primary_mesh_passes=sum(r.get('mesh_pass') is True for r in done),
        primary_qualified_execution_passes=sum(r.get('qualified_execution_pass') is True for r in done),
        primary_pending_execution=sum(r['execution_pending'] for r in done), primary_pooled_medians=groups,
        standalone_runtime=runtime,
        region_scenes=len(region), region_model_passes=region_model,
        region_qualified_execution_passes=sum(r['qualified_task_pass'] for r in region),
        feedback_experiments_completed=len(feedback),
        feedback_qualified_selected_passes=sum(r['results'][-1]['qualified_full_task_pass'] for r in feedback),
        candidate_banks_completed=len(banks), candidates_audited=sum(r['k'] for r in banks),
        candidate_qualified_passes=sum(r['qualified_full_task_passes'] for r in banks),
        pairing_status=pairing['status'], pairing_checks=pairing['completed_pairs'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(payload, indent=2)+'\n')
    page = f'''<!doctype html><meta charset="utf-8"><title>Guidance review — September 15</title>
<style>body{{font:17px system-ui;line-height:1.55;margin:35px auto;padding:0 22px;max-width:1150px;color:#222}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse;font-size:14px}}section{{margin-top:38px}}.status{{padding:18px;background:#f1f5f8;border-left:4px solid #0072b2}}a{{color:#0067a5}}</style>
<h1>Guidance review · September 15</h1>
<p class="status"><b>Noise optimization reduces the model's goal error. Reliable physical manipulation remains unresolved.</b><br>
{len(done)}/{len(primary['rows'])} primary comparison cells are complete. Full primary task passes: {payload['primary_mesh_passes']} in model geometry, {payload['primary_qualified_execution_passes']} in qualified independent execution. Pending primary execution audits: {payload['primary_pending_execution']}.</p>
<p>One fixed unconditional bidirectional checkpoint, EMA weights, clean observed context, no prescribed future actions. The primary set has eight pick and eight place scenes, with four development and four guidance-reserved scenes per task. Reserved does not establish that a scene was unseen during training. All primary columns share the same initial draw and target, seed8311, over1.8seconds.</p>
<section><h2>1. Does the standard optimizer work?</h2><p>Yes, as an optimizer through the frozen sampler. Production sampling parity, replay, cross-window gradients and zero-update behavior were checked. More updates substantially reduce predicted point error. Point accuracy alone does not establish sustained contact, release or executed success.</p>
{table}</table><p>These are pooled eight-scene medians. <a href="../primary_overview_v1/index.html">See every scene and separate development/reserved tables</a>. The primary physical objective uses the original geometric proxies, before the gripper-floor correction; the stronger mesh-cost and feedback experiments are separate. Actual primary objective and sampler modules are identical in frozen source_v4/source_v5. <a href="../primary_source_semantics_v1.json">Source check</a>. The first can and bowl's24-update entries use the retained24-update prefixes of their longer runs.</p>
<details><summary>Runtime and GPU memory for the primary standalone runs</summary>{runtime_table}</table>
<p>FP32, uncompiled RTX PRO6000 diagnostics; includes the paired initial sample and selection overhead. Model calls include gradient recomputation, but do not equalize backward FLOPs. The two retained24-update prefixes are excluded here because their enclosing100-update run's timing and memory are not24-update measurements. Allocated memory excludes other processes and the CUDA allocator's reserved pool. These timings are not a real-time deployment benchmark.</p></details></section>
<section><h2>2. Where do plausible predictions fail?</h2><p>The broader lift-workspace task produces {region_model}/{len(region)} model plans that pass its region and geometry screen, but {payload['region_qualified_execution_passes']}/{len(region)} qualified physical executions. This task asks for at least100mm lift within a bounded XY workspace, separately from the primary precise target.</p>
<p>In the movie below, orange is the object. Top: a recorded motion and its qualified controller replay. Bottom: the selected model prediction and its execution. Both right panels command only the gripper. The recorded lift replays, while the generated lift leaves the object on the table. The generated replay is stable at both physics steps; this example is not explained by timestep instability alone.</p>
<img src="../qualified_control_report_v1/recorded_vs_guided_8270.gif">
<p><a href="../qualified_control_report_v1/index.html">Recorded control versus generated plan, with limitations</a> · <a href="../region_replay_report_v1/index.html">All eight model-versus-execution lift movies</a> · <a href="../region_contact_timing_v1/index.html">Contact timing in all eight scenes</a> · <a href="../grasp_execution_report_v1/index.html">Contact-force explanation and the can's predicted lift versus actual drop</a></p></section>
<section><h2>3. Does observing and replanning solve it?</h2><p>{len(feedback)} completed feedback experiments; {payload['feedback_qualified_selected_passes']} selected continuations pass the full qualified task. Late corrections, early repeated corrections and sample selection remain separately labeled. A later correction cannot remove a violation that already occurred in its shared prefix.</p>
{feedback_table}</table>
<p>Final lift and point error here are descriptive, not success criteria. Large lift can accompany severe collision. All task, tracking and timestep checks remain in force. <a href="../feedback_replay_report_v1/index.html">Every original/fresh/optimized execution and failure</a> · <a href="../receding_prediction_report_v1/index.html">Does each next executed segment match its new prediction?</a></p>
<p>{len(banks)} completed49-candidate banks; {payload['candidates_audited']} candidates independently audited at both physics steps; {payload['candidate_qualified_passes']} complete qualified passes. The original candidate selection is preserved. <a href="../feedback_candidates_report_v1/index.html">Candidate coverage and selection plots</a>. Pairing audit: {payload['pairing_status']}, {payload['pairing_checks']} completed comparisons. <a href="../feedback_pairing_v1.json">Exact noise and shared-execution checks</a>.</p></section>
<section><h2>4. What should change next?</h2><p><b>Resolve the execution interface first.</b> The present floating-hand MuJoCo replay qualifies11/16 recorded scenes, with approximate collision, inertia and a fixed preload conversion from predicted achieved opening to motor commands. It is not the source PhysX controller or a robot. This limits attribution of prediction–execution differences.</p>
<p><b>Test contact and actuation consistency before adding more penalties.</b> Geometric proximity is useful but insufficient. Jointly generated object/gripper trajectories can satisfy a model-side objective and still fail to make the object follow the gripper in execution. More random restarts and more updates are tested controls, not a demonstrated solution.</p>
<p><b>Check that the requested task is represented in the recordings.</b> The primary held-point/released-place checks are stricter than the source task. All49 early bowl predictions also fail the5mm release-clearance criterion, as do their executions; this bank never offers a released ending. That is one scene and prefix, not a corpus-wide rate. <a href="../bowl_bank_release_v1/release_clearance.png">Predicted versus executed release clearance</a> · <a href="../bowl_bank_release_v1/report.json">Exact release measurements</a>.</p>
<p>The separate task audit retains the original definitions and examines recorded contact/release support; no thresholds were relaxed to manufacture success. <a href="../task_reference_report_v1/index.html">Task/reference audit</a> · <a href="../place_release_report_v1/index.html">Recorded placing-release study</a>.</p></section>
<p><a href="../index.html">Complete campaign report and receipts</a> · <a href="summary.json">This page's computed summary</a><br>Updated {html.escape(payload['updated_utc'])}. These are development experiments and independent-engine diagnostics, not a robot success rate.</p>'''
    (out/'index.html').write_text(page)
    print('Review:', payload['primary_cells_completed'], 'primary cells;', len(feedback), 'feedback experiments;', len(banks), 'candidate banks')


if __name__ == '__main__':
    main()
