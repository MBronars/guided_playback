"""Package observation experiments as standalone scientific plots, videos and HTML."""
from __future__ import annotations

import argparse
import collections
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from scripts.dev.observation_probe_common import reported_checkpoint, write_json


COLORS = ("#27a58f", "#517ad1", "#e99432", "#bb5f9e")


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(519)
    if len(values) == 1:
        return float(values[0]), float(values[0]), float(values[0])
    boot = values[rng.integers(len(values), size=(2000, len(values)))].mean(1)
    lo, hi = np.quantile(boot, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def axes3d(ax, lo, hi):
    center = (lo + hi) / 2
    half = max(float((hi - lo).max()) / 2, .08) * 1.1
    ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
           zlim=(center[2]-half, center[2]+half), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=25, azim=-55)
    ax.tick_params(labelsize=7)


def render_pair(path, clean_path, noisy_path, title):
    z, c = np.load(noisy_path), np.load(clean_path)
    reference, noisy, clean = z["reference"], z["prediction"], c["prediction"]
    C = int(z["context_len"]); ids = z["object_ids"]; act = z["action_mask"]
    lo = np.minimum(reference.min((0, 1)), noisy.min((0, 1)))
    hi = np.maximum(reference.max((0, 1)), noisy.max((0, 1)))
    fig = plt.figure(figsize=(12, 5.3), facecolor="white")
    writer = FFMpegWriter(fps=8, codec="libx264", extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "-crf", "22"])
    with writer.saving(fig, str(path), dpi=110):
        frames = [0]*8 + list(range(noisy.shape[0])) + [noisy.shape[0]-1]*12
        for f in frames:
            fig.clear()
            for panel, pred, subtitle in ((1, clean, "Clean observed history"), (2, noisy, "Imperfect observed history")):
                ax = fig.add_subplot(1, 2, panel, projection="3d")
                axes3d(ax, lo, hi)
                ax.scatter(*reference[C + f, ~act].T, c="#8d99a6", s=3, alpha=.35, label="Actual future")
                ax.scatter(*pred[f, ~act].T, c=COLORS[panel-1], s=3, alpha=.8, label="Predicted future")
                if act.any():
                    ax.scatter(*pred[f, act].T, c="#e99432", s=2, alpha=.7, label="Actuator")
                ax.set_title(subtitle, fontsize=11)
                if panel == 1:
                    ax.legend(loc="upper left", fontsize=7, markerscale=2)
            fig.suptitle(title, fontsize=11)
            fig.text(.5, .015, f"Future frame {f+1}/{len(noisy)} | playback slowed | simulation reference | display subset; full-particle metrics",
                     ha="center", fontsize=8)
            writer.grab_frame()
    plt.close(fig)


def robustness(root):
    out = root / "robustness"
    records = [json.loads(p.read_text()) for p in sorted((out / "cases").glob('*.json'))]
    good = [x for x in records if x["status"] == "ok"]
    run_status = json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    if not good:
        return {"status": "No completed robustness cases yet", "videos": []}
    clean = {(x["clip"], x["sampler_seed"]): x for x in good if x["family"] == "clean"}
    grouped = collections.defaultdict(list)
    rows = []
    for x in good:
        base = clean.get((x["clip"], x["sampler_seed"]))
        if base is None:
            continue
        err = x["metrics"]["passive_mean_mm"]
        delta = err - base["metrics"]["passive_mean_mm"]
        grouped[(x["domain"], x["family"], x["severity"], x["clip"])].append((err, delta))
        rows.append({"case": x["case"], "clip": x["clip"], "domain": x["domain"],
                     "family": x["family"], "severity": x["severity"],
                     "sampler_seed": x["sampler_seed"], "corruption_seed": x["corruption_seed"],
                     "passive_mean_mm": err, "paired_degradation_mm": delta,
                     "anchor_rms_mm": x["metrics"]["anchor_rms_mm"]})
    by_condition = collections.defaultdict(list)
    for (domain, family, severity, clip), vals in grouped.items():
        by_condition[(domain, family, severity)].append(np.mean(vals, axis=0))
    summary = []
    for (domain, family, severity), vals in sorted(by_condition.items()):
        v = np.array(vals)
        summary.append({"domain": domain, "family": family, "severity": severity,
                        "clips": len(v), "absolute_mm_mean_ci95": mean_ci(v[:,0]),
                        "paired_degradation_mm_mean_ci95": mean_ci(v[:,1])})
    write_json(out / "summary.json", {"conditions": summary, "successful_cases": len(good),
                                       "failed_cases": len(records)-len(good),
                                       "planned_cases": run_status.get('planned'),
                                       "all_planned_cases_attempted": run_status.get('all_planned_cases_attempted',False),
                                       "estimator": "average draws within clip, then equal clip weight; percentile bootstrap over clips",
                                       "diagnostic_only": True})
    with (out / "results.csv").open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    families = ("history_jitter", "history_drift", "pose_translation", "pose_rotation", "stiffness", "density")
    for metric, filename, ylabel in (("absolute_mm_mean_ci95", "absolute_error.png", "Passive particle error (mm)"),
                                     ("paired_degradation_mm_mean_ci95", "paired_degradation.png", "Increase over matching clean input (mm)")):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, family in zip(axes.flat, families):
            for domain, color in zip(("mpm", "pickup", "pushing", "cloth"), COLORS):
                data = sorted([x for x in summary if x['domain']==domain and x['family']==family], key=lambda x:x['severity'])
                if not data:
                    continue
                counts = [x['clips'] for x in data]
                clean_entry = next((x for x in summary if x['domain']==domain and x['family']=='clean'), None)
                if clean_entry is not None:
                    baseline = dict(clean_entry)
                    baseline['severity'] = 1 if family in ('stiffness','density') else 0
                    data = sorted([baseline,*data],key=lambda x:x['severity'])
                xx = [x['severity'] for x in data]; yy = np.array([x[metric] for x in data])
                count_label = str(counts[0]) if min(counts)==max(counts) else f'{min(counts)}–{max(counts)}'
                ax.plot(xx, yy[:,0], 'o-', color=color, label=f"{domain} (n={count_label})")
                ax.fill_between(xx, yy[:,1], yy[:,2], color=color, alpha=.13)
            ax.set_title(family.replace('_',' ')); ax.axhline(0, color='#aaaaaa', lw=.6)
            ax.set_xlabel("Rotation bias (degrees)" if family=='pose_rotation' else
                          "Property multiplier" if family in ('stiffness','density') else "Observation error severity (mm)")
            ax.set_ylabel(ylabel); ax.grid(alpha=.15)
            if ax.lines and ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
            elif not ax.get_legend_handles_labels()[0]:ax.text(.5,.5,'Pending results',transform=ax.transAxes,ha='center',color='#888888')
        progress = '' if run_status.get('all_planned_cases_attempted') else 'PARTIAL | '
        fig.suptitle(progress+"Frozen four-domain 40K UNet | 16-frame diagnostic | 95% intervals over clips", fontsize=13)
        fig.savefig(out / filename, dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    frame_summary = []
    for ax, domain in zip(axes.flat, ('mpm','pickup','pushing','cloth')):
        choices = [('clean',0,'Clean'),('history_jitter',10,'History jitter: 10 mm'),
                   ('history_drift',10,'History drift: 10 mm'),
                   ('stiffness',2,'Stiffness multiplied by 2') if domain=='mpm' else
                   ('pose_translation',10,'Object pose bias: 10 mm')]
        for (family,severity,label),color in zip(choices,COLORS):
            per_clip = collections.defaultdict(list)
            for case in good:
                if case['domain']==domain and case['family']==family and case['severity']==severity:
                    per_clip[case['clip']].append(case['metrics']['passive_per_frame_mm'])
            if not per_clip:continue
            values=np.array([np.mean(v,axis=0) for v in per_clip.values()])
            mean=values.mean(0); frames=np.arange(1,len(mean)+1)
            rng=np.random.default_rng(519)
            boot=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
            lo,hi=np.quantile(boot,[.025,.975],axis=0)
            ax.plot(frames,mean,color=color,label=label)
            ax.fill_between(frames,lo,hi,color=color,alpha=.10)
            frame_summary.append({'domain':domain,'family':family,'severity':severity,
                                  'n_clips':len(values),'passive_mean_mm':mean.tolist(),
                                  'pointwise_ci95_low_mm':lo.tolist(),'pointwise_ci95_high_mm':hi.tolist()})
        ax.set(title=domain,xlabel='Future frame',ylabel='Passive particle error (mm)')
        ax.grid(alpha=.15)
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
    fig.suptitle(progress+'Error growth over the prediction window | equal clip weight',fontsize=13)
    fig.savefig(out/'per_frame_error.png',dpi=160);plt.close(fig)
    write_json(out/'per_frame_summary.json',frame_summary)
    # Choose the largest paired degradation among saved primary-seed previews,
    # one per domain. Preserve all cases; this is an explicitly selected failure view.
    videos = []
    for domain in ("mpm", "pickup", "pushing", "cloth"):
        available = [x for x in good if x["domain"]==domain and x["family"]!='clean' and x.get('preview')
                     and (x['clip'],x['sampler_seed']) in clean]
        if not available:
            continue
        worst = max(available, key=lambda x:x['metrics']['passive_mean_mm']-clean[(x['clip'],x['sampler_seed'])]['metrics']['passive_mean_mm'])
        baseline = clean[(worst['clip'],worst['sampler_seed'])]
        if not baseline.get('preview'):
            continue
        filename = domain + '_selected_failure.mp4'
        target = out / filename
        selection_path = out / (domain + '_video_selection.json')
        previous = json.loads(selection_path.read_text()) if selection_path.exists() else {}
        if not target.exists() or previous.get('case') != worst['case']:
            render_pair(target, out/baseline['preview'], out/worst['preview'],
                        f"{domain}: {worst['family']} {worst['severity']:g} | {worst['clip']} | selected largest paired degradation")
            write_json(selection_path, {'case': worst['case'], 'clean_case': baseline['case']})
        videos.append({"file": 'robustness/'+filename, "case": worst['case'],
                       "selection": "largest mean-error increase among saved primary-seed cases"})
    write_json(out/'video_selections.json', videos)
    write_json(out/'report_status.json', {'state': 'finished', 'case_count': len(records),
                                         'successful_cases': len(good), 'failed_cases': len(records)-len(good)})
    return {"status": f"{len(good)} completed cases of {run_status.get('planned','?')} planned; {len(records)-len(good)} failures", "videos": videos,
            "summary": summary}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--index-only',action='store_true')
    a=ap.parse_args(); root=Path(a.root)
    reported_checkpoint(root)       # the titles and text below describe only this checkpoint
    result = robustness(root) if not a.index_only else {"status": "See saved lane reports", "videos": []}
    if a.index_only and (root/'robustness/status.json').exists():
        state=json.loads((root/'robustness/status.json').read_text())
        result['status']=f"{state.get('completed',0)} successful / {state.get('planned','?')} planned; {state.get('failed',0)} failures ({state.get('state','unknown')})"
    chunks = ['<!doctype html><html><head><meta charset="utf-8"><title>Overnight observation experiments</title>',
              '<style>body{max-width:1150px;margin:40px auto;padding:0 24px;font:17px/1.5 system-ui;color:#263344;background:#f7f9fc}img,video{width:100%;border-radius:8px}section{background:white;padding:24px;margin:24px 0;border-radius:12px}small{color:#607083}a{color:#2466a8}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}</style></head><body>',
              '<h1>Prediction from imperfect observations</h1>',
              '<p>Frozen four-domain 40K UNet. Diagnostic evaluation; historical training exclusions have not been audited for a paper benchmark.</p>',
              '<p><a href="FINDINGS.md">Read the findings and next experiments</a></p>',
              '<section><h2>GPU 1 — controlled observation errors</h2>',
              '<p>'+html.escape(result['status'])+'</p>',
              '<p>We give the same model either an accurate observed history or a deliberately inaccurate one, keep the future robot commands fixed, and compare both predictions with the original simulated future. The test covers geometry, motion history and MPM material values. It does not test segmentation failures or lost tracks.</p>',
              '<p>Gray points in the videos are the simulated reference. Colored points are model predictions. Rigid geometry and pose inputs receive the same perturbation. Commands, rigid templates and object identities stay known.</p>']
    explanations={
        'paired_degradation.png': '<h3>How much extra error does an imperfect observation cause?</h3><p>Zero means no increase over the matching clean-input prediction. Higher values mean worse predictions. The horizontal axis is the amount of injected input error; the shaded bands describe variation across sampled clips.</p>',
        'absolute_error.png': '<h3>How accurate are the predictions overall?</h3><p>These are distances to the original simulated particle trajectories. The clean-input point is included: a model can have little additional degradation while already making a large error.</p>',
        'per_frame_error.png': '<h3>Does the error grow as we predict further ahead?</h3><p>Each curve follows the 16-frame prediction window. These tests restart from observed data for each clip. They do not establish robustness over a long rollout.</p>'}
    for filename in ('paired_degradation.png','absolute_error.png','per_frame_error.png'):
        if (root/'robustness'/filename).exists():
            chunks.append(explanations[filename])
            chunks.append(f'<img src="robustness/{filename}">')
    if (root/'stability/input_response_vs_sample_spread.png').exists():
        chunks.append('<h3>Input sensitivity versus sampling variation</h3><p>Blue bars show how much predictions spread when we repeat the same clean input with different random seeds. Orange bars show how much predictions change when we add 1 mm of history error while pairing each random seed. Dots are individual clips. The two quantities have different definitions; equal heights are not a statistical threshold.</p><img src="stability/input_response_vs_sample_spread.png"><p>Six sampler seeds include diffusion noise and randomized FPS. Sample spread is not calibrated uncertainty about the observed physical state. <a href="stability/summary.json">Measurements</a></p>')
    if (root/'motion_noise/constant_vs_changing_history.png').exists():
        chunks.append('<h3>Does changing observation error create false motion?</h3><p>Both conditions have exactly the same noisy endpoint. Orange changes the estimation error from frame to frame; blue keeps each point or object offset constant, preserving the true observed motion. This controlled comparison tests temporal error structure. Constructing the blue input uses the clean reference; it is not a practical state estimator.</p><img src="motion_noise/constant_vs_changing_history.png"><p><a href="motion_noise/summary.json">Measurements and definitions</a></p>')
    chunks.append('<p><a href="robustness/results.csv">Per-case CSV</a> · <a href="robustness/summary.json">Summary and estimator</a> · <a href="robustness/run.json">Checkpoint and source receipt</a> · <a href="robustness/status.json">Run status</a></p>')
    chunks.append('<p>A small effect from changing material values does not by itself prove robustness: the selected motion may be insensitive to that property, or the model may not use it effectively. This experiment does not distinguish those explanations.</p>')
    for p in sorted((root/'robustness').glob('*_selected_failure.mp4')):
        chunks.append(f'<h3>{html.escape(p.stem.replace("_"," "))}</h3><video controls loop muted src="robustness/{p.name}"></video>')
    chunks += ['</section><section><h2>GPU 0 — real cloth</h2>',
               '<p>Recorded RGBench cloth and robot motion. Given future gripper actions make this action-conditioned prediction. Camera observations are partial; correspondence and material estimates remain assumptions.</p>']
    real = root/'real_cloth'
    if (real/'report.html').exists():
        chunks.append('<p><a href="real_cloth/report.html">Open real-cloth predictions, measurements, and limitations</a></p>')
    else:
        chunks.append('<p>Real-cloth adapter/results pending. Check the lane status and logs.</p>')
    if (root/'real_cloth_full/report.html').exists():
        chunks.append('<p><a href="real_cloth_full/report.html">Longer real-cloth rollouts through the recorded actions</a></p>')
    if (root/'real_cloth_annotated/report.html').exists():
        chunks.append('<p><a href="real_cloth_annotated/report.html">Real-cloth forecasts from the source-annotated evaluation phases</a></p>')
    if (root/'real_cloth_icp/report.html').exists():
        chunks.append('<p><a href="real_cloth_icp/report.html">Real-cloth forecasts with coherent prefix registration</a></p>')
    if (root/'real_cloth_short/report.html').exists():
        chunks.append('<p><a href="real_cloth_short/report.html">Independent half-second forecasts from fresh real observations</a></p>')
    if (root/'real_cloth_short_smooth/report.html').exists():
        chunks.append('<p><a href="real_cloth_short_smooth/report.html">Matched short forecasts with local smoothing of estimated history motion</a></p>')
    if (root/'real_motion/lift_height.png').exists():
        chunks.append('<h3>Does the model predict the real lift?</h3><p>Green follows the upper part of the measured cloth; orange follows the prediction. A large gap means the prediction missed the rise. This compares visible-geometry height, not tracked physical landmarks. The bottom row restarts from fresh measured data.</p><img src="real_motion/lift_height.png"><p><a href="real_motion/summary.json">Height and tail-distance measurements</a> · <a href="real_motion/motion_smoothing_comparison.json">Paired motion-smoothing results</a></p>')
    if (root/'real_world_montage.mp4').exists():
        chunks.append('<h3>Real-cloth montage</h3><video controls muted src="real_world_montage.mp4"></video><p><a href="real_results.json">All real-capture measurements and selections</a></p>')
    chunks += ['<p><a href="real_cloth/status.json">Real-cloth status</a></p></section>',
               '<p><small>Source dataset: <a href="https://rgbench.github.io/">RGBench, Hu et al., AAAI 2026</a>, CC BY 4.0. Source camera footage and our predicted output must be distinguished. No model-generated RGB video is used as a measured result.</small></p></body></html>']
    (root/'index.html').write_text('\n'.join(chunks))


if __name__=='__main__':
    main()
