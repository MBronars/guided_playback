"""Compact scorecard of declared model comparisons and independent executions."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    root=p.parse_args().root.resolve();gallery=json.loads((root/'gallery_summary.json').read_text())
    execution=json.loads((root/'execution_summary.json').read_text());replays={v['key']:v for v in execution['results']}
    unique={}
    for group in gallery['groups']:
        for row in group['variants']:unique.setdefault(row['entry']['key'],row)
    lines=['# Guidance scorecard','',
        '[Model movies](GALLERY.md) · [Compute measurements](COMPUTE.md) · [Prediction versus physics](SURFACE_GALLERY.md) · '+
        '[Obstacle before/after](OBSTACLE_GALLERY.md) · [All execution checks](EXECUTION_GALLERY.md)',
        '', 'These entries include shared optimization prefixes and controls; they are not independent trials. '+
        'Held error is the worst position error over the final six frames. The replay column takes the worse of the two solver timesteps.',
        '', 'A held-position hit is only one requirement. Full checks additionally test contact, penetration, support, joints and (for placement) resting/release/orientation.',
        '', '![Model goal reaching versus independent execution](figures/model_vs_execution_goals.png)','',
        '| Entry | Model held error | Replay held error | Model geometry | Full paired execution |',
        '|---|---:|---:|---|---|']
    rows=[]
    for key,row in unique.items():
        if row['state']!='completed':
            lines.append(f'| {key} | Pending | — | — | — |');continue
        m=row['metrics'];r=replays.get(key)
        value=max(v['held_worst_goal_mm'] for v in r['replays']) if r else None
        lines.append(f'| [{key}](exports/{key}/receipt.json) | {m["held_worst_goal_mm"]:.1f} mm | '+
            (f'{value:.1f} mm' if value is not None else 'Pending')+
            f' | {"PASS" if m["geometry_pass"] else "FAIL"} | '+
            (f'[{"PASS" if r["qualified_task_pass"] else "FAIL"}](replays/{key}/audit.json)' if r else 'Pending')+' |')
        rows.append(dict(key=key,model_held_mm=m['held_worst_goal_mm'],replay_held_mm=value,
            model_geometry_pass=m['geometry_pass'],execution_pass=r['qualified_task_pass'] if r else None))
    lines+=['', 'The longer horizon adds physical time, latent variables and computation; it is not a matched-compute comparison.',
        'Unguided entries are controls and were not asked to reach these lateral destinations. Obstacle controls reuse a frozen plan in a new replay world.',
        '', '[Ordinary pick bank: all 32 outcomes](ordinary_bank/README.md) · '+
        '[Two positive unguided lifts](ordinary_bank/SURFACE_GALLERY.md) · [Preload-controller diagnostic](PRELOAD_DIAGNOSTIC.md)','']
    if (root/'FIXED_GRIPPER.md').exists():
        lines.append('[Fixed-gripper object-noise resampling](FIXED_GRIPPER.md).')
    summary=dict(updated_utc=datetime.now(timezone.utc).isoformat(),completed_entries=len(rows),
        pending_entries=len(unique)-len(rows),entries=rows)
    (root/'scorecard_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plot(root,rows)
    compute_report(root, unique)
    tmp=root/'SCORECARD.pending.md';tmp.write_text('\n'.join(lines));tmp.replace(root/'SCORECARD.md')


def compute_report(root, unique):
    seen=set();rows=[]
    lines=['# Measured guidance compute','',
        'One RTX PRO 6000 Blackwell per experiment; frozen FP32 checkpoint, 25-step pyramid sampling, '+
        'activation checkpointing during differentiation. These are full completed runs, counted once per directory and seed.', '',
        'Elapsed time covers the initial sample and optimization, including in-loop objective measurements and prefix saves. '+
        'It excludes model loading, final audits/serialization, gallery rendering and independent physics replay. '+
        'Actual model calls include backward recomputation; forward NFE alone understates this cost. '+
        'Peak memory is PyTorch allocated memory, including resident weights, rather than total GPU process memory.', '',
        '| Run | Horizon / windows | Method / updates | Forward NFE | Actual model calls | Minutes | Peak allocated GiB |',
        '|---|---:|---|---:|---:|---:|---:|']
    for row in unique.values():
        entry=row['entry'];directory=Path(entry['run']);seed=entry.get('seed',8311)
        identity=(str(directory),seed)
        if identity in seen or not (directory/'results.json').exists():continue
        seen.add(identity);protocol=json.loads((directory/'protocol.json').read_text())
        if protocol.get('audit_adaptation'):continue
        record=next(v for v in json.loads((directory/'results.json').read_text()) if v['seed']==seed)
        args=protocol['arguments'];memory=record.get('max_allocated_bytes')
        rows.append(dict(run=str(directory),seed=seed,horizon=protocol['sampling']['horizon'],windows=len(record['windows']),
            method=args['method'],updates=args['iterations'],forward_nfe=record['forward_nfe'],
            actual_model_calls=record['actual_model_calls_including_recomputation'],seconds=record['seconds'],
            peak_allocated_gib=memory/2**30 if memory is not None else None))
        r=rows[-1]
        lines.append(f'| [{directory.name}]({directory.relative_to(root)}/results.json) | '+
            f'{r["horizon"]} / {r["windows"]} | {r["method"]} / {r["updates"]} | '+
            f'{r["forward_nfe"]:,} | {r["actual_model_calls"]:,} | {r["seconds"]/60:.2f} | '+
            (f'{r["peak_allocated_gib"]:.2f}' if memory is not None else '—')+' |')
    lines+=['','Saved 24-update prefixes do not count as separate full runs. '+
        'The 68-frame horizon adds physical duration and latent variables as well as compute. '+
        'Selection-plus-refinement also spends evaluations generating its candidate bank; it is not matched to a 64-update single-start run.', '',
        '[Numeric measurements](compute_summary.json) · [Outcomes](SCORECARD.md)','']
    (root/'compute_summary.json').write_text(json.dumps(dict(rows=rows),indent=2,allow_nan=False)+'\n')
    (root/'COMPUTE.md').write_text('\n'.join(lines))


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.8,7))
    seen=set();values=[]
    colors={'bottle':'#d49315','teacup':'#267da8','spice':'#579842','bowl':'#8756b1'}
    names={'bottle':'Bottle','teacup':'Teacup','spice':'Spice jar','bowl':'Bowl'}
    annotations={'bottle_8270_left_i64_r050':'Bottle: 64 / cap 0.50',
        'bottle_8270_left_long_h68_i64_r050':'Bottle: longer horizon',
        'teacup_7803_left_i64_r050':'Teacup: 64 / cap 0.50'}
    for row in rows:
        if row['replay_held_mm'] is None:continue
        x,y=row['model_held_mm'],row['replay_held_mm']
        if not np.isfinite([x,y]).all():raise ValueError('Nonfinite scorecard metric')
        kind=row['key'].split('_')[0];color=colors.get(kind,'#777777');name=names.get(kind,kind)
        marker='s' if 'obstacle' in row['key'] else '^' if 'no_physical' in row['key'] else 'o'
        ax.scatter(x,y,c=color,s=48,marker=marker,alpha=.75,edgecolors='white',linewidth=.5,label=name if kind not in seen else None)
        seen.add(kind);values.extend([x,y])
        if row['key'] in annotations:ax.annotate(annotations[row['key']],(x,y),xytext=(9,-15 if y>150 else 10),textcoords='offset points',fontsize=8)
    limit=max(240,max(values,default=0)*1.15)
    ax.set_xscale('symlog',linthresh=10);ax.set_yscale('symlog',linthresh=10)
    ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Worst held-position error in model (mm)',
        ylabel='Worst held-position error in physics (mm)',title='Predicted goal reaching versus executed goal reaching')
    ticks=[v for v in [0,5,20,50,100,200,500,1000] if v<=limit]
    ax.set_xticks(ticks,labels=[str(v) for v in ticks]);ax.set_yticks(ticks,labels=[str(v) for v in ticks])
    ax.axvline(20,color='#3a975b',ls='--',lw=1);ax.axhline(20,color='#3a975b',ls='--',lw=1)
    ax.fill_between([0,20],0,20,color='#79bc89',alpha=.13)
    ax.plot([0,limit],[0,limit],color='#999999',ls=':',lw=.8)
    ax.grid(alpha=.16);ax.legend(loc='lower right',fontsize=9)
    fig.text(.5,.015,'Position only. Lower-left means both hit the held-position tolerance; physical checks still apply.\nCircles: standard comparisons; squares: obstacle worlds; triangles: physical penalties disabled.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1));(root/'figures').mkdir(exist_ok=True)
    fig.savefig(root/'figures/model_vs_execution_goals.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
