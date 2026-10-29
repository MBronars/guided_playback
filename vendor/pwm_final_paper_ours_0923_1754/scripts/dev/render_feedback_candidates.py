"""Show candidate coverage and learned-cost ranking without promoting an oracle."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve();out=root/'feedback_candidates_report_v1';out.mkdir(exist_ok=True)
    sections=[];summaries=[]
    for path in sorted(root.glob('*/candidate_physics_v1/report.json')):
        report=json.loads(path.read_text());rows=report['results'];run=root/report['run'];driver=json.loads((run/'report.json').read_text())
        initial=rows[0];selected=rows[report['selected_by_model']]
        cost=np.array([r['model_cost'] for r in rows]);goal=np.array([[s['held_goal_violation_mm'] for s in r['replays']] for r in rows])
        violation=np.array([[s['continuation_validation']['max_violation_m']*1000 for s in r['replays']] for r in rows])
        a,z=[np.load(Path(driver['arguments'][key])/'execution.npz') for key in ('replay','fine_replay')]
        prefix=driver['prefix_frames']+1
        prefix_position=float(np.linalg.norm(a['model_T'][:prefix,0]-z['model_T'][:prefix,0],axis=-1).max()*1000)
        relative=a['model_R'][:prefix,0]@z['model_R'][:prefix,0].swapaxes(-1,-2)
        prefix_rotation=float(np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1))).max())
        inherited_stability=prefix_position<=5 and prefix_rotation<=5
        stable=np.array([r['timestep_stable'] for r in rows]);complete=np.array([r['qualified_full_task_pass'] for r in rows])
        continuation=np.array([r['timestep_stable'] and report['control_qualified'] and
            all(s['full_tracking_ok'] and s['continuation_validation']['accepted'] for s in r['replays']) for r in rows])
        figure=report['run']+'.png';fig,axes=plt.subplots(1,2,figsize=(12,4.7))
        for ax,values,title in zip(axes,[goal,violation],['Executed held-goal violation','Largest remaining-segment residual']):
            for i,(c,v) in enumerate(zip(cost,values)):
                ax.plot([c,c],v,color='#aaaaaa',alpha=.5,lw=.8)
                ax.scatter(c,v[0],c='#0072b2' if stable[i] else '#888888',s=24,marker='o')
                ax.scatter(c,v[1],c='#0072b2' if stable[i] else '#888888',s=24,marker='x')
            chosen=report['selected_by_model'];ax.scatter([cost[chosen]]*2,values[chosen],marker='*',s=190,facecolors='none',edgecolors='#d55e00',linewidths=1.5,label='Chosen by model cost')
            ax.scatter([cost[0]]*2,values[0],marker='s',s=85,facecolors='none',edgecolors='#cc0077',linewidths=1.5,label='Initial fresh draw')
            ax.axhline(0,color='#555555',ls=':',lw=1)
            if (cost>0).all():
                ax.set_xscale('log');ax.set_xlim(cost.min()*.75,cost.max()*1.4)
            else:
                ax.set_xscale('symlog',linthresh=.001);ax.set_xlim(0,max(cost.max()*1.2,.001))
            ax.set(title=title,xlabel='Learned guidance cost (lower is preferred)',ylabel='mm; circle 0.5 ms, cross 0.25 ms');ax.grid(alpha=.2);ax.legend(fontsize=8)
        kind='integration gate' if '_gate_' in report['run'] else 'experiment'
        fig.suptitle(f"{report['scene']} · {report['context_source']} context · {len(rows)} candidates · {kind}\nBlue: timestep-stable; gray: unstable. Complete qualified passes: {complete.sum()} / {len(rows)}.")
        fig.tight_layout(rect=(0,0,1,.86));fig.savefig(out/figure,dpi=140);plt.close(fig)
        record=dict(run=report['run'],kind=kind,k=len(rows),control_qualified=report['control_qualified'],
            timestep_stable_candidates=int(stable.sum()),qualified_full_task_passes=int(complete.sum()),
            continuation_screen_and_tracking_both_steps_stable=int(continuation.sum()),
            continuation_screen_passes_by_step=[sum(r['replays'][i]['continuation_validation']['accepted'] for r in rows) for i in (0,1)],
            inherited_prefix_timestep_stable=inherited_stability,inherited_prefix_position_max_mm=prefix_position,
            inherited_prefix_rotation_max_deg=prefix_rotation,
            selected_draw=report['selected_by_model'],selected_full_pass=selected['qualified_full_task_pass'],
            actual_model_calls=driver['actual_model_calls'],seconds=driver['optimization_seconds'])
        summaries.append(record)
        table='<table><tr><th>Draw</th><th>Model cost</th><th>Final lift, both steps</th><th>Held-goal violation, both steps</th><th>Timestep stable</th><th>Continuation-only screens</th><th>Complete qualified task</th></tr>'
        for row in rows:
            tag=' (model selected)' if row['selected_by_model'] else (' (initial)' if row['draw']==0 else '')
            lift=' / '.join(format(s['object_lift_mm'],'.1f') for s in row['replays'])
            goal_values=' / '.join(format(s['held_goal_violation_mm'],'.1f') for s in row['replays'])
            continuation_values=' / '.join('PASS' if s['continuation_validation']['accepted'] else 'FAIL' for s in row['replays'])
            full_value='PASS' if row['qualified_full_task_pass'] else 'FAIL'
            table+=f'<tr><td>{row["draw"]}{tag}</td><td>{row["model_cost"]:.6g}</td><td>{lift} mm</td><td>{goal_values} mm</td><td>{row["timestep_stable"]}</td><td>{continuation_values}</td><td>{full_value}</td></tr>'
        prefix_note=f'Before replanning, the two physics runs already differ by up to {prefix_position:.2f} mm and {prefix_rotation:.2f}°. '
        if not inherited_stability:prefix_note+='This fixed prefix already fails the full-rollout timestep criterion; a later continuation cannot remove that failure. '
        remaining=sum(all(s['continuation_validation']['accepted'] for s in r['replays']) for r in rows)
        sections.append(f'<section><h2>{html.escape(report["scene"])} · {report["context_source"]} context · {kind}</h2><p>{len(rows)} saved candidates; {complete.sum()} complete qualified passes. Independently of earlier violations, {remaining} candidates pass the remaining-segment task screens at both physics steps.</p><p>{prefix_note}The second plot therefore shows remaining-segment violations separately. {continuation.sum()} candidates also pass full tracking, controller qualification and full-rollout timestep stability.</p><p>Generation cost: {driver["actual_model_calls"]} model calls, {driver["optimization_seconds"]:.1f} seconds. The added physics audit is extra work. <a href="../{report["run"]}/candidate_physics_v1/report.json">All candidate metrics and exact replay checks</a></p><img src="{figure}"><details><summary>Every candidate, including failures</summary>{table}</table></details></section>')
    page='''<!doctype html><meta charset="utf-8"><title>Candidate coverage and selection</title><style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1450px}img{max-width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}section{margin-top:45px}</style><h1>Does the sampler offer usable continuations, and does the model cost select them?</h1><p>Every candidate in each completed best-of-K bank is replayed from the same saved physics state at both integration steps. Only the gripper is commanded; the object moves dynamically. The initial draw and model-selected draw reproduce the original driver bitwise. Candidate identity is never changed based on the audit.</p><p>These are diagnostics with extra simulator access, not a new learned-model planning result. The full task retains all earlier prefix violations. Continuation-only screens describe the remaining segment and cannot repair the past. Residual plots use the validator’s existing metre-equivalent scales, including angular constraints; they are not purely position errors.</p><p><a href="../index.html">Main report</a> · <a href="../feedback_replay_report_v1/index.html">Original selection movies</a> · <a href="../grasp_execution_report_v1/index.html">Why lift alone can mislead</a></p>'''
    (out/'index.html').write_text(page+''.join(sections));(out/'summary.json').write_text(json.dumps(dict(results=summaries),indent=2)+'\n')
    print('Rendered candidate banks:',len(summaries))


if __name__=='__main__':main()
