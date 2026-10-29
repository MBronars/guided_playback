"""One paired scene matrix for the primary precise-target guidance experiment."""
import argparse
from datetime import datetime,timezone
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve();out=root/'primary_overview_v1';out.mkdir(exist_ok=True)
    scenes=json.loads((root/'scenes_v2.json').read_text())['scenes']
    scenes.sort(key=lambda s:(s['task']!='pick',s['group']!='development',s['name']))
    summary=json.loads((root/'summary.json').read_text())
    prefixes=json.loads((root/'prefix_mesh_summary_v3.json').read_text())['results']
    replays=json.loads((root/'generated_replay_summary_v2.json').read_text())['results']
    replay_map={(v['run'],v['seed'],v['variant']):v for v in replays}
    selected={}
    for row in summary['trials']:
        if row['seed']!=8311 or row['arm'] not in ('goal6_h36','physical6_h36','goal24_h36','physical24_h36'):continue
        key=(row['scene'],row['arm'])
        if key in selected and selected[key]['link']!=row['link']:raise ValueError(f'Ambiguous primary trial: {key}')
        selected[key]=row
    columns=['Initial sample','Goal · 6','Physical · 6','Goal · 24','Physical · 24'];rows=[]
    expected_checkpoint=None;expected_sampling=None
    for scene in scenes:
        name=scene['name'];initial=selected.get((name,'goal6_h36'));reference=None
        if initial is not None:reference=np.load(root/initial['link']/'seed_8311.npz')
        for index,arm in enumerate([None,'goal6_h36','physical6_h36','goal24_h36','physical24_h36']):
            item=initial if index==0 else selected.get((name,arm));prefix=None
            if index>=3 and item is None:
                matches=[v for v in prefixes if v['scene']==name and v['seed']==8311 and v['updates']==24 and v['arm']==arm.replace('24','100')]
                if len(matches)>1:raise ValueError(f'Ambiguous 24-update prefix: {name} {arm}')
                prefix=matches[0] if matches else None
            row=dict(scene=name,task=scene['task'],group=scene['group'],method=columns[index],column=index,seed=8311,status='pending')
            if item is None and prefix is None:rows.append(row);continue
            run=prefix['run'] if prefix is not None else Path(item['link']).name
            protocol=json.loads((root/run/'protocol.json').read_text())
            if expected_checkpoint is None:
                expected_checkpoint=protocol['checkpoint_sha256'];expected_sampling=protocol['sampling']
            if protocol['checkpoint_sha256']!=expected_checkpoint or protocol['sampling']!=expected_sampling:
                raise ValueError(f'Primary checkpoint or sampling protocol changed: {run}')
            values=np.load(root/run/'seed_8311.npz')
            paired=bool(reference is not None and np.array_equal(values['initial'],reference['initial']) and np.array_equal(values['initial_T'],reference['initial_T']) and np.array_equal(values['target'],reference['target']))
            if not paired:raise ValueError(f'Primary initial trajectory or target is not exactly paired: {run}')
            variant='through_0024' if prefix is not None else ('initial' if index==0 else 'selected')
            if prefix is not None:
                error=prefix['goal_error_mm'];mesh=prefix;validation=prefix['validation']
                source=root/run/'seed_8311_iterates/through_0024_mesh_audit_v3.json'
            else:
                source=root/run/'mesh_audit_v2_8311.json';audit=json.loads(source.read_text()) if source.exists() else None
                error=item['initial_error_mm'] if index==0 else item['selected_error_mm']
                mesh=audit['initial' if index==0 else 'selected'] if audit else {};validation=mesh.get('validation',{})
            replay=replay_map.get((run,8311,variant))
            row.update(status='completed',run=run,variant=variant,goal_mm=error,penetration_mm=mesh.get('penetration_max_mm'),mesh_pass=validation.get('accepted'),paired_initial_and_target=True,mesh_source=str(source),qualified_execution_pass=replay.get('qualified_task_pass') if replay else None,control_qualified=replay.get('control_qualified') if replay else None,timestep_stable=replay.get('timestep_stable') if replay else None,execution_pending=replay is None)
            rows.append(row)
    fig,axes=plt.subplots(1,2,figsize=(12,10.5),sharey=True)
    labels=[s['name'].replace('_development_',' · dev ').replace('_reserved_',' · reserved ').replace('_asset_',' / ') for s in scenes]
    for ax,key,title,maximum in zip(axes,['goal_mm','penetration_mm'],['Final point-goal distance (mm)','Largest model mesh penetration (mm)'],[400,40]):
        data=np.full((len(scenes),len(columns)),np.nan)
        for row in rows:
            if row.get(key) is not None:data[[s['name'] for s in scenes].index(row['scene']),row['column']]=max(row[key],.01)
        cmap=plt.get_cmap('YlOrRd').copy();cmap.set_bad('#e6e6e6')
        ax.imshow(data,cmap=cmap,norm=LogNorm(vmin=.1,vmax=maximum),aspect='auto')
        for i in range(len(scenes)):
            for j in range(len(columns)):
                row=rows[i*len(columns)+j];value=row.get(key)
                label='pending' if value is None else f'{value:.1f}'
                ax.text(j,i,label,ha='center',va='center',fontsize=8,color='white' if value is not None and value>maximum*.35 else '#222222')
        ax.set(xticks=range(len(columns)),xticklabels=columns,yticks=range(len(scenes)),yticklabels=labels,title=title)
        ax.tick_params(axis='x',rotation=25)
        for y in [3.5,7.5,11.5]:ax.axhline(y,color='white',lw=2)
    fig.suptitle('Same initial draw and target for each scene · seed 8311\n36 future frames (1.8 s), fixed checkpoint, two objectives and update budgets',fontsize=12)
    fig.text(.02,.025,'Lower numbers are better; gray means pending. Point error alone does not check held goal, orientation, contact or release.\nThe first can and bowl use the retained 24-update prefix of their 100-update runs. These are model predictions, not execution success.',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(out/'primary_goal_and_collision.png',dpi=140);plt.close(fig)
    groups=[]
    for task in ['pick','place']:
        for group in ['development','reserved']:
            for column,label in enumerate(columns):
                subset=[v for v in rows if v['task']==task and v['group']==group and v['column']==column];done=[v for v in subset if v['status']=='completed'];mesh=[v for v in done if v['mesh_pass'] is not None];executed=[v for v in done if not v['execution_pending']]
                groups.append(dict(task=task,group=group,method=label,expected_scenes=len(subset),completed=len(done),median_goal_mm=float(np.median([v['goal_mm'] for v in done])) if done else None,mesh_checked=len(mesh),mesh_passes=sum(v['mesh_pass'] for v in mesh),executed=len(executed),qualified_execution_passes=sum(v['qualified_execution_pass'] is True for v in executed),control_qualified=sum(v['control_qualified'] is True for v in executed)))
    payload=dict(updated_utc=datetime.now(timezone.utc).isoformat(),checkpoint_sha256=expected_checkpoint,sampling=expected_sampling,seed=8311,rows=rows,groups=groups,note='Guidance-reserved scenes are not asserted unseen in checkpoint training. 24-update prefixes are not 100-update outcomes. Checkpoint/sampler metadata agree; paired initial full particle trajectory, translations and target verified bitwise. Independent floating-hand execution is qualified per scene and two timesteps; missing audits remain pending.')
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    table='<table><tr><th>Task / split / method</th><th>Complete / expected</th><th>Median final error</th><th>Full mesh task passes / audited</th><th>Qualified physics passes / replayed</th><th>Recorded control qualified / replayed</th></tr>'
    for g in groups:
        distance='pending' if g['median_goal_mm'] is None else f'{g["median_goal_mm"]:.1f} mm'
        table+=f'<tr><td>{g["task"]} / {g["group"]} / {g["method"]}</td><td>{g["completed"]}/{g["expected_scenes"]}</td><td>{distance}</td><td>{g["mesh_passes"]}/{g["mesh_checked"]}</td><td>{g["qualified_execution_passes"]}/{g["executed"]}</td><td>{g["control_qualified"]}/{g["executed"]}</td></tr>'
    page='''<!doctype html><meta charset="utf-8"><title>Paired primary guidance results</title><style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1450px}img{max-width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>What improved across the fixed pick/place scenes?</h1><p>Each row uses the same initial generated trajectory and requested target. The comparison changes only the objective or number of noise updates. There are eight pick and eight place scenes; development and guidance-reserved groups stay separate. Reserved does not mean unseen during checkpoint training.</p><p>This page uses seed8311 for every column. The separate seed8312 six-update trials remain in the full report. The first can and bowl's 24-update results come from saved best-so-far prefixes of their longer runs. Missing trials or execution audits are pending, never counted as successful.</p><p>Final point accuracy and mesh overlap are separate measurements. A complete task additionally requires a sustained goal, physical interaction, joint travel, and placing orientation/settling/release. Physics success here means passing the fixed screens, hand tracking and timestep checks under a scene-qualified independent floating-hand replay. It is not robot or source-simulator success.</p><p><a href="../index.html">All trials and movies</a> · <a href="../feedback_replay_report_v1/index.html">Executed feedback comparisons</a> · <a href="summary.json">Exact sources and paired checks</a></p><img src="primary_goal_and_collision.png">'''
    (out/'index.html').write_text(page+table+'</table><p>Updated '+html.escape(payload['updated_utc'])+'</p>')
    print('Primary cells:',sum(v['status']=='completed' for v in rows),'/',len(rows))


if __name__=='__main__':main()
