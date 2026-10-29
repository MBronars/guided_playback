"""Export an auditable table and scientific plot of completed pick diagnostics."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from guidance.pick_validation import PickValidator
from utils.reproducibility import file_sha256


def summarize(specs,out,title='Jar regression diagnostics'):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    rows=[]; pending=[]
    for spec in specs:
        label,directory=spec.split('=',1); d=Path(directory)
        if not (d/'results.json').exists(): pending.append(label); continue
        proto=json.loads((d/'protocol.json').read_text()); args=proto['arguments']
        b=torch.load(args['input'],map_location='cpu',weights_only=False)
        g=torch.load(args['geometry'],map_location='cpu',weights_only=False)
        # Older runs predate joint calibration in their CLI; the supplied common
        # scene calibration is recovered from the summary input only if present.
        grasp_path=args.get('grasp')
        if not grasp_path:
            candidate=Path('validation/guidance_staged_2026-09-14/grasp_scene.pt')
            if str(Path(args['input']).resolve()).endswith('/phantom_regeneration_2026-09-14/pick_input.pt') and candidate.exists():
                grasp_path=str(candidate)
        for result in json.loads((d/'results.json').read_text()):
            seed=result['seed']; m=result['selected']; v=result.get('validation')
            if v is None and grasp_path:
                grasp=torch.load(grasp_path,map_location='cpu',weights_only=False)
                if grasp['metadata']['input_hashes'].get(str(Path(args['input']).resolve()))!=file_sha256(args['input']):
                    raise ValueError('Post-hoc calibration does not match the source input')
                data=np.load(d/f'seed_{seed}.npz'); C=b.context_len
                validator=PickValidator(b,torch.tensor(data['target']),g,grasp['calibration'])
                v=validator(dict(positions=torch.tensor(data['positions'][:,C:]),R=torch.tensor(data['R'][:,C:]),
                    t=torch.tensor(data['T'][:,C:]),rigid_idx=torch.arange(4)))
            row=dict(method=label,seed=seed,final_goal_mm=m['target_distance_mm'],
                hold_goal_max_mm=1000*(v['residuals_m']['goal']+.02) if v else None,
                hold_finger_gap_max_mm=1000*(v['residuals_m']['contact']+.005) if v else None,
                penetration_max_mm=max(m['gripper_in_object_max_mm'],m['object_in_gripper_hull_max_mm']),
                interaction_pass=v['interaction_accepted'] if v else None,
                task_and_interaction_pass=v['accepted'] if v else None,seconds=result['seconds'],
                model_calls=result['actual_model_calls_including_recomputation'],
                checkpoint_sha256=proto['checkpoint_sha256'],protocol=str((d/'protocol.json').resolve()),
                result=str((d/f'seed_{seed}.json').resolve()),offline_reranked='offline_reranking' in proto)
            rows.append(row)
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,pending=pending),indent=2)+'\n')
    if not rows: return
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    labels=list(dict.fromkeys(r['method'] for r in rows)); seeds=sorted(set(r['seed'] for r in rows))
    fig,axes=plt.subplots(1,3,figsize=(15,max(5,.58*len(labels))),sharey=True)
    for ax,key,title,limit in zip(axes,('hold_goal_max_mm','hold_finger_gap_max_mm','penetration_max_mm'),
            ('Worst goal error during hold','Worst finger gap during hold','Worst intersection in trajectory'),(20,5,1)):
        for i,seed in enumerate(seeds):
            subset=[r for r in rows if r['seed']==seed and r[key] is not None]
            x=[r[key] for r in subset]; y=[labels.index(r['method'])+(i-(len(seeds)-1)/2)*.15 for r in subset]
            ax.scatter(x,y,s=35,label=str(seed))
        ax.axvline(limit,c='.4',ls='--',lw=1,label=f'Provisional limit: {limit} mm')
        ax.set_title(title,fontsize=11); ax.set_xlabel('mm'); ax.grid(axis='x',alpha=.2); ax.set_xlim(left=-1)
    axes[0].set_yticks(range(len(labels)),labels); axes[0].invert_yaxis(); axes[2].legend(fontsize=8)
    fig.suptitle(title+' · geometric screening, execution unverified',fontsize=13)
    fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=135); plt.close(fig)
    columns=['method','seed','final_goal_mm','hold_goal_max_mm','hold_finger_gap_max_mm','penetration_max_mm','task_and_interaction_pass','seconds']
    names=['Method','Seed','Final goal (mm)','Worst hold goal (mm)','Worst hold gap (mm)','Peak intersection (mm)','Passes all proxies','Seconds']
    def display(v):
        if v is None: return '—'
        if isinstance(v,bool): return 'Yes' if v else 'No'
        if isinstance(v,float): return f'{v:.1f}'
        return html.escape(str(v))
    table='<tr>'+''.join(f'<th>{n}</th>' for n in names)+'</tr>'
    table+=''.join('<tr>'+''.join('<td>'+display(r[k])+'</td>' for k in columns)+'</tr>' for r in rows)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Pick guidance diagnostics</title>'
        '<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 24px;color:#18202c}table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #dce1e8;text-align:left}th{background:#eef3f8}img{width:100%}p{max-width:1000px;line-height:1.5}</style>'
        '<h1>'+html.escape(title)+'</h1><p>Each row is a saved real-scene model rollout. The hold is the final six frames (0.25 seconds between endpoints). '
        'Both fingers must stay near the object; a low final goal error alone is insufficient. '
        'The displayed thresholds are provisional geometry checks, not simulator success. Repeated seeds from one scene are correlated.</p>'
        '<p>Frozen methods use different compute budgets. Inpainting prescribes actuator motion and leaves the object generated. '
        'Offline re-ranking reuses an existing bank; its listed time is the original bank runtime.</p>'
        '<img src="comparison.png" alt="Per-seed goal, contact, and collision measurements">'
        '<table>'+table+'</table><p>Pending: '+html.escape(', '.join(pending) or 'none in this snapshot')+'</p>'
        '<p><a href="summary.csv">CSV</a> · <a href="summary.json">Full provenance and results</a></p>')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out',required=True)
    p.add_argument('--title',default='Jar regression diagnostics')
    p.add_argument('--runs',nargs='+',required=True); a=p.parse_args(); summarize(a.runs,a.out,a.title)


if __name__=='__main__': main()
