"""Paired real-sample input sensitivity, with noise/goal identity checks."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); root=Path(a.root); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    control=root/'inpaint_actuated_p0'; specs=[('translation',[(0,None),(1,'translation_1mm'),(3,'translation_3mm'),(5,'translation_5mm')]),
        ('rotation',[(0,None),(1,'rotation_1deg'),(3,'rotation_3deg'),(5,'rotation_5deg')]),
        ('density',[(-25,'density_075'),(-10,'density_090'),(0,None),(10,'density_110'),(25,'density_125')])]
    # The original control precedes the bias CLI. Verify that every model,
    # sampler and waypoint implementation used for it is byte-identical.
    files=['guidance/sampler.py','guidance/unconditional.py','guidance/waypoints.py','models/unet_std.py',
        'utils/pose_native.py','utils/conditioning.py','inference.py','data/batch.py']
    for name in files:
        if (root/'source_v4'/name).read_bytes()!=(root/'source_v6'/name).read_bytes():
            raise ValueError('Control and perturbation core implementations differ: '+name)
    rows=[]; checks=[]
    for seed in (54321,54322):
        baseline=np.load(control/f'seed_{seed}.npz'); base_saved=torch.load(control/f'seed_{seed}.pt',map_location='cpu',weights_only=False)
        baseline_protocol=json.loads((control/'protocol.json').read_text())
        for kind,levels in specs:
            for level,name in levels:
                d=control if name is None else root/'input_sensitivity'/name
                result=json.loads((d/f'seed_{seed}.json').read_text()); v=result['validation']; m=result['selected']
                data=np.load(d/f'seed_{seed}.npz'); protocol=json.loads((d/'protocol.json').read_text())
                assert protocol['checkpoint_sha256']==baseline_protocol['checkpoint_sha256']
                assert protocol['input_sha256']==baseline_protocol['input_sha256']
                np.testing.assert_array_equal(data['target'],baseline['target'])
                saved=base_saved if name is None else torch.load(d/f'seed_{seed}.pt',map_location='cpu',weights_only=False)
                for left,right in zip(base_saved['initial']['noise'],saved['initial']['noise']):
                    for x,y in zip(left,right): torch.testing.assert_close(x,y,atol=0,rtol=0)
                checks.append(dict(kind=kind,level=level,seed=seed,same_checkpoint_input_goal_and_drawn_noise=True))
                C=int(data['context_len'])
                change=np.linalg.norm(data['T'][0,C:,0]-baseline['T'][0,C:,0],axis=-1)*1000
                rows.append(dict(kind=kind,level=level,seed=seed,max_object_path_change_mm=float(change.max()),
                    final_object_change_mm=float(change[-1]),final_goal_mm=m['target_distance_mm'],
                    hold_goal_max_mm=(v['residuals_m']['goal']+.02)*1000,
                    hold_contact_gap_max_mm=(v['residuals_m']['contact']+.005)*1000,
                    intersection_max_mm=max(m['gripper_in_object_max_mm'],m['object_in_gripper_hull_max_mm']),
                    interaction_pass=v['interaction_accepted'],task_and_interaction_pass=v['accepted']))
    (out/'sensitivity.json').write_text(json.dumps(dict(rows=rows,checks=checks,
        baseline=str(control),protocol='One jar; fixed task goal and Gaussian draws; input bias can change the prescribed actuator path; execution unverified'),indent=2)+'\n')
    with (out/'sensitivity.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig,axes=plt.subplots(3,3,figsize=(13,9))
    for col,((kind,_),xlabel) in enumerate(zip(specs,('Translation bias (mm)','Rotation bias (degrees)','Density estimate error (%)'))):
        for row,(key,ylabel) in enumerate([('max_object_path_change_mm','Largest object-path change\nrelative to clean input (mm)'),
                ('hold_contact_gap_max_mm','Worst hold finger gap (mm)'),('intersection_max_mm','Peak intersection (mm)')]):
            ax=axes[row,col]
            for seed in (54321,54322):
                values=[r for r in rows if r['seed']==seed and r['kind']==kind]
                ax.plot([r['level'] for r in values],[r[key] for r in values],'o-',label=str(seed))
            ax.grid(alpha=.2); ax.set_xlabel(xlabel)
            if col==0: ax.set_ylabel(ylabel)
            if row==1: ax.axhline(5,c='.5',ls='--',lw=1)
            if row==2: ax.axhline(1,c='.5',ls='--',lw=1)
            if row==0: ax.set_title(kind.capitalize()+' input bias')
    axes[0,2].legend(title='Paired sample seed',fontsize=8)
    fig.suptitle('Imperfect-state planning sensitivity · prescribed actuator path + generated object',fontsize=13)
    fig.text(.5,.01,'One jar and two seeds; the clean baseline already fails geometric acceptance. These are sensitivity measurements, not robust execution rates.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96)); fig.savefig(out/'sensitivity.png',dpi=140); plt.close(fig)


if __name__=='__main__': main()
