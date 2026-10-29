"""Inspect saved prior banks and K-dependent selection without resampling."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from guidance.pick_validation import selection_key


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--bank',default='old40k_bank16_v6')
    p.add_argument('--out',required=True)
    args=p.parse_args(); root=Path(args.root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    recorded={row['name']:row for row in json.loads((root/'recorded_development_validation.json').read_text())}
    scenes=sorted(root.glob('development_*'))
    candidates,curves,receipts=[],[],[]
    sampling=None
    for scene in scenes:
        bank=scene/args.bank; protocol=json.loads((bank/'protocol.json').read_text())
        if sampling is not None and protocol['sampling']!=sampling:
            raise ValueError('Mixed sampling protocols in one prior report')
        sampling=protocol['sampling']
        if protocol['arguments']['method']!='best_of_k' or protocol['arguments']['objective']!='goal':
            raise ValueError('Use an unguided bank scored by final goal distance')
        seeds=[int(s) for s in protocol['arguments']['seeds'].split(',')]
        for seed in seeds:
            result_path=bank/f'seed_{seed}.json'
            result=json.loads(result_path.read_text()); trace=result['trace']
            if len(trace)!=protocol['arguments']['k'] or [r['draw'] for r in trace]!=list(range(len(trace))):
                raise ValueError(f'Incomplete or reordered bank: {result_path}')
            receipts.append(dict(scene=scene.name,seed=seed,protocol=str(bank/'protocol.json'),
                protocol_sha256=hashlib.sha256((bank/'protocol.json').read_bytes()).hexdigest(),
                result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),
                checkpoint_sha256=protocol['checkpoint_sha256'],input_sha256=protocol['input_sha256']))
            for row in trace:
                v=row['validation']; r=v['residuals_m']
                candidates.append(dict(scene=scene.name,seed=seed,draw=row['draw'],
                    final_goal_mm=1000*row['loss'],contact_mm=1000*(r['contact']+.005),
                    penetration_mm=max(0.,1000*(r['penetration']+.001)),
                    opening_violation_mm=max(0.,1000*r['opening']),
                    interaction_accepted=v['interaction_accepted'],accepted=v['accepted']))
            for k in range(1,len(trace)+1):
                for policy,key in [('Goal only',lambda r:(r['loss'],)),
                                   ('Interaction first',lambda r:selection_key(r['loss'],r['validation']))]:
                    selected=min(trace[:k],key=key); v=selected['validation']
                    curves.append(dict(scene=scene.name,seed=seed,k=k,policy=policy,draw=selected['draw'],
                        final_goal_mm=1000*selected['loss'],
                        interaction_violation_mm=1000*v['max_interaction_violation_m'],
                        interaction_accepted=v['interaction_accepted'],accepted=v['accepted']))
    hashes={r['checkpoint_sha256'] for r in receipts}
    if len(hashes)!=1: raise ValueError('One checkpoint per prior report')
    title=(f'{args.bank} · checkpoint {next(iter(hashes))[:12]} · '
           f'F{sampling["future_len"]} / H{sampling["horizon"]} / {sampling["schedule"]}{sampling["steps"]}')
    fig,axes=plt.subplots(len(scenes),3,figsize=(13,3.1*len(scenes)),squeeze=False)
    columns=[('contact_mm','Worst held finger gap (mm)',5),
             ('penetration_mm','Maximum sampled intersection (mm)',1),
             ('opening_violation_mm','Opening violation after 0.1 mm slack (mm)',0)]
    for i,scene in enumerate(scenes):
        data=[r for r in candidates if r['scene']==scene.name]; v=recorded[scene.name]['residuals_m']
        reference=[1000*(v['contact']+.005),max(0.,1000*(v['penetration']+.001)),max(0.,1000*v['opening'])]
        for j,(key,label,threshold) in enumerate(columns):
            ax=axes[i,j]; values=np.sort([r[key] for r in data])
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color='#287fb8',label='Generated draws')
            ax.axvline(threshold,color='.3',ls='--',label='Provisional limit')
            ax.axvline(reference[j],color='#198754',label='Recorded interaction')
            ax.set(xlabel=label,ylim=(0,1.02)); ax.grid(alpha=.2)
            if j==0:
                passed=sum(r['interaction_accepted'] for r in data)
                ax.set_ylabel(f'{scene.name}\nFraction of {len(data)} draws')
                ax.set_title(f'Held interaction screen: {passed}/{len(data)}')
        if i==0: axes[i,2].legend(fontsize=8,loc='lower right')
    fig.suptitle(title+'\nA non-picking future can fail the required held-contact screen without being physically invalid.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(out/'prior_geometry.png',dpi=150); plt.close(fig)

    fig,axes=plt.subplots(len(scenes),2,figsize=(11,3.1*len(scenes)),squeeze=False)
    for i,scene in enumerate(scenes):
        data=[r for r in curves if r['scene']==scene.name]
        for policy,color in [('Goal only','#da7133'),('Interaction first','#287fb8')]:
            selected=[r for r in data if r['policy']==policy]
            for j,(key,label) in enumerate([('interaction_violation_mm','Worst interaction residual (mm)'),('final_goal_mm','Final object target error (mm)')]):
                ax=axes[i,j]
                for seed in sorted({r['seed'] for r in selected}):
                    rows=[r for r in selected if r['seed']==seed]
                    ax.plot([r['k'] for r in rows],[r[key] for r in rows],c=color,alpha=.25,lw=1)
                ks=sorted({r['k'] for r in selected})
                ax.plot(ks,[np.median([r[key] for r in selected if r['k']==k]) for k in ks],c=color,lw=2,label=policy)
                ax.set(xlabel='Candidates in the same ordered bank (K)',ylabel=label,xticks=[1,4,8,12,16]); ax.grid(alpha=.2)
        axes[i,0].set_title(scene.name); axes[i,1].legend(fontsize=8)
    fig.suptitle(title+'\nFaint: individual seed banks. Bold: median across banks. These curves reuse candidates; K is not wall time.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(out/'selection_k.png',dpi=150); plt.close(fig)
    for name,rows in [('candidates',candidates),('selection_k',curves)]:
        with (out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'report.json').write_text(json.dumps(dict(bank=args.bank,receipts=receipts,
        candidates=candidates,selection=curves,interpretation='Geometry proxies, not simulator execution. Recorded trajectories follow different goals.'),indent=2)+'\n')


if __name__=='__main__': main()
