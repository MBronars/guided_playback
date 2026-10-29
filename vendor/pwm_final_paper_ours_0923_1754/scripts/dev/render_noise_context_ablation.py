"""Matched H32 full-versus-detached gradient comparisons on completed scenes."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root.resolve()
    out=root/'context_gradient_report_v1';out.mkdir(exist_ok=True)
    pairs={}
    for path in sorted(root.glob('packet_*.json')):
        if path.stem.endswith('_status'):continue
        for entry in json.loads(path.read_text())['entries']:
            if entry['arm'] not in ('goal6_h32','goal6_h32_detach'):continue
            directory=Path(entry['out'])
            for result in directory.glob('seed_*.json'):
                if '_trace' in result.stem:continue
                seed=json.loads(result.read_text())['seed']
                pairs.setdefault((entry['scene'],seed),{})[entry['arm']]=directory
    rows,sections=[],[]
    for (scene,seed),pair in sorted(pairs.items()):
        if len(pair)!=2:continue
        dirs=[pair[k] for k in ('goal6_h32','goal6_h32_detach')]
        records=[json.loads((d/f'seed_{seed}.json').read_text()) for d in dirs]
        protocols=[json.loads((d/'protocol.json').read_text()) for d in dirs]
        saved=[torch.load(d/f'seed_{seed}.pt',weights_only=False,map_location='cpu') for d in dirs]
        arrays=[np.load(d/f'seed_{seed}.npz') for d in dirs]
        checks={k:torch.equal(saved[0]['initial'][k],saved[1]['initial'][k]) for k in ('positions','R','T')}
        checks['initial_noise']=all(torch.equal(a,b) for w0,w1 in zip(saved[0]['initial']['noise'],saved[1]['initial']['noise']) for a,b in zip(w0,w1))
        checks['target']=protocols[0]['target']==protocols[1]['target']
        checks['checkpoint']=protocols[0]['checkpoint_sha256']==protocols[1]['checkpoint_sha256']
        checks['sampling']={k:v for k,v in protocols[0]['sampling'].items() if k!='detach_context'}=={k:v for k,v in protocols[1]['sampling'].items() if k!='detach_context'}
        if not all(checks.values()):raise ValueError(f'Unmatched context-gradient comparison: {scene} {checks}')
        C=int(arrays[0]['context_len']);body=arrays[0]['particle_body'][0];target=arrays[0]['target'][0]
        tracks=[arrays[0]['initial'][0,C-1:],arrays[0]['positions'][0,C-1:],arrays[1]['positions'][0,C-1:]]
        pose_tracks=[arrays[0]['initial_T'][0,C-1:,0],arrays[0]['T'][0,C-1:,0],arrays[1]['T'][0,C-1:,0]]
        selected=np.concatenate([np.where(body==k)[0][::max(1,int((body==k).sum())//130)] for k in np.unique(body)])
        colors=np.array(['#e69f00','#888888','#0072b2','#009e73'])[body[selected]]
        cloud=np.concatenate([t[:,selected].reshape(-1,3) for t in tracks]+[target[None]])
        lo,hi=cloud.min(0),cloud.max(0);center=(lo+hi)/2;span=max(float((hi-lo).max())*1.1,.2)
        tag=f'{scene}_seed{seed}';movie=out/(tag+'.gif');still=out/(tag+'_final.png');chart=out/(tag+'_traces.png')
        labels=['Identical ordinary sample','Final goal adjusts both windows','Final goal adjusts window 2 only']
        if not movie.exists():
            fig=plt.figure(figsize=(13,5));artists=[];texts=[]
            for i,track in enumerate(tracks):
                ax=fig.add_subplot(1,3,i+1,projection='3d')
                artists.append(ax.scatter(*track[0,selected].T,c=colors,s=7))
                ax.scatter(*target,c='#cc0077',marker='x',s=60)
                ax.set(xlim=(center[0]-span/2,center[0]+span/2),ylim=(center[1]-span/2,center[1]+span/2),zlim=(center[2]-span/2,center[2]+span/2),xticks=[],yticks=[],zlabel='height (m)',title=labels[i])
                ax.view_init(22,-65);ax.set_box_aspect((1,1,1))
                texts.append(ax.text2D(.1,-.08,'',transform=ax.transAxes,fontsize=10))
            title=fig.suptitle('');fig.subplots_adjust(left=.02,right=.97,bottom=.13,top=.77,wspace=.08)
            def update(frame):
                for i,(artist,track) in enumerate(zip(artists,tracks)):
                    artist._offsets3d=tuple(track[frame,selected].T)
                    texts[i].set_text(f'Goal distance: {np.linalg.norm(pose_tracks[i][frame]-target)*1000:.1f} mm')
                title.set_text(f'{scene} · seed{seed} · {frame*.05:.2f}s\nTwo 16-frame windows; boundary at0.8s. Six noise updates. Orange: object; pink ×: goal.')
                return artists+texts+[title]
            FuncAnimation(fig,update,frames=len(tracks[0]),interval=85).save(movie,writer=PillowWriter(fps=12))
            update(len(tracks[0])-1);fig.savefig(still,dpi=140);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,7));methods=[]
        for d,record,value,label,color in zip(dirs,records,saved,labels[1:],['#0072b2','#d55e00']):
            trace=record['trace'];iterations=[v['iteration'] for v in trace]
            axes[0,0].plot(iterations,[v['target_distance_mm'] for v in trace],marker='o',label=label,c=color)
            axes[0,1].plot(np.arange(33)*.05,np.linalg.norm(value['result']['rollout']['T'][0,C-1:,0].numpy()-target,axis=-1)*1000,label=label,c=color)
            for window,ax in enumerate(axes[1]):
                ax.plot(iterations,[v['noise_rms_by_window'][window] for v in trace],marker='o',label=label,c=color)
            audit=json.loads((d/f'mesh_audit_v2_{seed}.json').read_text()) if (d/f'mesh_audit_v2_{seed}.json').exists() else {}
            methods.append(dict(run=d.name,label=label,final_goal_mm=record['selected']['target_distance_mm'],initial_gradient_by_window=trace[0]['gradient_norm_by_window'],final_update_rms_by_window=trace[-1]['noise_rms_by_window'],selected_iteration=value['result']['iteration'],model_calls=record['actual_model_calls_including_recomputation'],mesh_pass=audit.get('selected',{}).get('validation',{}).get('accepted')))
        for ax,title,ylabel in zip(axes.flat,['Goal distance during optimization','Goal distance along selected trajectory','Window 1 noise change','Window 2 noise change'],['mm','mm','RMS normalized noise','RMS normalized noise']):
            ax.set(title=title,ylabel=ylabel,xlabel='seconds' if ax is axes[0,1] else 'noise updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0,1].axvline(.8,c='#777777',ls='--',lw=1)
        fig.suptitle(scene+' · matched H32 gradient ablation');fig.tight_layout();fig.savefig(chart,dpi=140);plt.close(fig)
        rows.append(dict(scene=scene,seed=seed,pairing_checks=checks,methods=methods))
        table='<table><tr><th>Gradient route</th><th>Final goal mm</th><th>Initial gradient norm: windows1 /2</th><th>Actual model calls</th><th>Shared mesh task screen</th></tr>'
        for m in methods:table+=f'<tr><td>{m["label"]}</td><td>{m["final_goal_mm"]:.2f}</td><td>{m["initial_gradient_by_window"][0]:.5f} / {m["initial_gradient_by_window"][1]:.5f}</td><td>{m["model_calls"]}</td><td>{m["mesh_pass"]}</td></tr>'
        sections.append(f'<h2>{html.escape(scene)} · seed{seed}</h2><img src="{movie.name}">{table}</table><img src="{chart.name}">')
        print(tag,flush=True)
    header='''<!doctype html><meta charset="utf-8"><title>Guidance across two windows</title><style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1450px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>Can the final goal improve earlier actions?</h1><p>Both methods generate32 future frames in two16-frame windows. With full gradients, the final goal can adjust noise in both windows. With the gradient stopped at the boundary, the final goal adjusts only the second window. The forward sampler, starting noise, target and weights are identical.</p><p>This tests how optimization assigns credit across windows. It does not add observations or close the control loop. Smaller endpoint error does not establish a held goal, good contact or executed success; the independent mesh result remains alongside each comparison. H16 and H36 runs have different durations and are not part of this matched comparison.</p><p><a href="../index.html">All experiments</a> · <a href="summary.json">Pairing checks and exact results</a></p>'''
    (out/'summary.json').write_text(json.dumps(dict(results=rows),indent=2)+'\n')
    (out/'index.html').write_text(header+''.join(sections))


if __name__=='__main__':main()
