"""Build a local Markdown gallery from declared guidance comparisons and receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from scripts.dev.guidance_report_utils import task_validator, rollout_metrics
from utils.reproducibility import file_sha256


def load_variant(entry, export_root):
    run=Path(entry['run']);seed=entry.get('seed',8311)
    protocol_path=run/'protocol.json'
    if not protocol_path.exists():return None
    protocol=json.loads(protocol_path.read_text())
    if entry.get('evaluation_obstacles'):
        from scripts.dev.guidance_report_utils import obstacle_evaluation_protocol
        protocol=obstacle_evaluation_protocol(protocol,protocol_path,entry['evaluation_obstacles'])
    update=entry.get('updates')
    if update is not None:
        source=run/f'seed_{seed}_iterates'/f'through_{update:04d}.pt'
        marker=source.with_suffix('.json')
    else:
        source=run/f'seed_{seed}.pt';marker=run/'results.json'
    if not marker.exists():return None
    saved=torch.load(source,weights_only=False,map_location='cpu')
    if entry.get('ordinary'):
        rollout=saved['initial'];selected=0
    elif update is not None:
        rollout=saved['rollout'];selected=saved['iteration']
        if saved['evaluated_through_update']!=update:raise ValueError('Wrong optimization prefix')
    else:
        rollout=saved['result']['rollout'];selected=saved['result'].get('iteration')
    expected=entry.get('checkpoint_sha256')
    if expected is not None and protocol['checkpoint_sha256']!=expected:raise ValueError('Checkpoint mismatch')
    batch,geometry,collision,validator=task_validator(protocol)
    metrics=rollout_metrics(rollout,protocol,collision,validator)
    C=batch.context_len
    if not torch.equal(rollout['positions'][:,:C],batch.positions[:,:C]):
        raise ValueError('Rollout changed observed particle context')
    export=export_root/entry['key'];export.mkdir(parents=True,exist_ok=True)
    source_hash=file_sha256(source)
    receipt=dict(entry=entry,source_sha256=source_hash,protocol_sha256=file_sha256(protocol_path),
        selected_iteration=selected,metrics=metrics)
    npz=export/'trajectory.npz'
    if not npz.exists():
        np.savez_compressed(npz,positions=rollout['positions'].numpy(),R=rollout['R'].numpy(),T=rollout['T'].numpy(),
            particle_body=batch.particle_body.numpy(),target=np.asarray(protocol['target']),context_len=C)
        (export/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    else:
        old=json.loads((export/'receipt.json').read_text())
        if old['source_sha256']!=source_hash:raise ValueError('A completed gallery source changed')
        if json.loads((export/'protocol.json').read_text())!=protocol:raise ValueError('Export evaluation protocol changed')
    receipt['trajectory_sha256']=file_sha256(npz)
    (export/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return dict(entry=entry,protocol=protocol,rollout=rollout,batch=batch,geometry=geometry,
                metrics=metrics,selected_iteration=selected,source_sha256=source_hash,export=str(export))


def obstacle_vertices(box):
    import itertools
    corners=np.array(list(itertools.product([-1,1],repeat=3)))*np.asarray(box['size'])/2
    return corners @ np.asarray(box.get('rotation',np.eye(3))).T+np.asarray(box['centre'])


def draw_obstacles(ax,spec):
    if not spec:return
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    for box in spec['boxes']:
        v=obstacle_vertices(box)
        faces=[v[np.where(np.array([[(i>>2)&1,(i>>1)&1,i&1] for i in range(8)])[:,a]==s)[0]][[0,1,3,2]] for a in range(3) for s in range(2)]
        ax.add_collection3d(Poly3DCollection(faces,facecolor='#b74747',edgecolor='#752b2b',alpha=.17,linewidth=.7))


def render_group(group,variants,root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    available=[v for v in variants if v is not None]
    if not available:return None
    signature=hashlib.sha256(json.dumps(dict(renderer_version=2,group=group,
        sources=[None if v is None else (v['entry']['key'],v['source_sha256']) for v in variants])).encode()).hexdigest()[:12]
    stem=group['key']+'_'+signature;folder=root/'figures';folder.mkdir(exist_ok=True)
    movie=folder/(stem+'.gif');still=folder/(stem+'.png');paths=folder/(stem+'_paths.png')
    if movie.exists() and still.exists() and paths.exists():return dict(movie=str(movie.relative_to(root)),still=str(still.relative_to(root)),paths=str(paths.relative_to(root)))
    n=len(variants);cols=3 if group.get('kind')=='physics_replay' and n==3 else min(2,n);rows=math.ceil(n/cols)
    fig=plt.figure(figsize=((5.2 if cols==3 else 6)*cols,4.9*rows));artists=[];clouds=[];axes=[];texts=[]
    limits=[]
    for v in available:
        C=v['batch'].context_len;cloud=v['rollout']['positions'][0,C-1:].numpy()
        limits.append(cloud.reshape(-1,3));limits.append(np.asarray(v['protocol']['target']))
        for box in (group.get('display_obstacles') or v['protocol'].get('obstacle_spec') or {}).get('boxes',[]):limits.append(obstacle_vertices(box))
    points=np.concatenate(limits);low,high=points.min(0),points.max(0)
    centre=(low+high)/2;span=max(float((high-low).max())*1.15,.3)
    maximum=max(v['protocol']['sampling']['horizon'] for v in available)
    for i,(entry,v) in enumerate(zip(group['variants'],variants)):
        ax=fig.add_subplot(rows,cols,i+1,projection='3d');axes.append(ax)
        if v is None:
            ax.text2D(.2,.5,entry['label']+'\nPending',transform=ax.transAxes);ax.set_axis_off()
            artists.append(None);clouds.append(None);texts.append(None);continue
        C=v['batch'].context_len;body=v['batch'].particle_body[0].numpy()
        selected=np.concatenate([np.where(body==k)[0][::max(1,int((body==k).sum())//150)] for k in range(4)])
        cloud=v['rollout']['positions'][0,C-1:,selected].numpy();clouds.append(cloud)
        colors=np.asarray(['#df9500','#777777','#1677bc','#149477'])[body[selected]]
        artists.append(ax.scatter(*cloud[0].T,c=colors,s=7))
        target=np.asarray(v['protocol']['target'][0]);ax.scatter(*target,color='#d00045',marker='X',s=110)
        path=v['rollout']['T'][0,C-1:,0].numpy();ax.plot(*path.T,color='#df9500',lw=1,alpha=.3)
        draw_obstacles(ax,group.get('display_obstacles') or v['protocol'].get('obstacle_spec'))
        xx,yy=np.meshgrid(centre[0]+np.array([-.5,.5])*span,centre[1]+np.array([-.5,.5])*span)
        ax.plot_surface(xx,yy,np.full_like(xx,v['geometry']['metadata']['support_z']),color='#aaaaaa',alpha=.12)
        ax.set(xlim=(centre[0]-span/2,centre[0]+span/2),ylim=(centre[1]-span/2,centre[1]+span/2),
            zlim=(centre[2]-span/2,centre[2]+span/2),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)',title=entry['label'])
        ax.set_box_aspect((1,1,1));ax.view_init(23,-65)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):axis.set_major_locator(plt.MaxNLocator(4))
        texts.append(ax.text2D(.02,-.05,'',transform=ax.transAxes,fontsize=9))
    title=fig.suptitle('');fig.subplots_adjust(left=.02,right=.97,top=.9,bottom=.07,hspace=.15,wspace=.1)
    def update(frame):
        for v,artist,cloud,label in zip(variants,artists,clouds,texts):
            if v is None:continue
            f=min(frame,len(cloud)-1);artist._offsets3d=tuple(cloud[f].T)
            C=v['batch'].context_len;goal=np.asarray(v['protocol']['target'][0])
            error=np.linalg.norm(v['rollout']['T'][0,C-1+f,0].numpy()-goal)*1000
            label.set_text(f'Goal error: {error:.1f} mm'+(' · trajectory ended' if frame>f else ''))
        meaning=('Left: model prediction. Other panels: independent physics replay.' if group.get('kind')=='physics_replay' else 'Model predictions; execution is audited separately.')
        title.set_text(f'{group["title"]} · {frame*.05:.2f} s\nOrange object · grey palm · blue/green fingers · red X target. '+meaning)
        return [a for a in artists+texts+[title] if a is not None]
    temporary=movie.with_name(movie.stem+'.pending.gif')
    FuncAnimation(fig,update,frames=maximum+1,interval=80).save(temporary,writer=PillowWriter(fps=12))
    temporary.replace(movie);update(maximum);fig.savefig(still,dpi=130);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.2))
    for v in available:
        C=v['batch'].context_len;track=v['rollout']['T'][0,C-1:,0].numpy();label=v['entry']['label']
        axs[0].plot(track[:,0],track[:,1],label=label);axs[0].scatter(track[-1,0],track[-1,1],s=22)
        axs[1].plot(track[:,0],track[:,2],label=label)
        goal=np.asarray(v['protocol']['target'][0]);axs[0].scatter(goal[0],goal[1],marker='X',color='#d00045',s=60)
        axs[1].scatter(goal[0],goal[2],marker='X',color='#d00045',s=60)
        axs[2].plot(np.arange(len(track))*.05,np.linalg.norm(track-goal,axis=-1)*1000,label=label)
    axs[0].set(xlabel='x (m)',ylabel='y (m)',title='Object origin path, top view');axs[0].axis('equal')
    axs[1].set(xlabel='x (m)',ylabel='z (m)',title='Object origin path, side view');axs[1].axis('equal')
    axs[2].set(xlabel='Time (s)',ylabel='Goal error (mm)',title='Target reaching over time');axs[2].axhline(20,color='#d00045',ls='--')
    axs[0].legend(fontsize=8)
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(paths,dpi=140);plt.close(fig)
    return dict(movie=str(movie.relative_to(root)),still=str(still.relative_to(root)),paths=str(paths.relative_to(root)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path);args=p.parse_args()
    manifest=json.loads(args.manifest.read_text());root=args.manifest.resolve().parent
    os.environ.setdefault('MPLCONFIGDIR',str(root/'.mplconfig'))
    text=['# Guidance improvement gallery','',manifest.get('description',''),'',
        'Orange is the generated object; grey is the palm; blue/green are fingers. The red X is the requested destination. '+
        'Movies show model predictions. A good-looking path is not evidence of a successful physics replay.','',
        'The goal must be within 20 mm for all six held frames. Position, geometry and independent execution are reported separately.','']
    if (root/'EXECUTION_GALLERY.md').exists():
        text.extend(['[Model versus independent physics: execution gallery](EXECUTION_GALLERY.md).',''])
    if (root/'REPLAY_AUDIT.md').exists():
        text.extend(['[Replay audit of the earlier eight distinct goals](REPLAY_AUDIT.md).',''])
    if (root/'figures/optimization_progress.png').exists():
        text.extend(['![Optimization budget and noise cap](figures/optimization_progress.png)',''])
    results=[];cache={}
    for group in manifest['groups']:
        variants=[]
        for entry in group['variants']:
            if entry['key'] not in cache:cache[entry['key']]=load_variant(entry,root/'exports')
            variants.append(cache[entry['key']])
        figures=render_group(group,variants,root)
        text.extend(['## '+group['title'],'',group.get('description',''),''])
        if figures:text.extend([f'![{group["title"]}]({figures["movie"]})','',f'[Final frame]({figures["still"]}) · [Paths and goal errors]({figures["paths"]})',''])
        text.extend(['| Setting | Final error | Worst held error | Peak object–gripper overlap | Held goal | Full geometry |',
                     '|---|---:|---:|---:|---|---|'])
        rows=[]
        for entry,v in zip(group['variants'],variants):
            if v is None:
                text.append(f'| {entry["label"]} | Pending | — | — | — | — |');rows.append(dict(entry=entry,state='pending'));continue
            m=v['metrics'];yes=lambda value:'PASS' if value else 'FAIL'
            text.append(f'| {entry["label"]} | {m["final_goal_mm"]:.1f} mm | {m["held_worst_goal_mm"]:.1f} mm | {m["peak_overlap_mm"]:.2f} mm | {yes(m["goal_pass"])} | {yes(m["geometry_pass"])} |')
            rows.append(dict(entry=entry,state='completed',metrics=m,selected_iteration=v['selected_iteration'],export=v['export']))
        text.append('')
        for v in variants:
            if v is None:continue
            failed=[k for k,value in v['metrics']['validation']['residuals_m'].items() if value>0 and k!='goal']
            receipt=Path(v['export']).relative_to(root)/'receipt.json'
            text.append(f'- **{v["entry"]["label"]}**: selected update {v["selected_iteration"]}; additional failed checks: '+(', '.join(failed) or 'none')+f'. [Receipt]({receipt}).')
            if group.get('display_obstacles'):
                from guidance.obstacles import StaticObstacleDistance
                C=v['batch'].context_len;roll=v['rollout']
                probe=StaticObstacleDistance(v['geometry'],group['display_obstacles'],vendor_paths=v['protocol']['arguments']['collision_vendor'])
                signed=probe(roll['R'][:,C-1:],roll['T'][:,C-1:],roll['rigid_idx'],substeps=10)
                text.append(f'  Minimum clearance to the displayed obstacle: **{float(signed.min())*1000:.1f} mm**; negative means overlap. This includes every body and ten probes per transition.')
        text.append('');results.append(dict(group=group['key'],variants=rows,figures=figures))
    payload=dict(updated_utc=datetime.now(timezone.utc).isoformat(),manifest_sha256=file_sha256(args.manifest),groups=results)
    (root/'gallery_summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    text.extend(['This is an exploratory within-checkpoint comparison, not an estimate of population success rate.',
                 'Per-body interpolated obstacle checks are denser geometric probes, not exact continuous collision detection.',
                 'Generated trajectories retain the original observation and jointly generate object/gripper motion. '+
                 'Noise optimization changes all valid pose channels or only actuator channels as declared for each comparison. '+
                 'No recorded future is used as a task target.',''])
    tmp=root/'GALLERY.pending.md';tmp.write_text('\n'.join(text));tmp.replace(root/'GALLERY.md')
    print('Wrote',root/'GALLERY.md',flush=True)


if __name__=='__main__':main()
