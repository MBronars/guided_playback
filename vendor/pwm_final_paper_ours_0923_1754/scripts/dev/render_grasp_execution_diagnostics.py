"""Explain contact-direction and contact-force diagnostics on saved real samples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve();out=root/'grasp_execution_report_v1';out.mkdir(exist_ok=True)
    direction=root/'grasp_closing_direction_v2';directions=json.loads((direction/'report.json').read_text())['results']
    force_dir=root/'replay_contact_forces_v1';forces=json.loads((force_dir/'report.json').read_text())['results']
    # The first completed full feedback run is pinned for this explanation.
    rows=[v for v in forces if v['kind']!='recorded'];scene='pick_development_00_asset_5624'
    selected=[v for v in rows if v['step']==.0005]
    run=selected[0]['name'].removesuffix('_continue_original_step0')
    feedback_report=json.loads((root/run/'report.json').read_text())
    reference=np.load(Path(feedback_report['arguments']['replay'])/'execution.npz')
    anchor_z=reference['model_T'][0,0,2]
    movie=out/'optimized_prediction_vs_execution.gif'
    if not movie.exists():
        base=json.loads((Path(feedback_report['arguments']['base_run'])/'protocol.json').read_text())
        batch=torch.load(base['arguments']['input'],weights_only=False,map_location='cpu')
        geometry=torch.load(base['arguments']['geometry'],weights_only=False,map_location='cpu')
        predicted=np.load(root/run/'feedback_noise_opt_commands.npz')
        executed=np.load(root/run/'feedback_noise_opt_step0.npz');frame=feedback_report['prefix_frames']
        R=np.concatenate([reference['model_R'][:frame],predicted['R']])
        T=np.concatenate([reference['model_T'][:frame],predicted['T']])
        ids=batch.particle_body[0].numpy();subset=np.concatenate([np.where(ids==k)[0][::max(1,int((ids==k).sum())//160)] for k in range(4)])
        tracks=[]
        for rr,tt in [(R,T),(executed['R'],executed['T'])]:
            points=np.empty((len(tt),len(ids),3))
            for k in range(4):
                template=batch.body_templates[0,k,batch.body_template_mask[0,k]].numpy()
                points[:,ids==k]=np.einsum('tij,nj->tni',rr[:,k],template)+tt[:,k,None]
            tracks.append(points[:,subset])
        cloud=np.concatenate(tracks).reshape(-1,3);center=(cloud.min(0)+cloud.max(0))/2;span=max(np.ptp(cloud,axis=0).max()*1.2,.25)
        fig=plt.figure(figsize=(10,5.5));artists=[];labels=[];colors=np.array(['#e69f00','#777777','#0072b2','#009e73'])[ids[subset]]
        for i,(track,title) in enumerate(zip(tracks,['The model predicts this','Physics executes the gripper commands'])):
            ax=fig.add_subplot(1,2,i+1,projection='3d');artists.append(ax.scatter(*track[0].T,c=colors,s=8))
            ax.set(xlim=(center[0]-span/2,center[0]+span/2),ylim=(center[1]-span/2,center[1]+span/2),zlim=(center[2]-span/2,center[2]+span/2),title=title,xticks=[],yticks=[],zlabel='height (m)');ax.set_box_aspect((1,1,1));ax.view_init(22,-65)
            xx,yy=np.meshgrid(center[0]+np.array([-.5,.5])*span,center[1]+np.array([-.5,.5])*span);ax.plot_surface(xx,yy,np.full_like(xx,geometry['metadata']['support_z']),color='#999999',alpha=.12)
            labels.append(ax.text2D(.08,-.04,'',transform=ax.transAxes,fontsize=10))
        title=fig.suptitle('');fig.subplots_adjust(left=.01,right=.96,top=.75,bottom=.12,wspace=.12)
        def update(tick):
            for artist,track in zip(artists,tracks):artist._offsets3d=tuple(track[tick].T)
            for label,poses in zip(labels,[T,executed['T']]):label.set_text(f'Object height change: {(poses[tick,0,2]-anchor_z)*1000:.1f} mm')
            phase='Same executed prefix' if tick<=frame else 'Optimized continuation: imagined motion versus independent physics'
            title.set_text(f'Can 5624 · 24 noise updates · best iterate {feedback_report["selected_iteration"]} · {tick*.05:.2f} s\n{phase}\nOrange: object; gray/blue/green: gripper. Right: object is never commanded.')
            return artists+labels+[title]
        FuncAnimation(fig,update,frames=len(T),interval=85).save(movie,writer=PillowWriter(fps=12));update(len(T)-1);fig.savefig(out/'optimized_prediction_vs_execution_final.png',dpi=140);plt.close(fig)
    colors={'continue_original':'#555555','feedback_sample':'#0072b2','feedback_noise_opt':'#d55e00'}
    labels={'continue_original':'Continue original','feedback_sample':'Fresh sample','feedback_noise_opt':'24 noise updates'}
    fig,axes=plt.subplots(2,2,figsize=(13,8));summaries=[]
    for item in selected:
        z=np.load(force_dir/(item['name']+'.npz'));color=colors[item['kind']];label=labels[item['kind']]
        axes[0,0].plot(z['pose_time'],(z['T'][:,0,2]-anchor_z)*1000,color=color,label=label)
        # Signed log scale preserves short large impacts without hiding smaller
        # loads. Each plotted force is the actual substep value, not a mean.
        axes[0,1].plot(z['time'],z['force'][:,0,2],color=color,alpha=.85,label=label+' · table')
        axes[0,1].plot(z['time'],z['force'][:,1,2],color=color,alpha=.6,ls='--',label=label+' · hand')
        axes[1,0].plot(z['time'],z['force'][:,2:,2].sum(-1),color=color,label=label)
        axes[1,1].plot(z['time'],z['q'].max(-1)*1000,color=color,label=label)
        summaries.append(dict(variant=item['kind'],physics_step_ms=.5,peak_table_vertical_N=float(np.abs(z['force'][:,0,2]).max()),peak_hand_vertical_N=float(np.abs(z['force'][:,1,2]).max())))
    axes[0,0].axhline(100,color='#cc0077',ls=':',label='100 mm lift requirement')
    axes[0,0].set(title='Object height change after replanning',ylabel='mm relative to original observed anchor')
    axes[0,1].set(title='Table and hand forces on the object',ylabel='upward force (N; signed log scale)',yscale='symlog');axes[0,1].set_yscale('symlog',linthresh=10)
    axes[1,0].axhline(selected[0]['mass_kg']*9.81,color='#cc0077',ls=':',label='Object weight (3.43 N)')
    axes[1,0].set(title='Combined finger force on the object',ylabel='upward force (N; signed log scale)');axes[1,0].set_yscale('symlog',linthresh=10)
    axes[1,1].axhline(40,color='#cc0077',ls=':',label='40 mm finger joint limit')
    axes[1,1].set(title='Largest achieved finger joint position',ylabel='mm')
    for ax in axes.flat:ax.set_xlabel('Original rollout time (s)');ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
    fig.suptitle('First can · observed feedback at 0.8 s · seed 6381 · 0.5 ms physics step\nFresh-sample lift includes severe hand/table crushing; every complete task fails.')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/'feedback_forces.png',dpi=140);plt.close(fig)
    # Show every frozen direction-audit case, including recorded controls.
    names=[];values=[]
    for item in directions:
        asset=item['scene'].rsplit('_',1)[-1]
        label=f"{item['task']} {asset} · "+('recorded' if item['kind']=='recorded' else ('region24' if 'region_' in item['name'] else 'physical100'))
        names.append(label);n=max(1,item['near_finger_frames'])
        values.append([item[k]/n*100 for k in ('near_closing_compresses','near_opening_compresses','near_tangential','near_unstable')])
    values=np.array(values);fig,ax=plt.subplots(figsize=(11,8));left=np.zeros(len(names));yy=np.arange(len(names))
    for i,(label,color) in enumerate(zip(['Closing reduces gap','Opening reduces gap','Nearly tangential','Derivative changes with probe scale'],['#0072b2','#d55e00','#aaaaaa','#cc79a7'])):
        ax.barh(yy,values[:,i],left=left,color=color,label=label);left+=values[:,i]
    ax.set(yticks=yy,yticklabels=names,xlim=(0,100),xlabel='Percent of nearby finger/frame observations',title='Closing direction is mostly consistent in both recorded and generated plans')
    ax.invert_yaxis();ax.legend(fontsize=8,loc='lower right');fig.tight_layout();fig.savefig(out/'closing_direction.png',dpi=140);plt.close(fig)
    table='<table><tr><th>Variant</th><th>Physics step</th><th>Peak |table vertical force|</th><th>Peak |hand vertical force|</th></tr>'
    for item in rows:
        z=np.load(force_dir/(item['name']+'.npz'))
        table+=f'<tr><td>{labels[item["kind"]]}</td><td>{item["step"]*1000:g} ms</td><td>{np.abs(z["force"][:,0,2]).max():,.1f} N</td><td>{np.abs(z["force"][:,1,2]).max():,.1f} N</td></tr>'
    page='''<!doctype html><meta charset="utf-8"><title>Why a predicted grasp can fail</title><style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1450px}img{max-width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>More object height is not enough to establish a working grasp</h1>
<p>The first can feedback experiment illustrates why the full physical checks matter. A fresh continuation lifts the object, but first squeezes it between the hand and table with very large opposing forces. Optimization removes this extreme collision and loses the lift. Neither trajectory completes the declared task.</p>
<p>The plot shows measured forces from the independent MuJoCo replay, with only the hand and finger motors commanded. A stiff floating-hand controller can create forces that a real robot could not deliver. These values diagnose a failure of this replayed plan; they are not predictions of real robot forces.</p>
<p><a href="../index.html">Main report</a> · <a href="../feedback_replay_report_v1/index.html">Feedback movies and complete task checks</a> · <a href="../replay_contact_forces_v1/report.json">All 38 force/parity checks</a></p><img src="feedback_forces.png">'''
    page+=table+'</table><h2>The optimized prediction versus its execution</h2><p>The left panel is what the same selected noise-optimized plan predicts. The right panel executes only its gripper commands, using the 0.5 ms physics step. Both share the original executed prefix; after replanning, the predicted held lift and actual drop diverge.</p><img src="optimized_prediction_vs_execution.gif"><p>Logging forces leaves positions, joint states, and complete integration states bitwise unchanged on all 16 recorded trajectories at both physics steps, and on the three displayed feedback variants at both steps. At every substep, transformed contact forces sum to the engine’s passive-object translational constraint force. Recorded reference diagnostics use the qualified fixed mesh-preload controller.</p>'
    page+='''<h2>A tested explanation that did not account for these failures</h2><p>Closing in the wrong direction was a plausible cause: a finger touching a surface need not squeeze it when it moves. We measured the local mesh-gap change for 0.05 and 0.1 mm joint perturbations. Positive derivative means closing reduces the gap. The two generated lift examples mostly have the expected direction, as do the recorded picks. This measurement does not establish force closure, and does not explain the observed failure.</p><p>The analytic check separately verifies outside squeezing, inside squeezing that requires opening, and tangential motion. The fixed diagnostic set contains 16 recorded trajectories and the four available selected plans; nearby observations within a trajectory are correlated.</p><img src="closing_direction.png"><p><a href="../grasp_closing_direction_v2/report.json">Every direction-audit case and its source hashes</a> · <a href="../grasp_closing_direction_v2/analytic_gate.json">Analytic geometry checks</a> · <a href="https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-contactforce">MuJoCo contact-force API</a></p>'''
    (out/'index.html').write_text(page);(out/'summary.json').write_text(json.dumps(dict(force_examples=summaries,direction_cases=len(directions)),indent=2)+'\n')
    print('Rendered grasp execution diagnostics')


if __name__=='__main__':main()
