"""Show executed feedback continuations, including failures and the shared prefix."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve();out=root/'feedback_replay_report_v1';out.mkdir(exist_ok=True)
    audits=json.loads((root/'feedback_execution_summary_v1.json').read_text())['results']
    sections=[]
    for audit in sorted(audits,key=lambda r:(r['kind']=='integration_gate',r['scene'],r['iterations'],r['context_source'])):
        directory=root/audit['run'];report=json.loads((directory/'report.json').read_text());base=Path(report['arguments']['base_run'])
        protocol=json.loads((base/'protocol.json').read_text());b=torch.load(protocol['arguments']['input'],weights_only=False,map_location='cpu')
        g=torch.load(protocol['arguments']['geometry'],weights_only=False,map_location='cpu');body=b.particle_body[0].numpy()
        select=np.concatenate([np.where(body==k)[0][::max(1,int((body==k).sum())//110)] for k in range(4)])
        colors=np.array(['#e69f00','#777777','#0072b2','#009e73'])[body[select]]
        steps=len(audit['results'][0]['replays']);tracks=[];poses=[]
        def points(R,T):
            p=np.empty((len(T),len(body),3))
            for k in range(4):
                template=b.body_templates[0,k,b.body_template_mask[0,k]].numpy()
                p[:,body==k]=np.einsum('tij,nj->tni',R[:,k],template)+T[:,k,None]
            return p[:,select]
        for step in range(steps):
            for item in audit['results']:
                z=np.load(directory/f'{item["variant"]}_step{step}.npz');tracks.append(points(z['R'],z['T']));poses.append(z['T'])
        target=np.asarray(report['goal']).reshape(3);region=report['goal_region_bounds'];rectangle=None
        if region:
            lower,upper=[np.array([float(v) for v in region[k].split(',')]) for k in ('lower','upper')]
            rectangle=np.array([[lower[0],lower[1],lower[2]],[upper[0],lower[1],lower[2]],[upper[0],upper[1],lower[2]],[lower[0],upper[1],lower[2]],[lower[0],lower[1],lower[2]]])
        cloud=np.concatenate([v.reshape(-1,3) for v in tracks]+[rectangle if region else target[None]])
        lo,hi=cloud.min(0),cloud.max(0);center=(lo+hi)/2;span=max((hi-lo).max()*1.12,.25)
        C=b.context_len;frame=audit['prefix_frames'];H=audit['horizon'];tag=audit['run']
        artwork=tag+('_repeated' if audit.get('replan_schedule') else '')
        movie=out/(artwork+'.gif');still=out/(artwork+'_final.png');chart=out/(artwork+'_traces.png')
        context='executed observations' if audit['context_source']=='observed' else 'old model predictions'
        best_k=audit.get('method')=='best_of_k'
        repeated=bool(audit.get('replan_schedule'))
        budget=f'{audit["k"]} candidates' if best_k else f'{audit["iterations"]} updates'
        if repeated:budget=f'{len(audit["replan_schedule"])} replans × '+budget
        selected_label=f'Select from {audit["k"]} fresh samples' if best_k else f'Optimize fresh sample · {audit["iterations"]} update'+('s' if audit['iterations']!=1 else '')
        if repeated:selected_label=f'{audit["iterations"]} noise updates per replan'
        labels=['Continue original plan','Fresh continuation sample',selected_label]
        if repeated:labels[1]='Fresh sample at each replan'
        if not movie.exists():
            fig=plt.figure(figsize=(13,4.9*steps));artists=[];texts=[]
            for i,track in enumerate(tracks):
                ax=fig.add_subplot(steps,3,i+1,projection='3d');artists.append(ax.scatter(*track[0].T,c=colors,s=7))
                if region:ax.plot(*rectangle.T,color='#cc0077',lw=1.8)
                else:ax.scatter(*target,c='#cc0077',marker='x',s=60)
                dt=audit['results'][i%3]['replays'][i//3]['step']*1000
                ax.set(xlim=(center[0]-span/2,center[0]+span/2),ylim=(center[1]-span/2,center[1]+span/2),zlim=(center[2]-span/2,center[2]+span/2),xticks=[],yticks=[],zlabel='height (m)',title=labels[i%3]+f'\nPhysics step {dt:g} ms')
                ax.set_box_aspect((1,1,1));ax.view_init(22,-65)
                xx,yy=np.meshgrid(center[0]+np.array([-.5,.5])*span,center[1]+np.array([-.5,.5])*span)
                ax.plot_surface(xx,yy,np.full_like(xx,g['metadata']['support_z']),color='#999999',alpha=.12)
                texts.append(ax.text2D(.08,-.03,'',transform=ax.transAxes,fontsize=9))
            title=fig.suptitle('');fig.subplots_adjust(left=.01,right=.97,top=.83,bottom=.08,hspace=.22,wspace=.09)
            def update(tick):
                for artist,track,text,pose in zip(artists,tracks,texts,poses):
                    artist._offsets3d=tuple(track[tick].T)
                    lift=(pose[tick,0,2]-pose[0,0,2])*1000
                    text.set_text(f'Lift: {lift:.1f} mm; point distance: {np.linalg.norm(pose[tick,0]-target)*1000:.1f} mm')
                phase='Identical executed prefix within each physics step' if tick<=frame else f'New continuations use {context}'
                when=', '.join(f'{s["frame"]*.05:.2f}' for s in audit['replan_schedule']) if repeated else f'{frame*.05:.2f}'
                title.set_text(f'{audit["scene"]} · seed {audit["seed"]} · {tick*.05:.2f}s\n{phase}. Replanning at {when}s.\nAll panels show physics; orange object moves dynamically, only the gripper is commanded.')
                return artists+texts+[title]
            FuncAnimation(fig,update,frames=H+1,interval=85).save(movie,writer=PillowWriter(fps=12))
            update(H);fig.savefig(still,dpi=130);plt.close(fig)
        if not chart.exists():
            fig,axes=plt.subplots(1,3,figsize=(13,3.7));times=np.arange(H+1)*.05
            for i,label,color in zip(range(3),labels,['#555555','#0072b2','#d55e00']):
                for step in range(steps):
                    T=poses[step*3+i][:,0];style='-' if step==0 else '--'
                    axes[0].plot(times,(T[:,2]-T[0,2])*1000,style,color=color,label=label+(f' · {audit["results"][i]["replays"][step]["step"]*1000:g} ms'))
                    value=np.maximum(lower-T,T-upper).max(-1)*1000 if region else np.linalg.norm(T-target,axis=-1)*1000
                    axes[1].plot(times,value,style,color=color)
                if steps==2:axes[2].plot(times,np.linalg.norm(poses[i][:,0]-poses[3+i][:,0],axis=-1)*1000,color=color,label=label)
            if region:axes[0].axhline(100,color='#cc0077',ls=':')
            axes[1].axhline(0 if region else 20,color='#cc0077',ls=':')
            axes[2].axhline(5,color='#cc0077',ls=':')
            axes[0].legend(fontsize=6)
            for ax,title in zip(axes,['Executed object height change','Region violation (≤0 inside)' if region else 'Point-goal distance','Difference between physics steps']):
                ax.axvline(frame*.05,color='#777777',ls='--');ax.axvspan(0,frame*.05,color='#777777',alpha=.09)
                for stage in audit.get('replan_schedule',[])[1:]:ax.axvline(stage['frame']*.05,color='#777777',ls=':',alpha=.6)
                ax.set(title=title,xlabel='seconds',ylabel='mm');ax.grid(alpha=.2)
            fig.suptitle(f'{audit["scene"]} · context: {context}. Gray: original executed prefix.')
            fig.tight_layout();fig.savefig(chart,dpi=130);plt.close(fig)
        table='<table><tr><th>Continuation</th><th>Earlier hard violations (fixed prefix)</th><th>New segment task screen at each step</th><th>Complete qualified task</th></tr>'
        for label,item in zip(labels,audit['results']):
            failures=sorted(set(k for v in item['replays'] for k in v['prefix_hard_failures']))
            future=' / '.join('PASS' if v['continuation_validation']['accepted'] else 'FAIL' for v in item['replays'])
            table+=f'<tr><td>{label}</td><td>{", ".join(failures) or "none"}</td><td>{future}</td><td>{"PASS" if item["qualified_full_task_pass"] else "FAIL"}</td></tr>'
        selected=report.get('selected_draw') if best_k else report['selected_iteration']
        selection=f'Selected {"draw" if best_k else "update"} {selected}; '+('the fresh sample and selected output are identical.' if selected==0 else 'selection uses the declared model objective.')
        if repeated:
            selected=[p['selected_iteration'] for p in report['results'][-1]['replans']]
            selection=f'Retained updates at successive replans: {selected}. Each stage keeps the original deadline; both arms use paired noise and routing RNG.'
        sections.append(f'<details {"open" if not sections else ""}><summary>{html.escape(audit["scene"])} · {html.escape(context)} · {budget} · {audit["kind"]}</summary><p>Recorded-control qualification: {audit["control_qualified"]}. New planning: {report["actual_model_calls"]} model calls, {report["optimization_seconds"]:.1f}s; continuing the old plan uses no new planning compute. {selection} Source base plan and full receipts are linked in <a href="../{tag}/report.json">the report</a>.</p><img loading="lazy" src="{movie.name}">{table}</table><img loading="lazy" src="{chart.name}"></details>')
    header='''<!doctype html><meta charset="utf-8"><title>Replanning after a shared executed prefix</title><style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1450px}img{max-width:100%}summary{cursor:pointer;font-weight:600;margin:20px 0}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style>
<h1>Can a new continuation recover a failed interaction?</h1><p>Every column executes exactly the same original prefix. After the marked time, the left column continues the old plan. The middle samples a new continuation; the right optimizes that same new sample. All panels show independent physics execution, not the model moving the object directly.</p>
<p>There are two context conditions. <b>Executed observations</b> uses what actually happened. <b>Old model predictions</b> withholds that correction. The matched first-can probe uses identical fresh Gaussian noise and routing RNG in both conditions. Fresh sampling increases final height in both, but every complete task fails. The observed-context fresh sample includes severe hand/table crushing; greater height is not a better grasp. <a href="../grasp_execution_report_v1/index.html">See the contact-force diagnosis</a>.</p>
<p>Sample-and-select trials label the right column separately: select the lowest declared model objective from a candidate bank that includes the middle sample. A 49-candidate bank plus the separately evaluated paired sample uses the same 3,400 model calls as 24 noise updates at this remaining horizon. This matches forward invocations including gradient recomputation, not FLOPs or wall time; backward work differs. Every candidate's predicted poses are retained.</p>
<p>Repeated-feedback trials begin earlier, before the first can's original contact failure, and mark every replan with a vertical line. They execute a short part of each plan, rebuild context from the resulting state and plan again toward the same original deadline. The fresh-sampling and noise-optimization arms share their random draws at each replan but can observe different states after executing different controls. The original one-correction runs remain separate.</p>
<p>A later correction cannot undo a collision or joint-limit violation in the shared prefix. Those failures stay in the complete task result; the separate continuation screen describes only what happens afterward. Good final height alone is insufficient: the goal must also hold over the final six frames, with contact, release where applicable, collision and joint checks.</p>
<p>The finer physics step replays the same realized segmented command program. It does not optimize another feedback policy from finer-step observations. These are floating-hand MuJoCo diagnostics with approximate collision/inertia properties and separate recorded-control qualification, not source-PhysX or robot success. One-update integration gates are labeled and separate from longer experiments.</p>
<p><a href="../feedback_execution_summary_v1.json">Complete execution audits</a> · <a href="../index.html">Main guidance report</a></p>'''
    (out/'index.html').write_text(header+''.join(sections));print('Rendered feedback comparisons:',len(sections))


if __name__=='__main__':main()
