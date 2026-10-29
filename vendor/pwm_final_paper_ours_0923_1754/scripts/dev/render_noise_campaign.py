"""Scientific plots and paired movies for all completed pick/place noise trials."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import numpy as np


def visual(directory,seed):
    movie=directory/f'comparison_{seed}.gif';figure=directory/f'traces_{seed}.png'
    if movie.exists() and figure.exists():return
    data=np.load(directory/f'seed_{seed}.npz');record=json.loads((directory/f'seed_{seed}.json').read_text())
    protocol=json.loads((directory/'protocol.json').read_text());C=int(data['context_len']);H=protocol['sampling']['horizon']
    region=protocol.get('goal_kind')=='position_region'
    lower=upper=None
    if region:
        lower=np.array([float(v) for v in protocol['goal_region_bounds']['lower'].split(',')])
        upper=np.array([float(v) for v in protocol['goal_region_bounds']['upper'].split(',')])
    pb=data['particle_body'][0];target=data['target'][0];tracks=[data['initial'][0],data['positions'][0],data['gt'][0]]
    rows=np.concatenate([np.where(pb==k)[0][::max(1,len(np.where(pb==k)[0])//150)] for k in np.unique(pb)])
    colors=np.array(['#e69f00','#3568b0','#53aeb0','#a378bd'])[pb[rows]]
    allpoints=np.concatenate([a[:,rows].reshape(-1,3) for a in tracks]+[target[None]])
    if region and np.isfinite(np.r_[lower,upper[:2]]).all():
        corners=np.array([[x,y,lower[2]] for x in (lower[0],upper[0]) for y in (lower[1],upper[1])])
        allpoints=np.concatenate([allpoints,corners])
    lo=allpoints.min(0);hi=allpoints.max(0);mid=(lo+hi)/2;span=max(float((hi-lo).max())*1.1,.12)
    fig=plt.figure(figsize=(12,4));axes=[];artists=[]
    titles=['Ordinary sample','Noise optimized','Recorded simulation reference']
    for i,(a,title) in enumerate(zip(tracks,titles)):
        ax=fig.add_subplot(1,3,i+1,projection='3d');axes.append(ax)
        points=a[0,rows];artists.append(ax.scatter(*points.T,c=colors,s=5,alpha=.8))
        if region and np.isfinite(np.r_[lower,upper[:2]]).all():
            corners=np.array([[lower[0],lower[1],lower[2]],[upper[0],lower[1],lower[2]],
                              [upper[0],upper[1],lower[2]],[lower[0],upper[1],lower[2]],
                              [lower[0],lower[1],lower[2]]])
            ax.plot(*corners.T,c='red',lw=1.2)
        elif not region:ax.scatter(*target,c='red',marker='x',s=65)
        ax.set(xlim=(mid[0]-span/2,mid[0]+span/2),ylim=(mid[1]-span/2,mid[1]+span/2),zlim=(mid[2]-span/2,mid[2]+span/2),title=title,xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
        ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-65)
        support=protocol.get('geometry',{}).get('support_z')
        if support is not None:
            x=np.array([mid[0]-span/2,mid[0]+span/2]);y=np.array([mid[1]-span/2,mid[1]+span/2]);xx,yy=np.meshgrid(x,y)
            ax.plot_surface(xx,yy,np.full_like(xx,support),color='#999999',alpha=.12)
    title=fig.suptitle('');fig.tight_layout(rect=(0,0,1,.9))
    def update(t):
        for artist,a in zip(artists,tracks):artist._offsets3d=tuple(a[t,rows].T)
        future=t-C+1;phase='observed context' if t<C else f'future frame {future}/{H}; {future*.05:.2f}s'
        goal_caption='red frame: lower surface of allowed position region' if region else 'red ×: requested target'
        title.set_text(f'{protocol["arguments"].get("task","pick")} · seed {seed} · {phase}\nOrange: object; blue/purple: gripper; {goal_caption}. Geometry proxies; execution unverified.')
        return artists+[title]
    if not movie.exists():FuncAnimation(fig,update,frames=len(tracks[0]),interval=80).save(movie,writer=PillowWriter(fps=12))
    plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(12,9));axes=axes.flatten()
    for key,label,color in [('initial','Ordinary','#777777'),('selected','Optimized','#0072b2')]:
        t=record[key].get('traces',{});time=np.arange(H+1)*.05
        for name,idx in [('gaps',1),('inside_object',2),('inside_gripper_hull',2)]:
            if name in t:
                values=np.asarray(t[name]);values=values.max(-1)
                axes[idx].plot(time,values*1000,color=color,ls='--' if name=='inside_gripper_hull' else '-',label=label+(' reverse hull' if name=='inside_gripper_hull' else ''))
        a=tracks[0 if key=='initial' else 1];obj=a[:,pb==0].mean(1)
        poses=data['T' if key=='selected' else 'initial_T'][0,C:,0]
        axes[0].plot(np.arange(1,H+1)*.05,np.linalg.norm(poses-target,axis=-1)*1000,color=color,label=label)
        if 'bottom' in t:
            support=protocol['geometry']['support_z'];axes[3].plot(time,(np.asarray(t['bottom'])-support)*1000,color=color,label=label)
        steps=np.linalg.norm(np.diff(obj,axis=0),axis=-1)*1000
        axes[4].plot(np.arange(1,H+1)*.05,steps[C-1:],color=color,label=label)
        if 'target_rotation_deg' in t:
            axes[8].plot(np.arange(1,H+1)*.05,np.asarray(t['target_rotation_deg'])[0],color=color,label=label)
    for window in range(16,H,16):
        for ax in axes[:5]:ax.axvline(window*.05,color='#cc79a7',lw=1,alpha=.4)
    trace=record.get('trace',[])
    if trace:
        for w in range(len(trace[0].get('noise_rms_by_window',[]))):
            axes[5].plot([r.get('iteration',i) for i,r in enumerate(trace)],[r.get('noise_rms_by_window',[0]*(w+1))[w] for r in trace],label=f'window {w+1}')
            gradient_rows=[r for r in trace if 'gradient_norm_by_window' in r]
            axes[6].plot([r.get('iteration',i) for i,r in enumerate(gradient_rows)],[r['gradient_norm_by_window'][w] for r in gradient_rows],label=f'window {w+1}')
        axes[7].plot([r.get('iteration',i) for i,r in enumerate(trace)],[r['loss'] for r in trace])
    for i,(ax,title,ylabel) in enumerate(zip(axes,['Target distance (template origin)','Farther finger distance','Sampled penetration','Object bottom above support reference','Object centroid step','Noise change by window','Gradient by window','Declared objective','Orientation error (place)'],['mm','mm','mm','mm','mm/frame','RMS normalized noise','L2 norm','metres + weighted penalties','degrees'])):
        ax.set(title=title,ylabel=ylabel,xlabel='update' if i in (5,6,7) else 'seconds');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    fig.suptitle(f'{directory.name}; seed {seed}. Pink lines: internal window boundaries.',fontsize=10)
    fig.tight_layout();fig.savefig(figure,dpi=130);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root).resolve()
    rows=[];sections=[];errors=[];active=[]
    for packet in sorted(root.glob('packet_*.json')):
        if packet.stem.endswith('_status'):continue
        plan=json.loads(packet.read_text());status_path=packet.with_name(packet.stem+'_status.json')
        status=json.loads(status_path.read_text()) if status_path.exists() else {'state':'not started'}
        sections.append(f'<p>{html.escape(packet.stem)}: {html.escape(status["state"])}</p>')
        for running in status.get('entries',[]):
            if running['state']!='running':continue
            matches=[e for e in plan['entries'] if e['name']==running['name']]
            if not matches:continue
            directory=Path(matches[0]['out'])
            traces=sorted(directory.glob('seed_*_trace.jsonl'))
            if not traces and (directory/'trace.jsonl').exists():traces=[directory/'trace.jsonl']
            latest={}
            if traces:
                try:latest=json.loads(traces[-1].read_text().splitlines()[-1])
                except (ValueError,IndexError):pass  # Writer may be between bytes.
            active.append(dict(run=running['name'],**latest))
        for entry in plan['entries']:
            directory=Path(entry['out'])
            for result in sorted(directory.glob('seed_*.json')):
                if result.stem.endswith('_trace'):continue
                r=json.loads(result.read_text());seed=r['seed'];s=r['selected'];v=r.get('validation') or {}
                try:visual(directory,seed)
                except Exception as error:errors.append(dict(run=entry['name'],seed=seed,error=repr(error)))
                link=directory.relative_to(root).as_posix();scene=entry['scene'];arm=entry['arm']
                iv=r.get('initial_validation') or {}
                rows.append(dict(scene=scene,task=entry.get('task'),group=entry.get('group'),arm=arm,seed=seed,initial_error_mm=r['initial']['target_distance_mm'],selected_error_mm=s['target_distance_mm'],rotation_deg=s.get('target_rotation_deg'),initial_geometry_pass=iv.get('accepted'),geometry_pass=v.get('accepted'),max_violation_mm=1000*v.get('max_interaction_violation_m',float('nan')),unsupported_rise_mm=s.get('unsupported_rise_mm'),hover_proxy=s.get('sustained_hover_proxy'),seconds=r['seconds'],calls=r['actual_model_calls_including_recomputation'],peak_gpu_gb=(r.get('max_allocated_bytes') or 0)/1e9,link=link))
                audit_path=directory/f'mesh_audit_v2_{seed}.json'
                audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
                rows[-1].update(screen_kind=v.get('label','unspecified'),
                    mesh_pass=audit.get('selected',{}).get('validation',{}).get('accepted'),
                    mesh_initial_pass=audit.get('initial',{}).get('validation',{}).get('accepted'),
                    mesh_penetration_mm=audit.get('selected',{}).get('penetration_max_mm'))
    header='''<!doctype html><meta charset="utf-8"><title>Pick/place noise optimization</title><style>body{font:16px system-ui;margin:30px;max-width:1400px}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}summary{cursor:pointer;margin:12px 0}</style><h1>Pick and place: initial-noise optimization</h1><p>Each row is one pinned scene, method and seed. Lower target error is useful only alongside plausible motion. Geometric checks are proxies; robot/simulator execution is unverified. All completed trials and preparation failures remain listed.</p>'''
    header+='<p>Two-finger rise/hover flags describe sampled proximity; they do not prove force-free motion. The shared mesh screen (v2) checks full convex collision surfaces, release, and table intrusion by both the object and gripper. Every method is rescored with the same criteria. Neither screen certifies contact dynamics.</p>'
    header+='<p>Any arm labeled <b>region</b> is a separate exploratory minimum-height/workspace task. Its run-specific screen uses that region; the shared screen still evaluates the original precise-point task. Nominal point error is not the region objective. Recorded reference clips follow their source tasks and do not necessarily satisfy our requested point, hold or release.</p>'
    table='<table><tr><th>Scene / method / seed</th><th>Goal mm before → after</th><th>Run’s geometry screen before → after</th><th>Shared mesh screen before → after</th><th>Worst interaction excess mm</th><th>Two-finger rise mm / hover proxy</th><th>Seconds / calls / peak GB</th><th>View</th></tr>'
    details=[]
    for i,r in enumerate(rows):
        label=html.escape(f'{r["scene"]} · {r["arm"]} · {r["seed"]}')
        mesh=f'{r["mesh_initial_pass"]} → {r["mesh_pass"]}' if r['mesh_pass'] is not None else 'pending'
        table+=f'<tr><td>{label}</td><td>{r["initial_error_mm"]:.1f} → {r["selected_error_mm"]:.1f}</td><td title="{html.escape(r["screen_kind"])}">{r["initial_geometry_pass"]} → {r["geometry_pass"]}</td><td>{mesh}</td><td>{r["max_violation_mm"]:.2f}</td><td>{r["unsupported_rise_mm"]:.2f} / {r["hover_proxy"]}</td><td>{r["seconds"]:.1f} / {r["calls"]} / {r["peak_gpu_gb"]:.1f}</td><td><a href="#{i}">movie/traces</a></td></tr>'
        details.append(f'<details id="{i}"><summary>{label}</summary><img loading="lazy" src="{r["link"]}/comparison_{r["seed"]}.gif"><img loading="lazy" src="{r["link"]}/traces_{r["seed"]}.png"></details>')
    prep=root/'scenes_v2.json'
    if not prep.exists():prep=root/'scenes_v1.json'
    prep_html='<details><summary>Scene preparation inventory</summary><pre>'+html.escape(prep.read_text() if prep.exists() else 'pending')+'</pre></details>'
    active_html='<h2>Running experiments</h2>'
    for row in active:
        progress=f"seed {row.get('seed','pending')}, update/draw {row.get('iteration',row.get('draw','pending'))}, elapsed {row.get('elapsed_seconds',row.get('seconds',0)):.0f}s"
        active_html+=f'<p>{html.escape(row["run"])}: {html.escape(progress)}</p>'
    extra=''
    if (root/'review_11am/index.html').exists():
        extra+='<p><a href="review_11am/index.html">Start here: what improved, what failed, and what to try next</a></p>'
    if (root/'primary_overview_v1/index.html').exists():
        extra+='<p><a href="primary_overview_v1/index.html">Paired results across all 16 pick/place scenes</a></p>'
    if (root/'replay_report_v2/index.html').exists():
        extra+='<p><a href="replay_report_v2/index.html">Recorded-motion replay controls at matched timesteps</a></p>'
    if (root/'generated_replay_report_v2/index.html').exists():
        extra+='<p><a href="generated_replay_report_v2/index.html">Does the generated gripper actually move the object? Prediction versus independent physics movies</a></p>'
    if (root/'region_replay_report_v1/index.html').exists():
        extra+='<p><a href="region_replay_report_v1/index.html">Workspace lifts: model prediction versus physics at both integration steps</a></p>'
    if (root/'feedback_replay_report_v1/index.html').exists():
        extra+='<p><a href="feedback_replay_report_v1/index.html">After an executed prefix: continue the plan, sample again, or optimize a new continuation</a></p>'
    if (root/'receding_prediction_report_v1/index.html').exists():
        extra+='<p><a href="receding_prediction_report_v1/index.html">After each observation: predicted versus executed motion until the next replan</a></p>'
    if (root/'region_contact_timing_v1/index.html').exists():
        extra+='<p><a href="region_contact_timing_v1/index.html">Contact timing across all eight lift-workspace plans</a></p>'
    if (root/'qualified_control_report_v1/index.html').exists():
        extra+='<p><a href="qualified_control_report_v1/index.html">Same scene and controller: recorded lift replays, generated lift fails</a></p>'
    if (root/'grasp_execution_report_v1/index.html').exists():
        extra+='<p><a href="grasp_execution_report_v1/index.html">Why greater lift can still be a failure: executed forces and grasp geometry</a></p>'
    if (root/'feedback_candidates_report_v1/index.html').exists():
        extra+='<p><a href="feedback_candidates_report_v1/index.html">Replay every sampled candidate: generation coverage versus selection quality</a></p>'
    if (root/'feedback_pairing_v1.json').exists():
        extra+='<p><a href="feedback_pairing_v1.json">Paired feedback checks: identical noise and shared control trajectories</a></p>'
    if (root/'replay_limit_report_v1/index.html').exists():
        extra+='<p><a href="replay_limit_report_v1/index.html">Identical gripper commands with firmer simulator finger limits</a></p>'
    if (root/'task_reference_report_v1/index.html').exists():
        extra+='<p><a href="task_reference_report_v1/index.html">What task are we asking for? Recorded reference motion versus our precise target and release criteria</a></p>'
    if (root/'place_release_report_v1/index.html').exists():
        extra+='<p><a href="place_release_report_v1/index.html">Placing supervision: sampled-point clearance versus mesh contact in training segments</a></p>'
    if (root/'convergence_report_v1/index.html').exists():
        extra+='<p><a href="convergence_report_v1/index.html">More optimization: final-frame accuracy, sustained goal accuracy and collision checks</a></p>'
    if (root/'fixed_actuator_resampling_v2/index.html').exists():
        extra+='<p><a href="fixed_actuator_resampling_v2/index.html">Same gripper motion, resampled object noise: a model consistency diagnostic</a></p>'
    if (root/'context_gradient_report_v1/index.html').exists():
        extra+='<p><a href="context_gradient_report_v1/index.html">Can the final goal improve earlier actions? Matched two-window gradient comparison</a></p>'
    tmp=root/'index.pending';tmp.write_text(header+extra+active_html+''.join(sections)+table+'</table>'+prep_html+''.join(details));tmp.replace(root/'index.html')
    groups=[]
    for task,group,arm in sorted(set((r['task'],r['group'],r['arm']) for r in rows)):
        selected=[r for r in rows if (r['task'],r['group'],r['arm'])==(task,group,arm)]
        groups.append(dict(task=task,group=group,arm=arm,scenes=len(set(r['scene'] for r in selected)),trials=len(selected),
            median_goal_before_mm=float(np.median([r['initial_error_mm'] for r in selected])),
            median_goal_after_mm=float(np.median([r['selected_error_mm'] for r in selected])),
            initial_geometry_passes=sum(r['initial_geometry_pass'] is True for r in selected),geometry_passes=sum(r['geometry_pass'] is True for r in selected),
            mesh_audited_trials=sum(r['mesh_pass'] is not None for r in selected),mesh_passes=sum(r['mesh_pass'] is True for r in selected),
            note='seeds within one scene are correlated; these are completed-trial summaries, not execution success rates'))
    (root/'summary.json').write_text(json.dumps(dict(trials=rows,groups=groups,active=active,render_errors=errors),indent=2)+'\n')


if __name__=='__main__':main()
