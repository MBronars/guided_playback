"""Compare identical generated commands under baseline and firmer finger limits."""
import argparse,json,html
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve();out=root/'replay_limit_report_v1';out.mkdir(exist_ok=True)
    controls=json.loads((root/'mujoco_limit_controls_v2_summary.json').read_text())
    generated=json.loads((root/'mujoco_limit_generated_v1_summary.json').read_text())
    original_limits={v['scene']:v for v in json.loads((root/'recorded_joint_limit_audit_v1.json').read_text())['results']}
    pairs=[]
    for row in generated['results']:
        base=Path(row['source_audit']).parent;new=root/'mujoco_limit_generated_v1'/base.name
        for step in ('dt0005','dt00025'):
            a,b=np.load(base/step/'execution.npz'),np.load(new/step/'execution.npz')
            for key in ('desired_R','desired_T','commanded_q'):
                np.testing.assert_array_equal(a[key],b[key])
        pairs.append(dict(run=row['run'],seed=row['seed'],variant=row['variant'],same_hand_finger_commands=True,baseline_pass=row['baseline_controller']['qualified_task_pass'],firmer_limits_pass=row['qualified_task_pass']))
    chosen=[v for v in generated['results'] if v['variant']=='selected' and ((v['goal_kind']=='position_region' and v['scene'] in ('pick_development_00_asset_5624','pick_development_01_asset_8026')) or (v['arm']=='physical100_h36' and v['scene'] in ('pick_development_00_asset_5624','place_development_00_asset_4943')))]
    chosen.sort(key=lambda x:(x['goal_kind']!='position_region',x['scene']))
    fig,axes=plt.subplots(len(chosen),2,figsize=(12,3.5*len(chosen)),squeeze=False)
    for axrow,row in zip(axes,chosen):
        base=Path(row['source_audit']).parent;new=root/'mujoco_limit_generated_v1'/base.name
        for step,style in [('dt0005','-'),('dt00025','--')]:
            for directory,color,label in [(base,'#0072b2','Default limits'),(new,'#d55e00','1 ms limits')]:
                z=np.load(directory/step/'execution.npz');T=z['model_T'];times=np.arange(len(T))*.05
                axrow[0].plot(times,(T[:,0,2]-T[0,0,2])*1000,style,color=color,label=label+' · '+('0.5 ms' if step=='dt0005' else '0.25 ms'))
                excess=np.maximum(0,np.maximum(-z['joint_q'],z['joint_q']-.04).max(-1))*1000
                axrow[1].plot(times,excess,style,color=color)
        z=np.load(base/'dt0005/execution.npz');T=z['desired_T']
        axrow[0].plot(times,(T[:,0,2]-T[0,0,2])*1000,':',color='#333333',label='Model prediction')
        axrow[1].axhline(.1,color='#cc0077',ls=':',label='0.1 mm allowance')
        name=row['scene'].replace('_development_',' / dev ').replace('_asset_',' / ')
        axrow[0].set(title=name+' · '+row['arm'],ylabel='Object height change (mm)',xlabel='seconds')
        axrow[1].set(title='Executed finger travel beyond the allowed range',ylabel='Over-travel (mm)',xlabel='seconds')
        for ax in axrow:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Identical model plans and gripper commands · only the simulated finger limit time changes',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/'generated_limit_comparison.png',dpi=135);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5));names=[v['scene'].replace('_development_',' / dev ').replace('_reserved_',' / res ').replace('_asset_',' / ') for v in controls['results']];y=np.arange(len(names))
    baseline=np.array([max(s['replay_joint_limit_excess_mm'] for s in original_limits[v['scene']]['steps']) for v in controls['results']]);candidate=np.array([max(0,max(v['joint_limit_overshoot_mm'])) for v in controls['results']])
    axes[0].barh(y-.16,baseline,.32,label='Default limits',color='#0072b2');axes[0].barh(y+.16,candidate,.32,label='1 ms limits',color='#d55e00');axes[0].set(yticks=y,yticklabels=names,xlabel='Worst finger over-travel across both steps (mm)',title='Recorded-control replay');axes[0].invert_yaxis();axes[0].legend()
    ax=axes[1];ax.axis('off');ax.text(0,.93,f"Same {controls['qualified']}/16 recorded-control scenes qualify.\n\n{len(pairs)} frozen generated-plan diagnostics:\n{sum(v['baseline_pass'] for v in pairs)} complete task passes with default limits;\n{sum(v['firmer_limits_pass'] for v in pairs)} with 1 ms limits.\n\nEvery planned hand and finger command is\nverified bitwise equal across these replays.\n\nThe 2 ms candidate qualified 10/16 controls\nand failed the isolated 0.1 mm transient gate.\nIts separate results are retained.\n\nThese are independent floating-hand simulations,\nnot robot or source-PhysX trials.",va='top',fontsize=12)
    fig.tight_layout();fig.savefig(out/'recorded_limit_comparison.png',dpi=135);plt.close(fig)
    payload=dict(paired_commands=pairs,generated_count=len(pairs),baseline_task_passes=sum(v['baseline_pass'] for v in pairs),firmer_limit_task_passes=sum(v['firmer_limits_pass'] for v in pairs),control_qualified=controls['qualified'],baseline_control_qualified=controls['baseline_qualified'],control_qualification_scope='Original pose/timestep screen, with joint travel diagnostic shown separately.',limits_time=.001)
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    page='''<!doctype html><meta charset="utf-8"><title>Replay finger limits</title><style>body{font:16px system-ui;line-height:1.5;margin:30px;max-width:1450px}img{max-width:100%}</style><h1>Are soft simulator joint limits causing the failed lifts?</h1><p>The baseline simulator treats finger limits as soft constraints. The optional 1 ms limit setting keeps the same recorded-control qualification and reduces over-travel. The generated plans below use exactly the same commanded hand poses and finger motor targets. Model weights, generated object paths, friction, collision surfaces and task thresholds are unchanged.</p><p>This is a simulator ablation, not a learned-model improvement. All frozen generated-plan diagnostics remain in the paired summary. The four plots show the first two region picks and the first pick/place physical100 runs; no passing case is selected for display.</p><p><a href="../index.html">Main report</a> · <a href="summary.json">Paired command checks and counts</a> · <a href="../mujoco_limit_generated_v1_summary.json">Every generated result</a> · <a href="../mujoco_limit_controls_v2_summary.json">Every recorded control</a></p><img src="recorded_limit_comparison.png"><img src="generated_limit_comparison.png">'''
    (out/'index.html').write_text(page);print('Paired generated controller checks:',len(pairs))


if __name__=='__main__':main()
