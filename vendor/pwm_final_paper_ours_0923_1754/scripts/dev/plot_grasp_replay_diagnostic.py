"""Compare recorded-control and generated-grasp contact forces in replay."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=root.parent/'guidance_noise_2026-09-15'
    cases=[('Recorded control',root/'force_probe_recorded_bottle',
            old/'persistent_replay_controls_v1/pick_reserved_00_asset_8270/dt00025'),
           ('Generated leftward pick, 64 updates / cap 0.50',root/'force_probe_bottle64',
            root/'replays/bottle_8270_left_i64_r050/dt00025')]
    strong=root/'replays/bottle_8270_left_max_strong_interaction_i64_r050/dt00025'
    if (strong/'forces/report.json').exists():
        cases.append(('Stronger interaction costs + worst-held goal',strong/'forces',strong))
    fig,axes=plt.subplots(3,len(cases),figsize=(6.5*len(cases),10),sharex=True,sharey='row')
    summaries=[]
    for j,(name,force,replay) in enumerate(cases):
        z=np.load(force/'solver_steps.npz');a=np.load(replay/'execution.npz')
        report=json.loads((force/'report.json').read_text())
        if not report['bitwise_pose_joint_and_integration_replay']:raise ValueError('Force replay not qualified')
        t=z['pose_time'];origin=a['desired_T'][0,0,2]
        axes[0,j].plot(t,(a['desired_T'][:,0,2]-origin)*1000,label='Requested / predicted object',color='#bd2775')
        axes[0,j].plot(t,(z['T'][:,0,2]-origin)*1000,label='Object in physics',color='#222222')
        axes[0,j].set(title=name,ylabel='Object lift (mm)');axes[0,j].legend(fontsize=8)
        for k,(label,color) in enumerate([('Finger 0','#1677bc'),('Finger 1','#149477')]):
            axes[1,j].plot(z['time'],z['normal'][:,k+2],label=label,color=color,alpha=.8,lw=.8)
        fraction=report['both_fingers_positive_normal_fraction']*100
        axes[1,j].set(ylabel='Normal contact force (N)',title=f'Both fingers exert >0.01 N: {fraction:.2f}% of rollout')
        axes[1,j].set_yscale('symlog',linthresh=.05);axes[1,j].legend(fontsize=8)
        n=len(a['requested_q']);qtime=np.arange(1,n+1)*.05
        axes[2,j].plot(qtime,a['requested_q'].mean(-1)*1000,label='Requested / predicted',color='#bd2775')
        axes[2,j].plot(qtime,a['commanded_q'].mean(-1)*1000,label='Preloaded command',color='#222222')
        axes[2,j].set(xlabel='Time (s)',ylabel='Mean finger coordinate (mm)',title='Larger coordinate means a wider opening')
        axes[2,j].legend(fontsize=8)
        summaries.append(dict(label=name,both_fingers_positive_percent=fraction,
            force_report=str(force/'report.json'),execution=str(replay/'execution.npz')))
    for ax in axes.flat:ax.grid(alpha=.2);ax.set_xlim(0,1.8)
    fig.suptitle('Nearness in the predicted geometry does not establish a maintained grasp\nIndependent MuJoCo diagnostic at 0.25 ms; no source-PhysX or real-robot force claim')
    fig.tight_layout(rect=(0,0,1,.94));folder=root/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'grasp_replay_diagnostic.png',dpi=150);plt.close(fig)
    (root/'grasp_replay_diagnostic.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':main()
