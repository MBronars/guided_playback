"""Replay completed gallery exports at both qualified steps and render the results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from scripts.dev.guidance_report_utils import replay_pair, task_validator
from scripts.dev.render_guidance_push import render_group
from scripts.dev.audit_feedback_replays import physical_prediction
from scripts.dev.guidance_force_audit import audit_forces
from utils.reproducibility import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--controls',required=True,type=Path);p.add_argument('--vendor',action='append',default=[])
    args=p.parse_args();root=args.root.resolve();qualifications=json.loads(args.controls.read_text())['results']
    entries=json.loads((root/'gallery_summary.json').read_text())['groups'];unique={}
    for group in entries:
        for row in group['variants']:
            if row['state']=='completed':unique[row['entry']['key']]=row
    summaries=[];text=['# Model prediction versus independent physics','','Only the gripper is commanded in the physics panels. MuJoCo advances the object. '+
        'The two replay timesteps are 0.5 ms and 0.25 ms. The setup uses a floating hand and approximate inertia/contact, not the source simulator or a real robot.','']
    for key,row in unique.items():
        export=Path(row['export']);protocol=json.loads((export/'protocol.json').read_text());options=protocol['arguments']
        scene=options['task']+'_'+Path(options['input']).parent.name
        control=next(q for q in qualifications if q['scene']==scene)
        base=root/'replays'/key;base.mkdir(parents=True,exist_ok=True)
        directories=[base/tag for tag in ('dt0005','dt00025')]
        for directory,step in zip(directories,('.0005','.00025')):
            if (directory/'result.json').exists():continue
            if directory.exists():raise RuntimeError(f'Incomplete replay must be inspected before retrying: {directory}')
            command=[sys.executable,str(args.source/'scripts/dev/replay_pickup_mujoco.py'),
                '--scene',str(Path(options['input']).parent),'--trajectory',str(export/'trajectory.npz'),
                '--frames',str(protocol['sampling']['horizon']),'--finger-mode','mesh_preload','--step',step,
                '--save-frame-states','--out',str(directory)]
            for vendor in args.vendor:command+=['--dependency-path',vendor]
            if options.get('obstacles'):command+=['--obstacles',options['obstacles']]
            environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(args.source))
            with (base/(directory.name+'.log')).open('x') as log:
                subprocess.run(command,env=environment,cwd=args.source,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        audit=replay_pair(protocol,export/'trajectory.npz',directories,control)
        audit['task_pass_before_dense_contact_audit']=audit['qualified_task_pass']
        for replay,directory in zip(audit['replays'],directories):
            force_dir=directory/'forces'
            force=(json.loads((force_dir/'report.json').read_text()) if (force_dir/'report.json').exists()
                   else audit_forces(directory,force_dir,vendor_paths=args.vendor))
            if force['execution_sha256']!=file_sha256(directory/'execution.npz') or not force['bitwise_pose_joint_and_integration_replay']:
                raise ValueError('Force audit does not match this replay')
            replay['forces']=force
            replay['dense_contacts_pass']=(force['peak_object_gripper_depth_mm']<=options['mesh_penetration_tolerance_mm']
                and force['obstacle_contact_solver_steps']==0)
        audit['qualified_task_pass']=audit['qualified_task_pass'] and all(v['dense_contacts_pass'] for v in audit['replays'])
        audit.update(key=key,scene=scene,model_metrics=row['metrics'],controls_sha256=file_sha256(args.controls))
        (base/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n');summaries.append(audit)
        batch,geometry,_,_=task_validator(protocol);C=batch.context_len
        original=np.load(export/'trajectory.npz')
        plotvariants=[];descriptions=[]
        for index,label in enumerate(['Model prediction','Physics: 0.5 ms','Physics: 0.25 ms']):
            if index==0:
                R,T=original['R'],original['T'];positions=original['positions'];sha=file_sha256(export/'trajectory.npz')
            else:
                z=np.load(directories[index-1]/'execution.npz')
                R=np.concatenate([batch.body_poses[:,:C-1,:,:3,:3].numpy(),z['model_R'][None]],axis=1)
                T=np.concatenate([batch.body_poses[:,:C-1,:,:3,3].numpy(),z['model_T'][None]],axis=1)
                positions=physical_prediction(batch,R[0],T[0])['positions'].numpy();sha=file_sha256(directories[index-1]/'execution.npz')
            entry=dict(key=key+'_'+str(index),label=label);descriptions.append(entry)
            plotvariants.append(dict(entry=entry,source_sha256=sha,batch=batch,geometry=geometry,protocol=protocol,
                rollout=dict(R=torch.tensor(R),T=torch.tensor(T),positions=torch.tensor(positions),rigid_idx=torch.arange(4))))
        figures=render_group(dict(key=key+'_execution',title=scene+' / '+row['entry']['label'],kind='physics_replay',variants=descriptions),plotvariants,root)
        text.extend(['## '+scene+' / '+row['entry']['label'],'',f'![Model and physics]({figures["movie"]})','',
            f'Controller qualification: **{audit["control_qualified"]}**. Timestep agreement: **{audit["timestep_stable"]}** '+
            f'({audit["timestep_position_max_mm"]:.1f} mm / {audit["timestep_rotation_max_deg"]:.1f}°). '+
            f'Full qualified task: **{audit["qualified_task_pass"]}**.','',
            '| Track | Final goal error | Worst held error | Full task geometry |','|---|---:|---:|---|',
            f'| Model | {row["metrics"]["final_goal_mm"]:.1f} mm | {row["metrics"]["held_worst_goal_mm"]:.1f} mm | {row["metrics"]["geometry_pass"]} |'])
        for replay in audit['replays']:
            text.append(f'| Physics {replay["step"]*1000:g} ms | {replay["final_goal_mm"]:.1f} mm | {replay["held_worst_goal_mm"]:.1f} mm | {replay["validation"]["accepted"]} |')
        text.extend(['','Solver-step checks replay the exact saved commands and verify bitwise identical poses, joints and full integration state.'])
        for replay in audit['replays']:
            force=replay['forces']
            text.append(f'- {replay["step"]*1000:g} ms: peak object–gripper overlap {force["peak_object_gripper_depth_mm"]:.2f} mm; '+
                f'obstacle contact at {force["obstacle_contact_solver_steps"]} solver steps; '+
                f'both fingers exert positive normal force for {100*force["both_fingers_positive_normal_fraction"]:.2f}% of the rollout. Force fraction is descriptive, not a success threshold.')
        text.extend(['',f'[All checks and failures]({base.relative_to(root)}/audit.json) · [Paths and goal errors]({figures["paths"]}).',''])
        if audit['duration_beyond_recorded_control']:
            text.extend(['The generated horizon exceeds the duration of the recorded-control qualification; controller calibration is inherited from that shorter test.',''])
        print(key,'execution pass',audit['qualified_task_pass'],flush=True)
    summary=dict(results=summaries,count=len(summaries),qualified_task_passes=sum(v['qualified_task_pass'] for v in summaries))
    (root/'execution_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    temporary=root/'EXECUTION_GALLERY.pending.md';temporary.write_text('\n'.join(text));temporary.replace(root/'EXECUTION_GALLERY.md')


if __name__=='__main__':main()
