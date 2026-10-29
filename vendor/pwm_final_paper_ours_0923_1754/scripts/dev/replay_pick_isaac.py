"""Execute recorded joints or generated hand poses in the source Isaac setup.

Run with isaaclab_rlskills Python, after prepare_pick_replay.py. This reconstructs
the source scene with documented missing historical state; qualify recorded
commands before calling generated playback an execution benchmark.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',required=True)
parser.add_argument('--out',required=True)
parser.add_argument('--mode',choices=['recorded','hand'],default='recorded')
parser.add_argument('--frames',type=int,default=36)
parser.add_argument('--table-offset',type=float,default=0.)
parser.add_argument('--command-offset',type=int,default=0,help='Alignment diagnostic only; baseline is zero')
parser.add_argument('--finger-command-mode',choices=['achieved','recorded','preload'],default='achieved')
parser.add_argument('--close-frame',type=int,help='Explicit future closure frame for the preload controller diagnostic')
parser.add_argument('--finger-preload-mm',type=float,default=6.,help='Extra commanded closure after the declared closure frame; execution qualification required')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
out=Path(args.out).resolve()
if out.exists(): raise FileExistsError(out)
out.mkdir(parents=True)
launcher=AppLauncher(args)
app=launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from guidance.replay_control import finger_command, native_tracks_to_model


def main():
    raw=np.load(args.input)
    receipt=json.loads(Path(args.input).with_suffix('.json').read_text())
    if args.mode=='hand' and args.finger_command_mode=='recorded' and receipt['source_kind']!='recorded positive control':
        raise ValueError('Recorded future finger commands are reserved for the recorded-trajectory positive control')
    if args.mode=='hand' and args.finger_command_mode=='preload' and args.close_frame is None:
        raise ValueError('Specify the planned closure frame for preload control')
    source=Path(receipt['collector_config'])
    sys.path.insert(0,str(source.parents[3]))
    spec=importlib.util.spec_from_file_location('pickup_replay_source_config',source)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg=module.BaseEnvCfg()
    sim_cfg=cfg.sim.copy(); sim_cfg.device=args.device
    sim=sim_utils.SimulationContext(sim_cfg)
    robot_cfg=cfg.robot.copy(); robot_cfg.prim_path='/World/Robot'
    robot=Articulation(robot_cfg)
    table_cfg=cfg.table.copy(); table_cfg.prim_path='/World/Table'
    table_cfg.init_state.pos=(.5,0.,receipt['table_top_z']-.01+args.table_offset)
    table=RigidObject(table_cfg)
    obj_cfg=RigidObjectCfg(prim_path='/World/Object',
        spawn=sim_utils.UsdFileCfg(usd_path=receipt['object_usd'],
            rigid_props=cfg.object.spawn.rigid_props.copy()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(raw['initial_object_pose'][:3]),
                                                rot=tuple(raw['initial_object_pose'][3:])))
    obj=RigidObject(obj_cfg)
    sim.reset()
    device=sim.device
    to=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
    names=[f'panda_joint{i}' for i in range(1,8)]+['panda_finger_joint1','panda_finger_joint2']
    ids,found=robot.find_joints(names,preserve_order=True)
    if found!=names: raise ValueError(f'Joint names do not match collector order: {found}')
    bodies,body_names=robot.find_bodies(['panda_hand','panda_leftfinger','panda_rightfinger'],preserve_order=True)
    robot.write_joint_state_to_sim(to(raw['initial_joint_pos'])[None],to(raw['initial_joint_vel'])[None],joint_ids=ids)
    robot.set_joint_position_target(to(raw['initial_joint_pos'])[None],joint_ids=ids)
    obj.write_root_pose_to_sim(to(raw['initial_object_pose'])[None])
    obj.write_root_velocity_to_sim(to(raw['initial_object_velocity'])[None])
    robot.reset(); obj.reset(); table.reset()
    robot.update(0.); obj.update(0.); table.update(0.)
    controller=DifferentialIKController(DifferentialIKControllerCfg(command_type='pose',
        use_relative_mode=False,ik_method='dls'),num_envs=1,device=device)
    target=to(raw['desired_body_pose'])
    finger=to(raw['desired_finger_q'])
    commands=to(raw['recorded_commands'])
    frames=min(args.frames,len(target)-1)
    dt=sim.get_physics_dt(); substeps=round(float(raw['dt'])/dt)
    if abs(substeps*dt-float(raw['dt']))>1e-8: raise ValueError('Frame spacing not divisible by simulation step')
    pose_list=[]; q_list=[]; target_list=[]
    def capture():
        pose=torch.cat([obj.data.root_pose_w[:,None],robot.data.body_pose_w[:,bodies]],dim=1)
        pose_list.append(pose[0].detach().cpu().numpy().copy())
        q_list.append(robot.data.joint_pos[0,ids].detach().cpu().numpy().copy())
    capture()
    initial_hand_error=float(np.linalg.norm(pose_list[0][1,:3]-raw['reference_body_pose'][0,1,:3]))
    protocol=dict(arguments=vars(args),source=receipt,dt=dt,substeps=substeps,joint_names=found,
        body_names=body_names,initial_hand_error_m=initial_hand_error,
        actual_object_mass_kg=obj.root_physx_view.get_masses().cpu().tolist(),
        object_materials=obj.root_physx_view.get_material_properties().cpu().tolist(),
        controller='recorded joint targets' if args.mode=='recorded' else 'per-physics-step damped differential IK, source PD actuators',
        file_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),source,Path(receipt['object_usd']),Path(receipt['robot_usd']),Path(args.input))})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(dict(initial_hand_error_mm=initial_hand_error*1000,frames=frames,
        object_mass_kg=protocol['actual_object_mass_kg'])),flush=True)
    if initial_hand_error>.005:
        raise ValueError('Initial simulator hand differs from recorded state by more than 5 mm; inspect robot/config vintage')
    t0=time.perf_counter()
    for frame in range(frames):
        for substep in range(substeps):
            if args.mode=='recorded':
                index=frame+args.command_offset
                if not 0<=index<len(commands): raise ValueError('Command offset leaves available source sequence')
                q_des=commands[index][None]
            else:
                # Root is fixed at the world origin; Jacobian and desired hand use world axes.
                controller.set_command(target[frame+1,1][None])
                jacobian=robot.root_physx_view.get_jacobians()[:,bodies[0]-1,:,:][:,:,ids[:7]]
                current=robot.data.body_pose_w[:,bodies[0]]
                arm=controller.compute(current[:,:3],current[:,3:],jacobian,robot.data.joint_pos[:,ids[:7]])
                limits=robot.data.soft_joint_pos_limits[:,ids[:7]]
                arm=arm.clamp(min=limits[:,:,0],max=limits[:,:,1])
                recorded_fingers=None
                if args.finger_command_mode=='recorded':
                    index=frame+args.command_offset
                    if not 0<=index<len(commands): raise ValueError('Finger command offset leaves the recording')
                    recorded_fingers=commands[index,-2:]
                fq=finger_command(finger[frame+1],mode=args.finger_command_mode,frame=frame+1,
                    recorded=recorded_fingers,close_frame=args.close_frame,
                    preload_m=args.finger_preload_mm/1000)[None]
                q_des=torch.cat([arm,fq],dim=-1)
            robot.set_joint_position_target(q_des,joint_ids=ids)
            robot.write_data_to_sim(); obj.write_data_to_sim(); table.write_data_to_sim()
            sim.step(render=False)
            robot.update(dt); obj.update(dt); table.update(dt)
        capture(); target_list.append(q_des[0].detach().cpu().numpy().copy())
    track=np.array(pose_list); q=np.array(q_list)
    desired=raw['desired_body_pose'][:frames+1]
    ref=raw['reference_body_pose'][:frames+1]
    result=dict(seconds=time.perf_counter()-t0,frames=frames,
        hand_tracking_mean_mm=float(np.linalg.norm(track[1:,1,:3]-desired[1:,1,:3],axis=-1).mean()*1000),
        hand_tracking_max_mm=float(np.linalg.norm(track[1:,1,:3]-desired[1:,1,:3],axis=-1).max()*1000),
        object_prediction_final_error_mm=float(np.linalg.norm(track[-1,0,:3]-desired[-1,0,:3])*1000),
        object_recording_final_error_mm=float(np.linalg.norm(track[-1,0,:3]-ref[-1,0,:3])*1000),
        object_executed_lift_mm=float((track[-1,0,2]-track[0,0,2])*1000),
        recorded_object_lift_mm=float((ref[-1,0,2]-ref[0,0,2])*1000))
    result['finger_achieved_tracking_max_mm']=float(np.abs(q[1:,-2:]-raw['desired_finger_q'][1:frames+1]).max()*1000)
    executed_R,executed_t=native_tracks_to_model(track,raw['native_to_template_R'],
        raw['native_to_template_t'],raw['canonical_shift'])
    if 'requested_target' in raw and raw['requested_target'].shape==(3,):
        errors=(executed_t[:,0]-torch.as_tensor(raw['requested_target'])).norm(dim=-1)*1000
        result.update(requested_goal_final_error_mm=float(errors[-1]),requested_goal_last_six_max_error_mm=float(errors[-6:].max()))
    np.savez_compressed(out/'execution.npz',body_pose=track,joint_pos=q,joint_targets=np.array(target_list),
        desired=desired,recording=ref,native_to_template_R=raw['native_to_template_R'],
        native_to_template_t=raw['native_to_template_t'],canonical_shift=raw['canonical_shift'],
        model_R=executed_R.numpy(),model_T=executed_t.numpy())
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


try:
    main()
finally:
    app.close()
