"""One observed-state correction to a saved unconditional pick/place plan.

The original plan executes a fixed prefix. Standard initial-noise optimization
then replans the remaining duration from the latest observed poses. The goal is
unchanged. Compare continuing the original plan, fresh sampling after feedback,
and optimizing that same fresh draw. This is an independent-engine diagnostic.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from guidance.feedback import rigid_feedback_batch, reanchor_finger_calibration
from guidance.goal_region import PositionGoalRegion, PositionRegionValidator
from guidance.obstacles import load_obstacles, StaticObstacleDistance, ObstacleObjective, ObstacleValidator
from guidance.mesh_physical import MeshPickPhysicalObjective, MeshPlacePhysicalObjective, MeshPickValidator, MeshPlaceValidator
from guidance.persistent_replay import PersistentPickupReplay, planned_finger_commands
from guidance.unconditional import sample_rollout, noise_opt, best_of_k, prediction, _detach
from inference import load_model_and_diffusion
from utils.reproducibility import code_revision, file_sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run',required=True,type=Path)
    parser.add_argument('--base-seed',type=int,default=8311)
    parser.add_argument('--replay',required=True,type=Path,help='Saved base-plan replay with complete integration states')
    parser.add_argument('--fine-replay',type=Path,help='Replay the same realized command program at a finer step; no second feedback optimization')
    parser.add_argument('--frame',type=int,default=16,help='Number of original-plan frames executed before feedback')
    parser.add_argument('--context-source',choices=['observed','predicted'],default='observed',
                        help='Observed feedback, or a matched ablation that replans from the old model prediction')
    parser.add_argument('--iterations',type=int,default=24)
    parser.add_argument('--method',choices=['noise_opt','best_of_k'],default='noise_opt')
    parser.add_argument('--k',type=int,default=49,help='Candidate count for best_of_k; the paired initial draw is also evaluated separately')
    parser.add_argument('--replan-every',type=int,help='Execute this many frames between fresh observations and replanning')
    parser.add_argument('--seed',type=int,default=6381)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--dependency-path',action='append',default=[])
    args=parser.parse_args()
    if args.frame<1 or args.iterations<0:parser.error('Frame must be positive and updates nonnegative')
    if args.k<1:parser.error('Candidate count must be positive')
    if args.method=='best_of_k' and args.iterations!=0:
        parser.error('Use --iterations 0 for best_of_k; candidate count is --k')
    if args.replan_every is not None and (args.replan_every<1 or args.method!='noise_opt'):
        parser.error('Repeated feedback requires a positive interval and method noise_opt')
    if args.out.exists():raise FileExistsError(args.out)
    import sys
    for path in args.dependency_path:sys.path.append(str(Path(path).resolve()))
    from scipy.spatial.transform import Rotation
    base=args.base_run.resolve();protocol=json.loads((base/'protocol.json').read_text());options=protocol['arguments']
    if options['objective']!='physical' or options['collision_backend']!='fcl' or options.get('constraints'):
        raise ValueError('Feedback experiment requires a saved proposal-free mesh physical plan without AL')
    if options.get('observation_translation_mm',0) or options.get('observation_rotation_deg',0) or options.get('density_scale',1)!=1:
        raise ValueError('Observation-corruption experiments require a separately specified feedback protocol')
    original=torch.load(options['input'],weights_only=False,map_location='cpu')
    if original.body_type.shape!=(1,4) or original.pose_action_mask[0].tolist()!=[False,True,True,True]:
        raise ValueError('This replay experiment requires one passive object, a hand and two fingers')
    geometry=torch.load(options['geometry'],weights_only=False,map_location='cpu')
    calibration=torch.load(options['grasp'],weights_only=False,map_location='cpu')['calibration']
    obstacles=None
    if options.get('obstacles'):
        if file_sha256(options['obstacles'])!=protocol.get('obstacles_sha256'):
            raise ValueError('Obstacle geometry changed since the base plan')
        obstacles=StaticObstacleDistance(geometry,load_obstacles(options['obstacles']),vendor_paths=args.dependency_path)
    C=original.context_len;H=int(protocol['sampling']['horizon']);remaining=H-args.frame
    if args.frame<C-1 or remaining<options['hold_frames']:
        raise ValueError('Feedback must have a full observed context and leave the declared goal hold interval')
    if file_sha256(options['checkpoint'])!=protocol['checkpoint_sha256'] or file_sha256(options['input'])!=protocol['input_sha256']:
        raise ValueError('The checkpoint or input changed since the base plan')
    if file_sha256(options['geometry'])!=protocol['geometry_sha256'] or file_sha256(options['grasp'])!=protocol['grasp_sha256']:
        raise ValueError('Geometry or joint calibration changed since the base plan')
    base_arrays=np.load(base/f'seed_{args.base_seed}.npz')
    replay_dirs=[args.replay]+([args.fine_replay] if args.fine_replay else [])
    references=[np.load(path/'execution.npz') for path in replay_dirs]
    for path,reference in zip(replay_dirs,references):
        recorded=json.loads((path/'protocol.json').read_text())
        if recorded['arguments'].get('preload_policy','instant')!='instant':
            raise ValueError('Latched-preload feedback requires carrying its controller state; only stateless preload is supported here')
        replay_obstacles=recorded['arguments'].get('obstacles')
        if bool(replay_obstacles)!=bool(obstacles):
            raise ValueError('Feedback replay and planning obstacles must match')
        if replay_obstacles and recorded['sha256'].get(str(Path(replay_obstacles).resolve()))!=protocol['obstacles_sha256']:
            raise ValueError('Feedback replay used different obstacle geometry')
        if recorded['arguments']['finger_mode']!='mesh_preload' or recorded['arguments'].get('finger_velocity_feedforward',False):
            raise ValueError('Feedback uses the qualified fixed mesh-preload controller')
        if recorded['arguments'].get('joint_limit_time') is not None:
            raise ValueError('Altered finger limit dynamics require a separate feedback qualification')
        if Path(recorded['arguments']['scene']).resolve()!=Path(options['input']).resolve().parent:
            raise ValueError('Base replay belongs to a different observed scene')
        canonical=dict(preload=.006,mesh_contact_tolerance=.002,weld_time=.002,contact_time=.005,
                       friction_scale=1.,table_offset=0.)
        if any(recorded['arguments'].get(k)!=value for k,value in canonical.items()):
            raise ValueError('Feedback requires the fixed controller and contact settings qualified on recorded controls')
        if len(reference['model_T'])!=H+1:raise ValueError('Base replay does not cover the complete plan')
        np.testing.assert_array_equal(reference['desired_R'],base_arrays['R'][0,C-1:])
        np.testing.assert_array_equal(reference['desired_T'],base_arrays['T'][0,C-1:])
    source=references[0]
    prefix='model' if args.context_source=='observed' else 'desired'
    observed_R=source[prefix+'_R'][None,args.frame-C+1:args.frame+1]
    observed_T=source[prefix+'_T'][None,args.frame-C+1:args.frame+1]
    feedback=rigid_feedback_batch(original,observed_R,observed_T,horizon=remaining)
    refreshed=reanchor_finger_calibration(feedback,calibration)
    args.out.mkdir(parents=True)
    provenance=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        **code_revision(),runner_sha256=file_sha256(__file__),checkpoint_sha256=protocol['checkpoint_sha256'],
        base_protocol_sha256=file_sha256(base/'protocol.json'),base_plan_sha256=file_sha256(base/f'seed_{args.base_seed}.npz'),
        replay_state_sha256={str(path.resolve()):file_sha256(path/'integration_states.npz') for path in replay_dirs},
        goal=protocol['target'],goal_rotation=protocol.get('target_rotation'),goal_region_bounds=protocol.get('goal_region_bounds'))
    (args.out/'protocol.json').write_text(json.dumps(provenance,indent=2)+'\n')
    torch.save(dict(batch=feedback,calibration=refreshed),args.out/'feedback_input.pt')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch=feedback.to(device);target=torch.tensor(protocol['target'],device=device)
    target_R=torch.tensor(protocol['target_rotation'],device=device) if protocol.get('target_rotation') is not None else None
    region=None
    if protocol.get('goal_kind')=='position_region':
        region=PositionGoalRegion(*[[float(v) for v in protocol['goal_region_bounds'][key].split(',')] for key in ('lower','upper')])
    def make_objective(b,c):
        extra=dict(target_R=target_R,rotation_weight=options['rotation_weight'],settle_frames=options['hold_frames']) if options['task']=='place' else {}
        if options['task']=='pick':
            extra.update(goal_frames=options['hold_frames'] if region is not None or options.get('pick_goal_hold') else 1,goal_region=region)
        cls=MeshPlacePhysicalObjective if options['task']=='place' else MeshPickPhysicalObjective
        result=cls(b,target.to(b.device),geometry,c,vendor_paths=args.dependency_path,
            mesh_contact_tolerance=options['mesh_contact_tolerance_mm']/1000,
            mesh_penetration_tolerance=options['mesh_penetration_tolerance_mm']/1000,
            contact_weight=options['contact_weight'],rise_weight=options['rise_weight'],slip_weight=options['slip_weight'],
            penetration_weight=options['penetration_weight'],support_weight=options['support_weight'],
            joint_weight=options['joint_weight'],speed_weight=options['joint_speed_weight'],
            goal_weight=options.get('goal_weight',1.),goal_reduction=options.get('goal_reduction','mean'),
            physical_weight=options.get('physical_weight',1.),**extra)
        return ObstacleObjective(result,obstacles,weight=options['obstacle_weight'],
            clearance=options['obstacle_clearance_mm']/1000,substeps=options['obstacle_substeps']) if obstacles is not None else result
    objective=make_objective(batch,refreshed)
    def make_validator(b,c):
        cls=MeshPlaceValidator if options['task']=='place' else MeshPickValidator
        extra=dict(target_R=target_R.to(b.device)) if target_R is not None else {}
        v=cls(b,target.to(b.device),geometry,c,collision=objective.collision,hold_frames=options['hold_frames'],
              mesh_contact_tolerance=options['mesh_contact_tolerance_mm']/1000,
              mesh_penetration_tolerance=options['mesh_penetration_tolerance_mm']/1000,**extra)
        if region is not None:v=PositionRegionValidator(v,region,hold_frames=options['hold_frames'])
        return ObstacleValidator(v,b,obstacles,clearance=options['obstacle_clearance_mm']/1000,
            substeps=options['obstacle_audit_substeps']) if obstacles is not None else v
    predicted_validator=make_validator(batch,refreshed)
    execution_validator=make_validator(original,calibration)
    model,diffusion,cfg,steps=load_model_and_diffusion(options['checkpoint'],device)
    if float(cfg.train.action_dropout)!=1 or not bool(cfg.model.unet_bidirectional_temporal):
        raise ValueError('Feedback requires the pinned unconditional bidirectional checkpoint')
    for parameter in model.parameters():parameter.requires_grad_(False)
    sampling=dict(protocol['sampling'],horizon=remaining)
    if sampling['steps']!=steps or sampling['future_len']!=int(cfg.data.future_len):
        raise ValueError('Saved sampling protocol does not match the checkpoint')
    if args.replan_every is not None:
        from guidance.receding_feedback import run_receding_feedback
        run_receding_feedback(args,model,diffusion,original,calibration,protocol,
            replay_dirs,references,objective.collision,make_objective,make_validator,sampling,provenance)
        return
    calls=[0]
    hook=model.register_forward_pre_hook(lambda *_: calls.__setitem__(0,calls[0]+1))
    started=time.monotonic();torch.manual_seed(args.seed)
    with torch.no_grad():initial=sample_rollout(model,diffusion,batch,**sampling)
    def progress(row):
        value=dict(row,seconds=time.monotonic()-started)
        with (args.out/'trace.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    if args.method=='noise_opt':
        result=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=args.iterations,
            lr=options['lr'],scope='all',max_noise_rms=options['max_noise_rms'],prior_weight=options['prior_weight'],
            validator=predicted_validator if options['screen'] else None,callback=progress,**sampling)
    else:
        torch.manual_seed(args.seed)
        def candidate(draw,roll,row):
            if draw==0:
                for key in ('R','T','positions'):
                    torch.testing.assert_close(roll[key],initial[key],rtol=0,atol=0)
                for left,right in zip(roll['noise'],initial['noise']):
                    torch.testing.assert_close(left,right,rtol=0,atol=0)
                for left,right in zip(roll['rng_states'],initial['rng_states']):
                    for a,z in zip(left,right):
                        if a is not None:torch.testing.assert_close(a,z,rtol=0,atol=0)
            np.savez_compressed(args.out/f'candidate_{draw:03d}.npz',
                R=roll['R'].detach().cpu().numpy(),T=roll['T'].detach().cpu().numpy())
        result=best_of_k(model,diffusion,batch,objective,k=args.k,
            validator=predicted_validator if options['screen'] else None,
            callback=progress,candidate_callback=candidate,**sampling)
    hook.remove();seconds=time.monotonic()-started
    assert all(p.grad is None for p in model.parameters())
    torch.save(_detach(dict(initial=initial,result=result)),args.out/'plans.pt')
    c={k:v.double().numpy() if torch.is_tensor(v) else v for k,v in calibration.items()}
    # Use the original absolute joint convention for every realized command.
    variants=[('continue_original',None),('feedback_sample',initial),('feedback_'+args.method,result['rollout'])]
    rows=[]
    for name,roll in variants:
        if roll is None:
            R,T=source['desired_R'][args.frame:],source['desired_T'][args.frame:]
            command=dict(commanded=source['commanded_q'][args.frame:])
        else:
            R,T=[roll[k][0,C-1:].detach().double().cpu().numpy() for k in ('R','T')]
            command=planned_finger_commands(R,T,c,objective.collision)
        np.savez_compressed(args.out/(name+'_commands.npz'),R=R,T=T,**command)
        executions=[]
        for step,(path,reference) in enumerate(zip(replay_dirs,references)):
            engine=PersistentPickupReplay(path,frame=args.frame)
            executed=engine.advance(R[:,1],T[:,1],command['commanded'],dt=float(reference['dt']),joint_limits=c['limits'][:,:2])
            full_R=np.concatenate([reference['model_R'][:args.frame],executed['R']])
            full_T=np.concatenate([reference['model_T'][:args.frame],executed['T']])
            if roll is None:
                np.testing.assert_array_equal(full_R,reference['model_R'])
                np.testing.assert_array_equal(full_T,reference['model_T'])
            rr,tt=torch.from_numpy(full_R[None,1:]).float(),torch.from_numpy(full_T[None,1:]).float()
            positions=torch.empty(1,H,original.positions.shape[2],3)
            for k in range(4):
                template=original.body_templates[:,k,original.body_template_mask[0,k]]
                positions[:,:,original.particle_body[0]==k]=torch.einsum('btij,bnj->btni',rr[:,:,k],template)+tt[:,:,k,None]
            validation=execution_validator(dict(positions=positions,R=rr,t=tt,rigid_idx=torch.arange(4)))
            hand_error=np.linalg.norm(executed['T'][:,1]-T[:,1],axis=-1)*1000
            hand_rotation=np.rad2deg(Rotation.from_matrix(executed['R'][:,1] @ R[:,1].swapaxes(-1,-2)).magnitude())
            metrics=dict(step=float(engine.model.opt.timestep),validation=validation,
                final_goal_mm=float(np.linalg.norm(full_T[-1,0]-np.asarray(protocol['target']).reshape(3))*1000),
                object_lift_mm=float((full_T[-1,0,2]-full_T[0,0,2])*1000),
                hand_tracking_max_mm=float(hand_error.max()),hand_rotation_max_deg=float(hand_rotation.max()),
                warning_counts=executed['warning_counts'].tolist(),finite=bool(np.isfinite(full_R).all() and np.isfinite(full_T).all()))
            np.savez_compressed(args.out/f'{name}_step{step}.npz',R=full_R,T=full_T,
                continuation_start_frame=args.frame,pose_start_frame=0,
                **{k:v for k,v in executed.items() if k not in ('R','T')})
            executions.append((metrics,full_R,full_T))
        row=dict(variant=name,replays=[v[0] for v in executions],prediction_validation=predicted_validator(prediction(roll,C)) if roll is not None else None)
        if len(executions)==2:
            a,z=executions
            dp=np.linalg.norm(a[2][:,0]-z[2][:,0],axis=-1)*1000
            dr=np.rad2deg(Rotation.from_matrix(a[1][:,0] @ z[1][:,0].swapaxes(-1,-2)).magnitude())
            row.update(timestep_position_max_mm=float(dp.max()),timestep_rotation_max_deg=float(dr.max()),timestep_stable=bool(dp.max()<=5 and dr.max()<=5))
        rows.append(row)
    report=dict(status='completed',**provenance,sampling=sampling,seed=args.seed,
        selected_iteration=result.get('iteration'),selected_draw=result.get('draw'),
        optimization_seconds=seconds,actual_model_calls=calls[0],forward_nfe=initial['nfe']+result['nfe'],
        prefix_frames=args.frame,total_frames=H,
        context_source='Actual independent-engine replay poses' if args.context_source=='observed' else 'Old model prediction; observed feedback withheld',results=rows,
        limitations=['One feedback correction; same fixed goal and total execution duration.',
            'Fine-step run, if present, replays the same realized two-segment command program; it does not optimize a second policy from finer-step observations.',
            'Feedback optimization adds compute; continuing the original plan is not a matched-compute method comparison.',
            'Recorded-control qualification must still be checked per scene; these are independent-engine diagnostics, not robot success.'])
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('results','arguments','limitations')}),flush=True)


if __name__=='__main__':main()
