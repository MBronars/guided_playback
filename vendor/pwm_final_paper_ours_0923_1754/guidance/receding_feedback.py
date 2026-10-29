"""Repeated initial-noise planning while independent physics remains continuous."""
import json
import time

import numpy as np
import torch

from guidance.feedback import rigid_feedback_batch,reanchor_finger_calibration
from guidance.persistent_replay import PersistentPickupReplay,planned_finger_commands
from guidance.unconditional import sample_rollout,noise_opt,prediction,_detach


def replan_schedule(horizon,prefix,interval,hold):
    """Keep the original deadline and never plan less than the goal hold interval."""
    if any(isinstance(v,bool) or not isinstance(v,int) for v in (horizon,prefix,interval,hold)):
        raise ValueError('Planning frame counts must be integers')
    if min(interval,hold)<1 or not 0<=prefix<=horizon-hold:
        raise ValueError('Invalid repeated-feedback frame counts')
    rows=[];frame=prefix
    while frame<horizon:
        remaining=horizon-frame
        commit=remaining if remaining-interval<hold else interval
        rows.append(dict(frame=frame,horizon=remaining,commit=commit));frame+=commit
    return rows


def run_receding_feedback(args,model,diffusion,original,calibration,protocol,
        replay_dirs,references,collision,make_objective,make_validator,sampling,provenance):
    from scipy.spatial.transform import Rotation
    options=protocol['arguments'];C=original.context_len;H=protocol['sampling']['horizon']
    stages=replan_schedule(H,args.frame,args.replan_every,options['hold_frames'])
    c={k:v.double().numpy() if torch.is_tensor(v) else v for k,v in calibration.items()}
    device=next(model.parameters()).device;execution_validator=make_validator(original,calibration)
    source=references[0];calls=[0];hook=model.register_forward_pre_hook(lambda *_:calls.__setitem__(0,calls[0]+1))
    started=time.monotonic();planning_seconds=0.;nfe=0;rows=[];saved={};sample_initials=[]
    variants=['continue_original','feedback_sample','feedback_noise_opt']
    for variant in variants:
        plans=[];program=[];variant_calls=calls[0];variant_seconds=0.;variant_nfe=0
        engine=PersistentPickupReplay(replay_dirs[0],frame=args.frame)
        full_R=source['model_R'][:args.frame+1].copy();full_T=source['model_T'][:args.frame+1].copy()
        predicted_R=source['desired_R'][:args.frame+1].copy();predicted_T=source['desired_T'][:args.frame+1].copy()
        segments=[]
        if variant=='continue_original':
            program=[dict(R=source['desired_R'][args.frame:],T=source['desired_T'][args.frame:],
                          q=source['commanded_q'][args.frame:])]
        else:
            for index,stage in enumerate(stages):
                frame=stage['frame'];remaining=stage['horizon'];commit=stage['commit']
                assert len(full_R)==frame+1
                R,T=(full_R,full_T) if args.context_source=='observed' else (predicted_R,predicted_T)
                feedback=rigid_feedback_batch(original,R[None,-C:],T[None,-C:],horizon=remaining)
                refreshed=reanchor_finger_calibration(feedback,calibration);batch=feedback.to(device)
                objective=make_objective(batch,refreshed);validator=make_validator(batch,refreshed)
                sample_args=dict(sampling,horizon=remaining);before=calls[0];tick=time.monotonic()
                with torch.no_grad():
                    if variant=='feedback_sample':
                        torch.manual_seed(args.seed+index)
                        initial=sample_rollout(model,diffusion,batch,**sample_args)
                        sample_initials.append(_detach(initial))
                    else:
                        paired=sample_initials[index]
                        initial=sample_rollout(model,diffusion,batch,noise=paired['noise'],
                                               rng_states=paired['rng_states'],**sample_args)
                        if index==0:
                            for key in ('R','T','positions'):
                                torch.testing.assert_close(initial[key],paired[key].to(device),rtol=0,atol=0)
                variant_nfe+=initial['nfe'];result=None;roll=initial
                def progress(row):
                    value=dict(row,variant=variant,replan=index,frame=frame,remaining=remaining,
                               seed=args.seed+index,seconds=time.monotonic()-started)
                    with (args.out/'trace.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
                    print(json.dumps(value),flush=True)
                if variant=='feedback_noise_opt':
                    result=noise_opt(model,diffusion,batch,objective,initial=initial,iterations=args.iterations,
                        lr=options['lr'],scope='all',max_noise_rms=options['max_noise_rms'],prior_weight=options['prior_weight'],
                        validator=validator if options['screen'] else None,callback=progress,**sample_args)
                    roll=result['rollout'];variant_nfe+=result['nfe']
                elapsed=time.monotonic()-tick;variant_seconds+=elapsed
                with torch.no_grad():loss,diagnostics=objective(prediction(roll,C),batch)
                plan=dict(**stage,seed=args.seed+index,selected_iteration=result['iteration'] if result else None,
                    loss=float(loss),diagnostics=diagnostics,prediction_validation=validator(prediction(roll,C)),
                    actual_model_calls=calls[0]-before,seconds=elapsed)
                plans.append(plan)
                saved[f'{variant}_{index}']=_detach(dict(batch=feedback,initial=initial,result=result))
                R,T=[roll[key][0,C-1:].detach().double().cpu().numpy() for key in ('R','T')]
                q=planned_finger_commands(R,T,c,collision)['commanded']
                command=dict(R=R[:commit+1],T=T[:commit+1],q=q[:commit+1]);program.append(command)
                segment=engine.advance(command['R'][:,1],command['T'][:,1],command['q'],
                    dt=float(source['dt']),joint_limits=c['limits'][:,:2]);segments.append(segment)
                full_R=np.concatenate([full_R,segment['R'][1:]]);full_T=np.concatenate([full_T,segment['T'][1:]])
                predicted_R=np.concatenate([predicted_R,R[1:commit+1]]);predicted_T=np.concatenate([predicted_T,T[1:commit+1]])
                progress(dict(phase='committed',loss=float(loss),selected_iteration=plan['selected_iteration'],commit=commit))
        executions=[]
        for i,(directory,reference) in enumerate(zip(replay_dirs,references)):
            if variant=='continue_original' or i>0:
                replay=PersistentPickupReplay(directory,frame=args.frame);segments=[]
                for command in program:
                    segments.append(replay.advance(command['R'][:,1],command['T'][:,1],command['q'],
                        dt=float(reference['dt']),joint_limits=c['limits'][:,:2]))
            R=np.concatenate([reference['model_R'][:args.frame+1]]+[s['R'][1:] for s in segments])
            T=np.concatenate([reference['model_T'][:args.frame+1]]+[s['T'][1:] for s in segments])
            assert len(T)==H+1
            if variant=='continue_original':
                np.testing.assert_array_equal(R,reference['model_R']);np.testing.assert_array_equal(T,reference['model_T'])
            rr,tt=torch.from_numpy(R[None,1:]).float(),torch.from_numpy(T[None,1:]).float()
            points=torch.empty(1,H,original.positions.shape[2],3)
            for body in range(4):
                template=original.body_templates[:,body,original.body_template_mask[0,body]]
                points[:,:,original.particle_body[0]==body]=torch.einsum('btij,bnj->btni',rr[:,:,body],template)+tt[:,:,body,None]
            validation=execution_validator(dict(positions=points,R=rr,t=tt,rigid_idx=torch.arange(4)))
            position=max(float(np.linalg.norm(s['T'][:,1]-p['T'][:,1],axis=-1).max()*1000) for s,p in zip(segments,program))
            rotation=max(float(np.rad2deg(Rotation.from_matrix(s['R'][:,1] @ p['R'][:,1].swapaxes(-1,-2)).magnitude()).max()) for s,p in zip(segments,program))
            metrics=dict(step=float(json.loads((directory/'protocol.json').read_text())['arguments']['step']),validation=validation,
                final_goal_mm=float(np.linalg.norm(T[-1,0]-np.asarray(protocol['target']).reshape(3))*1000),
                object_lift_mm=float((T[-1,0,2]-T[0,0,2])*1000),hand_tracking_max_mm=position,hand_rotation_max_deg=rotation,
                warning_counts=np.max([s['warning_counts'] for s in segments],axis=0).tolist(),finite=bool(np.isfinite(R).all() and np.isfinite(T).all()))
            fields={key:np.concatenate([segments[0][key]]+[s[key][1:] for s in segments[1:]]) for key in ('q','time','integration_states')}
            np.savez_compressed(args.out/f'{variant}_step{i}.npz',R=R,T=T,pose_start_frame=0,continuation_start_frame=args.frame,**fields)
            executions.append((metrics,R,T))
        row=dict(variant=variant,replays=[v[0] for v in executions],prediction_validation=None,replans=plans,
                 actual_model_calls=calls[0]-variant_calls,planning_seconds=variant_seconds,forward_nfe=variant_nfe)
        if len(executions)==2:
            a,z=executions;dp=np.linalg.norm(a[2][:,0]-z[2][:,0],axis=-1)*1000
            dr=np.rad2deg(Rotation.from_matrix(a[1][:,0] @ z[1][:,0].swapaxes(-1,-2)).magnitude())
            row.update(timestep_position_max_mm=float(dp.max()),timestep_rotation_max_deg=float(dr.max()),timestep_stable=bool(dp.max()<=5 and dr.max()<=5))
        torch.save(program,args.out/f'{variant}_program.pt');rows.append(row);nfe+=variant_nfe;planning_seconds+=variant_seconds
    hook.remove();assert all(p.grad is None for p in model.parameters())
    torch.save(saved,args.out/'replans.pt')
    report=dict(status='completed',**provenance,sampling=sampling,seed=args.seed,selected_iteration=None,
        optimization_seconds=planning_seconds,actual_model_calls=calls[0],forward_nfe=nfe,prefix_frames=args.frame,total_frames=H,
        replan_schedule=stages,results=rows,context_source=args.context_source,
        limitations=['Repeated planning uses the same original goal and execution deadline.',
            'Fresh-sampling and optimization arms use exactly paired Gaussian noise and routing RNG at every replan; their observed histories can diverge after executing different controls.',
            'Fine-step execution replays the same realized segmented control program, including each new command anchor; it does not replan a separate fine-step policy.',
            'Full task and per-scene controller qualification remain required; this is an independent floating-hand diagnostic, not robot execution.'])
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='completed',actual_model_calls=calls[0],stages=stages)),flush=True)
