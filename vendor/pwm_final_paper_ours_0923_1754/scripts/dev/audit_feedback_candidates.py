"""Replay every saved best-of-K candidate to diagnose the learned cost's ranking.

Physics-based candidate ranking is an oracle diagnostic. It adds simulator access
and is not the deployed learned-model selection method or an independent test set.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from guidance.collision_distance import ConvexCollisionDistance
from guidance.feedback import rigid_feedback_batch, reanchor_finger_calibration
from guidance.goal_region import PositionGoalRegion, PositionRegionValidator
from guidance.mesh_physical import MeshPickValidator, MeshPlaceValidator
from guidance.persistent_replay import PersistentPickupReplay, planned_finger_commands
from scripts.dev.audit_feedback_replays import physical_prediction


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--out',type=Path,help='Fresh output directory for an explicit repeat qualification')
    args=parser.parse_args();root=args.root.resolve();run=args.run.resolve()
    report=json.loads((run/'report.json').read_text());a=report['arguments']
    if report['status']!='completed' or a.get('method')!='best_of_k':
        raise ValueError('Requires a completed saved best-of-K feedback experiment')
    report_hash=hashlib.sha256((run/'report.json').read_bytes()).hexdigest()
    out=args.out.resolve() if args.out else run/'candidate_physics_v1'
    if (out/'report.json').exists():
        assert json.loads((out/'report.json').read_text())['source_report_sha256']==report_hash
        return
    out.mkdir(exist_ok=False)
    for path in a['dependency_path']:sys.path.append(str(Path(path).resolve()))
    from scipy.spatial.transform import Rotation
    base=Path(a['base_run']);protocol=json.loads((base/'protocol.json').read_text());options=protocol['arguments']
    b=torch.load(options['input'],weights_only=False,map_location='cpu')
    g=torch.load(options['geometry'],weights_only=False,map_location='cpu')
    calibration=torch.load(options['grasp'],weights_only=False,map_location='cpu')['calibration']
    c={k:np.asarray(v,dtype=np.float64) for k,v in calibration.items()}
    query=ConvexCollisionDistance(g,vendor_paths=a['dependency_path'])
    scene=next(s for s in json.loads((root/'scenes_v2.json').read_text())['scenes']
               if Path(s['directory']).resolve()==Path(options['input']).resolve().parent)
    controls=next(s for s in json.loads((root/'mujoco_controls_matched_v2_summary.json').read_text())['results'] if s['scene']==scene['name'])
    paths=[Path(a['replay']),Path(a['fine_replay'])]
    references=[np.load(p/'execution.npz') for p in paths]
    target=torch.tensor(report['goal']).float();C=b.context_len;frame=report['prefix_frames'];H=report['total_frames']
    region=None
    if report['goal_region_bounds']:
        region=PositionGoalRegion(*[[float(v) for v in report['goal_region_bounds'][k].split(',')] for k in ('lower','upper')])
    def validator(batch,calibration):
        cls=MeshPickValidator if options['task']=='pick' else MeshPlaceValidator
        extra=dict(target_R=torch.tensor(report['goal_rotation']).float()) if report['goal_rotation'] is not None else {}
        v=cls(batch,target,g,calibration,collision=query,hold_frames=options['hold_frames'],
            mesh_contact_tolerance=options['mesh_contact_tolerance_mm']/1000,
            mesh_penetration_tolerance=options['mesh_penetration_tolerance_mm']/1000,**extra)
        return PositionRegionValidator(v,region,hold_frames=options['hold_frames']) if region is not None else v
    full_validator=validator(b,calibration);continuation_validators=[]
    for z in references:
        fresh=rigid_feedback_batch(b,z['model_R'][None,frame-C+1:frame+1],z['model_T'][None,frame-C+1:frame+1],horizon=H-frame)
        continuation_validators.append(validator(fresh,reanchor_finger_calibration(fresh,calibration)))
    traces=[json.loads(line) for line in (run/'trace.jsonl').read_text().splitlines()]
    assert [v['draw'] for v in traces]==list(range(a['k']))
    rows=[];parity=[]
    for draw in range(a['k']):
        source=run/f'candidate_{draw:03d}.npz';z=np.load(source)
        R,T=z['R'][0,C-1:].astype(np.float64),z['T'][0,C-1:].astype(np.float64)
        commands=planned_finger_commands(R,T,c,query)['commanded'];steps=[];full_poses=[]
        for step,(path,reference,continuation_validator) in enumerate(zip(paths,references,continuation_validators)):
            engine=PersistentPickupReplay(path,frame=frame)
            executed=engine.advance(R[:,1],T[:,1],commands,dt=float(reference['dt']),joint_limits=c['limits'][:,:2])
            full_R=np.concatenate([reference['model_R'][:frame],executed['R']])
            full_T=np.concatenate([reference['model_T'][:frame],executed['T']]);full_poses.append((full_R,full_T))
            validation=full_validator(physical_prediction(b,full_R[1:],full_T[1:]))
            continuation=continuation_validator(physical_prediction(b,executed['R'][1:],executed['T'][1:]))
            hand_error=float(np.linalg.norm(executed['T'][:,1]-T[:,1],axis=-1).max()*1000)
            hand_rotation=float(np.rad2deg(Rotation.from_matrix(executed['R'][:,1]@R[:,1].swapaxes(-1,-2)).magnitude()).max())
            prefix_hand=float(np.linalg.norm(reference['model_T'][:frame+1,1]-reference['desired_T'][:frame+1,1],axis=-1).max()*1000)
            prefix_rotation=float(np.rad2deg(Rotation.from_matrix(reference['model_R'][:frame+1,1]@reference['desired_R'][:frame+1,1].swapaxes(-1,-2)).magnitude()).max())
            original_result=json.loads((path/'result.json').read_text())
            finite=bool(np.isfinite(full_R).all() and np.isfinite(full_T).all())
            tracking=bool(finite and not any(executed['warning_counts']) and original_result['finite'] and not any(original_result['warning_counts'])
                          and max(hand_error,prefix_hand)<=3 and max(hand_rotation,prefix_rotation)<=3)
            goal_frames=full_T[-options['hold_frames']:,0]
            goal_violation=float(np.maximum(np.asarray(region.lower)-goal_frames,goal_frames-np.asarray(region.upper)).max()*1000) if region else float(np.linalg.norm(goal_frames-np.asarray(report['goal']).reshape(3),axis=-1).max()*1000-20)
            steps.append(dict(step=float(engine.model.opt.timestep),full_validation=validation,continuation_validation=continuation,
                full_tracking_ok=tracking,hand_tracking_max_mm=hand_error,hand_rotation_max_deg=hand_rotation,
                held_goal_violation_mm=goal_violation,object_lift_mm=float((full_T[-1,0,2]-full_T[0,0,2])*1000)))
            np.savez_compressed(out/f'candidate_{draw:03d}_step{step}.npz',R=full_R,T=full_T,
                q=executed['q'],time=executed['time'],integration_states=executed['integration_states'])
            for compare in (['feedback_sample'] if draw==0 else [])+(['feedback_best_of_k'] if draw==report['selected_draw'] else []):
                expected=np.load(run/f'{compare}_step{step}.npz')
                for key,value in [('R',full_R),('T',full_T)]+[(k,executed[k]) for k in ('q','time','integration_states')]:
                    np.testing.assert_array_equal(value,expected[key])
                expected_metrics=next(v for v in report['results'] if v['variant']==compare)['replays'][step]
                for key,value in validation['residuals_m'].items():
                    if abs(value-expected_metrics['validation']['residuals_m'][key])>1e-6:
                        raise ValueError('Candidate validation differs from the original driver')
                parity.append(dict(draw=draw,step=step,variant=compare,bitwise=True))
        dp=float(np.linalg.norm(full_poses[0][1][:,0]-full_poses[1][1][:,0],axis=-1).max()*1000)
        dr=float(np.rad2deg(Rotation.from_matrix(full_poses[0][0][:,0]@full_poses[1][0][:,0].swapaxes(-1,-2)).magnitude()).max())
        stable=dp<=5 and dr<=5
        accepted=bool(controls['qualified_for_generated_diagnostic'] and stable and all(s['full_tracking_ok'] and s['full_validation']['accepted'] for s in steps))
        row=dict(draw=draw,model_cost=traces[draw]['loss'],selected_by_model=draw==report['selected_draw'],replays=steps,
            timestep_stable=stable,timestep_position_max_mm=dp,timestep_rotation_max_deg=dr,qualified_full_task_pass=accepted,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        rows.append(row);print(draw,'lifts',[round(s['object_lift_mm'],2) for s in steps],'full pass',accepted,flush=True)
    output=dict(status='completed',source_report_sha256=report_hash,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        run=run.name,scene=scene['name'],task=scene['task'],context_source=a['context_source'],k=a['k'],seed=report['seed'],
        selected_by_model=report['selected_draw'],control_qualified=controls['qualified_for_generated_diagnostic'],
        bitwise_driver_replay_checks=parity,qualified_full_task_passes=sum(v['qualified_full_task_pass'] for v in rows),results=rows,
        limitations='Replays every saved candidate using added simulator access. Any oracle ranking is exploratory, not a learned-model result. Fixed prefix failures remain in the full task. Fine step replays the same commands, and criterion thresholds are unchanged.')
    (out/'report.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
