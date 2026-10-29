"""Separate immutable prefix failures, continuation quality and full task checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from guidance.feedback import rigid_feedback_batch, reanchor_finger_calibration
from guidance.goal_region import PositionGoalRegion, PositionRegionValidator
from guidance.collision_distance import ConvexCollisionDistance
from guidance.mesh_physical import MeshPickValidator, MeshPlaceValidator


def physical_prediction(batch, rotations, translations):
    R, t = torch.as_tensor(rotations).float()[None], torch.as_tensor(translations).float()[None]
    points = torch.empty(1, len(rotations), batch.positions.shape[2], 3)
    for k in range(4):
        template = batch.body_templates[:, k, batch.body_template_mask[0, k]]
        points[:, :, batch.particle_body[0] == k] = torch.einsum('btij,bnj->btni', R[:, :, k], template) + t[:, :, k, None]
    return dict(positions=points, R=R, t=t, rigid_idx=torch.arange(4))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve()
    scenes={str(Path(s['directory']).resolve()):s for s in json.loads((root/'scenes_v2.json').read_text())['scenes']}
    qualifications={s['scene']:s for s in json.loads((root/'mujoco_controls_matched_v2_summary.json').read_text())['results']}
    rows=[];cache={}
    hard_keys=('penetration','support','gripper_support','opening','speed')
    for path in sorted(root.glob('*/report.json')):
        report=json.loads(path.read_text())
        if report.get('status')!='completed' or 'prefix_frames' not in report or 'base_plan_sha256' not in report:
            continue
        args=report['arguments'];base=Path(args['base_run']);protocol=json.loads((base/'protocol.json').read_text());options=protocol['arguments']
        scene=scenes[str(Path(options['input']).resolve().parent)];name=scene['name']
        if name not in cache:
            b=torch.load(options['input'],weights_only=False,map_location='cpu')
            g=torch.load(options['geometry'],weights_only=False,map_location='cpu')
            c=torch.load(options['grasp'],weights_only=False,map_location='cpu')['calibration']
            query=ConvexCollisionDistance(g,vendor_paths=[root/'vendor_fcl',root.parent/'overnight_2026-09-14/vendor'])
            cache[name]=(b,g,c,query)
        b,g,c,query=cache[name];C=b.context_len;frame=report['prefix_frames'];H=report['total_frames']
        region=None
        if report['goal_region_bounds']:
            region=PositionGoalRegion(*[[float(v) for v in report['goal_region_bounds'][k].split(',')] for k in ('lower','upper')])
        def validator(batch,calibration,hold=options['hold_frames']):
            cls=MeshPickValidator if options['task']=='pick' else MeshPlaceValidator
            extra=dict(target_R=torch.tensor(report['goal_rotation'])) if report['goal_rotation'] is not None else {}
            v=cls(batch,torch.tensor(report['goal']),g,calibration,collision=query,hold_frames=hold,
                  mesh_contact_tolerance=options['mesh_contact_tolerance_mm']/1000,
                  mesh_penetration_tolerance=options['mesh_penetration_tolerance_mm']/1000,**extra)
            return PositionRegionValidator(v,region,hold_frames=hold) if region is not None else v
        qualified=qualifications[name]['qualified_for_generated_diagnostic']
        variants=[]
        for item in report['results']:
            parts=[]
            for i,result in enumerate(item['replays']):
                z=np.load(path.parent/f'{item["variant"]}_step{i}.npz');R,T=z['R'],z['T']
                full=validator(b,c)(physical_prediction(b,R[1:],T[1:]))
                # Same full criterion and saved realized poses as the driver.
                for k,v in full['residuals_m'].items():
                    if abs(v-result['validation']['residuals_m'][k])>1e-6:
                        raise ValueError(f'Full feedback audit mismatch: {path.parent.name} {item["variant"]} {k}')
                prefix=validator(b,c,min(frame,options['hold_frames']))(physical_prediction(b,R[1:frame+1],T[1:frame+1]))
                prefix_hard={k:prefix['residuals_m'][k] for k in hard_keys}
                refreshed=rigid_feedback_batch(b,R[None,frame-C+1:frame+1],T[None,frame-C+1:frame+1],horizon=H-frame)
                fresh_cal=reanchor_finger_calibration(refreshed,c)
                continuation=validator(refreshed,fresh_cal)(physical_prediction(refreshed,R[frame+1:],T[frame+1:]))
                replay_dir=Path(args['replay'] if i==0 else args['fine_replay'])
                reference=np.load(replay_dir/'execution.npz')
                np.testing.assert_array_equal(R[:frame+1],reference['model_R'][:frame+1])
                np.testing.assert_array_equal(T[:frame+1],reference['model_T'][:frame+1])
                prefix_position=np.linalg.norm(T[:frame+1,1]-reference['desired_T'][:frame+1,1],axis=-1)*1000
                relative=R[:frame+1,1] @ reference['desired_R'][:frame+1,1].swapaxes(-1,-2)
                prefix_rotation=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=-2,axis2=-1)-1)/2,-1,1)))
                # State checkpoints do not contain MuJoCo warning counters. A
                # warning-free original replay certifies its inherited prefix;
                # otherwise refuse full qualification conservatively.
                base_result=json.loads((replay_dir/'result.json').read_text())
                tracking=bool(result['finite'] and not any(result['warning_counts'])
                    and base_result['finite'] and not any(base_result['warning_counts'])
                    and max(result['hand_tracking_max_mm'],float(prefix_position.max()))<=3
                    and max(result['hand_rotation_max_deg'],float(prefix_rotation.max()))<=3)
                parts.append(dict(step=result['step'],prefix_hard_residuals_m=prefix_hard,
                    prefix_hard_failures=[k for k,v in prefix_hard.items() if v>0],
                    prefix_hard_pass=all(v<=0 for v in prefix_hard.values()),
                    continuation_validation=continuation,full_validation=full,tracking_ok=tracking,
                    prefix_hand_tracking_max_mm=float(prefix_position.max()),
                    prefix_hand_rotation_max_deg=float(prefix_rotation.max()),
                    object_lift_mm=result['object_lift_mm'],final_goal_mm=result['final_goal_mm']))
            full_pass=bool(qualified and len(parts)==2 and item.get('timestep_stable',False) and all(v['tracking_ok'] and v['full_validation']['accepted'] for v in parts))
            variants.append(dict(variant=item['variant'],replays=parts,qualified_full_task_pass=full_pass,
                timestep_stable=item.get('timestep_stable'),timestep_position_max_mm=item.get('timestep_position_max_mm'),
                timestep_rotation_max_deg=item.get('timestep_rotation_max_deg'),prediction_validation=item['prediction_validation'],
                replans=item.get('replans',[]),actual_model_calls=item.get('actual_model_calls')))
        row=dict(run=path.parent.name,scene=name,task=scene['task'],group=scene['group'],
            kind='integration_gate' if '_gate_' in path.parent.name else 'experiment',
            context_source=args.get('context_source','observed'),iterations=args['iterations'],seed=report['seed'],
            method=args.get('method','noise_opt'),k=args.get('k') if args.get('method')=='best_of_k' else None,
            prefix_frames=frame,horizon=H,control_qualified=qualified,results=variants,
            replan_schedule=report.get('replan_schedule',[]),
            source_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            note='Continuation-only checks cannot repair an earlier violation. Full-task criteria and original controller qualification are unchanged. The fine step replays the fixed realized command program.')
        (path.parent/'execution_audit_v1.json').write_text(json.dumps(row,indent=2)+'\n');rows.append(row)
    (root/'feedback_execution_summary_v1.json').write_text(json.dumps(dict(results=rows),indent=2)+'\n')
    print('Audited feedback reports:',len(rows))


if __name__=='__main__':main()
