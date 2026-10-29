"""Read-only task and replay audits shared by the local guidance galleries."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from guidance.collision_distance import ConvexCollisionDistance
from guidance.goal_region import PositionGoalRegion, PositionRegionValidator
from guidance.mesh_physical import MeshPickValidator, MeshPlaceValidator
from guidance.obstacles import StaticObstacleDistance, ObstacleValidator, load_obstacles
from guidance.unconditional import prediction
from scripts.dev.audit_feedback_replays import physical_prediction
from utils.pose_ops import rotation_geodesic_error
from utils.reproducibility import file_sha256


def obstacle_evaluation_protocol(protocol, source_path, obstacle_path, *, clearance_mm=5., substeps=10):
    """Re-evaluate an unchanged plan in a new world, retaining its optimizer inputs."""
    if protocol.get('obstacle_spec') or protocol.get('evaluation_override'):
        raise ValueError('Reference must be an original plan without obstacles')
    result=copy.deepcopy(protocol)
    result['evaluation_override']=dict(kind='static_obstacles_only',
        source_protocol=str(Path(source_path).resolve()),source_protocol_sha256=file_sha256(source_path),
        original_arguments=copy.deepcopy(protocol['arguments']),trajectory_reoptimized=False)
    result['obstacle_spec']=load_obstacles(obstacle_path)
    result['obstacles_sha256']=file_sha256(obstacle_path)
    result['arguments'].update(obstacles=str(Path(obstacle_path).resolve()),obstacle_clearance_mm=clearance_mm,
        obstacle_audit_substeps=substeps)
    return result


def task_validator(protocol):
    a=protocol['arguments']
    batch=torch.load(a['input'],weights_only=False,map_location='cpu')
    geometry=torch.load(a['geometry'],weights_only=False,map_location='cpu')
    calibration=torch.load(a['grasp'],weights_only=False,map_location='cpu')['calibration']
    for key,field in [('input','input_sha256'),('geometry','geometry_sha256'),('grasp','grasp_sha256')]:
        if file_sha256(a[key])!=protocol[field]:raise ValueError(f'Changed {key} fixture')
    collision=ConvexCollisionDistance(geometry,vendor_paths=a.get('collision_vendor',[]))
    cls=MeshPickValidator if a['task']=='pick' else MeshPlaceValidator
    extra=dict(target_R=torch.tensor(protocol['target_rotation'])) if protocol.get('target_rotation') is not None else {}
    validator=cls(batch,torch.tensor(protocol['target']),geometry,calibration,collision=collision,
        hold_frames=a['hold_frames'],mesh_contact_tolerance=a['mesh_contact_tolerance_mm']/1000,
        mesh_penetration_tolerance=a['mesh_penetration_tolerance_mm']/1000,**extra)
    if protocol.get('goal_region_bounds'):
        bounds=protocol['goal_region_bounds']
        region=PositionGoalRegion(*[[float(x) for x in bounds[k].split(',')] for k in ('lower','upper')])
        validator=PositionRegionValidator(validator,region,hold_frames=a['hold_frames'])
    if protocol.get('obstacle_spec'):
        if file_sha256(a['obstacles'])!=protocol['obstacles_sha256']:raise ValueError('Changed obstacles')
        obstacle=StaticObstacleDistance(geometry,protocol['obstacle_spec'],vendor_paths=a.get('collision_vendor',[]))
        validator=ObstacleValidator(validator,batch,obstacle,clearance=a['obstacle_clearance_mm']/1000,
            substeps=a['obstacle_audit_substeps'])
    return batch,geometry,collision,validator


def rollout_metrics(rollout, protocol, collision, validator):
    C=rollout['positions'].shape[1]-protocol['sampling']['horizon']
    pred=prediction(rollout,C);idx=pred['rigid_idx'];j=int((idx==0).nonzero()[0])
    distance=(pred['t'][0,:,j]-torch.tensor(protocol['target'][0])).norm(dim=-1)*1000
    signed=collision(pred['R'],pred['t'],idx)
    validation=validator(pred)
    return dict(final_goal_mm=float(distance[-1]),held_worst_goal_mm=float(distance[-protocol['arguments']['hold_frames']:].max()),
        held_mean_goal_mm=float(distance[-protocol['arguments']['hold_frames']:].mean()),
        peak_overlap_mm=float((-signed).clamp_min(0).max())*1000,
        goal_pass=validation['residuals_m']['goal']<=0,geometry_pass=validation['accepted'],
        non_goal_pass=validation['interaction_accepted'],validation=validation)


def replay_pair(protocol, run_file, directories, qualification):
    """Recompute task checks; demand identical commands and qualified time steps."""
    batch,geometry,collision,validator=task_validator(protocol)
    control_protocol=Path(qualification['paths']['mesh_preload'])/'protocol.json'
    control_receipt=json.loads(control_protocol.read_text())
    input_hashes=[value for name,value in control_receipt['sha256'].items() if Path(name).name=='input.pt']
    if input_hashes!=[protocol['input_sha256']]:raise ValueError('Controller qualification belongs to another fixture')
    arrays=np.load(run_file);C=int(arrays['context_len']);H=protocol['sampling']['horizon']
    rows=[];tracks=[]
    for directory,step in zip(directories,(.0005,.00025)):
        directory=Path(directory)
        if not (directory/'result.json').exists():return None
        receipt=json.loads((directory/'protocol.json').read_text())
        args=receipt['arguments'];result=json.loads((directory/'result.json').read_text())
        if args['step']!=step or args['frames']!=H or result['frames']!=H:
            raise ValueError('Replay duration/timestep mismatch')
        expected=dict(finger_mode='mesh_preload',preload=.006,mesh_contact_tolerance=.002,weld_time=.002,
            contact_time=.005,friction_scale=1.,table_offset=0.)
        if any(args.get(k)!=v for k,v in expected.items()) or args.get('finger_velocity_feedforward') or args.get('joint_limit_time') is not None:
            raise ValueError('Replay changed the qualified controller')
        policy=control_receipt['arguments'].get('preload_policy','instant')
        if args.get('preload_policy','instant')!=policy:
            raise ValueError('Replay preload policy does not match recorded-control qualification')
        if policy=='latched' and args.get('preload_release_mm')!=control_receipt['arguments']['preload_release_mm']:
            raise ValueError('Replay latch threshold does not match recorded-control qualification')
        if receipt['sha256'].get(str(Path(run_file).resolve()))!=file_sha256(run_file):
            raise ValueError('Replay trajectory hash does not match')
        if protocol.get('obstacles_sha256'):
            file=args.get('obstacles')
            if not file or receipt['sha256'].get(str(Path(file).resolve()))!=protocol['obstacles_sha256']:
                raise ValueError('Replay omitted or changed obstacles')
        z=np.load(directory/'execution.npz')
        np.testing.assert_array_equal(z['desired_T'],arrays['T'][0,C-1:])
        np.testing.assert_array_equal(z['desired_R'],arrays['R'][0,C-1:])
        pred=physical_prediction(batch,z['model_R'][1:],z['model_T'][1:])
        validation=validator(pred)
        goal=np.linalg.norm(z['model_T'][1:,0]-np.asarray(protocol['target'][0]),axis=-1)*1000
        tracking=bool(result['finite'] and not any(result['warning_counts']) and
            result['hand_tracking_max_mm']<=3 and result['hand_rotation_max_deg']<=3)
        rows.append(dict(step=step,final_goal_mm=float(goal[-1]),held_worst_goal_mm=float(goal[-protocol['arguments']['hold_frames']:].max()),
            validation=validation,tracking_ok=tracking,result=result,directory=str(directory),
            execution_sha256=file_sha256(directory/'execution.npz')))
        tracks.append(z)
    if len(tracks)!=2:raise ValueError('A replay pair requires exactly two integration steps')
    dp=np.linalg.norm(tracks[0]['model_T'][:,0]-tracks[1]['model_T'][:,0],axis=-1)*1000
    dr=rotation_geodesic_error(torch.tensor(tracks[0]['model_R'][:,0]),torch.tensor(tracks[1]['model_R'][:,0])).numpy()*180/np.pi
    stable=bool(dp.max()<=5 and dr.max()<=5)
    control=bool(qualification['qualified_for_generated_diagnostic'])
    return dict(replays=rows,control_qualified=control,timestep_stable=stable,
        control_qualified_frames=control_receipt['arguments']['frames'],
        duration_beyond_recorded_control=H>qualification['results']['mesh_preload']['frames'],
        control_protocol_sha256=file_sha256(control_protocol),
        timestep_position_max_mm=float(dp.max()),timestep_rotation_max_deg=float(dr.max()),
        qualified_task_pass=control and stable and all(r['tracking_ok'] and r['validation']['accepted'] for r in rows),
        note='Independent MuJoCo diagnostic; floating hand and approximate inertia/contact. Obstacle scenes inherit controller calibration, not source-simulator obstacle validation.')
