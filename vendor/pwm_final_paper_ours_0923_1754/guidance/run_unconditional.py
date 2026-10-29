"""Run a reproducible unconditional pickup guidance diagnostic on a saved fixture.

Example: python -m guidance.run_unconditional --checkpoint <final.pt>
  --input <pick_input.pt> --geometry <collision_geometry.pt> --out <new-directory>
  --method noise_opt --objective interaction --scope actuated --seeds 54321,54322
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from guidance.objectives import body_target
from guidance.pick_objective import PickObjective
from guidance.physical_objective import PickPhysicalObjective, PhysicalTermSubset
from guidance.place_objective import PlacePhysicalObjective, PlaceValidator
from utils.pose_ops import rotation_geodesic_error
from guidance.grasp import StagedPickObjective
from guidance.constraints import ConstrainedGraspObjective
from guidance.physical_constraints import PhysicalConstraints
from guidance.goal_region import PositionGoalRegion, PositionRegionValidator
from guidance.collision_distance import ConvexCollisionDistance
from guidance.mesh_physical import MeshPickPhysicalObjective, MeshPlacePhysicalObjective, MeshPickValidator, MeshPlaceValidator
from guidance.beam import beam_search
from guidance.control_search import waypoint_search
from guidance.observations import rigid_observation_bias, density_estimation_bias
from guidance.feedback import rigid_planning_batch
from guidance.obstacles import load_obstacles, StaticObstacleDistance, ObstacleObjective, ObstacleValidator
from guidance.waypoints import grasp_waypoints
from guidance.pick_validation import PickValidator
from guidance.unconditional import best_of_k, noise_opt, prediction, sample_rollout, select_then_refine, multistart
from inference import load_model_and_diffusion
from utils.reproducibility import code_revision, file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--input', required=True)
    p.add_argument('--geometry', help='Optional collision geometry for pickup costs and diagnostics')
    p.add_argument('--collision-backend',choices=['sampled','fcl'],default='sampled')
    p.add_argument('--collision-vendor',action='append',default=[],help='Optional dependency import directory; repeat for python-fcl/scipy')
    p.add_argument('--mesh-contact-tolerance-mm',type=float,default=2.)
    p.add_argument('--mesh-penetration-tolerance-mm',type=float,default=2.)
    p.add_argument('--obstacles', help='JSON file of supplied static boxes in world coordinates')
    p.add_argument('--obstacle-weight', type=float, default=5.)
    p.add_argument('--obstacle-clearance-mm', type=float, default=5.)
    p.add_argument('--obstacle-substeps', type=int, default=3, help='Geometric probes per frame transition during optimization')
    p.add_argument('--obstacle-audit-substeps', type=int, default=10, help='Denser geometric probes for reported acceptance')
    p.add_argument('--out', required=True)
    p.add_argument('--method', choices=['sample', 'best_of_k', 'noise_opt', 'select_refine', 'multistart', 'denoise', 'inpaint', 'beam', 'waypoint_search'], default='noise_opt')
    p.add_argument('--objective', choices=['goal', 'interaction', 'physical', 'staged'], default='interaction')
    p.add_argument('--task',choices=['pick','place'],default='pick')
    p.add_argument('--goal-body',type=int,default=0,help='Rigid body index for a mesh-free goal objective')
    p.add_argument('--grasp','--calibration',dest='grasp',help='Joint-calibration artifact; specialized staged methods additionally require grasp proposals')
    p.add_argument('--proposal', type=int, default=0)
    p.add_argument('--approach-frames', type=int, default=12)
    p.add_argument('--close-frame', type=int, default=18)
    p.add_argument('--hold-frames', type=int, default=6)
    p.add_argument('--pick-goal-hold',action='store_true',
                   help='For physical picking, average goal cost over --hold-frames; default targets the endpoint')
    p.add_argument('--goal-region-lower',help='Explicit lower x,y,z bounds in metres; -inf permits an unbounded axis')
    p.add_argument('--goal-region-upper',help='Explicit upper x,y,z bounds in metres; inf permits an unbounded axis')
    p.add_argument('--screen', action='store_true', help='Select by independent geometric acceptance before objective value')
    p.add_argument('--starts', type=int, default=2)
    p.add_argument('--constraints', action='store_true', help='Interleaved augmented-Lagrangian physical or staged constraints')
    p.add_argument('--constraint-penalty',type=float,default=None,
                   help='AL penalty; default2000 for unscaled residuals, required explicitly with tolerance scaling')
    p.add_argument('--constraint-scaling',choices=['none','tolerance'],default='none',
                   help='Optional dimensionless physical AL residuals; choose a penalty in the corresponding units explicitly')
    p.add_argument('--dual-every',type=int,default=2)
    p.add_argument('--save-candidates',action='store_true',help='Retain every candidate rollout for independent review')
    p.add_argument('--save-iterations',default='',help='Comma-separated noise_opt update counts at which to save the best-so-far rollout')
    p.add_argument('--denoise-scale',type=float,default=5.)
    p.add_argument('--denoise-threshold',type=float,default=.35)
    p.add_argument('--denoise-max-rms',type=float,default=.03)
    p.add_argument('--noise-knots',type=int,default=0,help='Temporal control points for noise updates; zero uses every frame')
    p.add_argument('--residual-max-rms',type=float,default=0.,help='Bound on a full-frame residual added to coarse noise updates')
    p.add_argument('--actuator-updates',type=int,default=0,help='Initial actuator-only updates under actuator_first scope')
    p.add_argument('--inpaint-scope',choices=['actuated','fingers'],default='actuated')
    p.add_argument('--beam-width',type=int,default=2)
    p.add_argument('--branches',type=int,default=4)
    p.add_argument('--grasp-proposals',type=int,default=8)
    p.add_argument('--stage-schedules',default='8:14,12:18,16:22',help='approach:closure frame pairs for waypoint search')
    p.add_argument('--observation-translation-mm',type=float,default=0.)
    p.add_argument('--observation-rotation-deg',type=float,default=0.)
    p.add_argument('--observation-seed',type=int,default=614)
    p.add_argument('--density-scale',type=float,default=1.)
    p.add_argument('--scope', choices=['all', 'actuated', 'actuator_first'], default='all')
    p.add_argument('--offset', default='0,0,0.1', help='Object goal offset from observed anchor, metres')
    p.add_argument('--target',help='Fixed absolute object target x,y,z; overrides --offset for observation-error studies')
    p.add_argument('--target-rotation',help='Nine row-major values of the requested rotation in model template coordinates')
    p.add_argument('--rotation-weight',type=float,default=.04,help='Metres per radian in the task objective')
    p.add_argument('--seeds', default='54321,54322')
    p.add_argument('--deterministic',action='store_true',help='Request deterministic PyTorch kernels; unsupported operations raise. For repeatability within a pinned environment.')
    p.add_argument('--horizon', type=int, default=36)
    p.add_argument('--context-only-future', action='store_true',
                   help='Construct rigid future placeholders from the last observation; permits horizons beyond the recording')
    p.add_argument('--detach-context',action='store_true',help='Gradient ablation: detach carried context at window boundaries; forward sampling is unchanged')
    p.add_argument('--iterations', type=int, default=6)
    p.add_argument('--lr', type=float, default=.03)
    p.add_argument('--lr-schedule',choices=['constant','cosine'],default='constant')
    p.add_argument('--warmup-updates',type=int,default=0)
    p.add_argument('--lr-min-ratio',type=float,default=.1)
    p.add_argument('--normalize-noise-gradient',action='store_true',help='Normalize the full trajectory gradient before Adam; B=1')
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--max-noise-rms', type=float, default=.25)
    p.add_argument('--prior-weight', type=float, default=.001)
    p.add_argument('--sampling-steps', type=int, help='Override the checkpoint inference-step count')
    p.add_argument('--no-activation-checkpointing', action='store_true',
                   help='Retain denoising activations instead of recomputing; can require much more memory')
    p.add_argument('--initial-rollout', help='Reuse the INITIAL noise/RNG of a saved seed_N.pt; regenerate with current sampler')
    p.add_argument('--physical-terms', help='Comma-separated auxiliary terms, or none; task goals and final audits are preserved')
    p.add_argument('--goal-weight', type=float, default=1., help='Weight for position goal only; rotation and physical weights stay independent')
    p.add_argument('--physical-weight', type=float, default=1.,
                   help='Global scale of physical penalties; preserves position/rotation goals, noise prior and obstacle cost')
    p.add_argument('--goal-reduction', choices=['mean','max'], default='mean',
                   help='Physical held-position objective: mean error or worst-frame error')
    p.add_argument('--contact-weight', type=float, default=2.)
    p.add_argument('--rise-weight', type=float, default=5.)
    p.add_argument('--slip-weight', type=float, default=1.)
    p.add_argument('--penetration-weight', type=float, default=5.)
    p.add_argument('--support-weight', type=float, default=5.)
    p.add_argument('--joint-weight',type=float,default=10.)
    p.add_argument('--joint-speed-weight',type=float,default=10.)
    args = p.parse_args()
    if args.sampling_steps is not None and args.sampling_steps < 1:
        p.error('Sampling steps must be positive')
    if args.physical_terms is not None and (args.objective != 'physical' or args.constraints):
        p.error('Physical-term subsets require an unconstrained physical objective')
    if args.initial_rollout and (args.method not in ('sample','noise_opt') or ',' in args.seeds):
        p.error('Saved initial noise requires sample/noise_opt and a single seed')
    obstacle_spec = load_obstacles(args.obstacles) if args.obstacles else None
    if obstacle_spec is not None:
        if args.objective != 'physical' or args.collision_backend != 'fcl' or args.constraints:
            p.error('Obstacle guidance currently requires mesh physical costs without augmented-Lagrangian constraints')
        if args.method not in ('sample','best_of_k','noise_opt','select_refine','multistart'):
            p.error('Obstacle guidance requires standard unconditional sampling or noise optimization')
        if (any(not np.isfinite(v) or v < 0 for v in (args.obstacle_weight,args.obstacle_clearance_mm))
                or args.obstacle_substeps < 1 or args.obstacle_audit_substeps < args.obstacle_substeps):
            p.error('Obstacle weights/clearance must be nonnegative; audit density must cover optimization density')
    if not np.isfinite(args.goal_weight) or args.goal_weight < 0:
        p.error('Goal weight must be finite and nonnegative')
    if not np.isfinite(args.physical_weight) or args.physical_weight < 0:
        p.error('Physical weight must be finite and nonnegative')
    if args.physical_weight != 1 and args.objective != 'physical':
        p.error('Physical-weight ablation requires the physical objective')
    if args.goal_weight != 1 and args.objective == 'staged':
        p.error('Goal weighting is supported for plain goal, interaction and physical objectives')
    if args.goal_reduction != 'mean' and args.objective != 'physical':
        p.error('Held-goal reduction requires a physical objective')
    goal_region=None
    if (args.goal_region_lower is None)!=(args.goal_region_upper is None):
        p.error('Supply both position-region bounds')
    if args.goal_region_lower is not None:
        if args.task!='pick' or args.objective!='physical' or not 2<=args.hold_frames<=args.horizon:
            p.error('Position-region guidance currently requires physical picking and a valid hold interval')
        if args.method not in ('sample','best_of_k','noise_opt','select_refine','multistart'):
            p.error('Position-region guidance currently supports standard noise and sampling methods')
        if args.target_rotation is not None:
            p.error('Position-region picking currently specifies position only')
        try:
            goal_region=PositionGoalRegion([float(v) for v in args.goal_region_lower.split(',')],
                                           [float(v) for v in args.goal_region_upper.split(',')])
        except ValueError as error:p.error(str(error))
    if args.pick_goal_hold and (args.task!='pick' or args.objective!='physical' or not 1<=args.hold_frames<=args.horizon):
        p.error('--pick-goal-hold requires physical picking and a hold interval within the horizon')
    if args.constraint_scaling=='tolerance' and not (args.constraints and args.objective=='physical'):
        p.error('Tolerance scaling requires proposal-free physical constraints')
    if args.constraint_penalty is None:
        if args.constraint_scaling=='tolerance':
            p.error('Tolerance scaling changes penalty units; provide --constraint-penalty explicitly')
        args.constraint_penalty=2000.
    save_iterations={int(v) for v in args.save_iterations.split(',') if v}
    if save_iterations and (args.method!='noise_opt' or min(save_iterations)<0 or max(save_iterations)>args.iterations):
        p.error('--save-iterations requires noise_opt and update counts within its budget')
    if args.collision_backend=='fcl' and (args.geometry is None or args.grasp is None or args.objective not in ('goal','physical')):
        p.error('FCL requires geometry, calibration, and a plain goal or physical objective')
    if args.collision_backend=='fcl' and any(not np.isfinite(v) or v<=0 for v in (args.mesh_contact_tolerance_mm,args.mesh_penetration_tolerance_mm)):
        p.error('Mesh tolerances must be finite and positive')
    if args.geometry is None and (args.objective!='goal' or args.screen or args.grasp is not None
            or args.method not in ('sample','best_of_k','noise_opt','select_refine','multistart')):
        p.error('Pickup diagnostics, screening and specialized objectives require --geometry; plain goal optimization does not')
    if args.geometry is not None and args.goal_body!=0:
        p.error('Pickup geometry diagnostics use object body 0; other body targets require the mesh-free goal path')
    if args.task=='place' and (args.objective in ('staged','interaction') or args.method in ('inpaint','denoise','beam','waypoint_search')):
        p.error('Placing supports plain goal or proposal-free physical objectives with standard noise/sampling methods')
    if args.task=='place' and args.target is None:
        p.error('Placing requires an explicit task target; a default upward pick offset is inappropriate')
    outdir = Path(args.out).resolve()
    if outdir.exists():
        raise FileExistsError(f'Use a fresh output directory: {outdir}')
    if args.deterministic:
        # Set before the first CUDA context/matmul, including batch transfer.
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = torch.load(args.input, weights_only=False, map_location='cpu').to(device)
    geometry = torch.load(args.geometry, weights_only=False, map_location='cpu') if args.geometry else None
    input_hash = file_sha256(args.input)
    if geometry is not None:
        hashes = geometry['metadata']['input_hashes']
        if hashes.get(str(Path(args.input).resolve())) != input_hash:
            raise ValueError('Collision geometry was prepared for a different input fixture')
    if batch.batch_size != 1 or args.horizon < 1 or (args.horizon > batch.num_future_frames and not args.context_only_future):
        raise ValueError('Expected one scene with enough future placeholder frames')
    model, diffusion, cfg, steps = load_model_and_diffusion(args.checkpoint, device)
    if args.sampling_steps is not None:
        steps = args.sampling_steps
    if float(cfg.train.get('action_dropout', 0)) != 1:
        raise ValueError('This entrypoint requires an unconditional action_dropout=1 checkpoint')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    offset = torch.tensor([float(v) for v in args.offset.split(',')], device=device)
    if offset.shape != (3,):
        raise ValueError('Expected --offset dx,dy,dz')
    if not 0<=args.goal_body<batch.body_type.shape[1] or not bool((batch.body_type[:,args.goal_body]==1).all()):
        raise ValueError('The goal body must be a valid rigid body')
    target = batch.body_poses[:, batch.context_len-1, args.goal_body, :3, 3] + offset
    if args.target is not None:
        target=torch.tensor([float(v) for v in args.target.split(',')],device=device)[None]
        if target.shape!=(1,3) or not bool(torch.isfinite(target).all()):
            raise ValueError('Expected a finite absolute target x,y,z')
    target_R=None
    if args.target_rotation is not None:
        target_R=torch.tensor([float(v) for v in args.target_rotation.split(',')],device=device).reshape(1,3,3)
        if not bool(torch.isfinite(target_R).all()) or not torch.allclose(target_R.transpose(-1,-2)@target_R,torch.eye(3,device=device)[None],atol=1e-4) or float(torch.linalg.det(target_R)[0])<.999:
            raise ValueError('The target rotation must be a proper orthogonal matrix')
    true_batch=batch
    batch,observation_receipt=rigid_observation_bias(batch,translation_m=args.observation_translation_mm/1000,
        rotation_deg=args.observation_rotation_deg,seed=args.observation_seed,roots=(args.goal_body,))
    batch,density_receipt=density_estimation_bias(batch,scale=args.density_scale,bodies=(args.goal_body,))
    recording_batch = batch
    if args.context_only_future:
        batch = rigid_planning_batch(batch, horizon=args.horizon)
    diagnostic = PickObjective(batch, target, geometry, contact_weight=args.contact_weight,
        rise_weight=args.rise_weight, slip_weight=args.slip_weight,
        penetration_weight=args.penetration_weight, support_weight=args.support_weight,
        goal_weight=args.goal_weight,physical_weight=args.physical_weight) if geometry is not None else None
    objective = diagnostic if args.objective == 'interaction' else body_target(args.goal_body, target,target_R,
        rotation_weight=args.rotation_weight,goal_weight=args.goal_weight)
    grasp_hash = None
    if args.grasp is not None:
        grasp = torch.load(args.grasp, weights_only=False, map_location='cpu')
        for path in (args.input, args.geometry):
            if grasp['metadata']['input_hashes'].get(str(Path(path).resolve())) != file_sha256(path):
                raise ValueError('Grasp proposals were prepared for a different input or collision geometry')
        # Calibration alone must not activate stage costs/diagnostics. A plain
        # 16-frame goal run has no 18-frame closure schedule.
        if grasp.get('proposals') and args.objective=='staged':
            diagnostic = StagedPickObjective(batch, target, geometry, grasp, proposal=args.proposal,
                approach_frames=args.approach_frames, close_frame=args.close_frame, hold_frames=args.hold_frames)
        grasp_hash = file_sha256(args.grasp)
    if args.objective=='physical':
        if grasp_hash is None: raise ValueError('Physical joint costs require --calibration')
        physical_class=PlacePhysicalObjective if args.task=='place' else PickPhysicalObjective
        extra=dict(target_R=target_R,rotation_weight=args.rotation_weight,settle_frames=args.hold_frames) if args.task=='place' else {}
        if args.pick_goal_hold:extra['goal_frames']=args.hold_frames
        if goal_region is not None:extra.update(goal_region=goal_region,goal_frames=args.hold_frames)
        if args.collision_backend=='fcl':
            physical_class=MeshPlacePhysicalObjective if args.task=='place' else MeshPickPhysicalObjective
            extra.update(vendor_paths=args.collision_vendor,
                mesh_contact_tolerance=args.mesh_contact_tolerance_mm/1000,
                mesh_penetration_tolerance=args.mesh_penetration_tolerance_mm/1000)
        diagnostic=physical_class(batch,target,geometry,grasp['calibration'],
            contact_weight=args.contact_weight,rise_weight=args.rise_weight,slip_weight=args.slip_weight,
            penetration_weight=args.penetration_weight,support_weight=args.support_weight,
            joint_weight=args.joint_weight,speed_weight=args.joint_speed_weight,goal_weight=args.goal_weight,
            goal_reduction=args.goal_reduction,physical_weight=args.physical_weight,**extra)
        objective=diagnostic
        if args.physical_terms is not None:
            enabled = [] if args.physical_terms == 'none' else args.physical_terms.split(',')
            objective = PhysicalTermSubset(diagnostic, enabled)
    if args.objective == 'staged':
        if grasp_hash is None:
            raise ValueError('Staged guidance requires --grasp')
        if not grasp.get('proposals'):
            raise ValueError('No geometric grasp proposal is available; preserve this planning failure in evaluation')
        objective = diagnostic
    evaluation_validator = None
    if grasp_hash is not None:
        validator_class=PlaceValidator if args.task=='place' else PickValidator
        extra=dict(target_R=target_R) if args.task=='place' else {}
        evaluation_validator=validator_class(batch,target,geometry,grasp['calibration'],hold_frames=args.hold_frames,**extra)
    sampled_evaluation_validator=evaluation_validator
    if args.collision_backend=='fcl':
        collision=diagnostic.collision if args.objective=='physical' else ConvexCollisionDistance(geometry,vendor_paths=args.collision_vendor)
        validator_class=MeshPlaceValidator if args.task=='place' else MeshPickValidator
        extra=dict(target_R=target_R) if args.task=='place' else {}
        evaluation_validator=validator_class(batch,target,geometry,grasp['calibration'],hold_frames=args.hold_frames,
            collision=collision,mesh_contact_tolerance=args.mesh_contact_tolerance_mm/1000,
            mesh_penetration_tolerance=args.mesh_penetration_tolerance_mm/1000,**extra)
    if goal_region is not None:
        sampled_evaluation_validator=PositionRegionValidator(sampled_evaluation_validator,goal_region,hold_frames=args.hold_frames)
        evaluation_validator=PositionRegionValidator(evaluation_validator,goal_region,hold_frames=args.hold_frames)
    if obstacle_spec is not None:
        obstacles = StaticObstacleDistance(geometry, obstacle_spec, vendor_paths=args.collision_vendor)
        base_objective = objective
        diagnostic = ObstacleObjective(diagnostic, obstacles, weight=args.obstacle_weight,
            clearance=args.obstacle_clearance_mm/1000, substeps=args.obstacle_substeps)
        objective = ObstacleObjective(base_objective, obstacles, weight=args.obstacle_weight,
            clearance=args.obstacle_clearance_mm/1000, substeps=args.obstacle_substeps)
        evaluation_validator = ObstacleValidator(evaluation_validator, batch, obstacles,
            clearance=args.obstacle_clearance_mm/1000, substeps=args.obstacle_audit_substeps)
    if args.screen:
        if grasp_hash is None:
            raise ValueError('Geometric screening requires --grasp for absolute joint calibration')
    validator=evaluation_validator if args.screen else None
    if args.constraints and args.objective not in ('staged','physical'):
        raise ValueError('Explicit constraint handling requires a physical or staged objective')
    if args.constraints and args.method not in ('noise_opt',):
        raise ValueError('Constraint state is currently isolated per noise_opt trial')
    if args.method in ('denoise','beam') and args.objective!='staged':
        raise ValueError('Prefix-aware guidance requires the staged objective')
    control=None
    if args.method=='waypoint_search' and (grasp_hash is None or validator is None):
        raise ValueError('Waypoint search requires --grasp and --screen')
    if args.method=='inpaint':
        if grasp_hash is None: raise ValueError('Grasp inpainting requires --grasp')
        control=grasp_waypoints(batch,target,grasp,proposal=args.proposal,horizon=args.horizon,
            approach_frames=args.approach_frames,close_frame=args.close_frame,hold_frames=args.hold_frames,
            scope=args.inpaint_scope)
    sampling = dict(future_len=int(cfg.data.future_len), horizon=args.horizon,
                    steps=int(steps), schedule=cfg.get('df_schedule', 'pyramid'),
                    uncertainty_scale=float(cfg.get('df_uncertainty_scale', 1.0)))
    if args.detach_context: sampling['detach_context']=True
    if args.no_activation_checkpointing: sampling['use_checkpoint']=False
    optimizer_options=dict(lr_schedule=args.lr_schedule,warmup_updates=args.warmup_updates,
        lr_min_ratio=args.lr_min_ratio,normalize_gradient=args.normalize_noise_gradient)
    def measure(pred):
        col=(pred['rigid_idx']==args.goal_body).nonzero(as_tuple=True)[0].item()
        distance=(pred['t'][:,:,col]-target[:,None]).norm(dim=-1)
        result=diagnostic.measure(pred) if diagnostic is not None else dict(target_distance_mm=float(distance[:,-1].mean())*1000,
            traces=dict(target_distance=distance.cpu().tolist()),physical_diagnostics='not requested')
        if target_R is not None:
            angle=rotation_geodesic_error(pred['R'][:,:,col],target_R)*180/torch.pi
            result['target_rotation_deg']=float(angle[:,-1].mean());result['traces']['target_rotation_deg']=angle.cpu().tolist()
        if goal_region is not None:
            track=goal_region.distance(pred['t'][:,:,col])
            violation=goal_region.signed_violation(pred['t'][:,:,col])
            result.update(goal_region_final_distance_mm=float(track[:,-1].mean())*1000,
                          goal_region_held_max_violation_mm=float(violation[:,-args.hold_frames:].max())*1000)
            result['traces']['goal_region_distance_m']=track.cpu().tolist()
        return result
    residual_scales=PhysicalConstraints(objective,task=args.task,hold_frames=args.hold_frames).tolerance_scales() if args.constraint_scaling=='tolerance' else None
    outdir.mkdir(parents=True)
    metadata = dict(arguments=vars(args), checkpoint_sha256=file_sha256(args.checkpoint),
                    input_sha256=input_hash, geometry_sha256=file_sha256(args.geometry) if geometry is not None else None,
                    geometry=geometry['metadata'] if geometry is not None else None, sampling=sampling,
                    normalization=OmegaConf.to_container(cfg.normalization, resolve=True),
                    weights='EMA', actions='hidden', actuator_generation=control['description'] if control is not None else
                        ('searched geometric actuator paths; passive object remains generated' if args.method=='waypoint_search' else 'jointly generated'),
                    precision='FP32; no autocast or compilation',
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                    cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    constraint_residual_scales=residual_scales,
                    recording_future_frames=min(args.horizon,recording_batch.num_future_frames),
                    obstacle_spec=obstacle_spec,
                    obstacles_sha256=file_sha256(args.obstacles) if args.obstacles else None,
                    future_placeholders='repeat last observation; zero actions and targets' if args.context_only_future else 'recording slots hidden from unconditional model',
                    interaction_weights=getattr(diagnostic,'weights',{}), contact_tolerance_m=getattr(diagnostic,'contact_tolerance',None),
                    physical_penalty_scale=args.physical_weight,
                    grasp_sha256=grasp_hash, stage_weights=getattr(diagnostic, 'stage_weights', None),
                    support_tolerance_m=getattr(diagnostic,'support_tolerance',None), target=target.cpu().tolist(),
                    goal_kind='position_region' if goal_region is not None else 'point',
                    goal_region_bounds=dict(lower=args.goal_region_lower,upper=args.goal_region_upper,
                                            units='metres',hold_frames=args.hold_frames) if goal_region is not None else None,
                    target_rotation=target_R.cpu().tolist() if target_R is not None else None,
                    observation=observation_receipt,density_estimate=density_receipt,
                    observation_protocol='fixed task goal; coherent object-pose bias; grasp candidates held fixed in object coordinates; true future retained',
                    torch=torch.__version__, cuda=torch.version.cuda, host=platform.node(),
                    gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                    source_root=str(Path(__file__).resolve().parent.parent), **code_revision())
    initial_reference = None
    if args.initial_rollout:
        reference_file = Path(args.initial_rollout).resolve()
        reference_protocol = json.loads((reference_file.parent/'protocol.json').read_text())
        for key in ('checkpoint_sha256','input_sha256','normalization','observation','density_estimate'):
            if reference_protocol[key] != metadata[key]:
                raise ValueError(f'Saved initial noise has incompatible {key}')
        for key in ('future_len','horizon'):
            if reference_protocol['sampling'][key] != sampling[key]:
                raise ValueError(f'Saved initial noise has incompatible {key}')
        initial_reference = torch.load(reference_file, map_location=device, weights_only=False)['initial']
        metadata['initial_noise_source'] = dict(path=str(reference_file), sha256=file_sha256(reference_file),
            protocol_sha256=file_sha256(reference_file.parent/'protocol.json'))
    (outdir/'protocol.json').write_text(json.dumps(metadata, indent=2)+'\n')
    C = batch.context_len
    ridx=(batch.body_type[0]==1).nonzero(as_tuple=True)[0]
    gt = dict(positions=recording_batch.positions[:, C:C+args.horizon],
              R=recording_batch.body_poses[:, C:C+args.horizon, ridx, :3, :3],
              t=recording_batch.body_poses[:, C:C+args.horizon, ridx, :3, 3], rigid_idx=ridx)
    (outdir/'recording.json').write_text(json.dumps(measure(gt), indent=2)+'\n')
    calls = [0]
    def check_forward(_, positional, kwargs):
        given = positional[0]
        if bool(given.pose_action_mask.any()) or bool(given.pose_actions.any()):
            raise AssertionError('Future actions reached the unconditional model')
        calls[0] += 1
    hook = model.register_forward_pre_hook(check_forward, with_kwargs=True)
    records = []
    base_diagnostic=diagnostic
    for seed in [int(v) for v in args.seeds.split(',')]:
        diagnostic=base_diagnostic
        residual_fn=PhysicalConstraints(objective,task=args.task,hold_frames=args.hold_frames) if args.constraints and args.objective=='physical' else None
        trial_objective = ConstrainedGraspObjective(objective,penalty=args.constraint_penalty,
            dual_every=args.dual_every,residual_fn=residual_fn,residual_scales=residual_scales) if args.constraints else objective
        torch.manual_seed(seed)
        calls[0] = 0
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        trace_file = outdir/f'seed_{seed}_trace.jsonl'
        def progress(row):
            line = json.dumps(dict(seed=seed,elapsed_seconds=time.perf_counter()-t0, **row), allow_nan=False)
            with trace_file.open('a') as f:
                f.write(line+'\n')
            print(line, flush=True)
        def save_candidate(draw,rollout,row):
            directory=outdir/f'seed_{seed}_candidates'
            directory.mkdir(exist_ok=True)
            torch.save(rollout,directory/f'draw_{draw:03d}.pt')
            (directory/f'draw_{draw:03d}.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
        def save_iteration(update,best):
            if update not in save_iterations: return
            directory=outdir/f'seed_{seed}_iterates'
            directory.mkdir(exist_ok=True)
            torch.save(dict(evaluated_through_update=update,**best),directory/f'through_{update:04d}.pt')
            receipt=dict(evaluated_through_update=update,selected_iteration=best['iteration'],
                selection_loss=best['loss'],validation=best['validation'])
            (directory/f'through_{update:04d}.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        with torch.no_grad():
            shared = dict(noise=initial_reference['noise'],rng_states=initial_reference['rng_states']) if initial_reference is not None else {}
            initial = sample_rollout(model, diffusion, batch, **shared, **sampling)
        initial_metric = measure(prediction(initial, C))
        if args.method == 'sample':
            result = dict(rollout=initial, nfe=0, trace=[])
        elif args.method=='denoise':
            with torch.no_grad():
                guided=sample_rollout(model,diffusion,batch,noise=initial['noise'],rng_states=initial['rng_states'],
                    guidance_objective=diagnostic,guidance_scale=args.denoise_scale,
                    guidance_threshold=args.denoise_threshold,guidance_max_rms=args.denoise_max_rms,**sampling)
            result=dict(rollout=guided,nfe=guided['nfe'],trace=guided['guidance_trace'])
            for row in result['trace']: progress(row)
        elif args.method=='inpaint':
            with torch.no_grad():
                guided=sample_rollout(model,diffusion,batch,noise=initial['noise'],rng_states=initial['rng_states'],
                    known_pose=control['known_pose'],known_mask=control['known_mask'],**sampling)
            result=dict(rollout=guided,nfe=guided['nfe'],trace=[],control=control)
        elif args.method=='beam':
            torch.manual_seed(seed)
            result=beam_search(model,diffusion,batch,diagnostic,width=args.beam_width,branches=args.branches,
                validator=validator,callback=progress,
                candidate_callback=save_candidate if args.save_candidates else None,**sampling)
        elif args.method=='waypoint_search':
            phases=tuple(tuple(int(v) for v in pair.split(':')) for pair in args.stage_schedules.split(','))
            result=waypoint_search(model,diffusion,batch,target,grasp,initial=initial,validator=validator,
                proposals=args.grasp_proposals,phases=phases,hold_frames=args.hold_frames,scope=args.inpaint_scope,
                callback=progress,candidate_callback=save_candidate if args.save_candidates else None,**sampling)
            diagnostic=StagedPickObjective(batch,target,geometry,grasp,proposal=result['proposal'],
                approach_frames=result['approach_frames'],close_frame=result['close_frame'],hold_frames=args.hold_frames)
        elif args.method == 'best_of_k':
            # Replay the same start as ordinary sampling, so candidate 0 is paired.
            torch.manual_seed(seed)
            result = best_of_k(model, diffusion, batch, objective, k=args.k,
                               callback=progress, validator=validator,
                               candidate_callback=save_candidate if args.save_candidates else None, **sampling)
        elif args.method == 'noise_opt':
            result = noise_opt(model, diffusion, batch, trial_objective, initial=initial,
                iterations=args.iterations, lr=args.lr, scope=args.scope,
                max_noise_rms=args.max_noise_rms, prior_weight=args.prior_weight,
                callback=progress, validator=validator,temporal_knots=args.noise_knots,
                checkpoint_callback=save_iteration if save_iterations else None,
                actuator_updates=args.actuator_updates,residual_max_rms=args.residual_max_rms, **optimizer_options,**sampling)
        else:
            torch.manual_seed(seed)
            planner = select_then_refine if args.method == 'select_refine' else multistart
            extra = dict(k=args.k) if args.method == 'select_refine' else dict(starts=args.starts)
            if args.method=='select_refine':
                extra['candidate_callback']=save_candidate if args.save_candidates else None
            result = planner(model,diffusion,batch,objective,iterations=args.iterations,
                lr=args.lr,scope=args.scope,max_noise_rms=args.max_noise_rms,prior_weight=args.prior_weight,
                validator=validator,callback=progress,temporal_knots=args.noise_knots,
                actuator_updates=args.actuator_updates,residual_max_rms=args.residual_max_rms,**optimizer_options,**extra,**sampling)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = time.perf_counter()-t0
        selected = result['rollout']
        metric = measure(prediction(selected, C))
        validation = evaluation_validator(prediction(selected,C)) if evaluation_validator is not None else None
        record = dict(seed=seed, initial=initial_metric, selected=metric,
                      selected_iteration=result.get('iteration'), selected_draw=result.get('draw'),
                      forward_nfe=initial['nfe']+result['nfe'],
                      actual_model_calls_including_recomputation=calls[0], seconds=seconds,
                      max_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                      windows=selected['windows'], trace=result['trace'])
        if args.screen:
            selection_status='accepted_by_geometry_proxy' if validation['accepted'] else 'no_acceptable_candidate; saved trajectory is a diagnostic fallback'
        else:
            selection_status='selected_by_objective; '+('geometric audit passed' if validation['accepted'] else 'geometric audit failed') if validation is not None else 'not_screened'
        record.update(validation=validation,selection_status=selection_status,
            initial_validation=evaluation_validator(prediction(initial,C)) if evaluation_validator is not None else None)
        if args.collision_backend=='fcl':
            record.update(sampled_validation=sampled_evaluation_validator(prediction(selected,C)),
                initial_sampled_validation=sampled_evaluation_validator(prediction(initial,C)))
        if args.method=='waypoint_search':
            record['selected_control']={k:result[k] for k in ('proposal','approach_frames','close_frame','draw')}
        torch.save(dict(initial=initial, result=result), outdir/f'seed_{seed}.pt')
        np.savez_compressed(outdir/f'seed_{seed}.npz',
            positions=selected['positions'].detach().cpu().numpy(),
            initial=initial['positions'].detach().cpu().numpy(),
            R=selected['R'].detach().cpu().numpy(), T=selected['T'].detach().cpu().numpy(),
            initial_T=initial['T'].detach().cpu().numpy(),
            gt=recording_batch.positions[:, :C+args.horizon].cpu().numpy(),
            true_positions=true_batch.positions[:,:C+args.horizon].cpu().numpy(),
            particle_body=batch.particle_body.cpu().numpy(), target=target.cpu().numpy(), context_len=C)
        (outdir/f'seed_{seed}.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        records.append(record)
        print(json.dumps(dict(seed=seed, seconds=seconds, selected={k:v for k,v in metric.items() if k!='traces'})),flush=True)
    hook.remove()
    (outdir/'results.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
