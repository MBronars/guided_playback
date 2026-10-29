"""Supplied static boxes for rigid trajectory planning and collision auditing.

Distances use full convex collision components, not particle samples or object
centres. The checkpoint is unchanged and does not observe these obstacles.
Interpolated checks are geometric probes, not a continuous collision guarantee
or a simulation of the motion between frames.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

from guidance.collision_distance import ConvexCollisionDistance
from utils.pose_ops import so3_exp, so3_log


def load_obstacles(path):
    """Explicit world coordinates in metres; size is full box extent, not radius."""
    return validate_obstacles(json.loads(Path(path).read_text()))


def validate_obstacles(spec):
    if spec.get('version') != 1 or spec.get('units') != 'metres':
        raise ValueError('Obstacle specification requires version 1 and units metres')
    boxes = spec.get('boxes')
    if not isinstance(boxes, list) or not boxes:
        raise ValueError('At least one static box is required')
    clean, names = [], set()
    for box in boxes:
        name = box.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Obstacle names must be unique nonempty strings')
        centre = np.asarray(box['centre'], dtype=np.float64)
        size = np.asarray(box['size'], dtype=np.float64)
        rotation = np.asarray(box.get('rotation', np.eye(3)), dtype=np.float64)
        if (centre.shape != (3,) or size.shape != (3,) or rotation.shape != (3, 3)
                or not all(np.isfinite(v).all() for v in (centre, size, rotation))
                or (size <= 0).any()):
            raise ValueError('Box centre, positive size and rotation must be finite')
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0)
                or abs(np.linalg.det(rotation)-1) > 1e-7):
            raise ValueError('Obstacle rotation must be a proper rotation')
        clean.append(dict(name=name, centre=centre.tolist(), size=size.tolist(), rotation=rotation.tolist()))
        names.add(name)
    return dict(version=1, units='metres', boxes=clean)


def box_geometry(size):
    half = torch.as_tensor(size, dtype=torch.float64)/2
    normals = torch.cat([torch.eye(3), -torch.eye(3)]).double()
    return dict(vertices=torch.tensor(list(itertools.product([-1., 1.], repeat=3)))*half,
                planes=[torch.cat([normals, -half.repeat(2)[:, None]], dim=1)])


def interpolate_poses(R, t, substeps):
    """Sample per-body linear translation and shortest SO(3) interpolation.

    Endpoints are preserved exactly. Independent body interpolation is a stated
    audit assumption; it does not reconstruct a robot's joint-space controller.
    """
    if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
        raise ValueError('Interpolation substeps must be a positive integer')
    if R.ndim != 5 or t.shape != R.shape[:-1] or R.shape[-2:] != (3, 3) or R.shape[1] < 1:
        raise ValueError('Expected [B,T,K,3,3] rotations and [B,T,K,3] translations')
    if substeps == 1 or R.shape[1] == 1:
        return R, t
    # Insert only the interior probes, so logged frames are bitwise unchanged.
    fractions = torch.arange(1, substeps, device=t.device, dtype=t.dtype)/substeps
    delta = so3_log(R[:, 1:] @ R[:, :-1].transpose(-1, -2))
    intermediate_R = so3_exp(delta[:, :, None]*fractions[None, None, :, None, None]) @ R[:, :-1, None]
    intermediate_t = t[:, :-1, None] + (t[:, 1:]-t[:, :-1])[:, :, None]*fractions[None, None, :, None, None]
    rotations = torch.cat([R[:, :-1, None], intermediate_R], dim=2).flatten(1, 2)
    translations = torch.cat([t[:, :-1, None], intermediate_t], dim=2).flatten(1, 2)
    return torch.cat([rotations, R[:, -1:]], dim=1), torch.cat([translations, t[:, -1:]], dim=1)


class StaticObstacleDistance:
    """Signed minimum component distance for every moving body / static box pair."""
    def __init__(self, geometry, spec, *, vendor_paths=()):
        self.spec = validate_obstacles(spec)
        self.bodies = tuple(sorted(geometry['bodies']))
        self.names = tuple(box['name'] for box in self.spec['boxes'])
        static = {i+1: box_geometry(box['size']) for i, box in enumerate(self.spec['boxes'])}
        self.queries = [ConvexCollisionDistance(dict(bodies={0: geometry['bodies'][key], **static}),
            other_bodies=tuple(static), vendor_paths=vendor_paths) for key in self.bodies]
        self.rotation = torch.tensor([box['rotation'] for box in self.spec['boxes']], dtype=torch.float64)
        self.translation = torch.tensor([box['centre'] for box in self.spec['boxes']], dtype=torch.float64)

    def __call__(self, R, t, rigid_idx, *, substeps=1):
        R, t = interpolate_poses(R, t, substeps)
        columns = {int(k): i for i, k in enumerate(rigid_idx.tolist())}
        if any(key not in columns for key in self.bodies):
            raise ValueError('Obstacle query requires poses for every supplied collision body')
        static_R = self.rotation.to(R).expand(*R.shape[:2], -1, -1, -1)
        static_t = self.translation.to(t).expand(*t.shape[:2], -1, -1)
        idx = torch.arange(len(self.names)+1, device=t.device)
        values = []
        for key, query in zip(self.bodies, self.queries):
            col = columns[key]
            values.append(query(torch.cat([R[:, :, col:col+1], static_R], dim=2),
                                torch.cat([t[:, :, col:col+1], static_t], dim=2), idx))
        return torch.stack(values, dim=-2)  # [B, dense time, moving body, obstacle]


def anchored_poses(batch, pred):
    C, idx = batch.context_len, pred['rigid_idx']
    return (torch.cat([batch.body_poses[:, C-1:C, idx, :3, :3], pred['R']], dim=1),
            torch.cat([batch.body_poses[:, C-1:C, idx, :3, 3], pred['t']], dim=1))


class ObstacleObjective:
    def __init__(self, base, distance, *, weight=5., clearance=.005, substeps=3):
        if any(not math.isfinite(v) or v < 0 for v in (weight, clearance)):
            raise ValueError('Obstacle weight and clearance must be finite and nonnegative')
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError('Obstacle substeps must be a positive integer')
        self.base, self.distance = base, distance
        self.weight, self.clearance, self.substeps = weight, clearance, substeps

    def __getattr__(self, name):
        return getattr(self.base, name)

    def __call__(self, pred, batch):
        loss, diagnostics = self.base(pred, batch)
        R, t = anchored_poses(batch, pred)
        signed = self.distance(R, t, pred['rigid_idx'], substeps=self.substeps)
        # Exclude the fixed anchor from the objective, retain the entire first
        # transition and every moving body. The audit includes the anchor too.
        excess = (self.clearance-signed[:, 1:]).clamp_min(0)
        cost = excess.mean()+excess.amax()
        diagnostics.update(obstacle_cost_mm=float(cost.detach())*1000,
                           obstacle_min_clearance_mm=float(signed.detach().amin())*1000)
        return loss+self.weight*cost, diagnostics


class ObstacleValidator:
    def __init__(self, base, batch, distance, *, clearance=.005, substeps=10):
        if not math.isfinite(clearance) or clearance < 0:
            raise ValueError('Obstacle clearance must be finite and nonnegative')
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps < 1:
            raise ValueError('Audit substeps must be a positive integer')
        self.base, self.batch, self.distance = base, batch, distance
        self.clearance, self.substeps = clearance, substeps

    @torch.no_grad()
    def __call__(self, pred):
        result = self.base(pred)
        R, t = anchored_poses(self.batch, pred)
        signed = self.distance(R, t, pred['rigid_idx'], substeps=self.substeps)
        values = dict(result['residuals_m'], obstacle=self.clearance-float(signed.amin()))
        physical = [v for k, v in values.items() if k not in ('goal', 'goal_rotation')]
        result.update(accepted=all(v <= 0 for v in values.values()),
                      interaction_accepted=all(v <= 0 for v in physical), residuals_m=values,
                      max_violation_m=max(0., max(values.values())),
                      max_interaction_violation_m=max(0., max(physical)),
                      obstacle_min_clearance_mm=float(signed.amin())*1000,
                      obstacle_clearance_by_body_mm=signed.amin(dim=(0, 1, 3)).mul(1000).tolist(),
                      obstacle_body_ids=list(self.distance.bodies), obstacle_substeps=self.substeps,
                      label=result['label']+'; interpolated static-obstacle geometry audit, not continuous collision detection')
        return result
