"""Optional full convex-surface contact costs and independent task screens.

The original sampled-point diagnostics remain available. Mesh distances improve
geometric coverage but still use the declared collision proxies, not contact
forces or simulator execution. All units are metres.
"""
import math

import numpy as np
import torch

from guidance.collision_distance import ConvexCollisionDistance
from guidance.physical_objective import PickPhysicalObjective
from guidance.pick_validation import PickValidator
from guidance.place_objective import PlacePhysicalObjective, PlaceValidator


class MeshContactCosts:
    def __init__(self, batch, target, geometry, calibration, *, vendor_paths=(),
                 mesh_contact_tolerance=.002, mesh_penetration_tolerance=.002, **kwargs):
        super().__init__(batch, target, geometry, calibration, **kwargs)
        if any(not math.isfinite(v) or v <= 0 for v in (mesh_contact_tolerance, mesh_penetration_tolerance)):
            raise ValueError('Mesh tolerances must be finite and positive')
        self.mesh_contact_tolerance = mesh_contact_tolerance
        self.mesh_penetration_tolerance = mesh_penetration_tolerance
        self.collision = ConvexCollisionDistance(geometry, object_body=self.object_body,
            other_bodies=(self.hand_body, *self.fingers), vendor_paths=vendor_paths)

    def features(self, pred):
        f = super().features(pred)
        b, C, idx = self.batch, self.batch.context_len, pred['rigid_idx']
        R = torch.cat([b.body_poses[:, C-1:C, idx, :3, :3], pred['R']], dim=1)
        t = torch.cat([b.body_poses[:, C-1:C, idx, :3, 3], pred['t']], dim=1)
        f['mesh_distance'] = self.collision(R, t, idx)
        return f

    def terms(self, f):
        terms = super().terms(f)
        signed = f['mesh_distance']
        gaps = signed[..., 1:].clamp_min(0)
        excess = (gaps[:, 1:].amax(-1)-self.mesh_contact_tolerance).clamp_min(0)
        lifted = ((f['bottom'][:, 1:]-self.support_z-self.support_tolerance)/.02).clamp(0, 1)
        missing = ((gaps[:, :-1].amax(-1)-self.mesh_contact_tolerance)/.01).clamp(0, 1)
        penetration = (-signed[:, 1:]-self.mesh_penetration_tolerance).clamp_min(0)
        terms.update(contact=(lifted*excess).mean()+excess[:, -min(4, excess.shape[1]):].mean(),
            unsupported_rise=(f['object_step'][..., 2].clamp_min(0)*missing).mean(),
            slip=(f['hand_step']*(1-missing)*lifted).mean(),
            penetration=penetration.amax(-1).mean()+penetration.amax())
        if isinstance(self, PlacePhysicalObjective):
            H = min(self.settle_frames, f['object_step'].shape[1])
            terms.update(contact=(lifted*excess).mean(),
                         release=(.005-signed[:, -H:, 1:].amin(-1)).clamp_min(0).mean())
        return terms

    @torch.no_grad()
    def measure(self, pred):
        result = super().measure(pred)
        distance = np.asarray(result['traces']['mesh_distance'])
        result.update(mesh_penetration_max_mm=float(np.maximum(0, -distance[1:]).max()*1000),
            mesh_final_finger_gap_mm=(distance[-1, 1:]*1000).tolist(),
            collision_cost_backend='FCL signed distances to convex collision components; sampled-point traces retained')
        return result


class MeshPickPhysicalObjective(MeshContactCosts, PickPhysicalObjective):
    pass


class MeshPlacePhysicalObjective(MeshContactCosts, PlacePhysicalObjective):
    pass


class MeshScreen:
    def __init__(self, *args, collision, mesh_contact_tolerance=.002,
                 mesh_penetration_tolerance=.002, **kwargs):
        super().__init__(*args, **kwargs)
        self.collision = collision
        self.mesh_contact_tolerance = mesh_contact_tolerance
        self.mesh_penetration_tolerance = mesh_penetration_tolerance

    @torch.no_grad()
    def __call__(self, pred):
        base = super().__call__(pred)
        d = self.collision(pred['R'], pred['t'], pred['rigid_idx'])
        values = dict(base['residuals_m'])
        values['penetration'] = float(-d.amin())-self.mesh_penetration_tolerance
        if isinstance(self, PlaceValidator):
            values['release'] = .005-float(d[:, -self.hold_frames:, 1:].amin())
        else:
            values['contact'] = float(d[:, -self.hold_frames:, 1:].amax())-self.mesh_contact_tolerance
        interaction = [v for k, v in values.items() if k not in ('goal', 'goal_rotation')]
        return dict(accepted=all(v <= 0 for v in values.values()),
            interaction_accepted=all(v <= 0 for v in interaction),
            max_violation_m=max(0., max(values.values())),
            max_interaction_violation_m=max(0., max(interaction)), residuals_m=values,
            label='Convex collision mesh task screen; contact forces and execution unverified')


class MeshPickValidator(MeshScreen, PickValidator):
    pass


class MeshPlaceValidator(MeshScreen, PlaceValidator):
    pass
