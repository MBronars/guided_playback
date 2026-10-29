"""Picking costs and diagnostics in metres, using explicitly supplied geometry.

Contact is a sampled-point proximity proxy. Collision depth measures sampled
points inside convex components; a zero value does not prove nonpenetration.
The support plane may be inferred: its provenance belongs in the geometry receipt.
"""
from __future__ import annotations

import math
import torch


def component_margin(points, components):
    """Signed half-space margin to a union of convex components (negative inside).

    Planes have unit outward normals. Inside one component, -margin is distance
    to its nearest supporting face. Outside, margin is a distance lower bound.
    For overlapping components the negative value is not the exact union SDF.
    """
    return torch.stack([(points @ p[:, :3].T + p[:, 3]).amax(-1)
                        for p in components], dim=-1).amin(-1)


class PickObjective:
    def __init__(self, batch, target, geometry, *, object_body=0, hand_body=1,
                 fingers=(2, 3), contact_tolerance=0.005, support_tolerance=0.003,
                 contact_weight=2.0, rise_weight=5.0, slip_weight=1.0,
                 penetration_weight=5.0, support_weight=5.0, goal_weight=1.0,
                 physical_weight=1.0):
        if batch.batch_size != 1:
            raise ValueError('Picking diagnostics currently require one scene')
        self.batch, self.target = batch, target.to(batch.positions)
        self.object_body, self.hand_body, self.fingers = object_body, hand_body, tuple(fingers)
        self.contact_tolerance, self.support_tolerance = contact_tolerance, support_tolerance
        if not math.isfinite(goal_weight) or goal_weight < 0:
            raise ValueError('Goal weight must be finite and nonnegative')
        self.goal_weight = goal_weight
        if not math.isfinite(physical_weight) or physical_weight < 0:
            raise ValueError('Physical weight must be finite and nonnegative')
        self.physical_weight = physical_weight
        self.weights = dict(contact=contact_weight, unsupported_rise=rise_weight,
                            slip=slip_weight, penetration=penetration_weight, support=support_weight)
        if min(self.weights.values()) < 0 or contact_tolerance <= 0 or support_tolerance < 0:
            raise ValueError('Invalid picking objective parameters')
        self.geometry = {int(k): dict(planes=[p.to(batch.positions) for p in v['planes']],
                                     vertices=v['vertices'].to(batch.positions))
                         for k, v in geometry['bodies'].items()}
        self.metadata = geometry['metadata']
        self.support_z = float(self.metadata['support_z'])
        self.rows = {k: ((batch.particle_body[0] == k) & batch.get_valid_mask()[0]).nonzero(as_tuple=True)[0]
                     for k in (object_body, hand_body, *fingers)}
        if any(len(rows) == 0 for rows in self.rows.values()):
            raise ValueError('Missing object, hand or finger particles')
        if bool(batch.pose_action_mask[0, object_body]) or not all(bool(batch.pose_action_mask[0, k]) for k in (hand_body, *fingers)):
            raise ValueError('Picking roles disagree with physical actuator identities')

    def features(self, pred):
        b, C = self.batch, self.batch.context_len
        ridx = pred['rigid_idx']
        columns = {int(k): i for i, k in enumerate(ridx.tolist())}
        pos = torch.cat([b.positions[:, C-1:C], pred['positions']], dim=1).float()
        R = torch.cat([b.body_poses[:, C-1:C, ridx, :3, :3], pred['R']], dim=1).float()
        t = torch.cat([b.body_poses[:, C-1:C, ridx, :3, 3], pred['t']], dim=1).float()
        obj = pos[:, :, self.rows[self.object_body]]
        gaps = torch.stack([torch.cdist(pos[:, :, self.rows[k]], obj).amin(dim=(-2, -1))
                            for k in self.fingers], dim=-1)
        j, h = columns[self.object_body], columns[self.hand_body]
        local_center = (R[:, :, h].transpose(-1, -2) @ (t[:, :, j] - t[:, :, h])[..., None])[..., 0]
        hand_step = (local_center[:, 1:] - local_center[:, :-1]).norm(dim=-1)
        dt = t[:, 1:, j] - t[:, :-1, j]
        vertices = self.geometry[self.object_body]['vertices']
        object_mesh = torch.einsum('bfij,nj->bfni', R[:, :, j], vertices) + t[:, :, j, None]
        bottom = object_mesh[..., 2].amin(-1)
        gripper_bottom = torch.stack([
            torch.einsum('bfi,ni->bfn', R[:, :, columns[k], 2], self.geometry[k]['vertices']).amin(-1)
            + t[:, :, columns[k], 2] for k in (self.hand_body, *self.fingers)], dim=-1)
        # Signed checks both ways; gripper-side geometry is explicitly a hull proxy.
        depths, reverse = [], []
        for k in (self.hand_body, *self.fingers):
            points = pos[:, :, self.rows[k]]
            local = (points - t[:, :, j, None]) @ R[:, :, j]
            depths.append((-component_margin(local, self.geometry[self.object_body]['planes'])).clamp_min(0).amax(-1))
            c = columns[k]
            object_local = (obj - t[:, :, c, None]) @ R[:, :, c]
            reverse.append((-component_margin(object_local, self.geometry[k]['planes'])).clamp_min(0).amax(-1))
        inside_object = torch.stack(depths, dim=-1)
        inside_gripper_hull = torch.stack(reverse, dim=-1)
        return dict(gaps=gaps, object_step=dt, hand_step=hand_step, bottom=bottom, gripper_bottom=gripper_bottom,
                    inside_object=inside_object, inside_gripper_hull=inside_gripper_hull,
                    target_distance=(pred['t'][:, -1, j] - self.target).norm(dim=-1))

    def terms(self, f):
        gap = f['gaps'][:, 1:].amax(-1)
        excess = (gap - self.contact_tolerance).clamp_min(0)
        lifted = ((f['bottom'][:, 1:] - self.support_z - self.support_tolerance) / 0.02).clamp(0, 1)
        missing = ((f['gaps'][:, :-1].amax(-1) - self.contact_tolerance) / 0.01).clamp(0, 1)
        held = 1 - missing
        penetration = torch.maximum(f['inside_object'].amax(-1), f['inside_gripper_hull'].amax(-1))[:, 1:]
        return dict(goal=f['target_distance'].mean(),
                    contact=(lifted * excess).mean() + excess[:, -min(4, excess.shape[1]):].mean(),
                    unsupported_rise=(f['object_step'][..., 2].clamp_min(0) * missing).mean(),
                    slip=(f['hand_step'] * held * lifted).mean(),
                    penetration=(penetration - 0.001).clamp_min(0).mean(),
                    support=(self.support_z - f['bottom'][:, 1:] - self.support_tolerance).clamp_min(0).mean())

    def __call__(self, pred, batch):
        f = self.features(pred)
        terms = self.terms(f)
        # Rotation specifies a task goal, so an ablation of physical penalties
        # must preserve it. Keep the original summation order for default parity.
        loss = self.goal_weight * terms['goal'] + sum(
            (self.weights[k] if k == 'rotation' else self.physical_weight * self.weights[k]) * terms[k]
            for k in self.weights)
        diag = {f'{k}_cost_mm': float(v.detach()) * 1000 for k, v in terms.items()}
        # Placing can average its goal cost over several frames. Keep the
        # endpoint diagnostic independent of that objective reduction.
        diag.update(target_distance_mm=float(f['target_distance'].detach().mean()) * 1000,
                    gripper_in_object_max_mm=float(f['inside_object'][:, 1:].detach().max()) * 1000,
                    object_in_gripper_hull_max_mm=float(f['inside_gripper_hull'][:, 1:].detach().max()) * 1000)
        return loss, diag

    @torch.no_grad()
    def measure(self, pred):
        f = self.features(pred)
        gap = f['gaps'][0].amax(-1)
        close = gap <= self.contact_tolerance
        above = f['bottom'][0] > self.support_z + 0.01
        dz = f['object_step'][0, :, 2]
        unsupported_rise = (dz > 0.0005) & ~close[:-1] & ~close[1:]
        hovering = above[1:] & ~close[1:] & (dz.abs() < 0.002)
        sustained_hover = any(bool(hovering[i:i+4].all()) for i in range(max(0, len(hovering)-3)))
        held_steps = close[:-1] & close[1:] & above[1:]
        collision = f['inside_object'][0, 1:].amax(-1)
        reverse = f['inside_gripper_hull'][0, 1:].amax(-1)
        data = dict(target_distance_mm=float(f['target_distance'][0]) * 1000,
                    final_farther_finger_gap_mm=float(gap[-1]) * 1000,
                    both_fingers_close_fraction=float(close[1:].float().mean()),
                    unsupported_rise_mm=float(dz.clamp_min(0)[unsupported_rise].sum()) * 1000,
                    unsupported_rise_transitions=int(unsupported_rise.sum()),
                    sustained_hover_proxy=bool(sustained_hover),
                    held_hand_frame_step_mm=(float(f['hand_step'][0, held_steps].mean())*1000 if bool(held_steps.any()) else None),
                    gripper_in_object_max_mm=float(collision.max())*1000,
                    gripper_in_object_over_1mm_frames=int((collision>0.001).sum()),
                    object_in_gripper_hull_max_mm=float(reverse.max())*1000,
                    object_in_gripper_hull_over_1mm_frames=int((reverse>0.001).sum()),
                    support_reference_intrusion_max_mm=float((self.support_z-f['bottom'][0,1:]).clamp_min(0).max())*1000,
                    gripper_support_intrusion_max_mm=float((self.support_z-f['gripper_bottom'][0,1:]).clamp_min(0).max())*1000,
                    final_object_bottom_above_reference_mm=float(f['bottom'][0,-1]-self.support_z)*1000)
        data['traces'] = {k: v[0].cpu().tolist() for k, v in f.items()}
        return data
