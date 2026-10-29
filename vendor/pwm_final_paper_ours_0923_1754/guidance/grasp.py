"""Grasp-stage costs for an observed, calibrated parallel-jaw scene.

Proposals are prepared from observed geometry, never the recorded future. Pad
contact and collision remain geometric proxies; execution is a separate test.
"""
from __future__ import annotations

import math

import torch

from guidance.pick_objective import PickObjective


def rotation_chord(R, target):
    """2 sin(theta/2), finite at identity and nonzero for an exact half-turn."""
    return torch.linalg.vector_norm(R - target, dim=(-2, -1)) / math.sqrt(2)


class StagedPickObjective(PickObjective):
    def __init__(self, batch, target, geometry, grasp, *, proposal=0,
                 approach_frames=12, close_frame=18, hold_frames=6,
                 contact_weight=4., alignment_weight=2., hold_weight=2.,
                 stationary_weight=4., penetration_weight=10., joint_weight=10.):
        super().__init__(batch, target, geometry)
        if not 0 <= approach_frames <= close_frame:
            raise ValueError('Require 0 <= approach_frames <= close_frame')
        if hold_frames < 2 or not 0 <= proposal < len(grasp['proposals']):
            raise ValueError('Invalid hold length or grasp proposal')
        self.approach_frames, self.close_frame, self.hold_frames = approach_frames, close_frame, hold_frames
        self.proposal_index = proposal
        self.calibration = {k: v.to(batch.positions) if torch.is_tensor(v) else v
                            for k, v in grasp['calibration'].items()}
        self.proposal = {k: v.to(batch.positions) if torch.is_tensor(v) else v
                         for k, v in grasp['proposals'][proposal].items()}
        self.stage_weights = dict(contact=contact_weight, alignment=alignment_weight,
            hold=hold_weight, stationary=stationary_weight, penetration=penetration_weight,
            joint=joint_weight, opening=1., normal=.03, rotation=.04, support=10.)
        if any(not math.isfinite(v) or v < 0 for v in self.stage_weights.values()):
            raise ValueError('Stage weights must be finite and nonnegative')

    def features(self, pred, *, allow_prefix=False):
        f = super().features(pred)
        C, b, c, p = self.batch.context_len, self.batch, self.calibration, self.proposal
        H = pred['positions'].shape[1]
        if not allow_prefix and (self.close_frame >= H or self.hold_frames > H - self.close_frame):
            raise ValueError('Rollout must contain closure followed by the complete hold')
        col = {int(k): i for i, k in enumerate(pred['rigid_idx'].tolist())}
        j, h = col[self.object_body], col[self.hand_body]
        fingers = [col[k] for k in self.fingers]
        R, t = pred['R'].float(), pred['t'].float()
        rel = (R[:, :, h].transpose(-1, -2)[:, :, None] @
               (t[:, :, fingers] - t[:, :, h, None])[..., None])[..., 0]
        q = c['q_anchor'] + ((rel - c['finger_anchor']) * c['axis']).sum(-1)
        pad = (R[:, :, fingers] @ c['pad_local'][None, None, :, :, None])[..., 0] + t[:, :, fingers]
        normal = (R[:, :, fingers] @ c['pad_normal_local'][None, None, :, :, None])[..., 0]
        contact = (R[:, :, j, None] @ p['contacts'][None, None, :, :, None])[..., 0] + t[:, :, j, None]
        object_normal = (R[:, :, j, None] @ p['normals'][None, None, :, :, None])[..., 0]
        pad_gap = (pad-contact).norm(dim=-1)
        normal_error = (1 + (normal*object_normal).sum(-1)).clamp_min(0)
        expected_R = R[:, :, j] @ p['hand_R_object']
        expected_t = (R[:, :, j] @ p['hand_t_object'][None, None, :, None])[..., 0] + t[:, :, j]
        R0 = b.body_poses[:, C-1, self.object_body, :3, :3]
        t0 = b.body_poses[:, C-1, self.object_body, :3, 3]
        grasp_t = (R0 @ p['hand_t_object'][None, :, None])[..., 0] + t0
        grasp_R = R0 @ p['hand_R_object']
        approach = (grasp_R @ c['approach_local'][None, :, None])[..., 0]
        pregrasp = grasp_t - float(p['approach_distance']) * approach
        ticks = torch.arange(1, H+1, device=t.device, dtype=t.dtype)
        reach = (ticks / max(1, self.approach_frames)).clamp(0, 1)
        close = ((ticks-self.approach_frames) / max(1, self.close_frame-self.approach_frames)).clamp(0, 1)
        observed_hand = b.body_poses[:, C-1, self.hand_body, :3, 3]
        path = torch.lerp(observed_hand[:, None], pregrasp[:, None], reach[None, :, None])
        path = torch.lerp(path, grasp_t[:, None], close[None, :, None])
        before = ticks < self.close_frame
        desired_t = torch.where(before[None, :, None], path, expected_t)
        desired_R = torch.where(before[None, :, None, None], grasp_R[:, None], expected_R)
        q_target = torch.lerp(p['q_open'][None].expand(H, -1),
                              p['q_close'][None].expand(H, -1), close[:, None])
        relative_R = R[:, :, h].transpose(-1, -2) @ R[:, :, j]
        relative_t = (R[:, :, h].transpose(-1, -2) @ (t[:, :, j]-t[:, :, h])[..., None])[..., 0]
        limits = c['limits']
        violation = torch.maximum(limits[:, 0]-q, q-limits[:, 1]).clamp_min(0)
        q_full = torch.cat([c['q_anchor'][None, None], q], dim=1)
        speed = (q_full[:, 1:]-q_full[:, :-1]).abs() / float(c['dt'])
        depth = torch.maximum(f['inside_object'].amax(-1), f['inside_gripper_hull'].amax(-1))[:, 1:]
        f.update(pad_gap=pad_gap, pad_world=pad, contact_world=contact, normal_error=normal_error,
                 q=q, q_target=q_target[None], joint_violation=violation, joint_speed=speed,
                 hand_alignment=(t[:, :, h]-desired_t).norm(dim=-1),
                 hand_rotation=rotation_chord(R[:, :, h], desired_R),
                 relative_translation=relative_t, relative_rotation=relative_R,
                 hold_translation_step=(relative_t[:, 1:]-relative_t[:, :-1]).norm(dim=-1),
                 hold_rotation_step=rotation_chord(relative_R[:, 1:], relative_R[:, :-1]),
                 object_anchor_motion=(t[:, :, j]-t0[:, None]).norm(dim=-1),
                 hold_goal=(t[:, :, j]-self.target[:, None]).norm(dim=-1), depth=depth)
        return f

    def stage_terms(self, f, *, complete=True):
        start = max(0, self.close_frame-1)
        zero = f['hold_goal'].sum()*0
        penetration = (f['depth']-.001).clamp_min(0)
        support = (self.support_z-f['bottom'][:, 1:]-self.support_tolerance).clamp_min(0)
        held_translation = f['hold_translation_step'][:, start:]
        held_rotation = f['hold_rotation_step'][:, start:]
        return dict(goal=f['hold_goal'][:, -self.hold_frames:].mean() if complete else zero,
            contact=f['pad_gap'][:, start:].mean() if start<f['pad_gap'].shape[1] else zero,
            normal=f['normal_error'][:, start:].mean() if start<f['normal_error'].shape[1] else zero,
            alignment=f['hand_alignment'].mean(), rotation=f['hand_rotation'].mean(),
            opening=(f['q']-f['q_target']).abs().mean(),
            hold=(held_translation.mean()+.04*held_rotation.mean()) if held_translation.numel() else zero,
            stationary=((f['object_anchor_motion'][:, :self.close_frame-1]-.002).clamp_min(0).mean()
                        if self.close_frame > 1 else zero),
            penetration=penetration.mean()+penetration.amax(),
            joint=f['joint_violation'].mean()+f['joint_violation'].amax(),
            support=support.mean()+support.amax())

    def prefix_loss(self,pred,*,complete=False):
        """Stage costs on a generated prefix; final goal applies only at full horizon."""
        terms=self.stage_terms(self.features(pred,allow_prefix=True),complete=complete)
        return terms['goal']+sum(self.stage_weights[k]*v for k,v in terms.items() if k!='goal')

    def residuals(self, f):
        """Signed geometric constraint residuals, metres; <=0 satisfies the proxy."""
        start = max(0, self.close_frame-1)
        return dict(penetration=f['depth'].amax()-.001,
            support=(self.support_z-f['bottom'][:, 1:]).amax()-self.support_tolerance,
            opening=f['joint_violation'].amax()-.0001,
            speed=(f['joint_speed']-self.calibration['limits'][:, 2]).amax()*float(self.calibration['dt']),
            contact=f['pad_gap'][:, start:].amax()-.005)

    def __call__(self, pred, batch):
        f = self.features(pred)
        terms = self.stage_terms(f)
        loss = terms['goal'] + sum(self.stage_weights[k]*v for k, v in terms.items() if k != 'goal')
        diag = {f'stage_{k}': float(v.detach()) for k, v in terms.items()}
        diag.update({f'constraint_{k}_m': float(v.detach()) for k, v in self.residuals(f).items()})
        return loss, diag

    @torch.no_grad()
    def measure(self, pred):
        data = super().measure(pred)
        f = self.features(pred)
        start = max(0, self.close_frame-1)
        data.update(proposal=self.proposal_index,
            stage_pad_gap_max_mm=float(f['pad_gap'][:, start:].max())*1000,
            stage_final_pad_gap_mm=float(f['pad_gap'][:, -1].max())*1000,
            stage_hold_target_mean_mm=float(f['hold_goal'][:, -self.hold_frames:].mean())*1000,
            stage_relative_translation_step_mm=float(f['hold_translation_step'][:, start:].mean())*1000,
            stage_relative_rotation_step_deg=float((2*torch.asin((f['hold_rotation_step'][:, start:]/2).clamp(0, 1))).mean())*180/math.pi,
            finger_limit_violation_max_mm=float(f['joint_violation'].max())*1000)
        data['constraint_residuals_m'] = {k: float(v) for k, v in self.residuals(f).items()}
        data['geometric_constraints_satisfied'] = all(v <= 0 for v in data['constraint_residuals_m'].values())
        return data
