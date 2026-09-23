"""Adapter interface: one class per model codebase. bin/guide.py, gp/guide.py and the replay backends only use this.

An adapter turns a checkpoint into (a) held-out episodes as a batch, (b) an optimization State (the noise draws that
define one streaming rollout per member, which of them are free, their Gaussian-shell radii) and (c) a differentiable
rollout(batch, state) -> positions [B, N+1, P, 3] in the checkpoint's own world frame (metres). `roles` and `export`
give the replay backends and the viewer what they need without knowing the codebase.
"""
import dataclasses
import numpy as np
import torch


@dataclasses.dataclass
class State:
    vars: list                      # tensors with requires_grad (optimized)
    free: list                      # per var: bool mask broadcastable to it, or None (= all free)
    shell: list                     # per var: per-draw shell radius [.., 1, 1] broadcastable, or None (no reprojection)
    fixed: dict                     # tensors that are part of the rollout but never optimized (e.g. particle draws)
    info: dict                      # printable facts (draws per row, free counts, model calls)


class Adapter:
    name = "base"
    frame_dt = None                 # seconds per model frame
    sim = None                      # default replay backend name
    joint_limits = None             # (q_min, q_max) metres for prismatic fingers (paper adapter only)

    def load(self, ckpt, device, compile=True):
        raise NotImplementedError

    def episodes(self, indices, transitions, domain=None):
        """Held-out episodes (repeated indices = population) as ONE batch on the device, N+1 frames; sets batch.gt_frames."""
        raise NotImplementedError

    def prepare(self, batch, transitions, steps, keep, seed, churn, t_freeze, fix_fps):
        raise NotImplementedError

    def rollout(self, batch, state, grad):
        raise NotImplementedError

    def object_mask(self, batch):
        """[B, P] bool: the passive object's particles (rigid body or deformable), the guided quantity."""
        raise NotImplementedError

    def roles(self, batch):
        """[B, P] int8: 0 actuator body, 1 actuator finger, 2 passive object, -1 padding."""
        raise NotImplementedError

    def export(self, batch):
        """Arrays for rollout.npz beyond positions (particle_body, pose_action_mask, joint_type, body_type, object_ids, body_poses0, ...)."""
        raise NotImplementedError

    def model_calls(self, state):
        return state.info.get("model_calls")


def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
