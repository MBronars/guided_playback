"""Frozen-checkpoint observation probes; all geometry is in metres/radians.

These helpers do not change training or production inference. References stay
separate from model inputs; future state tensors in inputs are placeholders.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from data.normalization import domain_normalizers
from inference import load_model_and_diffusion, run_diffusion_inference_df
from scripts.unified.eval_unified import hides_given_actions
from utils.conditioning import hide_actions
from utils.evaluation import TrajectoryScores
from utils.pose_native import materialize_pose_native
from utils.pose_ops import so3_exp
from utils.reproducibility import seed_for_device


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_frozen_run(path, receipt, fields):
    """Reject reuse across weights, sampling code or the caller's frozen inputs.

    Report/document edits do not change the sampling fingerprint. Legacy
    receipts without that fingerprint require a fresh output directory.
    """
    root = Path(__file__).resolve().parents[2]
    files = [root/'inference.py']
    for directory in ('models', 'diffusion', 'utils', 'data'):
        files += list((root/directory).rglob('*.py'))
    files += [root/'scripts/dev'/name for name in (
        'observation_probe_common.py', 'overnight_robustness.py', 'real_cloth_probe.py',
        'real_cloth_short_forecasts.py', 'observation_stability.py', 'observation_motion_noise.py')]
    digest = hashlib.sha256()
    for file in sorted(set(files)):
        digest.update(str(file.relative_to(root)).encode()+b'\0'+file.read_bytes()+b'\0')
    receipt = {**receipt, 'sampling_source_sha256': digest.hexdigest()}
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        canonical = lambda x: json.dumps(x, sort_keys=True, allow_nan=False)
        different = [key for key in ('checkpoint_sha256', 'sampling_source_sha256', *fields)
                     if canonical(old.get(key)) != canonical(receipt.get(key))]
        if different:
            raise ValueError(f'Existing experiment differs in {different}; use a fresh output directory. Nothing in its run receipt was overwritten.')
    write_json(path, receipt)


def require_given_actions(cfg):
    """Protocols that give and clamp the recorded robot motion (the real-cloth forecasts) cannot
    score a checkpoint trained with every action hidden (train.action_dropout=1)."""
    if hides_given_actions(cfg):
        raise ValueError("This protocol gives and clamps the recorded robot motion, but the checkpoint "
                         "was trained with every action hidden (train.action_dropout=1)")


# The overnight report text (package_overnight.py, overnight_report.py) describes this checkpoint
# only: the frozen four-domain 40K UNet, unified_unet_xy_p99_40k/final.pt.
REPORTED_CHECKPOINT_SHA256 = "1d08e5db8fd7b683e53aac18a52f598859fe6fdc69cdf67d372bd16569cb5302"


def reported_checkpoint(root):
    """{"path", "sha256"} of the checkpoint every run receipt under `root` names, refused unless
    it is the one the report text was written for."""
    receipts = {path.parent.name: json.loads(path.read_text()) for path in sorted(Path(root).glob("*/run.json"))}
    if not receipts:
        raise ValueError(f"Found no run receipts under {root}; nothing identifies the checkpoint to report")
    by_lane = {lane: receipt["checkpoint_sha256"] for lane, receipt in receipts.items()}
    if set(by_lane.values()) != {REPORTED_CHECKPOINT_SHA256}:
        raise ValueError(f"The overnight report text describes only the frozen 40K UNet (sha256 "
                         f"{REPORTED_CHECKPOINT_SHA256}), but the run receipts name {by_lane}; "
                         "update the report text before packaging another checkpoint")
    path = sorted({receipt["checkpoint"] for receipt in receipts.values()})[0]
    return {"path": path, "sha256": REPORTED_CHECKPOINT_SHA256}


def clone_batch(batch):
    return batch._map_tensors(lambda x: x.clone())


def single_clip(batch, row):
    if not 0 <= row < batch.batch_size:
        raise IndexError(row)
    return batch._map_tensors(lambda x: x[row:row + 1].clone())


def checkpoint_batch(batch, cfg, domain):
    """Use saved numerical scales; reject a fixture with a different window."""
    if batch.context_len != int(cfg.data.context_len):
        raise ValueError("Fixture context differs from checkpoint")
    if batch.num_future_frames != int(cfg.data.future_len):
        raise ValueError("Fixture horizon differs from checkpoint")
    if batch.joint_type is not None and not cfg.model.get("articulated_bodies", False):
        raise ValueError("Articulated fixture with a free-body checkpoint")
    delta, position, pose = domain_normalizers(cfg.normalization, domain)
    return dataclasses.replace(batch, delta_stats=delta, position_stats=position,
                               pose_delta_stats=pose if batch.body_poses is not None else None,
                               domain=domain)


def observed_input(reference):
    """Keep context and given action deltas; erase every future state/target.

The sampler only needs target shapes. This explicitly prevents a reference
future from entering a new observation adapter through a sidecar.
"""
    b = clone_batch(reference)
    C = b.context_len
    b.positions[:, C:] = b.anchor[:, None]
    if b.body_poses is not None:
        b.body_poses[:, C:] = b.body_poses[:, C - 1:C]
    if b.pose_deltas_gt is not None:
        b.pose_deltas_gt.zero_()
    return b


def _unit_noise(shape, rng):
    value = torch.randn(shape, generator=rng)
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def corrupt_observation(reference, family, severity, seed):
    """Independent observation errors, with perfect *world delta* commands.

Pose bias is shared by all links of an object, maintaining relative link poses.
History errors are pointwise on deformables and coherent translations on each
rigid object. Templates, identity and segmentation remain known in these tests.
"""
    if reference.device.type != "cpu" or reference.batch_size != 1:
        raise ValueError("Corrupt one CPU clip at a time")
    b = observed_input(reference)
    C, N = b.context_len, b.num_particles
    rng = torch.Generator().manual_seed(int(seed))
    if family == "clean" or (family not in ("stiffness", "density") and severity == 0):
        return b
    if family in ("stiffness", "density"):
        if reference.domain != "mpm" or severity <= 0:
            raise ValueError("Material multipliers require MPM and positive severity")
        field = b.E if family == "stiffness" else b.rho
        field[~b.action_mask] *= severity
        return b
    if family not in ("pose_translation", "pose_rotation", "history_jitter", "history_drift"):
        raise ValueError(family)
    if severity < 0:
        raise ValueError("Negative observation severity")
    valid = b.get_valid_mask()[0]
    rigid = (torch.zeros(N, dtype=torch.bool) if b.body_type is None else
             b.body_type[0, b.particle_body[0]] == 1) & valid
    deform = valid & ~rigid
    if family.startswith("history") and deform.any():
        eps = torch.randn((C, int(deform.sum()), 3), generator=rng)
        if family == "history_drift":
            eps[0].zero_()
            eps = eps.cumsum(0)
            eps /= eps[-1].square().sum(-1).mean().sqrt().clamp_min(1e-12)
        else:
            eps /= eps.square().sum(-1).mean(-1).sqrt()[:, None, None].clamp_min(1e-12)
        b.positions[0, :C, deform] += eps * (severity / 1000)
    for oid in b.object_ids[0, rigid].unique().tolist():
        rows = (b.object_ids[0] == oid) & rigid
        bodies = b.particle_body[0, rows].unique()
        center = reference.positions[0, C - 1, rows].mean(0)
        rotation = torch.eye(3).repeat(C, 1, 1)
        offset = torch.zeros(C, 3)
        if family == "pose_rotation":
            axis = _unit_noise((1, 3), rng) * np.deg2rad(severity)
            rotation[:] = so3_exp(axis)[0]
        elif family == "pose_translation":
            offset[:] = _unit_noise((1, 3), rng) * severity / 1000
        elif family == "history_jitter":
            offset = _unit_noise((C, 3), rng) * severity / 1000
        elif family == "history_drift":
            offset = torch.randn(C, 3, generator=rng)
            offset[0].zero_()
            offset = offset.cumsum(0)
            offset *= severity / (1000 * offset[-1].norm().clamp_min(1e-12))
        b.positions[0, :C, rows] = (torch.einsum(
            "tij,tpj->tpi", rotation, reference.positions[0, :C, rows] - center)
            + center + offset[:, None])
        for k in bodies.tolist():
            pose = reference.body_poses[0, :C, k]
            b.body_poses[0, :C, k, :3, :3] = rotation @ pose[:, :3, :3]
            b.body_poses[0, :C, k, :3, 3] = (torch.einsum(
                "tij,tj->ti", rotation, pose[:, :3, 3] - center) + center + offset)
    b.positions[:, C:] = b.anchor[:, None]
    if b.body_poses is not None:
        b.body_poses[:, C:] = b.body_poses[:, C - 1:C]
    return b


def geometry_residual_mm(batch):
    if batch.body_type is None:
        return 0.0
    worst = 0.0
    for k in (batch.body_type[0] == 1).nonzero().flatten().tolist():
        rows = (batch.particle_body[0] == k) & batch.get_valid_mask()[0]
        tmpl = batch.body_templates[0, k, batch.body_template_mask[0, k]]
        p = batch.body_poses[0, :batch.context_len, k]
        reconstructed = torch.einsum("tij,pj->tpi", p[:, :3, :3], tmpl) + p[:, None, :3, 3]
        worst = max(worst, float((reconstructed - batch.context[0, :, rows]).norm(dim=-1).max()) * 1000)
    return worst


class Predictor:
    def __init__(self, checkpoint, precision="bf16"):
        self.device = torch.device("cuda:0")
        self.model, self.diffusion, self.cfg, self.steps = load_model_and_diffusion(checkpoint, self.device)
        self.precision = precision
        self.calls = 0
        self.model.register_forward_pre_hook(self._count)

    def _count(self, module, inputs):
        self.calls += 1

    @torch.no_grad()
    def __call__(self, batch, seed):
        b = batch.to(self.device)
        if float(self.cfg.train.get("action_dropout", 0)) >= 1:
            b = hide_actions(b, 1.0)
        seed_for_device(seed, self.device)
        self.calls = 0
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.precision == "bf16"):
            x0 = run_diffusion_inference_df(
                self.model, b, self.diffusion, num_inference_steps=self.steps,
                schedule=self.cfg.get("df_schedule", "pyramid"),
                uncertainty_scale=float(self.cfg.get("df_uncertainty_scale", 1.0)))
        pred = (materialize_pose_native(b, *x0)[0] if isinstance(x0, tuple) else
                b.predictions_to_positions(b.delta_stats.denormalize(x0)))
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        if not torch.isfinite(pred).all():
            raise ValueError("Non-finite prediction")
        return pred.float().cpu(), {"seconds": seconds, "model_calls": self.calls,
                                   "peak_gpu_bytes": torch.cuda.max_memory_allocated()}


def score(reference, observed, prediction):
    scores = TrajectoryScores(per_frame=True)
    scores.add(reference, prediction, reference.future)
    out = scores.result()
    valid = reference.get_valid_mask()[0]
    delta = (observed.context[0, :, valid] - reference.context[0, :, valid]).norm(dim=-1)
    out["initial_context_rms_mm"] = float(delta.square().mean().sqrt()) * 1000
    out["anchor_rms_mm"] = float(delta[-1].square().mean().sqrt()) * 1000
    passive = valid & ~reference.action_mask[0]
    if passive.any():
        error = (prediction[0, :, passive] - reference.future[0, :, passive]).norm(dim=-1) * 1000
        out["passive_mean_mm"] = float(error.mean())
        out["passive_final_mm"] = float(error[-1].mean())
        out["passive_per_frame_mm"] = error.mean(-1).tolist()
    return out


def save_preview(path, reference, observed, prediction, max_points=2400):
    """A display subset only; metrics always use every valid particle."""
    rows = torch.linspace(0, int(reference.valid_lengths[0]) - 1,
                          min(max_points, int(reference.valid_lengths[0]))).long().unique()
    np.savez_compressed(path, reference=reference.positions[0, :, rows].numpy(),
                        observed=observed.context[0, :, rows].numpy(),
                        prediction=prediction[0, :, rows].numpy(),
                        object_ids=reference.object_ids[0, rows].numpy(),
                        action_mask=reference.action_mask[0, rows].numpy(),
                        rows=rows.numpy(), context_len=reference.context_len)
