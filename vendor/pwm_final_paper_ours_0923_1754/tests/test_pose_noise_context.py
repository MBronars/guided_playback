"""A GENERATED context: train.noise_context + train.pose_noise_context (models/factory.py,
utils/pose_native.py, utils/trajectory_objective.py, inference.py).
Standalone runner, house style:

    python tests/test_pose_noise_context.py

The retired train.noise_context noised every particle's history independently, which pulls a
rigid body's particles away from its pose. This recipe noises the context POSES instead and
rebuilds each body's context particle rows from them, so the body is rigid at every level and
the whole [context | future] window is one uniform, predicted stream. Covers:
  1. Clean context poses reproduce the clean context exactly (and the future path is unchanged).
  2. Noised context poses stay RIGID, with the retired per-particle noising as the control.
  3. The objective feeds the model noised context poses, scores them, and refuses the
     configurations that cannot work (no particle side, deformable rows).
  4. The sampler follows the checkpoint: a context-predicting one generates its history.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion import TrajectoryFlowMatching
from inference import run_diffusion_inference_df
from models.factory import context_predicted
from tests.test_unet_std_pose_native import (B, C, F, T, _model_cfg, _noisy_pose_gt, make_batch,
                                             open_gates)
from models.unet_std import UNetWrapper
from utils.pose_native import context_pose_deltas, pose_native_inputs


def _rigid_batch(seed=0):
    """Pure-rigid scene (the pickup / rl_push case): gripper links + a rigid object."""
    return make_batch(deform_n=0, rigid=((0, 24), (0, 16), (1, 16)), seed=seed)


def _ctx_pose_gt(b):
    ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
    return b.pose_delta_stats.normalize(context_pose_deltas(b, ridx), clip=False)


def test_context_predicted_truth_table():
    assert context_predicted({"train": {}}) is False
    assert context_predicted({"train": {"noise_context": True}}) is False
    assert context_predicted({"train": {"noise_context": True, "pose_noise_context": True}}) is True
    print("ok  context_predicted: only a pose-noised context is a generated one")


def test_clean_context_poses_reproduce_the_clean_context():
    b = _rigid_batch()
    n = b.positions.shape[2]
    x_t = torch.randn(B, F, n, 3)
    x_ctx = torch.randn(B, C - 1, n, 3)
    fut_ref, pack_ref = pose_native_inputs(b, x_t, _noisy_pose_gt(b))
    fut, ctx, pack = pose_native_inputs(b, x_t, _noisy_pose_gt(b),
                                        noisy_ctx_pose=_ctx_pose_gt(b), noisy_context=x_ctx)
    gt_ctx = b.delta_stats.normalize(b.context_deltas, clip=False)
    assert torch.allclose(ctx, gt_ctx, atol=1e-4), (ctx - gt_ctx).abs().max()
    # the future anchor is the pose the context integration reached = the clean frame C-1 pose
    assert torch.allclose(pack["anchor_centers"], pack_ref["anchor_centers"], atol=1e-5)
    assert torch.allclose(fut, fut_ref, atol=1e-5)
    assert pack["ctx_centers"].shape == (B, C - 1, pack["rigid_idx"].numel(), 3)
    print("ok  clean context poses rebuild the clean context and leave the future path unchanged")


def test_noised_context_stays_rigid():
    """Every body's context rows remain a rigid motion of its template; the retired
    per-particle noising (the control) does not."""
    b = _rigid_batch()
    n = b.positions.shape[2]
    torch.manual_seed(0)
    noisy_ctx_pose = _ctx_pose_gt(b) + 0.3 * torch.randn_like(_ctx_pose_gt(b))
    _, ctx, _ = pose_native_inputs(b, torch.randn(B, F, n, 3), _noisy_pose_gt(b),
                                   noisy_ctx_pose=noisy_ctx_pose,
                                   noisy_context=torch.randn(B, C - 1, n, 3))
    frame0 = b.positions[:, 0]
    pos = frame0[:, None] + b.delta_stats.denormalize(ctx).cumsum(1)     # [B, C-1, N, 3]
    control = frame0[:, None] + b.delta_stats.denormalize(
        b.delta_stats.normalize(b.context_deltas) + 0.3 * torch.randn(B, C - 1, n, 3)).cumsum(1)

    def worst_rigid_residual(P):
        out = 0.0
        for body in b.body_type[0].nonzero(as_tuple=True)[0].tolist():
            rows = (b.particle_body[0] == body).nonzero(as_tuple=True)[0]
            for e in range(B):
                for f in range(C - 1):
                    X, Y = frame0[e, rows], P[e, f, rows]                 # template, moved
                    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
                    U, _, Vh = torch.linalg.svd(Yc.T @ Xc)
                    d = torch.sign(torch.det(U @ Vh))
                    R = U @ torch.diag(torch.tensor([1.0, 1.0, d])) @ Vh
                    out = max(out, float((Yc - Xc @ R.T).abs().max()))
        return out

    assert worst_rigid_residual(pos) < 1e-5, worst_rigid_residual(pos)
    assert worst_rigid_residual(control) > 1e-3, "control: per-particle noise must break rigidity"
    print("ok  a noised context is still a rigid body (retired per-particle noising is not)")


def _objective_kwargs(**over):
    kw = dict(diffusion_forcing=True, loss_weighting="uniform", noise_context=True,
              pose_noise_context=True, pose_loss_space="sample", clip_deltas=False,
              clip_pose_deltas=False, timestep_schedule={"ramp": 1.0}, ramp_slope=0.04,
              ramp_jitter=0.02)
    kw.update(over)
    return kw


def _model(seed=0, **cfg):
    torch.manual_seed(seed)
    w = UNetWrapper(_model_cfg(**cfg))
    open_gates(w)
    return w


def test_objective_scores_the_generated_context():
    from utils.trajectory_objective import trajectory_objective
    b = _rigid_batch()
    w = _model()
    seen = {}

    def watch(_m, args, kw):
        seen.update(kw)
        seen["future_timesteps"] = args[2]

    hook = w.register_forward_pre_hook(watch, with_kwargs=True)
    diffusion = TrajectoryFlowMatching(num_train_timesteps=1000, prediction_type="sample")
    torch.manual_seed(0)
    loss, metrics, _ = trajectory_objective(w, b, diffusion, **_objective_kwargs())
    hook.remove()
    assert seen["noisy_ctx_pose"] is not None and seen["noisy_ctx_pose"].shape[1] == C - 1
    assert seen["context_timesteps"] is not None
    # ONE ramp across the window, not a staircase restarting at the boundary: the context rows
    # sit at the clean end of it. (Jitter makes this a statement about the window, not per row.)
    assert float(seen["context_timesteps"].float().mean()) < float(seen["future_timesteps"].float().mean())
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    grads = [p.grad for p in w.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "no parameter received gradient"
    print("ok  the objective noises, feeds and scores the context poses")


def test_objective_refuses_impossible_recipes():
    from utils.trajectory_objective import trajectory_objective
    diffusion = TrajectoryFlowMatching(num_train_timesteps=1000, prediction_type="sample")
    w = _model()
    for over, needle in ((dict(noise_context=False), "requires noise_context"),
                         (dict(), "deformable")):
        b = _rigid_batch() if needle == "requires noise_context" else make_batch()
        try:
            trajectory_objective(w, b, diffusion, **_objective_kwargs(**over))
            raise AssertionError(f"expected a refusal mentioning {needle!r}")
        except ValueError as e:
            assert needle in str(e), e
    print("ok  refusals: no particle side, deformable rows")


class _Recorder(torch.nn.Module):
    """Zero-velocity stand-in that records how the sampler queried it."""

    def __init__(self, predicted):
        super().__init__()
        from types import SimpleNamespace
        self.cfg = SimpleNamespace(all_relative=True, pose_native=True, per_object_noise=False)
        self._context_predicted = predicted
        self._last_ctx_pose_pred = None
        self.ctx_levels, self.fut_levels, self.ctx_tensors = [], [], []

    def forward(self, batch, noisy, times, *, noisy_pose=None, noisy_context=None,
                context_timesteps=None, noisy_ctx_pose=None, **kw):
        self.fut_levels.append(times.clone())
        self.ctx_levels.append(None if context_timesteps is None else context_timesteps.clone())
        self.ctx_tensors.append(None if noisy_context is None else noisy_context.clone())
        if noisy_ctx_pose is not None:
            self._last_ctx_pose_pred = torch.zeros_like(noisy_ctx_pose)
        return None, torch.zeros_like(noisy_pose)


def test_the_ladder_spans_the_whole_window():
    """The pyramid over [context | future]: the context rows lead, so they finish first —
    the sampler's counterpart of the ramp the objective trains on."""
    from diffusion.scheduling import generate_scheduling_matrix
    K = generate_scheduling_matrix((C - 1) + F, 4, "pyramid", 1.0)
    assert K.shape[1] == (C - 1) + F
    assert (K[:, 0] <= K[:, -1]).all(), "the earliest context row is never noisier than the last future row"
    assert K[-1].max() == 0 and K[0].min() > 0
    print("ok  the sampler ladder spans context and future as one window")


def test_sampler_follows_the_checkpoint():
    b = _rigid_batch()
    diffusion = TrajectoryFlowMatching(num_train_timesteps=1000, prediction_type="velocity")
    given = b.delta_stats.normalize(b.context_deltas)

    reader = _Recorder(False)
    run_diffusion_inference_df(reader, b, diffusion, num_inference_steps=4, schedule="pyramid")
    assert all(t is None or torch.equal(t, given) for t in reader.ctx_tensors), \
        "a context-reading checkpoint must be given its history, unchanged"
    assert all(t.shape[1] == F for t in reader.fut_levels)

    writer = _Recorder(True)
    run_diffusion_inference_df(writer, b, diffusion, num_inference_steps=4, schedule="pyramid")
    levels = torch.stack([t for t in writer.ctx_levels if t is not None])
    assert levels.shape[-1] == C - 1
    assert float(levels[0].max()) > float(levels[-1].max()) > -1, "context levels must descend"
    assert float(levels[0].max()) > 0, "the generated context must start noisy"
    assert all(t.shape[1] == F for t in writer.fut_levels), "future rows keep their own levels"
    print("ok  the sampler generates the context for a context-predicting checkpoint")


if __name__ == "__main__":
    test_context_predicted_truth_table()
    test_clean_context_poses_reproduce_the_clean_context()
    test_noised_context_stays_rigid()
    test_objective_scores_the_generated_context()
    test_objective_refuses_impossible_recipes()
    test_the_ladder_spans_the_whole_window()
    test_sampler_follows_the_checkpoint()
    print("ALL OK")
