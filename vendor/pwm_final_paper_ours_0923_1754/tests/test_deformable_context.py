"""model.unet_deformable_context: a GENERATED context for scenes with deformable rows
(models/unet_std.py, utils/trajectory_objective.py, inference.py). Standalone runner, house style:

    python tests/test_deformable_context.py

train.pose_noise_context generates the context through the rigid POSES and refuses deformable
rows. The flag extends it: the deformable rows' context particles are denoised and supervised
like the future rows, and the sampler steps them. Covers:
  1. Off (default): a deformable scene is still refused, and a rigid scene's loss is unchanged
     bit for bit whether the flag is on or off.
  2. On: the context particle prediction of DEFORMABLE rows receives gradient, rigid rows do not.
  3. Control: the objective with the flag but a model that does not stash refuses.
  4. Sampler: with the flag the deformable context rows move along the predicted velocity;
     without it the checkpoint is refused.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion import TrajectoryFlowMatching
from inference import run_diffusion_inference_df
from models.unet_std import UNetWrapper
from tests.test_pose_noise_context import _objective_kwargs, _rigid_batch
from tests.test_unet_std_pose_native import C, F, _model_cfg, make_batch, open_gates
from utils.trajectory_objective import trajectory_objective


def _model(seed=0, **cfg):
    torch.manual_seed(seed)
    w = UNetWrapper(_model_cfg(**cfg))
    open_gates(w)
    return w


def _fm():
    return TrajectoryFlowMatching(num_train_timesteps=1000, prediction_type="sample")


def test_default_off_is_unchanged():
    try:
        trajectory_objective(_model(), make_batch(), _fm(), **_objective_kwargs())
        raise AssertionError("a deformable scene must still be refused without the flag")
    except ValueError as e:
        assert "unet_deformable_context" in str(e), e
    losses = []
    for on in (False, True):
        w = _model(unet_deformable_context=on)
        torch.manual_seed(0)
        loss, _, _ = trajectory_objective(w, _rigid_batch(), _fm(),
                                          **_objective_kwargs(deformable_context=on))
        losses.append(loss.detach())
    assert torch.equal(losses[0], losses[1]), losses
    print("ok  off: deformable scenes refused; rigid scene loss bit-identical with the flag on")


def test_deformable_context_rows_are_supervised():
    b = make_batch()
    w = _model(unet_deformable_context=True)
    keep = {}
    orig = w.forward

    def fwd(*a, **k):
        out = orig(*a, **k)
        w._last_ctx_part_pred.retain_grad()
        keep["p"] = w._last_ctx_part_pred
        return out

    w.forward = fwd
    torch.manual_seed(0)
    loss, _, _ = trajectory_objective(w, b, _fm(), **_objective_kwargs(deformable_context=True))
    assert torch.isfinite(loss)
    loss.backward()
    g = keep["p"].grad                                                   # [B, C-1, N, 3]
    assert g is not None and g.shape[1] == C - 1
    deform = (b.body_type.gather(1, b.particle_body) == 0)[:, None, :, None].expand_as(g)
    assert g[deform].abs().sum() > 0, "deformable context rows must be supervised"
    assert g[~deform].abs().sum() == 0, "rigid context rows are the pose term's, not this one's"
    print("ok  on: deformable context particles get gradient, rigid rows none")


def test_flag_without_model_stash_refuses():
    """Control: the objective's term is live -- a model that does not stash cannot pass silently."""
    try:
        trajectory_objective(_model(), make_batch(), _fm(), **_objective_kwargs(deformable_context=True))
        raise AssertionError("expected a refusal")
    except RuntimeError as e:
        assert "did not stash" in str(e), e
    print("ok  control: the term refuses a model without the stash")


class _Recorder(torch.nn.Module):
    """Constant-velocity stand-in for the context particles; zero elsewhere."""

    def __init__(self, deform):
        super().__init__()
        self.cfg = SimpleNamespace(all_relative=True, pose_native=True, per_object_noise=False,
                                   unet_deformable_context=deform)
        self._context_predicted = True
        self._last_ctx_pose_pred = self._last_ctx_part_pred = None
        self.ctx_tensors = []

    def forward(self, batch, noisy, times, *, noisy_pose=None, noisy_context=None,
                context_timesteps=None, noisy_ctx_pose=None, **kw):
        self.ctx_tensors.append(noisy_context.clone())
        self._last_ctx_pose_pred = torch.zeros_like(noisy_ctx_pose)
        self._last_ctx_part_pred = torch.ones_like(noisy_context)       # velocity 1 everywhere
        return torch.zeros_like(noisy), torch.zeros_like(noisy_pose)


def test_sampler_steps_deformable_context():
    b = make_batch()
    diffusion = TrajectoryFlowMatching(num_train_timesteps=1000, prediction_type="velocity")
    try:
        run_diffusion_inference_df(_Recorder(False), b, diffusion, num_inference_steps=4, schedule="pyramid")
        raise AssertionError("a deformable scene must be refused without the flag")
    except ValueError as e:
        assert "deformable" in str(e), e
    r = _Recorder(True)
    run_diffusion_inference_df(r, b, diffusion, num_inference_steps=4, schedule="pyramid")
    first, last = r.ctx_tensors[0], r.ctx_tensors[-1]
    deform = (b.body_type.gather(1, b.particle_body) == 0)[0]
    moved = (first - last)[:, :, deform]                  # Euler: x -= dt * v, v = 1 -> moves down
    assert (moved > 0).all(), "every deformable context row must move along the predicted velocity"
    print("ok  sampler: deformable context rows are denoised with the flag, refused without it")


if __name__ == "__main__":
    test_default_off_is_unchanged()
    test_deformable_context_rows_are_supervised()
    test_flag_without_model_stash_refuses()
    test_sampler_steps_deformable_context()
    print("ALL OK")
