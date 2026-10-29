"""Regression tests that exercise the production train_step, not a toy loop."""
from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator

from diffusion.flow_matching import TrajectoryFlowMatching
from tests.test_train import create_dummy_batch
from tests.test_unet_std_pose_native import make_batch
from train import train_step


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        return torch.full_like(grad, float("nan"))


class ConstantModel(torch.nn.Module):
    def __init__(self, *, pose=False, failure=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.cfg = SimpleNamespace(pose_native=pose)
        self.failure = failure

    def forward(self, batch, noisy_future, timesteps=None, **kwargs):
        weight = self.weight
        if self.failure == "loss":
            weight = weight * float("nan")
        elif self.failure == "gradient":
            weight = BadGradient.apply(weight)
        if self.cfg.pose_native:
            return None, weight.expand_as(kwargs["noisy_pose"])
        return weight.expand_as(noisy_future)


def scalar_batch(target=1.0):
    batch = create_dummy_batch(B=1, T=3, N=1, context_len=2)
    batch.positions.zero_()
    batch.positions[:, 2].fill_(target)
    return batch


def setup(*, pose=False, failure=None, accumulation=1):
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=accumulation)
    model = ConstantModel(pose=pose, failure=failure)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model, optimizer = accelerator.prepare(model, optimizer)
    diffusion = TrajectoryFlowMatching(prediction_type="sample")
    return model, optimizer, accelerator, diffusion


@pytest.mark.parametrize("failure", ["loss", "gradient"])
@pytest.mark.parametrize("grad_clip", [None, 1.0])
def test_nonfinite_step_never_mutates_weights(failure, grad_clip):
    model, optimizer, accelerator, diffusion = setup(failure=failure)
    before = model.weight.detach().clone()
    _, metrics = train_step(model, scalar_batch(), diffusion, optimizer, accelerator,
                            grad_clip=grad_clip)
    assert metrics["skipped"], metrics
    assert torch.equal(model.weight, before)
    assert torch.isfinite(model.weight)


def test_rigid_only_spike_has_valid_diagnostics():
    model, optimizer, accelerator, diffusion = setup(pose=True)
    batch = make_batch(deform_n=0, rigid=((1, 24), (2, 16)))
    before = model.weight.detach().clone()
    _, metrics = train_step(model, batch, diffusion, optimizer, accelerator,
                            grad_clip=1.0, loss_ema=1e-12, loss_spike_threshold=2.0)
    assert metrics["skipped"]
    assert torch.equal(model.weight, before)
    assert metrics["spike_frame_mse"].shape == (batch.num_future_frames,)
    assert torch.isfinite(metrics["spike_frame_mse"]).all()


def test_spike_diagnostics_allow_fewer_than_ten_particles():
    model, optimizer, accelerator, diffusion = setup()
    _, metrics = train_step(model, scalar_batch(), diffusion, optimizer, accelerator,
                            grad_clip=1.0, loss_ema=1e-12, loss_spike_threshold=2.0)
    assert metrics["skipped"]
    assert metrics["spike_top_errs"].numel() == 1


def test_gradient_accumulation_uses_both_microbatches():
    model, optimizer, accelerator, diffusion = setup(accumulation=2)
    for i, target in enumerate((1.0, 3.0)):
        with accelerator.accumulate(model):
            train_step(model, scalar_batch(target), diffusion, optimizer, accelerator,
                       grad_clip=None, grad_norm_spike_threshold=float("inf"),
                       clip_deltas=False)
        if i == 0:
            assert model.weight.item() == pytest.approx(0.25)
    # mean d/dw[(w-1)^2, (w-3)^2] at w=.25 = -3.5, SGD lr=.1.
    assert model.weight.item() == pytest.approx(0.6, abs=1e-6)


def test_nonfinite_microbatch_discards_the_whole_accumulation_group():
    model, optimizer, accelerator, diffusion = setup(accumulation=2)
    for failure in ("loss", None):
        model.failure = failure
        with accelerator.accumulate(model):
            train_step(model, scalar_batch(), diffusion, optimizer, accelerator,
                       grad_clip=None, grad_norm_spike_threshold=float("inf"))
    assert model.weight.item() == pytest.approx(0.25)
    # A subsequent healthy group must be able to train.
    for _ in range(2):
        with accelerator.accumulate(model):
            train_step(model, scalar_batch(), diffusion, optimizer, accelerator,
                       grad_clip=None, grad_norm_spike_threshold=float("inf"))
    assert model.weight.item() == pytest.approx(0.4, abs=1e-6)


def test_rejected_step_keeps_adam_moments_and_step_counter():
    import copy
    model, _, accelerator, diffusion = setup()
    optimizer = accelerator.prepare(torch.optim.AdamW(model.parameters(), lr=0.01))
    train_step(model, scalar_batch(), diffusion, optimizer, accelerator, grad_clip=1.0)
    before_weights = model.weight.detach().clone()
    before_state = copy.deepcopy(optimizer.state_dict())
    model.failure = "gradient"
    _, metrics = train_step(model, scalar_batch(), diffusion, optimizer, accelerator, grad_clip=1.0)
    assert metrics["skipped"]
    assert torch.equal(model.weight, before_weights)
    torch.testing.assert_close(optimizer.state_dict(), before_state, rtol=0, atol=0)
