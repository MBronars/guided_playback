"""An analytic GT oracle separates sampler/integration errors from learned quality.

This deliberately uses future labels. It tests numerical plumbing, never model
prediction or generalization. The SAME oracle must satisfy the training objective
and the actual per-frame production sampler, including pose-only outputs.
"""
from types import SimpleNamespace

import pytest
import torch

from diffusion import TrajectoryFlowMatching
from inference import run_diffusion_inference_df
from tests.test_unet_std_pose_native import make_batch
from train import trajectory_objective
from utils.pose_native import materialize_pose_native


class GroundTruthOracle(torch.nn.Module):
    def __init__(self, diffusion):
        super().__init__()
        self.diffusion = diffusion
        self.cfg = SimpleNamespace(all_relative=True, pose_native=True, per_object_noise=False)

    def field(self, noisy, target, times):
        if self.diffusion.prediction_type == "sample":
            return target
        time = times.float().clamp_min(1e-6) / self.diffusion.num_train_timesteps
        while time.ndim < noisy.ndim:
            time = time.unsqueeze(-1)
        return (noisy - target) / time

    def forward(self, batch, noisy, times, *, noisy_pose=None, **kwargs):
        target = batch.prediction_stats.normalize(batch.prediction_x0)
        particles = self.field(noisy, target, times)
        if noisy_pose is None:
            return particles
        rigid = batch.body_type[0] == 1
        pose_target = batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, rigid])
        if bool(rigid.all()):
            particles = None
        return particles, self.field(noisy_pose, pose_target, times)


def scene(kind):
    options = {"deformable": dict(rigid=(), sidecar=False), "mixed": {},
               "rigid": dict(deform_n=0, rigid=((1, 24), (2, 16)))}
    return make_batch(**options[kind])


@pytest.mark.parametrize("kind", ["deformable", "mixed", "rigid"])
@pytest.mark.parametrize("prediction", ["velocity", "sample"])
@pytest.mark.parametrize("schedule", ["full_sequence", "pyramid"])
@pytest.mark.parametrize("timesteps,steps", [(37, 1), (1000, 4)])
def test_production_sampler_recovers_oracle_target(kind, prediction, schedule, timesteps, steps):
    batch = scene(kind)
    diffusion = TrajectoryFlowMatching(num_train_timesteps=timesteps, prediction_type=prediction)
    oracle = GroundTruthOracle(diffusion)
    result = run_diffusion_inference_df(oracle, batch, diffusion, num_inference_steps=steps,
                                       schedule=schedule)
    if isinstance(result, tuple):
        predicted = materialize_pose_native(batch, *result)[0]
    else:
        predicted = batch.predictions_to_positions(batch.prediction_stats.denormalize(result))
    torch.testing.assert_close(predicted, batch.future, rtol=0, atol=2e-6)


@pytest.mark.parametrize("kind", ["deformable", "mixed", "rigid"])
@pytest.mark.parametrize("prediction", ["velocity", "sample"])
def test_sampler_oracle_also_satisfies_the_production_training_objective(kind, prediction):
    batch = scene(kind)
    diffusion = TrajectoryFlowMatching(prediction_type=prediction)
    oracle = GroundTruthOracle(diffusion)
    torch.manual_seed(314)
    loss, _, _ = trajectory_objective(oracle, batch, diffusion, diffusion_forcing=True,
                                      noise_context=True, clip_deltas=False)
    assert float(loss) < 1e-8


@pytest.mark.parametrize("kind", ["deformable", "mixed", "rigid"])
@pytest.mark.parametrize("prediction", ["velocity", "sample"])
def test_training_visualizations_use_the_pose_native_sampler(kind, prediction):
    from train import run_inference_df, run_one_step_denoising
    batch = scene(kind)
    fm = TrajectoryFlowMatching(prediction_type=prediction)
    oracle = GroundTruthOracle(fm)
    result = run_inference_df(oracle, batch, fm, num_inference_steps=4)
    assert result["position_mse"] < 1e-10
    for timestep in (100, 500, 900, "random"):
        result = run_one_step_denoising(oracle, batch, fm, timestep=timestep)
        assert result["mse"] < 1e-10
        torch.testing.assert_close(result["est_positions"], batch.future, rtol=0, atol=2e-6)


def test_training_visualization_frame_curve_ignores_padding():
    from train import _prediction_metrics
    batch = scene("deformable")
    batch.valid_lengths.fill_(batch.num_particles - 1)
    predicted = batch.future.clone()
    predicted[:, :, -1] = float("nan")
    result = _prediction_metrics(batch, predicted)
    assert result["position_mse"] == 0
    assert result["per_frame_mse"] == [0] * batch.num_future_frames


class MetadataWrapper(torch.nn.Module):
    """DDP/compiled attribute layout with an observable forward policy."""
    def __init__(self, module, attribute="module", *, autocast=False):
        super().__init__()
        self.attribute, self.calls, self.autocast = attribute, 0, autocast
        setattr(self, attribute, module)

    def forward(self, *args, **kwargs):
        self.calls += 1
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.autocast):
            return getattr(self, self.attribute)(*args, **kwargs)


@pytest.mark.parametrize("kind", ["deformable", "mixed", "rigid"])
@pytest.mark.parametrize("prediction", ["velocity", "sample"])
@pytest.mark.parametrize("wrappers", [("module",), ("_orig_mod",), ("module", "_orig_mod")])
def test_wrapped_sampler_and_diagnostic_keep_pose_context_and_forward_policy(kind, prediction, wrappers):
    from train import run_one_step_denoising
    batch = scene(kind)
    fm = TrajectoryFlowMatching(prediction_type=prediction)

    class ContextOracle(GroundTruthOracle):
        _clean_context_timestep = 1

        def forward(self, *args, **kwargs):
            assert torch.is_autocast_enabled("cpu"), "Sampling bypassed the forward wrapper"
            assert kwargs["noisy_context"] is not None
            assert bool((kwargs["context_timesteps"] == 1).all())
            return super().forward(*args, **kwargs)

    wrapped = ContextOracle(fm)
    layers = []
    for attribute in wrappers:
        wrapped = MetadataWrapper(wrapped, attribute, autocast=True)
        layers.append(wrapped)
    result = run_diffusion_inference_df(wrapped, batch, fm, num_inference_steps=4, schedule="full_sequence")
    assert isinstance(result, tuple) == (kind != "deformable")
    predicted = (materialize_pose_native(batch, *result)[0] if isinstance(result, tuple) else
                 batch.predictions_to_positions(batch.prediction_stats.denormalize(result)))
    torch.testing.assert_close(predicted, batch.future, rtol=0, atol=2e-6)
    diagnostic = run_one_step_denoising(wrapped, batch, fm, timestep=500)
    assert diagnostic["mse"] < 1e-10
    assert all(layer.calls == 5 for layer in layers)


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("kind", ["deformable", "mixed", "rigid"])
def test_wrapped_retained_model_sampler_matches_raw_model_exactly(architecture, kind):
    from tests.test_model_scope import _construct
    batch, model = scene(kind), _construct(architecture, False)
    fm = TrajectoryFlowMatching()
    torch.manual_seed(231)
    raw = run_diffusion_inference_df(model, batch, fm, num_inference_steps=4, schedule="full_sequence")
    wrapper = MetadataWrapper(MetadataWrapper(model, "_orig_mod"))
    torch.manual_seed(231)
    wrapped = run_diffusion_inference_df(wrapper, batch, fm, num_inference_steps=4, schedule="full_sequence")
    torch.testing.assert_close(raw, wrapped, rtol=0, atol=0)
    assert wrapper.calls == wrapper.module.calls == 4


@pytest.mark.parametrize("prediction", ["velocity", "sample"])
@pytest.mark.parametrize("future_len", [1, 2, 3, 20])
@pytest.mark.parametrize("direct", [False, True])
def test_streaming_rolls_every_frame_and_conditions_on_its_predictions(monkeypatch, prediction, future_len, direct):
    from inference import AutoregressiveConfig, autoregressive_rollout, streaming_rollout
    from tests.test_rollout_batch import sequence

    batch = sequence()
    batch.positions.mul_(.001)
    batch.actions.mul_(.001)
    batch.action_mask.fill_(False)
    fm = TrajectoryFlowMatching(prediction_type=prediction)
    context_len, increment = 3, .004
    expected = batch.positions.clone()
    expected[:, context_len:] = expected[:, context_len - 1:context_len] + increment * torch.arange(
        1, batch.num_frames - context_len + 1)[None, :, None, None]
    get_window, starts = batch.get_window, []

    def recorded_window(*args, **kwargs):
        window = get_window(*args, **kwargs)
        start = kwargs["start_frame"]
        starts.append(start)
        torch.testing.assert_close(window.context, expected[:, start:start + context_len], rtol=0, atol=1e-6)
        return window

    monkeypatch.setattr(batch, "get_window", recorded_window)

    class ConstantIncrement(GroundTruthOracle):
        def forward(self, window, noisy, times, **kwargs):
            target = window.delta_stats.normalize(torch.full_like(noisy, increment))
            return self.field(noisy, target, times)

    config = AutoregressiveConfig(context_len=context_len, future_len=future_len,
                                  num_inference_steps=3, streaming=True)
    result = (streaming_rollout if direct else autoregressive_rollout)(ConstantIncrement(fm), fm, batch, config)
    torch.testing.assert_close(result["predicted_positions"], expected, rtol=0, atol=1e-6)
    assert max(starts) > 0, "The rolling buffer must consume its own predicted context"
    assert len(result["per_frame_mse"]) == batch.num_frames - context_len
    assert all(value > 0 for value in result["per_frame_mse"]), "No uncomputed zero-error tail"
