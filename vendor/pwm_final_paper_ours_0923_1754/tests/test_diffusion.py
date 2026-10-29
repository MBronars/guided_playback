"""Retained flow-matching arithmetic, clocks and regression oracles."""
import pytest
import torch

from diffusion import TrajectoryFlowMatching, generate_scheduling_matrix
from tests.test_model_scope import _reference_module


def test_get_x0_est_recovers_true_x0():
    """Recovered from the retired guidance tests: this is ordinary FM algebra."""
    fm = TrajectoryFlowMatching()
    fm.scheduler.set_timesteps(25)
    x_0 = torch.randn(2, 4, 8, 3)
    noise = torch.randn_like(x_0)
    velocity = noise - x_0
    for timestep in fm.scheduler.timesteps[1:-1]:
        time = timestep.float().view(1, 1, 1, 1) / 1000
        sample = (1 - time) * x_0 + time * noise
        torch.testing.assert_close(fm.get_x0_est(velocity, timestep, sample), x_0,
                                   rtol=0, atol=1e-4)


@pytest.mark.parametrize("mode", ["velocity", "sample"])
@pytest.mark.parametrize("shape", [(), (2,), (2, 4)])
def test_sample_velocity_clean_roundtrip(mode, shape):
    torch.manual_seed(574)
    fm = TrajectoryFlowMatching(prediction_type=mode)
    clean = torch.randn(2, 4, 8, 3)
    noise = torch.randn_like(clean)
    times = torch.rand(shape) * 800 + 100
    sample = fm.add_noise(clean, noise, times)
    output = clean if mode == "sample" else noise - clean
    torch.testing.assert_close(fm.get_x0_est(output, times, sample), clean, rtol=0, atol=1e-5)
    torch.testing.assert_close(fm.as_velocity(output, sample, times), noise - clean,
                               rtol=0, atol=1e-5)


@pytest.mark.parametrize("mode", ["velocity", "sample"])
@pytest.mark.parametrize("sampler", ["uniform", "beta"])
@pytest.mark.parametrize("frames", [None, 4])
def test_flow_primitives_match_frozen_baseline_exactly(mode, sampler, frames):
    reference_cls = _reference_module("diffusion/flow_matching.py").TrajectoryFlowMatching
    outputs = []
    for cls in (reference_cls, TrajectoryFlowMatching):
        torch.manual_seed(72187)
        fm = cls(prediction_type=mode, time_sampler=sampler)
        clean, noise = torch.randn(2, 4, 8, 3), torch.randn(2, 4, 8, 3)
        times = fm.sample_timesteps(2, torch.device("cpu"), num_frames=frames, low=1)
        sample = fm.add_noise(clean, noise, times)
        output = clean * 0.3 if mode == "sample" else noise * 0.3
        outputs.append((times, sample, fm.get_target(clean, noise, times),
                        fm.get_x0_est(output, times, sample), fm.as_velocity(output, sample, times)))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_beta_time_sampler_distribution():
    fm = TrajectoryFlowMatching(prediction_type="sample", time_sampler="beta")
    torch.manual_seed(0)
    normalized = fm.sample_timesteps(20000, torch.device("cpu")) / 1000
    assert float(normalized.min()) >= 0.001 - 1e-6
    assert abs(float(normalized.mean()) - 0.6) < 0.02


@pytest.mark.parametrize("options", [
    {"prediction_type": "epsilon"}, {"num_train_timesteps": 0}, {"shift": float("nan")},
    {"beta_alpha": 0}, {"beta_beta": -1}, {"beta_t_min": 1}, {"time_sampler": "unknown"},
])
def test_invalid_process_settings_fail_before_training(options):
    with pytest.raises(ValueError):
        TrajectoryFlowMatching(**options)


@pytest.mark.parametrize("shape", [(4,), (1, 4), (2, 4, 8)])
def test_mismatched_or_retired_clock_shapes_are_rejected(shape):
    fm = TrajectoryFlowMatching()
    sample = torch.randn(2, 4, 8, 3)
    with pytest.raises(ValueError, match="Timesteps must be"):
        fm.add_noise(sample, torch.zeros_like(sample), torch.ones(shape))


@pytest.mark.parametrize("schedule", ["full_sequence", "pyramid"])
@pytest.mark.parametrize("scale", [0, 0.25, 1, 2.3])
def test_schedules_match_frozen_baseline(schedule, scale):
    reference = _reference_module("diffusion/scheduling.py").generate_scheduling_matrix
    expected = reference(7, 5, schedule, scale)
    observed = generate_scheduling_matrix(7, 5, schedule, scale)
    assert (observed == expected).all()


@pytest.mark.parametrize("frames,steps,scale", [(0, 5, 1), (4, 0, 1), (4, 5, -1), (4, 5, float("nan"))])
def test_invalid_schedules_fail_explicitly(frames, steps, scale):
    with pytest.raises(ValueError):
        generate_scheduling_matrix(frames, steps, uncertainty_scale=scale)
