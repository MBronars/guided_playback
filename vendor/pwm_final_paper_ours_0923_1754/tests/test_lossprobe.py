"""The loss-visualization rules, as tests (docs/loss_viz_rules.md).

These encode the three mistakes that made the round-2 overfit charts unreadable:
a single-sample statistic used as a measurement, a trend read off noise, and a printf
format that became the data path.
"""
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from diffusion import TrajectoryFlowMatching

from scripts.unified.lossprobe import T_LARGE, fixed_t_loss, guard, large_t_loss, tail


def _series(start, decades_per_1k, n, jitter, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    return list(10 ** (np.log10(start) + decades_per_1k * x / 1000 + rng.normal(0, jitter, n)))


def test_tail_reports_the_noise_floor_beside_the_effect():
    # scatter has to be visible, because an effect smaller than it is not a result
    t = tail(_series(1e-2, 0.0, 900, jitter=0.5))
    assert t["scatter"] > 5, "a series that jitters half a decade must report large scatter"
    assert t["min"] < t["median"] < t["max"]


def test_tail_calls_a_descending_series_descending():
    assert tail(_series(1e-1, -0.5, 900, jitter=0.1))["still_descending"]


def test_tail_does_not_call_a_converged_series_descending():
    assert not tail(_series(1e-3, 0.0, 900, jitter=0.1))["still_descending"]


def test_the_last_point_is_not_a_measurement():
    """The old gate. A flat series with realistic jitter passes or fails on the draw."""
    h = _series(1e-2, 0.0, 900, jitter=0.5)
    draws = [h[i] for i in range(600, 900, 7)]
    assert max(draws) / min(draws) > 3, "consecutive draws scatter more than 3x"
    assert min(draws) < 0.05 < max(draws), "a 0.05 threshold on one draw is a coin flip"


def test_guard_catches_a_printf_flattened_zero():
    h = _series(1e-3, 0.0, 400, jitter=0.2)
    h[123] = 0.0                      # what f"{5e-6:.4f}" leaves behind
    with pytest.raises(ValueError, match="non-positive"):
        guard("t", h)


def test_guard_catches_a_series_too_short_to_estimate():
    with pytest.raises(ValueError, match="need >="):
        guard("t", _series(1e-3, 0.0, 50, jitter=0.1))


def test_guard_catches_nan():
    h = _series(1e-3, 0.0, 400, jitter=0.2)
    h[7] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        guard("t", h)


def test_large_t_loss_averages_only_the_region_free_of_the_1_over_t_tail():
    sweep = {0.02: (0.27, 0.0), 0.05: (0.04, 0.0), 0.5: (0.001, 0.0), 0.95: (0.003, 0.0)}
    assert large_t_loss(sweep) == pytest.approx(0.002)
    assert all(t >= T_LARGE for t in (0.5, 0.95))


def _Batch():
    from tests.test_train import create_dummy_batch
    return create_dummy_batch(B=2, T=5, N=5, context_len=2)


class _Model(torch.nn.Module):
    """Deterministic in its inputs, but with a t-dependent error like the real one."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))
        self.cfg = SimpleNamespace(pose_native=False)
    def forward(self, batch, x_t, t, noisy_context=None, context_timesteps=None):
        scale = 1.0 / (t.float().mean() / 1000.0 + 1e-3)
        return x_t * self.w * 0.01 * scale


_Diffusion = TrajectoryFlowMatching


def test_fixed_t_loss_is_repeatable():
    m, b, d = _Model(), _Batch(), _Diffusion()
    torch.manual_seed(7)
    a = fixed_t_loss(m, b, d, grid=(0.2, 0.5, 0.95), draws=3)
    torch.manual_seed(999)                 # a different ambient seed must not matter
    c = fixed_t_loss(m, b, d, grid=(0.2, 0.5, 0.95), draws=3)
    assert a == c, "the gate must reproduce exactly, or arm-vs-arm differences are noise"


def test_fixed_t_loss_restores_training_mode():
    m = _Model(); m.train()
    fixed_t_loss(m, _Batch(), _Diffusion(), grid=(0.5,), draws=1)
    assert m.training


def test_fixed_probe_restores_the_ambient_rng():
    m, b, d = _Model(), _Batch(), _Diffusion()
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    fixed_t_loss(m, b, d, grid=(0.5,), draws=1)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_fixed_probe_supports_a_pure_rigid_production_head():
    from diffusion import TrajectoryFlowMatching
    from tests.test_training_safety import ConstantModel
    from tests.test_unet_std_pose_native import make_batch
    batch = make_batch(deform_n=0, rigid=((1, 24), (2, 16)))
    values = fixed_t_loss(ConstantModel(pose=True), batch, TrajectoryFlowMatching(), grid=(0.5,), draws=1)
    assert np.isfinite(values[0.5][0]) and values[0.5][0] > 0


def test_fixed_probe_ignores_padded_particles():
    from diffusion import TrajectoryFlowMatching
    from tests.test_train import create_dummy_batch
    from tests.test_training_safety import ConstantModel
    batch = create_dummy_batch(B=1, N=3, T=4, context_len=2)
    batch.valid_lengths.fill_(1)
    model, diffusion = ConstantModel(), TrajectoryFlowMatching()
    before = fixed_t_loss(model, batch, diffusion, grid=(0.5,), draws=1)
    batch.positions[:, -1, 1:].fill_(999)
    after = fixed_t_loss(model, batch, diffusion, grid=(0.5,), draws=1)
    assert before == after


def test_sample_head_probe_uses_the_configured_x0_objective():
    from tests.test_training_safety import ConstantModel, scalar_batch
    model = ConstantModel()
    values = fixed_t_loss(model, scalar_batch(0.5), TrajectoryFlowMatching(prediction_type="sample"),
                          grid=(0.2, 0.5, 0.9), draws=2)
    for mean, std in values.values():
        assert mean == pytest.approx((0.5 - 0.25) ** 2)
        assert std == 0.0
