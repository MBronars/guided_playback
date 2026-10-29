"""The recovery regression itself must reject a deliberately broken estimator."""
import pytest
import torch

from diffusion.flow_matching import TrajectoryFlowMatching
from tests.test_diffusion import test_get_x0_est_recovers_true_x0 as recovery_check


def test_recovery_check_fails_when_estimator_is_broken(monkeypatch):
    monkeypatch.setattr(TrajectoryFlowMatching, "get_x0_est",
                        lambda self, velocity, timestep, sample: torch.zeros_like(sample))
    with pytest.raises(AssertionError):
        recovery_check()


def test_parameterization_roundtrip_rejects_a_broken_converter(monkeypatch):
    from tests.test_diffusion import test_sample_velocity_clean_roundtrip
    monkeypatch.setattr(TrajectoryFlowMatching, "as_velocity",
                        lambda self, output, sample, timestep: torch.zeros_like(sample))
    with pytest.raises(AssertionError):
        test_sample_velocity_clean_roundtrip("sample", (2, 4))
