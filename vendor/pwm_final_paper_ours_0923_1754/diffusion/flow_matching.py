"""Optimal-transport conditional flow matching for unified trajectories.

Corruption: x_t = (1 - t) x_0 + t noise. Network output is either clean x_0
(`sample`) or velocity noise - x_0 (`velocity`). The production objective owns
loss weighting; inference owns rollout/scheduling. No alternate sampler lives here.
"""
import math

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


class TrajectoryFlowMatching:
    def __init__(self, num_train_timesteps=1000, shift=1.0, prediction_type="velocity",
                 time_sampler="uniform", beta_alpha=1.5, beta_beta=1.0, beta_t_min=0.001):
        if prediction_type not in ("velocity", "sample"):
            raise ValueError(f"prediction_type must be velocity or sample, got {prediction_type!r}")
        if isinstance(num_train_timesteps, bool) or not isinstance(num_train_timesteps, int) \
                or num_train_timesteps < 1:
            raise ValueError("num_train_timesteps must be a positive integer")
        if not math.isfinite(float(shift)) or shift <= 0:
            raise ValueError("shift must be finite and positive")
        if time_sampler not in ("uniform", "beta"):
            raise ValueError(f"unknown time_sampler: {time_sampler}")
        if not all(math.isfinite(float(value)) and value > 0 for value in (beta_alpha, beta_beta)):
            raise ValueError("Beta concentrations must be finite and positive")
        if not math.isfinite(float(beta_t_min)) or not 0 <= beta_t_min < 1:
            raise ValueError("beta_t_min must be in [0, 1)")
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.time_sampler = time_sampler
        self.beta_alpha = float(beta_alpha)
        self.beta_beta = float(beta_beta)
        self.beta_t_min = float(beta_t_min)
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps, shift=shift)

    def _normalized_time(self, timesteps, sample):
        """Scalar, [B] or [B,F] clock, expanded only along trailing channels."""
        if timesteps.ndim > 2 or (timesteps.ndim and tuple(timesteps.shape) !=
                                 tuple(sample.shape[:timesteps.ndim])):
            raise ValueError("Timesteps must be scalar, [B] or [B,F] matching the sample")
        time = timesteps.float() / self.num_train_timesteps
        while time.ndim < sample.ndim:
            time = time.unsqueeze(-1)
        return time

    def add_noise(self, x_0, noise, timesteps):
        """Linear interpolation, with time measured in units of num_train_timesteps."""
        time = self._normalized_time(timesteps, x_0)
        return (1 - time) * x_0 + time * noise

    def get_target(self, x_0, noise, timesteps=None):
        """Always velocity. A clean-target particle loss compares directly to x_0.

        Native pose loss uses this velocity target even for sample-parameterized
        heads; converting a clean-pose error to velocity weights it by 1/t².
        """
        return noise - x_0

    def sample_timesteps(self, batch_size, device, num_frames=None, low=0):
        """Continuous clocks [B] or [B,F], preserving the historical RNG/arithmetic.

        Uniform draws are in [low,T). Beta draws use Beta(a,b)*(1-t_min)+t_min,
        scaled by T and clamped to [low,T]. Future draws use low=1 to exclude
        singular sample-to-velocity conversion at t=0; context draws use low=0.
        """
        if batch_size < 1 or (num_frames is not None and num_frames < 1):
            raise ValueError("Positive batch_size and num_frames required")
        if not math.isfinite(float(low)) or not 0 <= low < self.num_train_timesteps:
            raise ValueError("low must be in [0, num_train_timesteps)")
        shape = (batch_size,) if num_frames is None else (batch_size, num_frames)
        if self.time_sampler == "beta":
            a = torch.full(shape, float(self.beta_alpha), device=device)
            b = torch.full(shape, float(self.beta_beta), device=device)
            time = torch.distributions.Beta(a, b).sample()
            time = time * (1.0 - self.beta_t_min) + self.beta_t_min
            return (time * self.num_train_timesteps).clamp(float(low), float(self.num_train_timesteps))
        return torch.rand(shape, device=device) * (self.num_train_timesteps - low) + low

    def get_x0_est(self, model_output, timestep, sample):
        """Recover clean target: x_0 = x_t - t*v; sample heads already return x_0."""
        if self.prediction_type == "sample":
            return model_output
        return sample - self._normalized_time(timestep, sample) * model_output

    def as_velocity(self, model_output, sample, timestep):
        """Convert network output into the velocity required by Euler integration."""
        if self.prediction_type == "velocity":
            return model_output
        time = self._normalized_time(timestep, sample)
        return (sample - model_output) / time.clamp(min=1e-8)
