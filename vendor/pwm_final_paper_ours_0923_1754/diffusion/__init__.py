"""Unified flow matching and per-frame denoising schedules."""

from diffusion.flow_matching import TrajectoryFlowMatching
from diffusion.scheduling import generate_scheduling_matrix

__all__ = ["TrajectoryFlowMatching", "generate_scheduling_matrix"]
