"""
Scheduling matrices for Diffusion Forcing inference.

Controls per-frame noise levels during iterative denoising.
Each schedule defines a matrix K[m, t] where:
  m = denoising step (row), t = frame index (column)
  K[m, t] = noise level for frame t at step m (0 = clean, max = fully noisy)
"""

import math

import numpy as np


def generate_scheduling_matrix(
    num_frames: int,
    num_inference_steps: int,
    schedule: str = "full_sequence",
    uncertainty_scale: float = 1.0,
    keep_clean: int = 0,
) -> np.ndarray:
    """
    Generate denoising schedule K[m, t] for each frame at each step.

    Args:
        num_frames: Number of future frames F
        num_inference_steps: Number of denoising steps S
        schedule: "full_sequence", "pyramid" or "stream"
        uncertainty_scale: Controls pyramid slope (higher = later frames stay noisy longer)
        keep_clean: "stream" only -- the c rows a streaming rollout keeps clean at the window front.
            The remaining m = num_frames - c rows carry the rollout's own ramp, slope 1/m per row,
            so this matrix is one window of that rollout (diffusion/stream_states.window_states
            with g=0, N=n). Fractional levels: the caller rounds to the training clock.

    Returns:
        scheduling_matrix: [M, F] integer noise levels in [0, num_inference_steps]
            M = num_inference_steps + 1 for full_sequence
            M = num_inference_steps + int((num_frames - 1) * uncertainty_scale) + 1 for pyramid
            First row = max noise, last row = all zeros (clean)
    """
    if num_frames < 1 or num_inference_steps < 1:
        raise ValueError("Positive num_frames and num_inference_steps required")
    if not np.isfinite(uncertainty_scale) or uncertainty_scale < 0:
        raise ValueError("uncertainty_scale must be finite and nonnegative")
    S = num_inference_steps

    if schedule == "full_sequence":
        # All frames share the same timestep at each step — standard diffusion
        col = np.arange(S, -1, -1, dtype=np.int64)  # [S, S-1, ..., 0]
        return np.broadcast_to(col[:, None], (S + 1, num_frames)).copy()

    elif schedule == "pyramid":
        # Earlier frames denoised faster, later frames stay noisy longer
        height = S + int((num_frames - 1) * uncertainty_scale) + 1
        matrix = np.zeros((height, num_frames), dtype=np.int64)
        for m in range(height):
            for t in range(num_frames):
                matrix[m, t] = S + int(t * uncertainty_scale) - m
        return np.clip(matrix, 0, S)

    elif schedule == "stream":
        # The streaming rollout's staircase over one window: row t at clip(S + t*S/m - k, 0, S),
        # m = n - keep_clean. Float, because S/m is not an integer; the caller rounds.
        if not 0 <= keep_clean < num_frames:
            raise ValueError(f"stream keep_clean must be in [0, {num_frames}), got {keep_clean}")
        m = num_frames - keep_clean
        rows = np.arange(num_frames, dtype=np.float64)
        height = math.ceil(S + (num_frames - 1) * S / m) + 1
        calls = np.arange(height, dtype=np.float64)[:, None]
        return np.clip(S + rows[None] * S / m - calls, 0.0, S)

    else:
        raise ValueError(f"Unknown schedule: {schedule}")
