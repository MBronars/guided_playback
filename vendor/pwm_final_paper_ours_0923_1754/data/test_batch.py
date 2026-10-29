"""
LongTrajectoryBatch: Data structure for long test sequences.

Analogous to TrajectoryBatch but for sequences longer than a single
inference window (e.g., 96 frames vs 25 frames for training).

Used for autoregressive inference where we predict multiple windows.
"""

from dataclasses import dataclass, fields, replace
from typing import Optional, Any

import torch

from data.batch import TrajectoryBatch


@dataclass
class LongTrajectoryBatch:
    """
    Data structure for long particle trajectory sequences.

    This is the test/inference analog of TrajectoryBatch. It stores full
    trajectories that can be split into TrajectoryBatch windows for
    autoregressive rollout.

    Attributes:
        positions: [B, T, N, 3] - Full particle trajectory
        actions: [B, T-1, N, 3] - Actions for each frame transition (t -> t+1)
        action_mask: [B, N] - True for actuated particles (e.g., gripper)
        valid_lengths: [B] - Number of valid (non-padded) particles per sample
        E: [B, N] - Young's modulus
        nu: [B, N] - Poisson's ratio
        rho: [B, N] - Density
        delta_stats: NormalizationStats for per-particle deltas (used for both
                     diffusion target and action conditioning)
        position_stats: NormalizationStats for positions (context normalization)
        prediction_target: "delta" (canonical displacement targets)

    Shapes:
        B = batch size
        T = total frames (e.g., 96)
        N = total particles (may include padding)

    Note:
        Unlike TrajectoryBatch which has actions [B, F, N, 3] for future frames only,
        LongTrajectoryBatch stores actions for ALL transitions [B, T-1, N, 3].
    """
    positions: torch.Tensor        # [B, T, N, 3]
    actions: torch.Tensor          # [B, T-1, N, 3]
    action_mask: torch.Tensor      # [B, N]
    object_ids: torch.Tensor       # [B, N] object type per particle
    valid_lengths: torch.Tensor    # [B]
    E: torch.Tensor                # [B, N]
    nu: torch.Tensor               # [B, N]
    rho: torch.Tensor              # [B, N]
    mat_type: Optional[torch.Tensor] = None  # [B, N] Material type index (long)
    delta_stats: Optional[Any] = None     # NormalizationStats for per-particle deltas (diffusion target + actions)
    position_stats: Optional[Any] = None  # NormalizationStats for positions
    prediction_target: str = "delta"
    action_active: Optional[torch.Tensor] = None  # [B,T-1,N], commanded drag activity

    def __post_init__(self):
        """Validate tensor shapes after initialization."""
        self._validate_shapes()

    def _validate_shapes(self):
        """Ensure all tensors have consistent shapes."""
        if self.prediction_target != "delta":
            raise ValueError("LongTrajectoryBatch supports prediction_target='delta' only")
        B, T, N, _ = self.positions.shape

        assert self.positions.shape == (B, T, N, 3), \
            f"positions shape mismatch: {self.positions.shape} != ({B}, {T}, {N}, 3)"

        assert self.actions.shape == (B, T - 1, N, 3), \
            f"actions shape mismatch: {self.actions.shape} != ({B}, {T-1}, {N}, 3)"

        assert self.action_mask.shape == (B, N), \
            f"action_mask shape mismatch: {self.action_mask.shape} != ({B}, {N})"
        if self.action_active is not None:
            if self.action_active.shape!=(B,T-1,N) or self.action_active.dtype!=torch.bool:
                raise ValueError('Invalid long-sequence action activity')
            if torch.any(self.action_active & ~self.action_mask[:,None,:]) or torch.any(
                    self.actions.masked_select(~self.action_active[...,None])):
                raise ValueError('Commands/activity outside selected active patch')

        assert self.object_ids.shape == (B, N), \
            f"object_ids shape mismatch: {self.object_ids.shape} != ({B}, {N})"

        assert self.valid_lengths.shape == (B,), \
            f"valid_lengths shape mismatch: {self.valid_lengths.shape} != ({B},)"

        assert self.E.shape == (B, N), f"E shape mismatch: {self.E.shape} != ({B}, {N})"
        assert self.nu.shape == (B, N), f"nu shape mismatch: {self.nu.shape} != ({B}, {N})"
        assert self.rho.shape == (B, N), f"rho shape mismatch: {self.rho.shape} != ({B}, {N})"
        if self.mat_type is not None:
            assert self.mat_type.shape == (B, N), f"mat_type shape mismatch: {self.mat_type.shape} != ({B}, {N})"

    # ==================== Properties ====================

    @property
    def batch_size(self) -> int:
        """Number of samples in batch."""
        return self.positions.shape[0]

    @property
    def num_frames(self) -> int:
        """Total number of frames."""
        return self.positions.shape[1]

    @property
    def num_particles(self) -> int:
        """Number of particles per sample (fixed, may include padding)."""
        return self.positions.shape[2]

    @property
    def device(self) -> torch.device:
        """Device where tensors are stored."""
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Data type of position tensors."""
        return self.positions.dtype

    # ==================== Utility Methods ====================

    def to(self, device: torch.device) -> 'LongTrajectoryBatch':
        """Move all tensors to specified device."""
        return self._map_tensors(lambda value: value.to(device))

    def _map_tensors(self, transform):
        """One metadata-preserving path for every declared tensor field."""
        return replace(self, **{field.name: transform(value) for field in fields(self)
                                if torch.is_tensor(value := getattr(self, field.name))})

    def slice(self, max_batch_size: int) -> 'LongTrajectoryBatch':
        """Slice batch to a smaller size."""
        if isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1:
            raise ValueError("max_batch_size must be a positive integer")
        if self.batch_size <= max_batch_size:
            return self
        return self._map_tensors(lambda value: value[:max_batch_size])

    def get_valid_mask(self) -> torch.Tensor:
        """
        Create boolean mask for valid (non-padded) particles.

        Returns:
            [B, N] boolean tensor where True = valid particle
        """
        N = self.num_particles
        indices = torch.arange(N, device=self.device)
        return indices.unsqueeze(0) < self.valid_lengths.unsqueeze(1)

    # ==================== Window Extraction ====================

    def get_window(
        self,
        start_frame: int,
        context_len: int,
        future_len: int,
        predicted_positions: Optional[torch.Tensor] = None,
        pad_to_length: bool = False,
    ) -> TrajectoryBatch:
        """
        Extract a TrajectoryBatch window for inference.

        Args:
            start_frame: Starting frame index for context
            context_len: Number of context frames (C)
            future_len: Number of future frames to predict (F)
            predicted_positions: [B, T_pred, N, 3] If provided, use these for
                                context instead of ground truth (autoregressive mode)
            pad_to_length: If True, pad with the final frame when the window
                        extends beyond available frames. Useful for guided
                        rollouts that extend past the GT horizon.

        Returns:
            TrajectoryBatch with positions [B, C+F, N, 3]
        """
        T = self.num_frames
        for name, value, minimum in (("start_frame", start_frame, 0), ("context_len", context_len, 1),
                                      ("future_len", future_len, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

        # Get context positions
        if predicted_positions is not None and start_frame > 0:
            context_pos = predicted_positions[:, start_frame:start_frame + context_len]
        else:
            context_pos = self.positions[:, start_frame:start_frame + context_len]
        if context_pos.shape != (self.batch_size, context_len, self.num_particles, 3):
            raise ValueError("Requested context is not fully available in the selected trajectory")

        # Get future positions, padding if requested and needed
        future_start = start_frame + context_len
        future_end = min(future_start + future_len, T)
        available = max(0, future_end - future_start)  # could be 0 if start_frame >= T
        future_pos = self.positions[:, future_start:future_end]

        if pad_to_length and available < future_len:
            pad_len = future_len - available
            last_frame = self.positions[:, -1:].expand(-1, pad_len, -1, -1)
            future_pos = torch.cat([future_pos, last_frame], dim=1)

        actual_future_len = future_pos.shape[1]
        if actual_future_len < 1:
            raise ValueError("Requested window has no future frames; enable pad_to_length to extend it")

        window_positions = torch.cat([context_pos, future_pos], dim=1)

        # Actions: zeros for the full window, fill in what's available
        window_actions = torch.zeros(
            self.batch_size, actual_future_len, self.num_particles, 3,
            device=self.device, dtype=self.actions.dtype
        )
        action_start = start_frame + context_len - 1
        action_end = min(action_start + actual_future_len, self.actions.shape[1])
        available = action_end - action_start
        if available > 0:
            window_actions[:, :available] = self.actions[:, action_start:action_end]
        activity={}
        if self.action_active is not None:
            for key,start,length in (('action_active',action_start,actual_future_len),
                                     ('context_action_active',start_frame,context_len-1)):
                gate=torch.zeros(self.batch_size,length,self.num_particles,dtype=torch.bool,device=self.device)
                count=max(0,min(start+length,self.action_active.shape[1])-start)
                if count:gate[:,:count]=self.action_active[:,start:start+count]
                activity[key]=gate
        return TrajectoryBatch(
            positions=window_positions,
            actions=window_actions,
            action_mask=self.action_mask,
            object_ids=self.object_ids,
            valid_lengths=self.valid_lengths,
            E=self.E,
            nu=self.nu,
            rho=self.rho,
            mat_type=self.mat_type,
            context_len=context_len,
            delta_stats=self.delta_stats,
            position_stats=self.position_stats,
            prediction_target=self.prediction_target,
            **activity,
        )

    def __repr__(self) -> str:
        return (
            f"LongTrajectoryBatch(\n"
            f"  batch_size={self.batch_size},\n"
            f"  num_frames={self.num_frames},\n"
            f"  num_particles={self.num_particles},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype}\n"
            f")"
        )
