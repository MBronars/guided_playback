"""
TrajectoryBatch: Clean, typed data structure for particle trajectories.

This is the single source of truth for batch structure throughout the codebase.
All data flows through this contract.
"""

from dataclasses import dataclass, fields, replace
from typing import Any, List, Optional
import torch


@dataclass
class TrajectoryBatch:
    """
    Clean, typed batch structure for particle trajectory prediction.

    All tensors are on the same device with consistent shapes.

    Attributes:
        positions: [B, T, N, 3] - Full particle trajectory (context + future)
        actions: [B, F, N, 3] - Rigid actuator displacements, or explicit MPM
                               commanded velocity times transition duration.
                               MPM commands are NOT achieved future motion.
        action_mask: [B, N] - True = particle has actuator-motion conditioning (gripper)
                              False = particle has no action (object)
        object_ids: [B, N] - Object type ID per particle for stratified encoding
                             e.g., 0=finger1, 1=finger2, 2=deformable_object
        valid_lengths: [B] - Number of valid (non-padded) particles per sample
        context_len: int - Number of context frames (default 2)

        Physics parameters (per-particle, optional):
        E: [B, N] - Young's modulus
        nu: [B, N] - Poisson's ratio
        rho: [B, N] - Density
        mat_type: [B, N] - Material type index (0=elastic, 1=plasticine, 2=sand)

        Normalization:
        delta_stats: NormalizationStats for per-particle deltas. Used for both
                     the diffusion target (realized position deltas) and action
                     conditioning (realized actuator deltas, masked to actuated
                     particles). Drag uses commanded displacements in metres;
                     these share units but are not the observed target values.
        position_stats: NormalizationStats for positions (for context normalization)
        prediction_target: "delta" (the only supported particle target)

    Shapes:
        B = batch size
        T = total frames = context_len + F
        F = future frames to predict
        N = number of particles (fixed, padded if necessary)
    """
    positions: torch.Tensor           # [B, T, N, 3]
    actions: torch.Tensor             # [B, F, N, 3]
    action_mask: torch.Tensor         # [B, N]
    object_ids: torch.Tensor          # [B, N] object type per particle
    valid_lengths: torch.Tensor       # [B]
    E: torch.Tensor                   # [B, N] Young's modulus
    nu: torch.Tensor                  # [B, N] Poisson's ratio
    rho: torch.Tensor                 # [B, N] Density
    mat_type: Optional[torch.Tensor] = None  # [B, N] Material type index (long), None if not available
    context_len: int = 2
    delta_stats: Optional[Any] = None     # NormalizationStats for per-particle deltas (diffusion target + actions)
    position_stats: Optional[Any] = None  # NormalizationStats for positions (context normalization)
    prediction_target: str = "delta"
    pose_delta_stats: Optional[Any] = None  # Six world-frame [translation, rotation-vector] channels
    # Per-body rigid sidecar (unified Tier-2 routing head; None everywhere else).
    # Bodies are at LINK granularity: a gripper is one rigid body per link. Without the
    # articulation fields below every link is a free body (no FK); with them a child link
    # follows its parent through one joint.
    body_poses: Optional[torch.Tensor] = None          # [B, T, K, 4, 4] body->world
    body_templates: Optional[torch.Tensor] = None      # [B, K, P, 3] body-frame points
    body_template_mask: Optional[torch.Tensor] = None  # [B, K, P] valid template points
    body_type: Optional[torch.Tensor] = None           # [B, K] 0=deformable 1=rigid
    particle_body: Optional[torch.Tensor] = None       # [B, N] body index per particle
    # Tier-2 v2 pose-native channels (docs/tier2_v2_design.md): first-class pose
    # targets over the future window + pose-space actions. Raw units (m / rad).
    # Particle and rigid pose channels use the same resolved normalization profile.
    pose_deltas_gt: Optional[torch.Tensor] = None      # [B, F, K, 6] [dt, daa] world deltas
    pose_actions: Optional[torch.Tensor] = None        # [B, F, K, 6] deltas of ACTUATED bodies, zeros else
    pose_action_mask: Optional[torch.Tensor] = None    # [B, K] True = actuated rigid body
    # Articulation (optional; None = every rigid body is a free six-channel body). A child body
    # moves with its parent through one 1-DOF joint and its pose channels carry only the joint
    # coordinate change: prismatic in channel 0 (m), revolute in channel 3 (rad), the other five
    # zero (utils/pose_native.py JOINT_*). Axis and pivot are in the PARENT's body frame.
    body_parent: Optional[torch.Tensor] = None     # [B, K] long, -1 = no parent
    joint_type: Optional[torch.Tensor] = None      # [B, K] long, JOINT_FREE / FIXED / PRISMATIC / REVOLUTE
    joint_axis: Optional[torch.Tensor] = None      # [B, K, 3] unit axis (prismatic, revolute), zeros otherwise
    joint_pivot: Optional[torch.Tensor] = None     # [B, K, 3] revolute pivot, zeros otherwise
    # Source-corpus tag set by the unified loaders. A REAL field (not a plain
    # attribute) so .to(device)/.collate() carry it — validate()'s per-domain
    # accumulation reads it after the device transfer.
    domain: Optional[str] = None
    # Optional commanded-velocity actuation (MPM drag). Static action_mask is
    # patch membership; these gates distinguish zero-velocity HOLD from RELEASE.
    # None preserves the historical rigid-actuator/passive contract exactly.
    action_active: Optional[torch.Tensor] = None          # [B,F,N] bool
    context_action_active: Optional[torch.Tensor] = None  # [B,C-1,N] bool

    def __post_init__(self):
        """Validate tensor shapes after initialization."""
        self._validate_shapes()

    def _validate_shapes(self):
        """Ensure all tensors have consistent shapes."""
        if self.prediction_target != "delta":
            raise ValueError(f"Unsupported prediction_target {self.prediction_target!r}; unified batches require delta")
        B, T, N, _ = self.positions.shape
        F = T - self.context_len

        assert self.positions.shape == (B, T, N, 3), \
            f"positions shape mismatch: {self.positions.shape} != ({B}, {T}, {N}, 3)"

        assert self.actions.shape == (B, F, N, 3), \
            f"actions shape mismatch: {self.actions.shape} != ({B}, {F}, {N}, 3)"

        assert self.action_mask.shape == (B, N), \
            f"action_mask shape mismatch: {self.action_mask.shape} != ({B}, {N})"
        gates=(self.action_active,self.context_action_active)
        if any(v is not None for v in gates):
            if any(v is None for v in gates):raise ValueError('Both temporal action gates required')
            for value,length in zip(gates,(F,self.context_len-1)):
                if value.shape!=(B,length,N) or value.dtype!=torch.bool:
                    raise ValueError('Invalid temporal action gate shape/dtype')
                if torch.any(value & ~self.action_mask[:,None,:]):
                    raise ValueError('Temporal actuation outside selected patch')
            if torch.any(self.actions.masked_select(~self.action_active[...,None])):
                raise ValueError('Command outside active transitions')

        assert self.object_ids.shape == (B, N), \
            f"object_ids shape mismatch: {self.object_ids.shape} != ({B}, {N})"

        assert self.valid_lengths.shape == (B,), \
            f"valid_lengths shape mismatch: {self.valid_lengths.shape} != ({B},)"

        assert self.context_len > 0, "context_len must be positive"
        assert self.context_len < T, "context_len must be less than total frames"

        # Validate physics parameters
        assert self.E.shape == (B, N), f"E shape mismatch: {self.E.shape} != ({B}, {N})"
        assert self.nu.shape == (B, N), f"nu shape mismatch: {self.nu.shape} != ({B}, {N})"
        assert self.rho.shape == (B, N), f"rho shape mismatch: {self.rho.shape} != ({B}, {N})"
        if self.mat_type is not None:
            assert self.mat_type.shape == (B, N), f"mat_type shape mismatch: {self.mat_type.shape} != ({B}, {N})"

        joints = (self.body_parent, self.joint_type, self.joint_axis, self.joint_pivot)
        if any(v is not None for v in joints):
            assert all(v is not None for v in joints) and self.body_type is not None, \
                "articulation needs body_parent, joint_type, joint_axis, joint_pivot and the rigid sidecar"
            K = self.body_type.shape[1]
            assert self.body_parent.shape == (B, K) and self.joint_type.shape == (B, K), \
                f"body_parent / joint_type shape mismatch: {self.body_parent.shape}, {self.joint_type.shape} != ({B}, {K})"
            assert self.joint_axis.shape == (B, K, 3) and self.joint_pivot.shape == (B, K, 3), \
                f"joint_axis / joint_pivot shape mismatch: {self.joint_axis.shape}, {self.joint_pivot.shape} != ({B}, {K}, 3)"

    # ==================== Properties ====================

    @property
    def batch_size(self) -> int:
        """Number of samples in batch."""
        return self.positions.shape[0]

    @property
    def num_frames(self) -> int:
        """Total number of frames (context + future)."""
        return self.positions.shape[1]

    @property
    def num_future_frames(self) -> int:
        """Number of future frames to predict."""
        return self.num_frames - self.context_len

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

    # ==================== Slicing Properties ====================

    @property
    def context(self) -> torch.Tensor:
        """
        Context frames for conditioning.

        Returns:
            [B, C, N, 3] where C = context_len
        """
        return self.positions[:, :self.context_len]

    @property
    def future(self) -> torch.Tensor:
        """
        Future frames to predict.

        Returns:
            [B, F, N, 3] where F = num_frames - context_len
        """
        return self.positions[:, self.context_len:]

    @property
    def anchor(self) -> torch.Tensor:
        """
        Last context frame - used as reference/anchor point.

        Returns:
            [B, N, 3]
        """
        return self.positions[:, self.context_len - 1]

    @property
    def future_deltas(self) -> torch.Tensor:
        """
        Per-frame displacements for future frames (what diffusion predicts).

        future_deltas[t] = positions[context_len + t] - positions[context_len + t - 1]

        This is what the diffusion model should predict:
        - Small values centered around 0 (good for diffusion)
        - Scale-invariant (doesn't depend on absolute position)

        Returns:
            [B, F, N, 3] where F = num_future_frames
        """
        # positions[:, context_len-1:] gives us [anchor, future_0, future_1, ...]
        # diff along time axis gives [future_0 - anchor, future_1 - future_0, ...]
        trajectory_from_anchor = self.positions[:, self.context_len - 1:]  # [B, F+1, N, 3]
        return trajectory_from_anchor[:, 1:] - trajectory_from_anchor[:, :-1]  # [B, F, N, 3]

    @property
    def prediction_stats(self):
        """Normalizer for the particle denoising target."""
        return self.delta_stats

    @property
    def prediction_x0(self) -> torch.Tensor:
        """Clean per-frame physical displacement target."""
        return self.future_deltas

    def predictions_to_positions(self, preds: torch.Tensor) -> torch.Tensor:
        """Integrate denormalized displacement predictions from the context anchor."""
        return self.deltas_to_positions(preds)

    @property
    def context_deltas(self) -> torch.Tensor:
        """
        Per-frame displacements for context frames.

        Useful for conditioning - shows recent motion history.

        Returns:
            [B, C-1, N, 3] where C = context_len
        """
        return self.context[:, 1:] - self.context[:, :-1]

    # ==================== Utility Methods ====================

    def _map_tensors(self, transform):
        """Apply to tensor fields only; preserve stats and metadata by identity."""
        return replace(self, **{
            field.name: transform(getattr(self, field.name))
            for field in fields(self) if torch.is_tensor(getattr(self, field.name))
        })

    def to(self, device: torch.device) -> 'TrajectoryBatch':
        """Move all tensor fields; normalizers move their bounds when consumed."""
        return self._map_tensors(lambda value: value.to(device))

    def slice(self, max_batch_size: int) -> 'TrajectoryBatch':
        """Keep at most max_batch_size elements, including all sidecar metadata."""
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if self.batch_size <= max_batch_size:
            return self
        return self._map_tensors(lambda value: value[:max_batch_size])

    def float(self) -> 'TrajectoryBatch':
        """Cast floating tensors; preserve integer IDs and boolean masks."""
        return self._map_tensors(lambda value: value.float() if value.is_floating_point() else value)

    def half(self) -> 'TrajectoryBatch':
        """Cast floating tensors; preserve integer IDs and boolean masks."""
        return self._map_tensors(lambda value: value.half() if value.is_floating_point() else value)

    def get_valid_mask(self) -> torch.Tensor:
        """
        Create boolean mask for valid (non-padded) particles.

        Returns:
            [B, N] boolean tensor where True = valid particle
        """
        N = self.num_particles
        device = self.device

        # Create index tensor [N]
        indices = torch.arange(N, device=device)

        # Broadcast comparison: indices < valid_lengths
        # [N] < [B, 1] -> [B, N]
        return indices.unsqueeze(0) < self.valid_lengths.unsqueeze(1)

    def deltas_to_positions(self, deltas: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct future positions from predicted deltas.

        Used at inference time: model predicts deltas, we accumulate from anchor.

        Args:
            deltas: [B, F, N, 3] predicted per-frame displacements

        Returns:
            [B, F, N, 3] reconstructed future positions
        """
        # Start from anchor and accumulate deltas
        # cumsum gives [delta_0, delta_0+delta_1, delta_0+delta_1+delta_2, ...]
        cumulative = torch.cumsum(deltas, dim=1)  # [B, F, N, 3]
        # Add anchor position
        return self.anchor.unsqueeze(1) + cumulative  # [B, F, N, 3]

    @property
    def unique_object_ids(self) -> torch.Tensor:
        """Get unique object IDs in this batch."""
        return self.object_ids[0].unique()

    @property
    def num_objects(self) -> int:
        """Number of distinct object types."""
        return len(self.unique_object_ids)

    def __repr__(self) -> str:
        return (
            f"TrajectoryBatch(\n"
            f"  batch_size={self.batch_size},\n"
            f"  num_frames={self.num_frames} (context={self.context_len}, future={self.num_future_frames}),\n"
            f"  num_particles={self.num_particles},\n"
            f"  num_objects={self.num_objects},\n"
            f"  device={self.device},\n"
            f"  dtype={self.dtype}\n"
            f")"
        )

    @classmethod
    def collate(cls, batches: List['TrajectoryBatch']) -> 'TrajectoryBatch':
        """Concatenate tensor fields only when the complete batch contract agrees."""
        if not batches:
            raise ValueError("Cannot collate an empty batch list")
        if len(batches) == 1:
            return batches[0]
        values = {}
        for field in fields(cls):
            items = [getattr(batch, field.name) for batch in batches]
            first = items[0]
            if torch.is_tensor(first):
                if any(not torch.is_tensor(v) or v.shape[1:] != first.shape[1:]
                       or v.dtype != first.dtype or v.device != first.device for v in items):
                    raise ValueError(f"Cannot collate incompatible tensor field {field.name}")
                values[field.name] = torch.cat(items, dim=0)
            else:
                for value in items[1:]:
                    if value is first:
                        continue
                    if field.name.endswith("_stats") and first is not None and value is not None:
                        equal = (type(value) is type(first)
                                 and torch.equal(value.min_val, first.min_val)
                                 and torch.equal(value.max_val, first.max_val))
                    else:
                        equal = not torch.is_tensor(value) and value == first
                    if not equal:
                        raise ValueError(f"Cannot collate differing metadata field {field.name}")
                values[field.name] = first
        return cls(**values)
