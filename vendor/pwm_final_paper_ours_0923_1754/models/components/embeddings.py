"""
Embedding modules for the trajectory transformer.

Components:
- SinusoidalTimeEmbed: Diffusion timestep embedding
- PointEmbed: 3D coordinate embedding
- RotaryPositionEncoding: 1D rotary embeddings (temporal)
- RotaryPositionEncoding3D: 3D rotary embeddings (spatial)
"""

import math
import torch
import torch.nn as nn
from typing import Tuple


class SinusoidalTimeEmbed(nn.Module):
    """
    Sinusoidal timestep embedding for diffusion models.

    Follows OpenAI/DiT style: sinusoidal encoding + MLP projection.

    Args:
        dim: Output embedding dimension
        max_period: Maximum period for sinusoidal encoding (default 10000)
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # MLP to project sinusoidal embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] integer or float timesteps

        Returns:
            [B, dim] timestep embeddings
        """
        # Do NOT cast to long here: this module computes sinusoids on the raw
        # value (it indexes nothing — the "indexing" comment was copied from
        # TemporalEmbed). Flooring to int collapses continuous flow-matching
        # timesteps to integers, destroying continuous noise-level conditioning.
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=timesteps.device)
            / half
        )

        # [B, half]
        args = timesteps[:, None].float() * freqs[None]

        # [B, dim]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))

        return self.mlp(emb)


class PointEmbed(nn.Module):
    """
    Simple MLP embedding for 3D point coordinates.

    Args:
        out_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension (default: same as out_dim)
    """

    def __init__(self, out_dim: int = 256, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or out_dim

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., 3] point coordinates

        Returns:
            [..., out_dim] point embeddings
        """
        return self.mlp(x)


class SimplePointEmbed(nn.Module):
    """
    Minimal point embedding: single linear layer.

    Args:
        out_dim: Output embedding dimension
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(3, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., 3] point coordinates

        Returns:
            [..., out_dim] point embeddings
        """
        return self.linear(x)


class RotaryPositionEncoding(nn.Module):
    """
    1D Rotary Position Encoding for temporal sequences.

    Rotary embeddings encode position information multiplicatively rather than
    additively, which provides better extrapolation and relative position awareness.

    Args:
        dim: Embedding dimension (should match attention head dimension)
        max_period: Maximum period for frequency computation (default 10000)
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor.

        Rotates adjacent pairs of dimensions:
        [x0, x1, x2, x3, ...] -> [x0*cos - x1*sin, x0*sin + x1*cos, ...]

        Args:
            x: [..., dim] input tensor
            cos: [..., dim] cosine values (interleaved)
            sin: [..., dim] sine values (interleaved)

        Returns:
            [..., dim] rotated tensor
        """
        # Create rotated version: [-x1, x0, -x3, x2, ...]
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x_rotated = x_rotated.reshape_as(x)

        return x * cos + x_rotated * sin

    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for given positions.

        Args:
            positions: [B, N] or [B, N, 1] position indices (can be float for interpolation)

        Returns:
            (cos, sin): Each [B, N, dim] for rotary application
        """
        # Handle shape
        if positions.dim() == 3:
            positions = positions.squeeze(-1)  # [B, N, 1] -> [B, N]

        device = positions.device

        # Compute frequencies: [dim/2]
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, self.dim, 2, dtype=torch.float32, device=device)
            / self.dim
        )

        # Compute angles: [B, N, dim/2]
        angles = positions.unsqueeze(-1).float() * freqs

        # Compute sin/cos and interleave: [B, N, dim]
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        # Interleave: [s0, s0, s1, s1, ...] for proper pairing
        sin = torch.stack([sin, sin], dim=-1).reshape(*sin.shape[:-1], -1)
        cos = torch.stack([cos, cos], dim=-1).reshape(*cos.shape[:-1], -1)

        return cos, sin


class RotaryPositionEncoding3D(nn.Module):
    """
    3D Rotary Position Encoding for spatial point clouds.

    Encodes XYZ coordinates separately and concatenates. Each axis gets
    1/3 of the embedding dimension.

    Args:
        dim: Total embedding dimension (should be divisible by 6 for clean split)
        max_period: Maximum period for frequency computation
    """

    def __init__(
        self,
        dim: int,
        max_period: float = 10000.0,
        rotary_dim: int | None = None,
        f_min: float | None = None,
        f_max: float | None = None,
    ):
        super().__init__()
        # `rotary_dim` (partial-rotary pass-through): number of channels to actually
        # rotate; the remaining (dim - rotary_dim) channels are identity-padded — a
        # true non-positional pass-through block for content/identity. None => legacy
        # (rotate as much of `dim` as divides cleanly into 6).
        rot = dim if rotary_dim is None else rotary_dim
        # Round down to multiple of 6 for clean 3-axis split
        self.feature_dim = (rot // 6) * 6
        self.pad_length = dim - self.feature_dim
        self.dim_per_axis = self.feature_dim // 3
        self.max_period = max_period
        # Explicit log-spaced frequency band (banded RoPE). When both are set they
        # OVERRIDE the max_period inverse-base formula. None => legacy max_period path.
        self.f_min = f_min
        self.f_max = f_max

    @staticmethod
    def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotary embedding (same as 1D version)."""
        x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x_rotated = x_rotated.reshape_as(x)
        return x * cos + x_rotated * sin

    @torch.no_grad()
    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for 3D coordinates.

        Args:
            xyz: [B, N, 3] point coordinates

        Returns:
            (cos, sin): Each [B, N, dim] for rotary application
        """
        B, N, _ = xyz.shape
        device = xyz.device

        # Frequencies for each axis: [dim_per_axis/2]
        if self.f_min is not None and self.f_max is not None:
            # Explicit log-spaced band (banded RoPE): dim_per_axis//2 frequencies
            # log-spaced from f_min to f_max (radians per normalized position unit).
            n_freq = self.dim_per_axis // 2
            freqs = torch.exp(
                torch.linspace(
                    math.log(self.f_min), math.log(self.f_max), n_freq,
                    dtype=torch.float32, device=device,
                )
            )
        else:
            # Legacy inverse-base (max_period) formula.
            freqs = torch.exp(
                -math.log(self.max_period)
                * torch.arange(0, self.dim_per_axis, 2, dtype=torch.float32, device=device)
                / self.dim_per_axis
            )

        # Process each axis
        cos_parts = []
        sin_parts = []

        for axis in range(3):
            coord = xyz[..., axis:axis+1]  # [B, N, 1]
            angles = coord * freqs  # [B, N, dim_per_axis/2]

            cos_axis = torch.cos(angles)
            sin_axis = torch.sin(angles)

            # Interleave for rotary application
            cos_axis = torch.stack([cos_axis, cos_axis], dim=-1).reshape(B, N, -1)
            sin_axis = torch.stack([sin_axis, sin_axis], dim=-1).reshape(B, N, -1)

            cos_parts.append(cos_axis)
            sin_parts.append(sin_axis)

        # Concatenate all axes
        cos = torch.cat(cos_parts, dim=-1)  # [B, N, feature_dim]
        sin = torch.cat(sin_parts, dim=-1)  # [B, N, feature_dim]

        # Pad if needed (pad with 1 for cos, 0 for sin to be identity)
        if self.pad_length > 0:
            cos_pad = torch.ones(B, N, self.pad_length, device=device)
            sin_pad = torch.zeros(B, N, self.pad_length, device=device)
            cos = torch.cat([cos, cos_pad], dim=-1)
            sin = torch.cat([sin, sin_pad], dim=-1)

        return cos, sin

    @torch.no_grad()
    def forward_pooled(self, xyz_set: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Patch-pooled (set-aggregated) rotary descriptor.

        Mean-pools cos and sin SEPARATELY over a set of points (e.g. a latent's KNN
        patch) rather than rotating by a single point. Pooling cos/sin (not the raw
        angles) gives a vector with magnitude <= 1: the phase encodes the patch
        centroid and the magnitude encodes its spread. Permutation-invariant over the
        set; the identity pass-through tail pools (1, 0) -> (1, 0).

        Args:
            xyz_set: [..., k, 3] coordinates of each set member.
        Returns:
            (cos, sin): each [..., dim], the pooled descriptor.
        """
        lead = xyz_set.shape[:-2]
        k = xyz_set.shape[-2]
        flat = xyz_set.reshape(-1, k, 3)          # [M, k, 3]
        cos, sin = self.forward(flat)             # [M, k, dim]
        cos = cos.mean(dim=1).reshape(*lead, -1)  # [..., dim]
        sin = sin.mean(dim=1).reshape(*lead, -1)
        return cos, sin


def get_sincos_pos_embed_1d(num_positions: int, dim: int, device=None) -> torch.Tensor:
    """
    Generate 1D sinusoidal positional embeddings.

    Args:
        num_positions: Number of positions
        dim: Embedding dimension
        device: Target device

    Returns:
        [num_positions, dim] positional embeddings
    """
    position = torch.arange(num_positions, dtype=torch.float32, device=device)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / dim)
    )

    pe = torch.zeros(num_positions, dim, device=device)
    pe[:, 0::2] = torch.sin(position[:, None] * div_term)
    pe[:, 1::2] = torch.cos(position[:, None] * div_term)

    return pe


class FourierSpatialEmbed(nn.Module):
    """
    Fourier feature embedding for 3D spatial coordinates.

    Encodes xyz coordinates using sinusoidal functions at multiple frequencies,
    then projects to output dimension. This is the standard additive positional
    encoding for 3D data (similar to NeRF positional encoding).

    Args:
        out_dim: Output embedding dimension
        num_bands: Number of frequency bands per axis (total features = 3 * 2 * num_bands)
        max_freq_log2: Log2 of maximum frequency (frequencies are 2^0 to 2^max_freq_log2)
    """

    def __init__(self, out_dim: int = 256, num_bands: int = 6, max_freq_log2: int = 5):
        super().__init__()
        self.out_dim = out_dim
        self.num_bands = num_bands

        # Frequencies: 2^0, 2^1, ..., 2^max_freq_log2
        # Linearly spaced in log space
        freqs = 2.0 ** torch.linspace(0, max_freq_log2, num_bands)
        self.register_buffer('freqs', freqs)  # [num_bands]

        # Input dimension: 3 (xyz) + 3 * 2 * num_bands (sin/cos for each axis)
        fourier_dim = 3 + 3 * 2 * num_bands

        # Project to output dimension
        self.proj = nn.Sequential(
            nn.Linear(fourier_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: [..., 3] point coordinates

        Returns:
            [..., out_dim] spatial embeddings
        """
        # Store original shape for reshaping later
        orig_shape = xyz.shape[:-1]
        xyz_flat = xyz.reshape(-1, 3)  # [N, 3]

        # Compute Fourier features for each axis
        # xyz_flat: [N, 3], freqs: [num_bands]
        # angles: [N, 3, num_bands]
        angles = xyz_flat.unsqueeze(-1) * self.freqs * math.pi  # [N, 3, num_bands]

        # Sin and cos: [N, 3, num_bands] each
        sin_features = torch.sin(angles)
        cos_features = torch.cos(angles)

        # Flatten: [N, 3 * num_bands] each
        sin_flat = sin_features.reshape(xyz_flat.shape[0], -1)
        cos_flat = cos_features.reshape(xyz_flat.shape[0], -1)

        # Concatenate: [N, 3 + 3*num_bands + 3*num_bands] = [N, 3 + 6*num_bands]
        features = torch.cat([xyz_flat, sin_flat, cos_flat], dim=-1)

        # Project to output dimension
        out = self.proj(features)  # [N, out_dim]

        # Reshape back to original batch shape
        return out.reshape(*orig_shape, self.out_dim)


class TemporalEmbed(nn.Module):
    """
    Sinusoidal temporal embedding for frame indices.

    Embeds frame indices [0, 1, 2, ..., T-1] using sinusoidal functions.
    This is the additive positional encoding for the temporal dimension.

    Args:
        out_dim: Output embedding dimension
        max_len: Maximum sequence length (default 100)
    """

    def __init__(self, out_dim: int = 256, max_len: int = 100):
        super().__init__()
        self.out_dim = out_dim

        # Precompute sinusoidal embeddings for all positions
        pe = get_sincos_pos_embed_1d(max_len, out_dim)
        self.register_buffer('pe', pe)  # [max_len, out_dim]

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_indices: [...] integer frame indices

        Returns:
            [..., out_dim] temporal embeddings
        """
        # Lookup embeddings
        return self.pe[frame_indices.long()]
