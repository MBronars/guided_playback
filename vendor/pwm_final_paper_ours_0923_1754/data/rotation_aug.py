"""Yaw (about-Z) rotation augmentation for particle trajectories.

The model's spatial RoPE encodes *absolute* orientation, so a single-orientation
overfit memorizes that frame. Augmenting with random rotations breaks the
orientation overfit. For gravity-driven MPM data the Z axis is gravity and must
NOT be rotated, so we only rotate in the XY plane (yaw about Z).

This augmentation is currently supported only for passive MPM: it rotates
particle positions/actions, not rigid-body sidecars. The shared config preflight
rejects its use on the rigid or mixed domains. XY-isotropic displacement scales
are rotation-compatible; absolute positions are not re-centered or re-fitted.
"""

import math
import torch


def xy_centroid(positions: torch.Tensor, valid_lengths: torch.Tensor | None = None) -> torch.Tensor:
    """Mass centroid in XY over valid particles, per sample. positions [B,T,N,3] -> [B,2]."""
    xy = positions[..., :2]                      # [B, T, N, 2]
    B, T, N, _ = xy.shape
    if valid_lengths is not None:
        mask = (torch.arange(N, device=positions.device)[None, :] < valid_lengths[:, None]).to(xy.dtype)  # [B,N]
        w = mask[:, None, :, None]               # [B,1,N,1]
        s = (xy * w).sum(dim=(1, 2))             # [B,2] — sums over T AND N
        # cnt must count T*N terms (w broadcasts over T in the numerator). The
        # pre-2026-07-22 version divided by N only -> centroid inflated T-fold ->
        # rotation about a far-outside point = correct rotation + a large XY
        # translation. Benign for delta-target + relative-RoPE arms (translation
        # cancels everywhere), but it broke the "centroid stays fixed / nothing
        # clips" invariant and would poison absolute-position channels.
        cnt = (w.sum(dim=(1, 2)) * T).clamp_min(1.0)   # [B,1]
        return s / cnt
    return xy.mean(dim=(1, 2))                    # [B,2]


def random_yaw_matrices(B: int, device, dtype=torch.float32) -> torch.Tensor:
    """B independent uniform-random yaw rotation matrices, [B,2,2]."""
    angles = torch.rand(B, device=device, dtype=dtype) * (2 * math.pi)
    c, s = torch.cos(angles), torch.sin(angles)
    row0 = torch.stack([c, -s], dim=-1)
    row1 = torch.stack([s, c], dim=-1)
    return torch.stack([row0, row1], dim=-2)     # [B,2,2]


def _rotate_xy(coords: torch.Tensor, R: torch.Tensor, center_xy: torch.Tensor) -> torch.Tensor:
    """Rotate the XY of coords [B, ..., 3] about per-sample center_xy [B,2] by R [B,2,2]. Z untouched."""
    xy = coords[..., :2]
    n_mid = xy.dim() - 2                          # number of dims between batch and the xy dim
    cen = center_xy.reshape(center_xy.shape[0], *([1] * n_mid), 2)
    xy_rot = torch.einsum('bij,b...j->b...i', R, xy - cen) + cen
    out = coords.clone()
    out[..., :2] = xy_rot
    return out


def apply_yaw_aug(batch, center_xy: torch.Tensor):
    """Rotate `batch` in place by a fresh random yaw per sample, about center_xy [B,2].

    Rotates `positions` (and `actions`, treated as displacements about origin). All
    derived quantities (anchor/deltas) follow from positions. Z is preserved.
    """
    B = batch.positions.shape[0]
    R = random_yaw_matrices(B, batch.positions.device, batch.positions.dtype)
    batch.positions = _rotate_xy(batch.positions, R, center_xy)
    if batch.actions is not None and batch.actions.numel() > 0:
        zero = torch.zeros(B, 2, device=batch.positions.device, dtype=batch.positions.dtype)
        batch.actions = _rotate_xy(batch.actions, R, zero)
    return batch
