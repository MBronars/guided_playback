"""Weighted batched rigid alignment used to construct unified pose sidecars."""
from typing import Optional, Tuple

import torch


def kabsch_batched(
    P: torch.Tensor, Q: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find SE(3) (R, t) per batch element such that R @ P + t ≈ Q.

    Args:
        P: [..., N, 3] source point cloud
        Q: [..., N, 3] target point cloud
        weights: [..., N] optional per-point weights (default uniform)

    Returns:
        R: [..., 3, 3]
        t: [..., 3]
    """
    if weights is None:
        weights = torch.ones_like(P[..., 0])  # [..., N]
    w = weights.unsqueeze(-1)  # [..., N, 1]
    w_sum = w.sum(dim=-2).clamp_min(1e-12)
    cP = (P * w).sum(dim=-2) / w_sum  # [..., 3]
    cQ = (Q * w).sum(dim=-2) / w_sum
    PP = P - cP.unsqueeze(-2)
    QQ = Q - cQ.unsqueeze(-2)
    H = torch.einsum("...ni,...nj->...ij", PP * w, QQ)
    U, _, Vt = torch.linalg.svd(H)
    # Ensure proper rotation (det = +1)
    d = torch.sign(torch.linalg.det(torch.einsum("...ij,...kj->...ik", Vt, U)))
    D = torch.eye(3, device=P.device, dtype=P.dtype).expand(*d.shape, 3, 3).clone()
    D[..., 2, 2] = d
    R = torch.einsum("...ji,...jk,...kl->...il", Vt, D, U.transpose(-1, -2))
    t = cQ - torch.einsum("...ij,...j->...i", R, cP)
    return R, t
