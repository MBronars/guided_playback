"""Random attention groups (model.unet_tube_grouping=random).

A tubed level cuts its tokens into k = ceil(n / P) groups and runs one attention call per group.
`curves` cuts a space-filling curve, so a group is a compact blob of space. `random` instead draws
uniformly random equal-size groups, re-drawn on every forward: a group has no locality at all, every
call sees a scattered sample of the level, and no token pair is systematically co-located. The core
draws `unet_tube_group_draws` independent groupings per level and cycles blocks through them, so a
token meets a different set of neighbours in every block.

Randomness comes from the global torch RNG, so a seeded evaluation is reproducible.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def random_cells(X: torch.Tensor, P: int):
    """X [G, n, 3] (positions, read only for shape/device) -> (labels [G, n] in 0..k-1, k, cap).
    k = ceil(n / P) uniformly random groups of (almost) equal size; cap = ceil(n / k) slots per group."""
    G, n, _ = X.shape
    k = math.ceil(n / P)
    cap = math.ceil(n / k)
    rank = torch.rand(G, n, device=X.device).argsort(1).argsort(1)
    return rank // cap, k, cap


@torch.no_grad()
def label_slots(glab: torch.Tensor, K: int, S: int):
    """Tube layout from group labels. glab [B, n] in 0..K-1, every group holding at most S tokens.
    Returns (src [B, K*S] long: index into [tokens (n) | one zero row], pads read row n;
             back [B, n] long: the flat slot of every token)."""
    B, n = glab.shape
    ones = torch.ones(B, n, dtype=torch.long, device=glab.device)
    sizes = torch.zeros(B, K, dtype=torch.long, device=glab.device).scatter_add_(1, glab, ones)
    assert int(sizes.max()) <= S, (int(sizes.max()), S)
    order = torch.argsort(glab, dim=1, stable=True)
    inv = torch.empty_like(order).scatter_(1, order, torch.arange(n, device=glab.device).expand(B, n))
    start = torch.cumsum(sizes, 1) - sizes
    back = glab * S + (inv - start.gather(1, glab))
    src = torch.full((B, K * S), n, dtype=torch.long, device=glab.device)
    src.scatter_(1, back, torch.arange(n, device=glab.device).expand(B, n))
    return src, back
