"""JointBlock: the transformer block of the joint-attention core (model.unet_core=tube_joint).

One block = pre-RMSNorm + adaLN-Zero -> ONE scaled_dot_product_attention over the call's whole
(rows x tokens) sequence -> GEGLU feed-forward. Nothing is factorized: a token attends to every
other token of the call in every row in a single kernel. The caller decides what a call contains
(a P-token tube x all rows, or a whole level x all rows) and supplies the RoPE tables; the block
itself only ever sees a flat [B, L, w] sequence plus `n_per_frame`.

adaLN-Zero carries two sources, summed into the same six slots (shift/scale/gate x 2):
  per row     the frame's noise-level embedding (always). One level per frame, shared by its tokens.
  per token   model.unet_token_adaln: a zero-init projection of the token's material + physics
              vector, constant over rows. No bias, so a token with no material (a register, or a
              padded tube slot, which arrive as cond = 0) contributes exactly nothing.
Both projections are zero-init, so the block is the identity at step 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, h, n, hd] with cos/sin [B, 1, n, hd]; adjacent-pair rotation (the UNet's RoPE convention)."""
    xr = torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape_as(x)
    return x * cos.to(x.dtype) + xr * sin.to(x.dtype)


class JointBlock(nn.Module):
    """pre-RMSNorm, adaLN-Zero, one joint flash attention over the whole call, single GEGLU FFN."""

    ckpt_ingraph = False        # set by the core: checkpoint INSIDE forward, so a regionally compiled
                                # block carries its own compiled recompute subgraph (the UNet Block's pattern)

    def __init__(self, w: int, head_dim: int, t_dim: int, expansion: float, dropout: float,
                 skip: bool = False, token_adaln: bool = False):
        super().__init__()
        # skip (decoder blocks under model.unet_tube_long_skip): the UNet Block's HunyuanDiT-style long skip --
        # the paired encoder block's output re-enters as concat -> LayerNorm -> linear down to w, REPLACING the
        # stream before the sub-ops.
        self.skip_norm = nn.LayerNorm(2 * w) if skip else None
        self.skip_linear = nn.Linear(2 * w, w) if skip else None
        hd = min(head_dim, w)
        assert w % hd == 0, f"width {w} not divisible by head_dim {hd}"
        self.w, self.hd, self.h = w, hd, w // hd
        self.norm1 = nn.RMSNorm(w, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(w, elementwise_affine=False)
        self.qkv = nn.Linear(w, 3 * w, bias=False)
        self.out = nn.Linear(w, w)
        self.q_norm, self.k_norm = nn.RMSNorm(hd), nn.RMSNorm(hd)
        inner = int(round(w * expansion))
        self.ff_in, self.ff_out = nn.Linear(w, 2 * inner), nn.Linear(inner, w)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.adaln = nn.Linear(t_dim, 6 * w)
        nn.init.zeros_(self.adaln.weight)                      # identity block at init (adaLN-Zero)
        nn.init.zeros_(self.adaln.bias)
        self.adaln_obj = None
        if token_adaln:
            self.adaln_obj = nn.Linear(t_dim, 6 * w, bias=False)
            nn.init.zeros_(self.adaln_obj.weight)

    def forward(self, x, cos, sin, t_emb, n_per_frame: int, attn_mask=None, skip=None, obj_cond=None):
        if self.ckpt_ingraph and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x, cos, sin, t_emb, n_per_frame, attn_mask, skip,
                              obj_cond, use_reentrant=False)
        return self._forward_impl(x, cos, sin, t_emb, n_per_frame, attn_mask, skip, obj_cond)

    def _forward_impl(self, x, cos, sin, t_emb, n_per_frame: int, attn_mask=None, skip=None, obj_cond=None):
        """x [B, T*n, w] in (row, token) order; t_emb [B, T, t_dim] (one noise level per row, shared by its n
        tokens); cos/sin [B, 1, T*n, hd]; obj_cond [B, n, t_dim] per-token material conditioning, constant over
        rows (model.unet_token_adaln), None = off.
        attn_mask [B, 1, 1, T*n] bool (True = a real key): padded tube slots are excluded as KEYS; as queries they
        compute garbage the caller drops. None keeps the unmasked (flash) path."""
        if self.skip_linear is not None:
            assert skip is not None, "skip-block called without its skip tensor"
            x = self.skip_linear(self.skip_norm(torch.cat([x, skip], dim=-1)))
        else:
            assert skip is None, "skip tensor passed to a non-skip block"
        B, L, w = x.shape
        T = t_emb.shape[1]
        mods = self.adaln(t_emb).reshape(B, T, 1, 6, w)
        if self.adaln_obj is not None and obj_cond is not None:
            mods = mods + self.adaln_obj(obj_cond).reshape(B, 1, n_per_frame, 6, w)
        s1, c1, g1, s2, c2, g2 = mods.expand(B, T, n_per_frame, 6, w).reshape(B, L, 6, w).unbind(2)
        h = self.norm1(x) * (1 + c1) + s1
        q, k, v = self.qkv(h).reshape(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), cos, sin), apply_rope(self.k_norm(k), cos, sin)
        a = F.scaled_dot_product_attention(q, k, v.to(q.dtype), attn_mask=attn_mask).transpose(1, 2).reshape(B, L, w)
        x = x + g1 * self.drop(self.out(a))
        h = self.norm2(x) * (1 + c2) + s2
        u, g = self.ff_in(h).chunk(2, dim=-1)
        return x + g2 * self.drop(self.ff_out(F.gelu(g) * u))
