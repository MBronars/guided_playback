"""
Attention components for the trajectory transformer.

Components:
- RotaryAttnProcessor: Attention processor with rotary embedding support
- KNNAttn: KNN-based local attention for encoder
- AdaLayerNorm: Adaptive layer norm with timestep conditioning
- TransformerBlock: Main transformer block with temporal/spatial attention

Uses diffusers for Attention and FeedForward.
"""

from typing import Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.normalization import AdaLayerNormZeroSingle


# =============================================================================
# Windowed Temporal Attention Mask
# =============================================================================

def create_sink_window_mask(
    num_frames: int,
    window_size: int = 5,
    use_sink: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Create attention mask for sink + causal window pattern.

    Each frame t attends to:
    1. Frame 0 (sink token) - provides global reference point
    2. Frames [t-W+1, t] (causal window) - local temporal context

    This reduces O(F²) attention to O(F*W) while maintaining:
    - Long-range reference via sink (frame 0 = first predicted frame)
    - Local temporal coherence via causal window

    Args:
        num_frames: Number of frames F
        window_size: Size of causal window W (includes current frame)
        use_sink: Whether to include frame 0 as global sink
        device: Target device
        dtype: Target dtype (should be float for attention scores)

    Returns:
        mask: [F, F] float tensor where 0.0 = attend, -inf = blocked
              (Compatible with F.scaled_dot_product_attention)
    """
    # Start with all positions blocked
    mask = torch.full(
        (num_frames, num_frames),
        float('-inf'),
        device=device,
        dtype=dtype or torch.float32,
    )

    for t in range(num_frames):
        # Sink: always attend to frame 0
        if use_sink:
            mask[t, 0] = 0.0

        # Causal window: attend to [t-W+1, t] inclusive
        window_start = max(0, t - window_size + 1)
        mask[t, window_start:t + 1] = 0.0

    return mask


# =============================================================================
# Component 1: RotaryAttnProcessor
# =============================================================================

_SDPA_BATCH_LIMIT = 65535


def batched_sdpa(query, key, value, *, attn_mask=None, dropout_p=0.0, is_causal=False):
    """Bound the flash-attention launch grid for large native particle clouds.

Temporal attention flattens batch × particles into its batch dimension. New
folding exports reach 34,739 points, exceeding CUDA's 65,535 grid dimension at
batch two. Rows attend independently, so chunk that dimension only. Ordinary
sizes retain the exact existing call; query/key sequence lengths are untouched.
    """
    if query.shape[0] <= _SDPA_BATCH_LIMIT:
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask,
                                             dropout_p=dropout_p, is_causal=is_causal)
    outputs = []
    for start in range(0, query.shape[0], _SDPA_BATCH_LIMIT):
        end = start + _SDPA_BATCH_LIMIT
        mask = attn_mask
        if mask is not None and mask.ndim == 4 and mask.shape[0] != 1:
            mask = mask[start:end]
        outputs.append(F.scaled_dot_product_attention(query[start:end], key[start:end], value[start:end],
            attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal))
    return torch.cat(outputs, dim=0)


class RotaryAttnProcessor:
    """
    Attention processor with Rotary Position Embedding (RoPE) support.

    This processor is used with diffusers' Attention module. It applies
    rotary embeddings to queries and keys before computing attention.

    Uses PyTorch 2.0's scaled_dot_product_attention for efficiency.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "RotaryAttnProcessor requires PyTorch >= 2.0 for "
                "scaled_dot_product_attention"
            )

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos_sin: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        num_heads: int,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.

        Args:
            x: [B, H, S, Dh] - queries or keys in multi-head format
            cos_sin: Either (cos, sin) tuple or [B, S, Dh, 2] stacked tensor
                     cos/sin can be: [S, Dh], [B, S, Dh], or [B, S, D] where D=H*Dh
            num_heads: Number of attention heads (H)

        Returns:
            [B, H, S, Dh] - rotated tensor
        """
        # Unpack cos/sin
        if isinstance(cos_sin, tuple):
            cos, sin = cos_sin
        else:
            # Stacked format [B, S, Dh, 2]
            cos, sin = cos_sin[..., 0], cos_sin[..., 1]

        B, H, S, Dh = x.shape
        assert H == num_heads, f"Expected {num_heads} heads, got {H}"

        # Broadcast cos/sin to match [B, H, S, Dh]
        if cos.dim() == 2:
            # [S, Dh] -> [1, 1, S, Dh]
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif cos.dim() == 3:
            if cos.shape[-1] == H * Dh:
                # [B, S, D] -> [B, H, S, Dh]
                cos = cos.view(B, S, H, Dh).permute(0, 2, 1, 3)
                sin = sin.view(B, S, H, Dh).permute(0, 2, 1, 3)
            else:
                # [B, S, Dh] -> [B, 1, S, Dh]
                cos = cos[:, None, :, :]
                sin = sin[:, None, :, :]
        else:
            raise ValueError(f"Unexpected cos shape: {cos.shape}")

        # Ensure dtype and device match
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)

        # Apply rotation to adjacent pairs
        # x = [x0, x1, x2, x3, ...] -> x_rot = [-x1, x0, -x3, x2, ...]
        x_pairs = x.reshape(B, H, S, Dh // 2, 2)
        x0, x1 = x_pairs.unbind(-1)
        x_rotated = torch.stack([-x1, x0], dim=-1).flatten(-2)

        # Apply rotation: x' = x * cos + x_rot * sin
        return x * cos + x_rotated * sin

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Union[Tuple, torch.Tensor]] = None,
        rotary_emb_k: Optional[Union[Tuple, torch.Tensor]] = None,
        is_causal: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Process attention with optional rotary embeddings.

        Args:
            attn: Diffusers Attention module
            hidden_states: [B, N, D] - queries (and keys/values for self-attention)
            encoder_hidden_states: [B, M, D] - keys/values for cross-attention
            attention_mask: Optional attention mask
            rotary_emb: Rotary embedding for queries (and keys if self-attention)
            rotary_emb_k: Rotary embedding for keys (cross-attention with different positions)

        Returns:
            [B, N, D] - attention output
        """
        B, N, _ = hidden_states.shape

        # Self-attention or cross-attention?
        is_cross = encoder_hidden_states is not None
        if not is_cross:
            encoder_hidden_states = hidden_states

        M = encoder_hidden_states.shape[1]  # Key/value sequence length

        # Project to Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Reshape to multi-head: [B, N, D] -> [B, H, N, Dh]
        head_dim = query.shape[-1] // attn.heads
        query = query.view(B, N, attn.heads, head_dim).transpose(1, 2)
        key = key.view(B, M, attn.heads, head_dim).transpose(1, 2)
        value = value.view(B, M, attn.heads, head_dim).transpose(1, 2)

        # Apply QK normalization if available (helps training stability)
        if hasattr(attn, 'norm_q') and attn.norm_q is not None:
            query = attn.norm_q(query)
        if hasattr(attn, 'norm_k') and attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply rotary embeddings
        if rotary_emb is not None:
            query = self.apply_rotary_emb(query, rotary_emb, attn.heads)

            # For self-attention, use same RoPE for keys unless specified
            if rotary_emb_k is not None:
                key = self.apply_rotary_emb(key, rotary_emb_k, attn.heads)
            elif not is_cross:
                key = self.apply_rotary_emb(key, rotary_emb, attn.heads)

        # Efficient attention using SDPA. is_causal=True (with no mask) keeps
        # the flash fast path; an explicit attn_mask forces the fallback
        # kernels that materialize/save the [B,h,S,S] scores.
        out = batched_sdpa(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        # Reshape back: [B, H, N, Dh] -> [B, N, D]
        out = out.transpose(1, 2).reshape(B, N, attn.heads * head_dim)

        # Output projection
        out = attn.to_out[0](out)  # Linear
        if len(attn.to_out) > 1:
            out = attn.to_out[1](out)  # Dropout

        return out


# =============================================================================
# Component 2: KNN Attention
# =============================================================================

class KNNAttention(nn.Module):
    """
    Multi-head KNN attention with learnable Q/K/V projections.

    Each query attends only to its K nearest neighbors. Unlike standard attention
    where all queries share the same key/value set, here each query has its own
    unique set of K neighbors.

    Used in encoder to compress N points to L latent tokens via FPS + KNN.

    Supports optional Rotary Position Embeddings (RoPE) for spatial encoding.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=bias),
            nn.Dropout(dropout),
        )

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to tensor x.

        Args:
            x: [..., Dh] tensor to rotate
            cos: [..., Dh] cosine values (interleaved)
            sin: [..., Dh] sine values (interleaved)

        Returns:
            [..., Dh] rotated tensor
        """
        # Rotate adjacent pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        x_pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x0, x1 = x_pairs.unbind(-1)
        x_rotated = torch.stack([-x1, x0], dim=-1).flatten(-2)

        # Apply rotation: x' = x * cos + x_rot * sin
        return x * cos + x_rotated * sin

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
        rotary_emb_q: Tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_emb_k: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, L, D] - query embeddings (FPS-sampled points)
            kv: [B, L, K, D] - neighbor embeddings for each query (from KNN)
            kv_mask: [B, L, K] - True = valid neighbor, False = masked
            rotary_emb_q: Optional (cos, sin) each [B, L, Dh] for query positions
            rotary_emb_k: Optional (cos, sin) each [B, L, K, Dh] for key positions

        Returns:
            [B, L, D] - attended output
        """
        B, L, D = queries.shape
        K = kv.shape[2]
        H, Dh = self.heads, self.dim_head

        # Project: queries [B, L, D] -> [B, H, L, Dh]
        q = self.to_q(queries).view(B, L, H, Dh).permute(0, 2, 1, 3)

        # Project: kv [B, L, K, D] -> [B, H, L, K, Dh]
        k = self.to_k(kv).view(B, L, K, H, Dh).permute(0, 3, 1, 2, 4)
        v = self.to_v(kv).view(B, L, K, H, Dh).permute(0, 3, 1, 2, 4)

        # Apply rotary embeddings if provided
        if rotary_emb_q is not None:
            cos_q, sin_q = rotary_emb_q
            # cos_q, sin_q: [B, L, Dh] -> [B, 1, L, Dh] for broadcasting over H
            cos_q = cos_q[:, None, :, :]
            sin_q = sin_q[:, None, :, :]
            q = self._apply_rotary_emb(q, cos_q, sin_q)

        if rotary_emb_k is not None:
            cos_k, sin_k = rotary_emb_k
            # cos_k, sin_k: [B, L, K, Dh] -> [B, 1, L, K, Dh] for broadcasting over H
            cos_k = cos_k[:, None, :, :, :]
            sin_k = sin_k[:, None, :, :, :]
            k = self._apply_rotary_emb(k, cos_k, sin_k)

        # Attention scores: [B, H, L, 1, K]
        q = q.unsqueeze(3)
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Mask invalid neighbors before softmax
        if kv_mask is not None:
            # kv_mask [B, L, K] -> [B, 1, L, 1, K] for broadcasting
            attn = attn.masked_fill(~kv_mask[:, None, :, None, :], float('-inf'))

        attn = F.softmax(attn, dim=-1)

        # Safety: if all neighbors masked, softmax produces NaN -> zero output
        attn = torch.nan_to_num(attn, nan=0.0)

        # Output: [B, H, L, 1, K] @ [B, H, L, K, Dh] -> [B, H, L, Dh]
        out = torch.matmul(attn, v).squeeze(3)

        # Reshape: [B, H, L, Dh] -> [B, L, D]
        out = out.permute(0, 2, 1, 3).reshape(B, L, H * Dh)

        return self.to_out(out)


# =============================================================================
# Component 3: AdaLayerNorm (Legacy) - kept for backward compatibility
# =============================================================================

class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization with timestep conditioning.

    Produces scale and shift from timestep embedding.
    Zero-initialized for identity at training start.

    DEPRECATED: Use AdaLayerNormZeroSingle from diffusers for new code.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, dim * 2)

        # Zero-init for stable training (starts as identity)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, ..., dim] input tensor
            t_emb: [B, dim] timestep embedding

        Returns:
            [B, ..., dim] normalized and modulated tensor
        """
        # Broadcast t_emb to match x's shape
        while t_emb.dim() < x.dim():
            t_emb = t_emb.unsqueeze(1)

        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


# =============================================================================
# Component 4: TransformerBlock with AdaLN-Zero (DiT-style)
# =============================================================================

class TransformerBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning (DiT-style).

    Structure:
    1. Temporal cross-attention (future ↔ context) with GATED residual
    2. Temporal self-attention (per-particle across frames) with GATED residual
    3. Spatial self-attention (per-frame across particles) with GATED residual
    4. FeedForward with GATED residual

    Each residual branch is gated by the timestep embedding. Explicitly zeroed
    modulation projections make blocks start as identity; the gates learn how
    much each branch contributes during training (DiT's AdaLN-Zero convention).

    Uses diffusers' Attention with RotaryAttnProcessor for RoPE support.
    Uses diffusers' AdaLayerNormZeroSingle for timestep conditioning.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        head_dim = dim // heads

        # Shared processor for RoPE
        self.attn_processor = RotaryAttnProcessor()

        # Temporal cross-attention with AdaLN-Zero (future ↔ context)
        self.norm1 = AdaLayerNormZeroSingle(dim)
        self.cross_attn = Attention(
            query_dim=dim,
            cross_attention_dim=dim,
            heads=heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=False,  # Modern best practice: no bias in attention
        )

        # Temporal self-attention with AdaLN-Zero
        self.norm2 = AdaLayerNormZeroSingle(dim)
        self.temporal_attn = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=False,
        )

        # Spatial self-attention with AdaLN-Zero
        self.norm3 = AdaLayerNormZeroSingle(dim)
        self.spatial_attn = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=False,
        )

        # FeedForward with AdaLN-Zero (GEGLU = gated linear unit, similar to SwiGLU)
        self.norm4 = AdaLayerNormZeroSingle(dim)
        self.ffn = FeedForward(
            dim,
            mult=int(mlp_ratio),
            dropout=dropout,
            activation_fn="geglu",  # GEGLU > GELU (gated linear unit for better quality)
        )

        # AdaLayerNormZeroSingle constructs ordinary random Linear parameters.
        # Apply identity initialization here, before any checkpoint is loaded.
        for norm in (self.norm1, self.norm2, self.norm3, self.norm4):
            nn.init.zeros_(norm.linear.weight)
            nn.init.zeros_(norm.linear.bias)

    def forward(
        self,
        x: torch.Tensor,                              # [B, F, L, d]
        context: torch.Tensor,                        # [B, C, L, d]
        actions: Optional[torch.Tensor],               # [B, F, L, d] or None
        t_emb: torch.Tensor,                          # [B, F, d] or [B, F, L, d] (per-token)
        ctx_t_emb: Optional[torch.Tensor] = None,     # [B, C, d] or [B, C, L, d] (per-token)
        rotary_emb: Optional[Dict[str, Tuple]] = None,  # RoPE embeddings
        temporal_attn_mask: Optional[torch.Tensor] = None,  # [F, F] causal mask
    ) -> torch.Tensor:
        """
        Args:
            x: Future frame tokens [B, F, L, d]
            context: Context tokens [B, C, L, d] (initial frames, physics, etc.)
            actions: Action tokens [B, F, L, d] or None (unconditional mode)
            t_emb: Timestep embedding [B, F, d] (per-frame) or [B, F, L, d] (per-token, for per-object noise)
            ctx_t_emb: Context timestep embedding [B, C, d] or [B, C, L, d] (per-token)
            rotary_emb: Optional dict with 'temporal' and 'spatial' RoPE
            temporal_attn_mask: Optional [F, F] mask for temporal self-attention (causal)

        Returns:
            Updated x [B, F, L, d]
        """
        B, F, L, d = x.shape
        C = context.shape[1]
        BL = B * L
        per_token = (t_emb.ndim == 4)  # [B, F, L, d] vs [B, F, d]

        # Get RoPE if provided
        temporal_rope_q = rotary_emb.get('temporal_q') if rotary_emb else None
        temporal_rope_kv = rotary_emb.get('temporal_kv') if rotary_emb else None
        spatial_rope = rotary_emb.get('spatial') if rotary_emb else None

        # =====================================================================
        # 1. Temporal cross-attention with GATED residual (AdaLN-Zero)
        # =====================================================================
        # Reshape: [B, F, L, d] -> [B*L, F, d]
        x_temporal = x.permute(0, 2, 1, 3).reshape(BL, F, d)
        ctx_temporal = context.permute(0, 2, 1, 3).reshape(BL, C, d)

        # Per-token: [B, F, L, d] -> [B, L, F, d] -> [B*L, F, d] (each latent has its own t_emb)
        # Per-frame: [B, F, d] -> [B, L, F, d] -> [B*L, F, d] (all latents share t_emb)
        if per_token:
            t_emb_temporal = t_emb.permute(0, 2, 1, 3).reshape(BL, F, d)
        else:
            t_emb_temporal = t_emb.unsqueeze(1).expand(-1, L, -1, -1).reshape(BL, F, d)

        # AdaLN-Zero for queries (future frames)
        x_for_norm = x_temporal.reshape(BL * F, 1, d)
        t_for_norm = t_emb_temporal.reshape(BL * F, d)
        x_mod, gate1 = self.norm1(x_for_norm, emb=t_for_norm)
        x_mod = x_mod.reshape(BL, F, d)
        gate1 = gate1.reshape(BL, F, d)

        # AdaLN-Zero for context KV — same norm, conditioned on context timestep
        if ctx_t_emb is not None:
            if ctx_t_emb.ndim == 4:  # [B, C, L, d] per-token
                ctx_t_emb_temporal = ctx_t_emb.permute(0, 2, 1, 3).reshape(BL, C, d)
            else:  # [B, C, d] per-frame
                ctx_t_emb_temporal = ctx_t_emb.unsqueeze(1).expand(-1, L, -1, -1).reshape(BL, C, d)
            ctx_for_norm = ctx_temporal.reshape(BL * C, 1, d)
            ctx_t_for_norm = ctx_t_emb_temporal.reshape(BL * C, d)
            ctx_mod, _ = self.norm1(ctx_for_norm, emb=ctx_t_for_norm)
            ctx_mod = ctx_mod.reshape(BL, C, d)
        else:
            ctx_mod = ctx_temporal

        if actions is None:
            kv = ctx_mod  # [B*L, C, d]
        else:
            act_temporal = actions.permute(0, 2, 1, 3).reshape(BL, F, d)
            kv = torch.cat([ctx_mod, act_temporal], dim=1)  # [B*L, C+F, d]

        attn_out = self.attn_processor(
            self.cross_attn,
            x_mod,
            encoder_hidden_states=kv,
            rotary_emb=temporal_rope_q,
            rotary_emb_k=temporal_rope_kv,
        )
        x_temporal = x_temporal + gate1 * attn_out  # GATED residual (per-frame)

        # =====================================================================
        # 2. Temporal self-attention with GATED residual
        # =====================================================================
        x_for_norm = x_temporal.reshape(BL * F, 1, d)
        x_mod, gate2 = self.norm2(x_for_norm, emb=t_for_norm)
        x_mod = x_mod.reshape(BL, F, d)
        gate2 = gate2.reshape(BL, F, d)

        attn_out = self.attn_processor(
            self.temporal_attn,
            x_mod,
            attention_mask=temporal_attn_mask,
            rotary_emb=temporal_rope_q,
        )
        x_temporal = x_temporal + gate2 * attn_out  # GATED residual (per-frame)

        # Reshape back: [B*L, F, d] -> [B, F, L, d]
        x = x_temporal.reshape(B, L, F, d).permute(0, 2, 1, 3)

        # =====================================================================
        # 3. Spatial self-attention with GATED residual
        # =====================================================================
        # Reshape: [B, F, L, d] -> [B*F, L, d]
        x_spatial = x.reshape(B * F, L, d)

        if per_token:
            # Per-token AdaLN: each token gets independent modulation
            # Reshape so each token is its own "batch element" for AdaLayerNormZeroSingle
            t_emb_spatial = t_emb.reshape(B * F * L, d)
            x_mod, gate3 = self.norm3(x_spatial.reshape(B * F * L, 1, d), emb=t_emb_spatial)
            x_mod = x_mod.reshape(B * F, L, d)
            gate3 = gate3.reshape(B * F, L, d)
        else:
            t_emb_spatial = t_emb.reshape(B * F, d)
            x_mod, gate3 = self.norm3(x_spatial, emb=t_emb_spatial)

        attn_out = self.attn_processor(
            self.spatial_attn,
            x_mod,
            rotary_emb=spatial_rope,
        )

        if per_token:
            x_spatial = x_spatial + gate3 * attn_out
        else:
            x_spatial = x_spatial + gate3[:, None] * attn_out  # GATED residual!

        # =====================================================================
        # 4. FeedForward with GATED residual
        # =====================================================================
        if per_token:
            x_mod, gate4 = self.norm4(x_spatial.reshape(B * F * L, 1, d), emb=t_emb_spatial)
            x_mod = x_mod.reshape(B * F, L, d)
            gate4 = gate4.reshape(B * F, L, d)
            x_spatial = x_spatial + gate4 * self.ffn(x_mod)
        else:
            x_mod, gate4 = self.norm4(x_spatial, emb=t_emb_spatial)
            x_spatial = x_spatial + gate4[:, None] * self.ffn(x_mod)  # GATED residual!

        # Reshape back: [B*F, L, d] -> [B, F, L, d]
        x = x_spatial.reshape(B, F, L, d)

        return x


# =============================================================================
# Component 5: JointTransformerBlock (joint context+future temporal stream)
# =============================================================================

class JointTransformerBlock(nn.Module):
    """
    Transformer block for joint context+future stream (MM-DiT style for a
    single modality), used by TrajectoryTransformer.

    Replaces the (cross_attn → temporal_attn) two-step of TransformerBlock
    with a single temporal_attn over the concatenated [context, future]
    sequence under a block-causal mask. Eliminates the cross-attention
    parameters and the structurally degenerate 1-token KV cross-attention
    that occurs with all_relative=True + context_len=2 (C_ctx=1).

    Three stages remain, each with PixArt adaLN-single conditioning:
    1. Temporal self-attention over [B*L, T_joint, d] with block-causal mask
    2. Spatial self-attention per frame over [B*T_joint, L, d]
    3. FeedForward (GEGLU)

    Each block holds a per-block adapter [9, d] = 3 stages × (shift, scale,
    gate). The shared `adaln_proj: Linear(d, 9*d)` lives in the parent model
    (one Linear shared across all blocks).

    Note: PoseTransformer continues to use the original TransformerBlock
    (with cross-attention + 4× AdaLayerNormZeroSingle); this class is only
    used by TrajectoryTransformer.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 2.67,
        qk_norm: str | None = "rms_norm",
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        head_dim = dim // heads

        self.attn_processor = RotaryAttnProcessor()

        # adaLN-single per-block adapter: 3 stages × (shift, scale, gate)
        self.scale_shift_table = nn.Parameter(torch.randn(9, dim) / dim**0.5)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Temporal self-attention over joint [context, future] sequence
        self.temporal_attn = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=False,
            qk_norm=qk_norm,
        )

        # Spatial self-attention per frame
        self.spatial_attn = Attention(
            query_dim=dim,
            heads=heads,
            dim_head=head_dim,
            dropout=dropout,
            bias=False,
            qk_norm=qk_norm,
        )

        self.ffn = FeedForward(
            dim,
            inner_dim=int(round(dim * mlp_ratio)),
            dropout=dropout,
            activation_fn="geglu",
        )

    def forward(
        self,
        x: torch.Tensor,                              # [B, T, L, d] joint sequence
        t_mods: torch.Tensor,                         # [B, T, 9, d] or [B, T, L, 9, d]
        rotary_emb: Optional[Dict[str, Tuple]] = None,
        temporal_attn_mask: Optional[torch.Tensor] = None,  # [T, T] block-causal
        spatial_attn_bias: Optional[torch.Tensor] = None,   # [B*T, H, L, L] or [B, L, L] (v7: cross-obj edge)
    ) -> torch.Tensor:
        """
        Args:
            x: Joint stream tokens [B, T, L, d] where T = C_ctx + F.
                The caller (TrajectoryTransformer) concatenates context and
                future tokens along the time axis before passing here.
            t_mods: Pre-projected per-stream-token modulations
                [B, T, 9, d] (per-frame) or [B, T, L, 9, d] (per-token, for
                per-object noise). The 9 dim packs (shift, scale, gate) × 3
                stages.
            rotary_emb: dict with 'temporal_q' and 'spatial' keys.
            temporal_attn_mask: [T, T] block-causal mask. Context can attend
                to context bidirectionally; future attends to all context +
                causal within future.
        """
        B, T, L, d = x.shape
        BL = B * L
        per_token = (t_mods.ndim == 5)

        temporal_rope_q = rotary_emb.get('temporal_q') if rotary_emb else None
        spatial_rope = rotary_emb.get('spatial') if rotary_emb else None

        # Per-block adapter, then unpack 9 modulations.
        mods = t_mods + self.scale_shift_table  # broadcasts
        s1, sc1, g1, s2, sc2, g2, s3, sc3, g3 = mods.unbind(dim=-2)

        # =====================================================================
        # 1. Temporal self-attention over joint sequence (block-causal)
        # =====================================================================
        x_temporal = x.permute(0, 2, 1, 3).reshape(BL, T, d)

        if per_token:
            s1_t  = s1.permute(0, 2, 1, 3).reshape(BL, T, d)
            sc1_t = sc1.permute(0, 2, 1, 3).reshape(BL, T, d)
            g1_t  = g1.permute(0, 2, 1, 3).reshape(BL, T, d)
        else:
            s1_t  = s1.unsqueeze(1).expand(-1, L, -1, -1).reshape(BL, T, d)
            sc1_t = sc1.unsqueeze(1).expand(-1, L, -1, -1).reshape(BL, T, d)
            g1_t  = g1.unsqueeze(1).expand(-1, L, -1, -1).reshape(BL, T, d)

        x_mod = self.norm1(x_temporal) * (1 + sc1_t) + s1_t
        attn_out = self.attn_processor(
            self.temporal_attn, x_mod,
            attention_mask=temporal_attn_mask,
            rotary_emb=temporal_rope_q,
        )
        x_temporal = x_temporal + g1_t * attn_out

        # Back to [B, T, L, d]
        x = x_temporal.reshape(B, L, T, d).permute(0, 2, 1, 3)

        # =====================================================================
        # 2. Spatial self-attention per frame
        # =====================================================================
        x_spatial = x.reshape(B * T, L, d)
        if per_token:
            s2_t  = s2.reshape(B * T, L, d)
            sc2_t = sc2.reshape(B * T, L, d)
            g2_t  = g2.reshape(B * T, L, d)
            s3_t  = s3.reshape(B * T, L, d)
            sc3_t = sc3.reshape(B * T, L, d)
            g3_t  = g3.reshape(B * T, L, d)
        else:
            s2_t  = s2.reshape(B * T, 1, d)
            sc2_t = sc2.reshape(B * T, 1, d)
            g2_t  = g2.reshape(B * T, 1, d)
            s3_t  = s3.reshape(B * T, 1, d)
            sc3_t = sc3.reshape(B * T, 1, d)
            g3_t  = g3.reshape(B * T, 1, d)

        x_mod = self.norm2(x_spatial) * (1 + sc2_t) + s2_t
        # v7 cross-object edge: an additive bias on attention logits keyed by
        # whether (Q, K) latents come from the same object. Reshape to [B*T, ...]
        # to match x_spatial's flattened layout. SDPA's attn_mask is additive.
        sa_mask = None
        if spatial_attn_bias is not None:
            if spatial_attn_bias.ndim == 3:
                # [B, L, L] → repeat for each frame, no per-head dim (broadcast)
                Bx = spatial_attn_bias.shape[0]
                Tx = (B * T) // Bx
                sa_mask = spatial_attn_bias.unsqueeze(1).expand(Bx, Tx, L, L).reshape(B * T, L, L)
                # SDPA needs [B*T, H, L, L] OR [B*T, 1, L, L] for broadcast over heads.
                sa_mask = sa_mask.unsqueeze(1)   # [B*T, 1, L, L]
            elif spatial_attn_bias.ndim == 4:
                # [B, H, L, L] → repeat over time
                Bx, H, L_in, _ = spatial_attn_bias.shape
                Tx = (B * T) // Bx
                sa_mask = spatial_attn_bias.unsqueeze(1).expand(Bx, Tx, H, L_in, L_in)
                sa_mask = sa_mask.reshape(B * T, H, L_in, L_in)
            else:
                sa_mask = spatial_attn_bias
            # SDPA broadcasts over heads if dim==1; ensure dtype matches Q/K
            sa_mask = sa_mask.to(dtype=x_mod.dtype)
        attn_out = self.attn_processor(self.spatial_attn, x_mod, rotary_emb=spatial_rope,
                                        attention_mask=sa_mask)
        x_spatial = x_spatial + g2_t * attn_out

        # =====================================================================
        # 3. FeedForward
        # =====================================================================
        x_mod = self.norm3(x_spatial) * (1 + sc3_t) + s3_t
        x_spatial = x_spatial + g3_t * self.ffn(x_mod)

        return x_spatial.reshape(B, T, L, d)
