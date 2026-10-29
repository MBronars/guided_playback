"""Tests for attention components."""

import torch
from diffusers.models.attention import Attention

from models.components.attention import (
    RotaryAttnProcessor,
    KNNAttention,
    AdaLayerNorm,
    TransformerBlock,
)
from models.components.embeddings import (
    RotaryPositionEncoding,
    RotaryPositionEncoding3D,
)


def test_rotary_attn_processor_self_attention():
    """Test RotaryAttnProcessor with self-attention."""
    print("\n" + "=" * 60)
    print("TEST: RotaryAttnProcessor - Self Attention")
    print("=" * 60)

    B, N, D = 4, 100, 256
    num_heads = 8

    # Create diffusers Attention module
    attn = Attention(
        query_dim=D,
        heads=num_heads,
        dim_head=D // num_heads,
        dropout=0.0,
        bias=True,
    )

    # Create our processor
    processor = RotaryAttnProcessor()

    # Input
    x = torch.randn(B, N, D)

    # Test without RoPE
    out_no_rope = processor(attn, x)
    print(f"Input: {x.shape}")
    print(f"Output (no RoPE): {out_no_rope.shape}")
    assert out_no_rope.shape == x.shape

    # Test with RoPE
    rope = RotaryPositionEncoding(dim=D // num_heads)
    positions = torch.arange(N).unsqueeze(0).expand(B, -1).float()
    cos, sin = rope(positions)  # [B, N, Dh]

    out_with_rope = processor(attn, x, rotary_emb=(cos, sin))
    print(f"Output (with RoPE): {out_with_rope.shape}")
    assert out_with_rope.shape == x.shape

    # Verify RoPE changes the output
    assert not torch.allclose(out_no_rope, out_with_rope, atol=1e-5), \
        "RoPE should change attention output"
    print("RoPE effect verified: outputs differ")

    print("RotaryAttnProcessor self-attention test passed!")


def test_rotary_attn_processor_cross_attention():
    """Test RotaryAttnProcessor with cross-attention."""
    print("\n" + "=" * 60)
    print("TEST: RotaryAttnProcessor - Cross Attention")
    print("=" * 60)

    B, N, M, D = 4, 50, 100, 256
    num_heads = 8

    # Create diffusers Attention with cross_attention_dim
    attn = Attention(
        query_dim=D,
        cross_attention_dim=D,
        heads=num_heads,
        dim_head=D // num_heads,
        dropout=0.0,
        bias=True,
    )

    processor = RotaryAttnProcessor()

    # Inputs
    queries = torch.randn(B, N, D)
    context = torch.randn(B, M, D)

    # Test cross-attention
    out = processor(attn, queries, encoder_hidden_states=context)
    print(f"Queries: {queries.shape}")
    print(f"Context: {context.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == queries.shape

    # Test with different RoPE for Q and K
    rope = RotaryPositionEncoding(dim=D // num_heads)

    q_positions = torch.arange(N).unsqueeze(0).expand(B, -1).float()
    k_positions = torch.arange(M).unsqueeze(0).expand(B, -1).float()

    cos_q, sin_q = rope(q_positions)
    cos_k, sin_k = rope(k_positions)

    out_rope = processor(
        attn, queries,
        encoder_hidden_states=context,
        rotary_emb=(cos_q, sin_q),
        rotary_emb_k=(cos_k, sin_k),
    )
    print(f"Output (with RoPE): {out_rope.shape}")
    assert out_rope.shape == queries.shape

    print("RotaryAttnProcessor cross-attention test passed!")


def test_rotary_attn_processor_3d_rope():
    """Test RotaryAttnProcessor with 3D spatial RoPE."""
    print("\n" + "=" * 60)
    print("TEST: RotaryAttnProcessor - 3D Spatial RoPE")
    print("=" * 60)

    B, N, D = 4, 100, 256
    num_heads = 8
    head_dim = D // num_heads

    attn = Attention(
        query_dim=D,
        heads=num_heads,
        dim_head=head_dim,
        dropout=0.0,
        bias=True,
    )

    processor = RotaryAttnProcessor()

    # Input with 3D positions
    x = torch.randn(B, N, D)
    xyz = torch.randn(B, N, 3)  # 3D coordinates

    # Get 3D RoPE
    rope_3d = RotaryPositionEncoding3D(dim=head_dim)
    cos, sin = rope_3d(xyz)  # [B, N, head_dim]

    out = processor(attn, x, rotary_emb=(cos, sin))
    print(f"Input: {x.shape}")
    print(f"3D positions: {xyz.shape}")
    print(f"RoPE cos/sin: {cos.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == x.shape

    print("RotaryAttnProcessor 3D RoPE test passed!")


def test_knn_attention():
    """Test KNNAttention module."""
    print("\n" + "=" * 60)
    print("TEST: KNNAttention")
    print("=" * 60)

    B, L, K, D = 4, 64, 32, 256
    heads = 8
    dim_head = D // heads

    # Create module
    knn_attn = KNNAttention(dim=D, heads=heads, dim_head=dim_head)

    queries = torch.randn(B, L, D)
    kv = torch.randn(B, L, K, D)

    out = knn_attn(queries, kv)
    print(f"Queries: {queries.shape}")
    print(f"KV (K neighbors per query): {kv.shape}")
    print(f"Output: {out.shape}")

    assert out.shape == queries.shape

    # Test with mask
    kv_mask = torch.ones(B, L, K, dtype=torch.bool)
    kv_mask[:, :, -5:] = False  # Mask last 5 neighbors
    out_masked = knn_attn(queries, kv, kv_mask=kv_mask)
    print(f"Output with mask: {out_masked.shape}")
    assert out_masked.shape == queries.shape

    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print(f"Gradient flows through Q projection: {knn_attn.to_q.weight.grad is not None}")

    print("KNNAttention test passed!")


def test_attention_determinism():
    """Test that attention is deterministic."""
    print("\n" + "=" * 60)
    print("TEST: Attention Determinism")
    print("=" * 60)

    B, N, D = 2, 50, 128
    num_heads = 4

    attn = Attention(
        query_dim=D,
        heads=num_heads,
        dim_head=D // num_heads,
    )
    processor = RotaryAttnProcessor()

    x = torch.randn(B, N, D)
    rope = RotaryPositionEncoding(dim=D // num_heads)
    positions = torch.arange(N).unsqueeze(0).expand(B, -1).float()
    cos, sin = rope(positions)

    # Run twice
    out1 = processor(attn, x.clone(), rotary_emb=(cos.clone(), sin.clone()))
    out2 = processor(attn, x.clone(), rotary_emb=(cos.clone(), sin.clone()))

    assert torch.allclose(out1, out2), "Attention should be deterministic"
    print("Determinism: PASSED")

    print("Attention determinism test passed!")


def test_attention_equivariance():
    """Test that RoPE provides position equivariance (relative position invariance)."""
    print("\n" + "=" * 60)
    print("TEST: RoPE Position Equivariance")
    print("=" * 60)

    B, N, D = 1, 20, 64
    num_heads = 4

    attn = Attention(
        query_dim=D,
        heads=num_heads,
        dim_head=D // num_heads,
    )
    processor = RotaryAttnProcessor()
    rope = RotaryPositionEncoding(dim=D // num_heads)

    x = torch.randn(B, N, D)

    # Test 1: Shifting all positions should NOT change output (relative positions same)
    pos1 = torch.arange(N).unsqueeze(0).float()  # 0, 1, 2, ..., N-1
    pos2 = torch.arange(10, N + 10).unsqueeze(0).float()  # 10, 11, ..., N+9

    cos1, sin1 = rope(pos1)
    cos2, sin2 = rope(pos2)

    out1 = processor(attn, x, rotary_emb=(cos1, sin1))
    out2 = processor(attn, x, rotary_emb=(cos2, sin2))

    # Outputs should be THE SAME because relative positions are preserved
    assert torch.allclose(out1, out2, atol=1e-5), \
        "Shifting all positions should preserve output (relative positions unchanged)"
    print("Relative position invariance: PASSED")

    # Test 2: Changing relative positions SHOULD change output
    # Use non-sequential positions
    pos3 = torch.tensor([[0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
                          1, 3, 5, 7, 9, 11, 13, 15, 17, 19]]).float()

    cos3, sin3 = rope(pos3)
    out3 = processor(attn, x, rotary_emb=(cos3, sin3))

    assert not torch.allclose(out1, out3, atol=1e-5), \
        "Different relative positions should give different outputs"
    print("Relative position sensitivity: PASSED")

    print("RoPE equivariance test passed!")


def test_ada_layer_norm():
    """Test AdaLayerNorm module."""
    print("\n" + "=" * 60)
    print("TEST: AdaLayerNorm")
    print("=" * 60)

    dim = 256
    adaln = AdaLayerNorm(dim)

    # Test basic forward
    B, N = 4, 100
    x = torch.randn(B, N, dim)
    t_emb = torch.randn(B, dim)

    out = adaln(x, t_emb)
    print(f"Input: {x.shape}")
    print(f"t_emb: {t_emb.shape}")
    print(f"Output: {out.shape}")
    assert out.shape == x.shape

    # Test zero initialization (should start near identity)
    # With zero-init, scale=0, shift=0, so output = norm(x) * 1 + 0 = norm(x)
    adaln_fresh = AdaLayerNorm(dim)
    out_fresh = adaln_fresh(x, t_emb)
    x_normed = torch.nn.functional.layer_norm(x, [dim])
    assert torch.allclose(out_fresh, x_normed, atol=1e-5), \
        "Zero-init should give identity (just LayerNorm)"
    print("Zero initialization (identity at start): PASSED")

    # Test that after training, different timesteps give different outputs
    # Initialize with non-zero weights to simulate trained model
    adaln_trained = AdaLayerNorm(dim)
    torch.nn.init.normal_(adaln_trained.proj.weight, std=0.02)
    torch.nn.init.zeros_(adaln_trained.proj.bias)

    t_emb1 = torch.randn(B, dim)
    t_emb2 = torch.randn(B, dim)
    out1 = adaln_trained(x, t_emb1)
    out2 = adaln_trained(x, t_emb2)
    assert not torch.allclose(out1, out2), "Different t_emb should give different outputs"
    print("Different timesteps produce different outputs: PASSED")

    # Test gradient flow
    x_grad = torch.randn(B, N, dim, requires_grad=True)
    t_grad = torch.randn(B, dim, requires_grad=True)
    out_grad = adaln_trained(x_grad, t_grad)
    loss = out_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradients should flow to input"
    assert t_grad.grad is not None, "Gradients should flow to t_emb"
    print("Gradient flow: PASSED")

    print("AdaLayerNorm test passed!")


def test_transformer_block():
    """Test TransformerBlock module."""
    print("\n" + "=" * 60)
    print("TEST: TransformerBlock")
    print("=" * 60)

    B, F, L, C, d = 2, 8, 64, 2, 128
    heads = 4

    block = TransformerBlock(dim=d, heads=heads, dropout=0.0)
    # Exercise active attention; fresh AdaLN-Zero blocks are exactly identity.
    with torch.no_grad():
        for norm in (block.norm1, block.norm2, block.norm3, block.norm4):
            norm.linear.bias[2 * d:].fill_(1)

    # Create inputs
    x = torch.randn(B, F, L, d)        # Future frames
    context = torch.randn(B, C, L, d)  # Context (initial frames, physics)
    actions = torch.randn(B, F, L, d)  # Actions
    t_emb = torch.randn(B, F, d)        # Per-frame timestep embedding

    print(f"x (future): {x.shape}")
    print(f"context: {context.shape}")
    print(f"actions: {actions.shape}")
    print(f"t_emb: {t_emb.shape}")

    # Forward without RoPE
    out = block(x, context, actions, t_emb)
    print(f"Output: {out.shape}")
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    # Forward with RoPE
    head_dim = d // heads
    rope = RotaryPositionEncoding(dim=head_dim)

    # Temporal RoPE (F frames)
    temporal_pos = torch.arange(F).unsqueeze(0).expand(B * L, -1).float()
    temporal_cos, temporal_sin = rope(temporal_pos)

    # Spatial RoPE (L particles)
    spatial_pos = torch.arange(L).unsqueeze(0).expand(B * F, -1).float()
    spatial_cos, spatial_sin = rope(spatial_pos)

    rotary_emb = {
        'temporal': (temporal_cos, temporal_sin),
        'spatial': (spatial_cos, spatial_sin),
    }

    out_rope = block(x, context, actions, t_emb, rotary_emb=rotary_emb)
    print(f"Output (with RoPE): {out_rope.shape}")
    assert out_rope.shape == x.shape

    # Verify RoPE changes output
    assert not torch.allclose(out, out_rope, atol=1e-5), \
        "RoPE should change output"
    print("RoPE effect verified: PASSED")

    # Test gradient flow
    x_grad = torch.randn(B, F, L, d, requires_grad=True)
    out_grad = block(x_grad, context, actions, t_emb)
    loss = out_grad.sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradients should flow"
    print("Gradient flow: PASSED")

    print("TransformerBlock test passed!")


if __name__ == "__main__":
    test_rotary_attn_processor_self_attention()
    test_rotary_attn_processor_cross_attention()
    test_rotary_attn_processor_3d_rope()
    test_knn_attention()
    test_attention_determinism()
    test_attention_equivariance()
    test_ada_layer_norm()
    test_transformer_block()

    print("\n" + "=" * 60)
    print("ALL ATTENTION TESTS PASSED!")
    print("=" * 60)
