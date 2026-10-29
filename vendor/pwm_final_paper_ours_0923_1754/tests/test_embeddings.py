"""Tests for embedding modules."""

import torch
from models.components.embeddings import (
    SinusoidalTimeEmbed,
    PointEmbed,
    SimplePointEmbed,
    RotaryPositionEncoding,
    RotaryPositionEncoding3D,
    get_sincos_pos_embed_1d,
)


def test_sinusoidal_time_embed():
    """Test timestep embedding."""
    print("\n" + "=" * 60)
    print("TEST: SinusoidalTimeEmbed")
    print("=" * 60)

    dim = 256
    embed = SinusoidalTimeEmbed(dim=dim)

    # Test with different batch sizes
    for B in [1, 4, 16]:
        timesteps = torch.randint(0, 1000, (B,))
        out = embed(timesteps)

        print(f"  B={B}: timesteps {timesteps.shape} -> embedding {out.shape}")
        assert out.shape == (B, dim), f"Expected ({B}, {dim}), got {out.shape}"

    # Test that different timesteps give different embeddings
    t1 = torch.tensor([0])
    t2 = torch.tensor([500])
    t3 = torch.tensor([999])

    e1 = embed(t1)
    e2 = embed(t2)
    e3 = embed(t3)

    assert not torch.allclose(e1, e2), "Different timesteps should give different embeddings"
    assert not torch.allclose(e2, e3), "Different timesteps should give different embeddings"

    # Test that same timestep gives same embedding (deterministic)
    e1_again = embed(t1)
    assert torch.allclose(e1, e1_again), "Same timestep should give same embedding"

    print("  Determinism: PASSED")
    print("  Uniqueness: PASSED")
    print("SinusoidalTimeEmbed test passed!")


def test_point_embed():
    """Test point coordinate embedding."""
    print("\n" + "=" * 60)
    print("TEST: PointEmbed")
    print("=" * 60)

    out_dim = 256
    embed = PointEmbed(out_dim=out_dim)

    # Test different input shapes
    test_cases = [
        (4, 100, 3),      # [B, N, 3]
        (2, 50, 3),       # [B, N, 3]
        (8, 1000, 3),     # [B, N, 3] larger
    ]

    for shape in test_cases:
        x = torch.randn(shape)
        out = embed(x)
        expected_shape = shape[:-1] + (out_dim,)
        print(f"  {shape} -> {out.shape}")
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    # Test that it handles arbitrary leading dimensions
    x = torch.randn(2, 10, 50, 3)  # [B, T, N, 3]
    out = embed(x)
    assert out.shape == (2, 10, 50, out_dim)
    print(f"  (2, 10, 50, 3) -> {out.shape}")

    print("PointEmbed test passed!")


def test_simple_point_embed():
    """Test simple (single linear) point embedding."""
    print("\n" + "=" * 60)
    print("TEST: SimplePointEmbed")
    print("=" * 60)

    out_dim = 256
    embed = SimplePointEmbed(out_dim=out_dim)

    x = torch.randn(4, 100, 3)
    out = embed(x)

    print(f"  (4, 100, 3) -> {out.shape}")
    assert out.shape == (4, 100, out_dim)

    # Check it's a single linear (no hidden layer)
    assert len(list(embed.parameters())) == 2, "Should have only weight and bias"

    print("SimplePointEmbed test passed!")


def test_rotary_1d():
    """Test 1D rotary position encoding."""
    print("\n" + "=" * 60)
    print("TEST: RotaryPositionEncoding (1D)")
    print("=" * 60)

    dim = 64  # Typical head dimension
    rope = RotaryPositionEncoding(dim=dim)

    # Test basic shapes
    B, N = 4, 100
    positions = torch.arange(N).unsqueeze(0).expand(B, -1).float()  # [B, N]

    cos, sin = rope(positions)

    print(f"  positions: {positions.shape}")
    print(f"  cos: {cos.shape}, sin: {sin.shape}")

    assert cos.shape == (B, N, dim), f"Expected ({B}, {N}, {dim}), got {cos.shape}"
    assert sin.shape == (B, N, dim), f"Expected ({B}, {N}, {dim}), got {sin.shape}"

    # Test that cos/sin are bounded
    assert cos.abs().max() <= 1.0, "Cosine should be in [-1, 1]"
    assert sin.abs().max() <= 1.0, "Sine should be in [-1, 1]"

    # Test apply_rotary
    x = torch.randn(B, N, dim)
    x_rotated = RotaryPositionEncoding.apply_rotary(x, cos, sin)
    assert x_rotated.shape == x.shape, "Output shape should match input"
    print(f"  apply_rotary: {x.shape} -> {x_rotated.shape}")

    # Test that rotation is not identity (unless position is 0)
    assert not torch.allclose(x, x_rotated), "Rotation should change the tensor"

    # Test with 3D input [B, N, 1]
    positions_3d = positions.unsqueeze(-1)
    cos_3d, _ = rope(positions_3d)
    assert cos_3d.shape == (B, N, dim)

    print("RotaryPositionEncoding (1D) test passed!")


def test_rotary_3d():
    """Test 3D rotary position encoding."""
    print("\n" + "=" * 60)
    print("TEST: RotaryPositionEncoding3D")
    print("=" * 60)

    dim = 96  # Should be divisible by 6
    rope = RotaryPositionEncoding3D(dim=dim)

    B, N = 4, 100
    xyz = torch.randn(B, N, 3)  # Random 3D coordinates

    cos, sin = rope(xyz)

    print(f"  xyz: {xyz.shape}")
    print(f"  cos: {cos.shape}, sin: {sin.shape}")
    print(f"  feature_dim: {rope.feature_dim}, pad_length: {rope.pad_length}")

    assert cos.shape == (B, N, dim), f"Expected ({B}, {N}, {dim}), got {cos.shape}"
    assert sin.shape == (B, N, dim), f"Expected ({B}, {N}, {dim}), got {sin.shape}"

    # Test apply_rotary
    x = torch.randn(B, N, dim)
    x_rotated = RotaryPositionEncoding3D.apply_rotary(x, cos, sin)
    assert x_rotated.shape == x.shape

    # Test with non-divisible dimension
    dim_odd = 100  # Not divisible by 6
    rope_odd = RotaryPositionEncoding3D(dim=dim_odd)
    cos_odd, sin_odd = rope_odd(xyz)

    print(f"  dim={dim_odd}: feature_dim={rope_odd.feature_dim}, pad_length={rope_odd.pad_length}")
    assert cos_odd.shape == (B, N, dim_odd)
    assert sin_odd.shape == (B, N, dim_odd)

    # Check padding region
    if rope_odd.pad_length > 0:
        # Cos padding should be 1, sin padding should be 0
        assert torch.allclose(cos_odd[..., -rope_odd.pad_length:], torch.ones_like(cos_odd[..., -rope_odd.pad_length:]))
        assert torch.allclose(sin_odd[..., -rope_odd.pad_length:], torch.zeros_like(sin_odd[..., -rope_odd.pad_length:]))
        print("  Padding values: CORRECT (cos=1, sin=0)")

    print("RotaryPositionEncoding3D test passed!")


def test_rotary_relative_position():
    """Test that rotary embeddings preserve relative position information."""
    print("\n" + "=" * 60)
    print("TEST: Rotary Relative Position Property")
    print("=" * 60)

    dim = 64
    rope = RotaryPositionEncoding(dim=dim)

    # Create positions with known relative distances
    pos1 = torch.tensor([[0, 1, 2, 3]]).float()  # Positions 0, 1, 2, 3
    pos2 = torch.tensor([[10, 11, 12, 13]]).float()  # Shifted by 10

    cos1, sin1 = rope(pos1)
    cos2, sin2 = rope(pos2)

    # Create queries and keys
    q = torch.randn(1, 4, dim)
    k = torch.randn(1, 4, dim)

    # Apply rotary to both
    q1 = RotaryPositionEncoding.apply_rotary(q, cos1, sin1)
    k1 = RotaryPositionEncoding.apply_rotary(k, cos1, sin1)

    q2 = RotaryPositionEncoding.apply_rotary(q, cos2, sin2)
    k2 = RotaryPositionEncoding.apply_rotary(k, cos2, sin2)

    # Compute attention scores (dot product)
    # For positions i and j, the dot product depends on (i - j)
    attn1 = torch.einsum('bqd,bkd->bqk', q1, k1)  # [1, 4, 4]
    attn2 = torch.einsum('bqd,bkd->bqk', q2, k2)  # [1, 4, 4]

    # The diagonal (self-attention) and off-diagonals should be similar
    # because relative distances are the same
    print(f"  Attention scores (pos 0-3):\n{attn1[0][:2, :2]}")
    print(f"  Attention scores (pos 10-13):\n{attn2[0][:2, :2]}")

    # They should be approximately equal (relative position is preserved)
    assert torch.allclose(attn1, attn2, atol=1e-5), "Relative position should be preserved"

    print("  Relative position property: PRESERVED")
    print("Rotary relative position test passed!")


def test_sincos_pos_embed_1d():
    """Test 1D sinusoidal positional embedding function."""
    print("\n" + "=" * 60)
    print("TEST: get_sincos_pos_embed_1d")
    print("=" * 60)

    num_positions = 100
    dim = 256

    pe = get_sincos_pos_embed_1d(num_positions, dim)

    print(f"  Shape: {pe.shape}")
    assert pe.shape == (num_positions, dim)

    # Check that different positions have different embeddings
    assert not torch.allclose(pe[0], pe[1])
    assert not torch.allclose(pe[0], pe[-1])

    # Check that embeddings are bounded (sin/cos in [-1, 1])
    assert pe.abs().max() <= 1.0

    print("get_sincos_pos_embed_1d test passed!")


def test_embedding_gradients():
    """Test that embeddings have proper gradients for training."""
    print("\n" + "=" * 60)
    print("TEST: Embedding Gradients")
    print("=" * 60)

    # Time embedding
    time_embed = SinusoidalTimeEmbed(dim=256)
    t = torch.randint(0, 1000, (4,))
    out = time_embed(t)
    loss = out.sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in time_embed.parameters())
    print(f"  SinusoidalTimeEmbed has gradients: {has_grad}")
    assert has_grad, "Time embedding should have gradients"

    # Point embedding
    point_embed = PointEmbed(out_dim=256)
    x = torch.randn(4, 100, 3, requires_grad=True)
    out = point_embed(x)
    loss = out.sum()
    loss.backward()

    has_grad = any(p.grad is not None for p in point_embed.parameters())
    print(f"  PointEmbed has gradients: {has_grad}")
    assert has_grad, "Point embedding should have gradients"
    assert x.grad is not None, "Input should have gradient"

    print("Embedding gradients test passed!")


if __name__ == "__main__":
    test_sinusoidal_time_embed()
    test_point_embed()
    test_simple_point_embed()
    test_rotary_1d()
    test_rotary_3d()
    test_rotary_relative_position()
    test_sincos_pos_embed_1d()
    test_embedding_gradients()

    print("\n" + "=" * 60)
    print("ALL EMBEDDING TESTS PASSED!")
    print("=" * 60)
