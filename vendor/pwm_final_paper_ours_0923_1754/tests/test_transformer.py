"""Tests for TrajectoryTransformer."""

import torch
from data.batch import TrajectoryBatch
from data.dataset import NormalizationStats
from models.transformer import TrajectoryTransformer, ModelConfig


def create_dummy_batch(
    B: int = 2,
    T: int = 10,
    N: int = 256,
    context_len: int = 2,
    device: str = "cpu",
) -> TrajectoryBatch:
    """Create a dummy TrajectoryBatch for testing."""
    F = T - context_len

    # Random positions
    positions = torch.randn(B, T, N, 3, device=device)

    # Random actions (only for gripper particles)
    actions = torch.randn(B, F, N, 3, device=device)

    # Action mask: first 10% of particles are grippers
    num_grippers = max(1, N // 10)
    action_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    action_mask[:, :num_grippers] = True

    # Valid lengths (all particles valid for simplicity)
    valid_lengths = torch.full((B,), N, dtype=torch.long, device=device)

    # Object ids: single object type (stratified encoding defaults off)
    object_ids = torch.zeros(B, N, dtype=torch.long, device=device)

    # Identity normalization stats (min=-1, max=1 => denormalize is a no-op),
    # required by the model forward to map normalized targets back to raw deltas.
    ident = NormalizationStats(
        min_val=torch.tensor([-1.0, -1.0, -1.0], device=device),
        max_val=torch.tensor([1.0, 1.0, 1.0], device=device),
    )

    # Physics parameters
    E = torch.rand(B, N, device=device) * 1e6  # Young's modulus
    nu = torch.rand(B, N, device=device) * 0.5  # Poisson's ratio
    rho = torch.rand(B, N, device=device) * 1000  # Density

    return TrajectoryBatch(
        positions=positions,
        actions=actions,
        action_mask=action_mask,
        object_ids=object_ids,
        valid_lengths=valid_lengths,
        E=E,
        nu=nu,
        rho=rho,
        context_len=context_len,
        delta_stats=ident,
        position_stats=ident,
    )


def test_model_config():
    """Test ModelConfig dataclass."""
    print("\n" + "=" * 60)
    print("TEST: ModelConfig")
    print("=" * 60)

    # Default config
    cfg = ModelConfig()
    print(f"Default config: dim={cfg.dim}, depth={cfg.depth}, heads={cfg.heads}")
    assert cfg.dim == 256
    assert cfg.depth == 4
    assert cfg.heads == 8

    # Custom config
    cfg = ModelConfig(dim=128, depth=2, heads=4)
    print(f"Custom config: dim={cfg.dim}, depth={cfg.depth}, heads={cfg.heads}")
    assert cfg.dim == 128
    assert cfg.depth == 2

    print("ModelConfig test passed!")


def test_transformer_forward():
    """Test TrajectoryTransformer forward pass."""
    print("\n" + "=" * 60)
    print("TEST: TrajectoryTransformer Forward")
    print("=" * 60)

    # Small config for testing
    cfg = ModelConfig(
        dim=64,
        depth=2,
        heads=4,
        num_latents=32,
        k_neighbors=8,
        dropout=0.0,
        context_len=2,
    )

    model = TrajectoryTransformer(cfg)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create dummy batch
    B, T, N = 2, 10, 64
    batch = create_dummy_batch(B=B, T=T, N=N, context_len=cfg.context_len)
    print(f"Batch: {batch}")

    # Noisy future (what diffusion provides)
    F = batch.num_future_frames
    noisy_future = torch.randn(B, F, N, 3)

    # Timestep
    timestep = torch.randint(0, 1000, (B,))

    print(f"noisy_future: {noisy_future.shape}")
    print(f"timestep: {timestep.shape}")

    # Forward pass
    out = model(batch, noisy_future, timestep)
    print(f"Output: {out.shape}")

    assert out.shape == (B, F, N, 3), f"Expected {(B, F, N, 3)}, got {out.shape}"

    print("TrajectoryTransformer forward test passed!")


def test_transformer_gradients():
    """Test that gradients flow through the model."""
    print("\n" + "=" * 60)
    print("TEST: TrajectoryTransformer Gradients")
    print("=" * 60)

    cfg = ModelConfig(
        dim=64,
        depth=2,
        heads=4,
        num_latents=32,
        k_neighbors=8,
        dropout=0.0,
        context_len=2,
    )

    model = TrajectoryTransformer(cfg)

    B, T, N = 2, 10, 64
    batch = create_dummy_batch(B=B, T=T, N=N, context_len=cfg.context_len)

    F = batch.num_future_frames
    noisy_future = torch.randn(B, F, N, 3, requires_grad=True)
    timestep = torch.randint(0, 1000, (B,))

    # Forward
    out = model(batch, noisy_future, timestep)

    # Backward
    loss = out.sum()
    loss.backward()

    # Check gradients
    assert noisy_future.grad is not None, "Gradients should flow to input"
    print("Input gradient: PASSED")

    # Check model gradients
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Model should have gradients"
    print("Model gradients: PASSED")

    print("TrajectoryTransformer gradients test passed!")


def test_transformer_determinism():
    """Test that model is deterministic."""
    print("\n" + "=" * 60)
    print("TEST: TrajectoryTransformer Determinism")
    print("=" * 60)

    cfg = ModelConfig(
        dim=64,
        depth=2,
        heads=4,
        num_latents=32,
        k_neighbors=8,
        dropout=0.0,  # No dropout for determinism
        context_len=2,
    )

    model = TrajectoryTransformer(cfg)
    model.eval()  # Ensure eval mode

    B, T, N = 2, 10, 64
    batch = create_dummy_batch(B=B, T=T, N=N, context_len=cfg.context_len)

    F = batch.num_future_frames
    noisy_future = torch.randn(B, F, N, 3)
    timestep = torch.randint(0, 1000, (B,))

    # Two forward passes
    with torch.no_grad():
        out1 = model(batch, noisy_future.clone(), timestep.clone())
        out2 = model(batch, noisy_future.clone(), timestep.clone())

    assert torch.allclose(out1, out2), "Model should be deterministic"
    print("Determinism: PASSED")

    print("TrajectoryTransformer determinism test passed!")


if __name__ == "__main__":
    test_model_config()
    test_transformer_forward()
    test_transformer_gradients()
    test_transformer_determinism()

    print("\n" + "=" * 60)
    print("ALL TRANSFORMER TESTS PASSED!")
    print("=" * 60)
