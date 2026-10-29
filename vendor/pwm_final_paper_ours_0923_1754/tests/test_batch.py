"""Test TrajectoryBatch creation and basic properties."""

import torch
from data.batch import TrajectoryBatch

def create_dummy_batch(
    batch_size: int = 2,
    num_frames: int = 24,
    num_particles: int = 1000,
    context_len: int = 2,
    num_gripper_particles: int = 500,
    device: str = 'cpu',
) -> TrajectoryBatch:
    """
    Create a dummy batch for testing.

    Args:
        batch_size: Number of samples
        num_frames: Total frames (context + future)
        num_particles: Particles per sample
        context_len: Number of context frames
        num_gripper_particles: Number of gripper particles (with actions)
        device: Target device

    Returns:
        TrajectoryBatch with random data
    """
    F = num_frames - context_len

    # Random positions
    positions = torch.randn(batch_size, num_frames, num_particles, 3, device=device)

    # Actions: only for gripper particles, only for future frames
    actions = torch.zeros(batch_size, F, num_particles, 3, device=device)
    actions[:, :, :num_gripper_particles] = torch.randn(
        batch_size, F, num_gripper_particles, 3, device=device
    ) * 0.1  # Small random actions

    # Action mask: True for gripper particles
    action_mask = torch.zeros(batch_size, num_particles, dtype=torch.bool, device=device)
    action_mask[:, :num_gripper_particles] = True

    # All particles valid (no padding)
    valid_lengths = torch.full((batch_size,), num_particles, dtype=torch.long, device=device)

    # Object ids: gripper particles = type 1, object particles = type 0
    object_ids = torch.zeros(batch_size, num_particles, dtype=torch.long, device=device)
    object_ids[:, :num_gripper_particles] = 1

    # Physics parameters (per-particle)
    E = torch.ones(batch_size, num_particles, device=device) * 5.0      # Young's modulus
    nu = torch.ones(batch_size, num_particles, device=device) * 0.3    # Poisson's ratio
    rho = torch.ones(batch_size, num_particles, device=device) * 1.0   # Density

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
    )

def test_batch():
    """Basic creation and property access."""
    batch = create_dummy_batch(
        batch_size=2,
        num_frames=24,
        num_particles=1000,
        context_len=2,
        num_gripper_particles=500,
    )

    # Shapes
    assert batch.context.shape == (2, 2, 1000, 3)
    assert batch.future.shape == (2, 22, 1000, 3)
    assert batch.anchor.shape == (2, 1000, 3)

    # Properties
    assert batch.batch_size == 2
    assert batch.num_frames == 24
    assert batch.num_future_frames == 22

    print(batch)
    print("✓ test_batch passed")


def test_deltas():
    """Test delta computation and reconstruction."""
    batch = create_dummy_batch(
        batch_size=2,
        num_frames=10,
        num_particles=100,
        context_len=2,
        num_gripper_particles=50,
    )

    # Check delta shapes
    assert batch.future_deltas.shape == (2, 8, 100, 3)  # F = 10 - 2 = 8
    assert batch.context_deltas.shape == (2, 1, 100, 3)  # C - 1 = 1

    # Verify delta computation is correct
    # future_deltas[0] should be positions[context_len] - positions[context_len - 1]
    expected_first_delta = batch.positions[:, 2] - batch.positions[:, 1]  # frame 2 - frame 1
    assert torch.allclose(batch.future_deltas[:, 0], expected_first_delta)

    # Verify reconstruction: deltas_to_positions(future_deltas) should equal future
    reconstructed = batch.deltas_to_positions(batch.future_deltas)
    assert torch.allclose(reconstructed, batch.future, atol=1e-6)

    print("✓ test_deltas passed")


if __name__ == "__main__":
    test_batch()
    test_deltas()
