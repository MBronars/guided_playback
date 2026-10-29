"""Tests for utilities."""

import torch
import torch.nn as nn
from utils.ema import EMA


def test_ema_basic():
    """Test EMA basic functionality."""
    print("\n" + "=" * 60)
    print("TEST: EMA Basic")
    print("=" * 60)

    # Simple model
    model = nn.Linear(10, 10)
    ema = EMA(model, decay=0.9)

    # Store initial weights
    initial_weight = model.weight.data.clone()

    # Simulate training step
    model.weight.data += 1.0

    # Update EMA
    ema.update()

    # Shadow should be: 0.9 * initial + 0.1 * (initial + 1)
    expected = 0.9 * initial_weight + 0.1 * (initial_weight + 1.0)
    assert torch.allclose(ema.shadow['weight'], expected), "EMA update incorrect"
    print("EMA update: PASSED")

    print("EMA basic test passed!")


def test_ema_apply_restore():
    """Test EMA apply_shadow and restore."""
    print("\n" + "=" * 60)
    print("TEST: EMA Apply/Restore")
    print("=" * 60)

    model = nn.Linear(10, 10)
    ema = EMA(model, decay=0.9)

    # Store original weights
    original_weight = model.weight.data.clone()

    # Simulate training
    model.weight.data += 1.0
    ema.update()

    # Store training weights
    training_weight = model.weight.data.clone()

    # Apply shadow (for eval)
    ema.apply_shadow()
    shadow_weight = model.weight.data.clone()

    # Model should have shadow weights now
    assert torch.allclose(model.weight.data, ema.shadow['weight']), "apply_shadow failed"
    print("apply_shadow: PASSED")

    # Restore original
    ema.restore()

    # Model should have training weights back
    assert torch.allclose(model.weight.data, training_weight), "restore failed"
    print("restore: PASSED")

    print("EMA apply/restore test passed!")


def test_ema_state_dict():
    """Test EMA state dict for checkpointing."""
    print("\n" + "=" * 60)
    print("TEST: EMA State Dict")
    print("=" * 60)

    model = nn.Linear(10, 10)
    ema = EMA(model, decay=0.9)

    # Simulate training
    for _ in range(5):
        model.weight.data += 0.1
        ema.update()

    # Save state
    state = ema.state_dict()

    # Create new EMA and load
    ema2 = EMA(model, decay=0.9)
    ema2.load_state_dict(state)

    # Should match
    assert torch.allclose(ema.shadow['weight'], ema2.shadow['weight']), "state_dict mismatch"
    print("state_dict save/load: PASSED")

    print("EMA state dict test passed!")


if __name__ == "__main__":
    test_ema_basic()
    test_ema_apply_restore()
    test_ema_state_dict()

    print("\n" + "=" * 60)
    print("ALL UTILS TESTS PASSED!")
    print("=" * 60)
