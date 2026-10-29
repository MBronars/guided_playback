"""Retained long MPM sequences preserve contracts at autoregressive boundaries."""
from dataclasses import fields

import pytest
import torch

from data.normalization import NormalizationStats
from data.test_batch import LongTrajectoryBatch


def sequence():
    positions = torch.arange(2 * 10 * 4 * 3).reshape(2, 10, 4, 3).float()
    stats = NormalizationStats(-torch.ones(3), torch.ones(3))
    return LongTrajectoryBatch(positions=positions, actions=positions[:, 1:] - positions[:, :-1],
        action_mask=torch.ones(2, 4, dtype=torch.bool), object_ids=torch.zeros(2, 4, dtype=torch.long),
        valid_lengths=torch.tensor([3, 4]), E=torch.ones(2, 4), nu=torch.ones(2, 4), rho=torch.ones(2, 4),
        mat_type=torch.ones(2, 4, dtype=torch.long), delta_stats=stats, position_stats=stats)


@pytest.mark.parametrize("operation", ["to", "slice"])
def test_all_long_batch_fields_survive_transform(operation):
    original = sequence()
    result = original.to("cpu") if operation == "to" else original.slice(1)
    for field in fields(original):
        before, after = getattr(original, field.name), getattr(result, field.name)
        if torch.is_tensor(before):
            torch.testing.assert_close(after, before if operation == "to" else before[:1], atol=0, rtol=0)
        else:
            assert after is before


@pytest.mark.parametrize("pad", [True, False])
def test_partial_last_window_actions_match_actual_future(pad):
    original = sequence()
    result = original.get_window(start_frame=5, context_len=3, future_len=4, pad_to_length=pad)
    assert result.positions.shape[1] == (7 if pad else 5)
    assert result.actions.shape[1] == (4 if pad else 2)
    torch.testing.assert_close(result.actions[:, :2], original.actions[:, 7:9], atol=0, rtol=0)
    if pad:
        assert torch.count_nonzero(result.actions[:, 2:]) == 0
        torch.testing.assert_close(result.future[:, -1], original.positions[:, -1], atol=0, rtol=0)


def test_autoregressive_context_uses_predictions_and_preserves_gt_future():
    original = sequence()
    predictions = original.positions + 17
    result = original.get_window(3, 3, 2, predicted_positions=predictions)
    torch.testing.assert_close(result.positions[:, :3], predictions[:, 3:6], atol=0, rtol=0)
    torch.testing.assert_close(result.future, original.positions[:, 6:8], atol=0, rtol=0)


def test_padded_window_can_extend_past_gt_with_available_predicted_context():
    original = sequence()
    predictions = original.positions[:, -1:].expand(-1, 16, -1, -1).clone()
    result = original.get_window(11, 3, 2, predicted_positions=predictions, pad_to_length=True)
    assert result.positions.shape[1] == 5 and result.actions.shape[1] == 2
    assert torch.count_nonzero(result.actions) == 0


@pytest.mark.parametrize("kwargs", [{"start_frame": -1}, {"context_len": 0}, {"future_len": 0},
                                    {"start_frame": 9}, {"start_frame": 7}])
def test_invalid_windows_fail_clearly(kwargs):
    options = dict(start_frame=0, context_len=3, future_len=4)
    options.update(kwargs)
    with pytest.raises(ValueError):
        sequence().get_window(**options)
