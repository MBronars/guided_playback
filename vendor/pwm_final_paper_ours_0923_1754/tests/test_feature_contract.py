"""Ground truth travels in a batch for loss/metrics, not as a future model input."""
from dataclasses import replace

import pytest
import torch

from models.transformer import ModelConfig, TrajectoryTransformer
from models.unet_std import UNetWrapper
from tests.test_unet_std_pose_native import _model_cfg, _noisy_pose_gt, make_batch


@pytest.mark.parametrize("architecture", ["classic", "unet_std"])
@pytest.mark.parametrize("scene", ["deformable", "mixed", "rigid"])
def test_passive_future_ground_truth_is_not_a_feature(architecture, scene):
    torch.manual_seed(273)
    options = {"deformable": dict(rigid=(), sidecar=False), "mixed": {},
               "rigid": dict(deform_n=0, rigid=((1, 24), (2, 16)))}[scene]
    batch = make_batch(**options)
    if scene == "deformable":
        batch.mat_type.zero_()
    if architecture == "unet_std":
        model = UNetWrapper(_model_cfg())
    else:
        model = TrajectoryTransformer(ModelConfig(
            dim=64, depth=2, heads=4, num_latents=24, k_neighbors=8, dropout=0,
            context_len=batch.context_len, all_relative=True, num_mat_types=6,
            bypass_compression=False,
            pose_native=True, use_stratified_encoding=True, use_stratified_decoding=True))
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.025)

    changed = replace(batch, positions=batch.positions.clone(),
                      body_poses=None if batch.body_poses is None else batch.body_poses.clone(),
                      pose_deltas_gt=None if batch.pose_deltas_gt is None else batch.pose_deltas_gt.clone())
    # Context, actions, templates, physics, normalized noisy input and times stay fixed.
    changed.positions[:, batch.context_len:] += (~batch.action_mask)[:, None, :, None] * 0.123
    if changed.body_poses is not None:
        passive = ~batch.pose_action_mask
        changed.body_poses[:, batch.context_len:, :, :3, 3] += passive[:, None, :, None] * 0.123
        changed.pose_deltas_gt += passive[:, None, :, None] * 0.456
    assert not torch.equal(batch.future, changed.future)
    noisy = torch.randn_like(batch.future_deltas)
    times = torch.full(noisy.shape[:2], 500, dtype=torch.long)
    kwargs = {} if scene == "deformable" else {"noisy_pose": _noisy_pose_gt(batch)}
    with torch.no_grad():
        original = model(batch, noisy, times, **kwargs)
        perturbed = model(changed, noisy, times, **kwargs)
        different_noise = model(batch, noisy + 0.2, times,
                                **({"noisy_pose": kwargs["noisy_pose"] + 0.2} if kwargs else {}))
    def tensors(result):
        return [x for x in (result if isinstance(result, tuple) else (result,)) if x is not None]
    for actual, expected in zip(tensors(perturbed), tensors(original), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The equality above must not be a vacuous check of zero/constant heads.
    assert any(not torch.equal(a, b) for a, b in zip(tensors(original), tensors(different_noise), strict=True))
