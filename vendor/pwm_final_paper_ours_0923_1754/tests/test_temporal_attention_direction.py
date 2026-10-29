"""Temporal attention direction is a config choice on both retained backbones.

Causal (transformer `causal_temporal=True`; UNet default block-causal mask):
a noisy-state perturbation on the LAST future frame must not reach any earlier
frame's output. Bidirectional (transformer `causal_temporal=False`; UNet
`unet_bidirectional_temporal=True`): it must. The last frame is perturbed so
that the delta->position integration (a forward cumsum) cannot carry the
perturbation to earlier frames by itself.

Known actions are held fixed here. Their access policy differs: the Transformer
can read the full supplied plan even in causal mode, while the causal UNet can
read actions only through the predicted frame, including that frame's action.
"""

import pytest
import torch

from models.config import ModelConfig
from models.transformer import TrajectoryTransformer
from models.unet_std import UNetWrapper
from tests.test_unet_std_pose_native import _model_cfg, _noisy_pose_gt, make_batch


def _transformer_cfg(causal):
    return ModelConfig(
        dim=32, depth=2, heads=4, num_latents=24, k_neighbors=8, dropout=0,
        context_len=3, all_relative=True, num_mat_types=6, pose_native=True,
        use_stratified_encoding=True, use_stratified_decoding=True,
        bypass_compression=False, per_frame_knn=True, concat_anchor_z=True,
        embed_actuatable=True, causal_temporal=causal)


def _build(architecture, bidirectional):
    if architecture == "transformer":
        model = TrajectoryTransformer(_transformer_cfg(causal=not bidirectional))
    else:
        model = UNetWrapper(_model_cfg(unet_bidirectional_temporal=bidirectional))
    torch.manual_seed(29417)
    model = model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.025)
    return model


def _outputs(model, batch, noisy, noisy_pose):
    times = torch.full(noisy.shape[:2], 500, dtype=torch.long)
    context = batch.delta_stats.normalize(batch.context_deltas)
    context_times = torch.zeros(context.shape[:2], dtype=torch.long)
    with torch.no_grad():
        output = model(batch, noisy, times, noisy_context=context,
                       context_timesteps=context_times, noisy_pose=noisy_pose)
    return output if isinstance(output, tuple) else (output, None)


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("bidirectional", [False, True])
def test_last_frame_perturbation_reaches_earlier_frames_iff_bidirectional(architecture, bidirectional):
    batch = make_batch()
    model = _build(architecture, bidirectional)
    torch.manual_seed(7293)
    noisy = torch.randn_like(batch.future_deltas)
    noisy_pose = _noisy_pose_gt(batch) + 0.15
    base_particles, base_pose = _outputs(model, batch, noisy, noisy_pose)
    bumped = noisy.clone()
    bumped[:, -1] += 0.5
    bumped_pose = noisy_pose.clone()
    bumped_pose[:, -1] += 0.5
    particles, pose = _outputs(model, batch, bumped, bumped_pose)
    assert base_particles is not None and base_pose is not None
    earlier = slice(None, -1)
    if bidirectional:
        assert not torch.allclose(base_particles[:, earlier], particles[:, earlier])
        assert not torch.allclose(base_pose[:, earlier], pose[:, earlier])
    else:
        torch.testing.assert_close(base_particles[:, earlier], particles[:, earlier], rtol=0, atol=0)
        torch.testing.assert_close(base_pose[:, earlier], pose[:, earlier], rtol=0, atol=0)
    # the perturbed frame itself always changes
    assert not torch.allclose(base_particles[:, -1], particles[:, -1])


def test_unet_bidirectional_and_fully_causal_are_exclusive():
    model = UNetWrapper(_model_cfg(unet_bidirectional_temporal=True, unet_causal_context=True)).eval()
    batch = make_batch()
    noisy = torch.randn_like(batch.future_deltas)
    with pytest.raises(AssertionError, match="exclusive"):
        _outputs(model, batch, noisy, _noisy_pose_gt(batch))
