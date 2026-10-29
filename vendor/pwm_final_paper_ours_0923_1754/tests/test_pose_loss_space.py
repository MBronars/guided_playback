"""train.pose_loss_space: the rigid pose term of a sample-parameterized (x0) model.

"velocity" (default) is the historical recipe: the clean pose output is converted to a velocity and compared with the
velocity target, which weights a clean pose error by 1/t^2. "sample" compares clean pose with clean pose, flat in t like
the particle term of the same mode. The default's exact parity with the frozen objective is tests/test_objective_parity.py.
"""
import pytest
import torch
from omegaconf import OmegaConf

from diffusion import TrajectoryFlowMatching
from tests.test_model_scope import _construct
from tests.test_unet_std_pose_native import make_batch
from utils.config import resolve_config
from utils.trajectory_objective import trajectory_loss_options, trajectory_objective

SCENES = {"mixed": {}, "rigid": dict(deform_n=0, rigid=((1, 24), (2, 16)))}


def _objective(architecture, scene, space, prediction_type="sample", seed=4711):
    batch = make_batch(**SCENES[scene])
    torch.manual_seed(7)
    model = _construct(architecture, False)
    fm = TrajectoryFlowMatching(prediction_type=prediction_type)
    torch.manual_seed(seed)
    options = {} if space is None else dict(pose_loss_space=space)
    loss, metrics, draw = trajectory_objective(model, batch, fm, diffusion_forcing=True, clip_deltas=False,
                                               clip_pose_deltas=False, **options)
    return model, fm, loss, metrics, draw


def _masked_mean(square, draw):
    mask = draw.get("pose_mask")
    if mask is None:
        return square.mean()
    return (square * mask).sum() / (mask.sum() * square.shape[0] * square.shape[1])


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("scene", ["mixed", "rigid"])
def test_sample_space_pose_loss_is_the_flat_clean_pose_mse(architecture, scene):
    _, _, loss, metrics, draw = _objective(architecture, scene, "sample")
    assert torch.equal(draw["pose_target"], draw["pose_x0"])              # clean target, not noise - x0
    expected = _masked_mean((draw["pose_prediction"].float() - draw["pose_x0"].float()) ** 2, draw)
    assert metrics["loss_pose_native"] == pytest.approx(float(expected), rel=1e-6)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("scene", ["mixed", "rigid"])
def test_the_two_spaces_differ_by_exactly_one_over_t_squared(architecture, scene):
    # Same seed, same weights: identical timesteps, noise and network output; only the comparison space differs.
    _, fm, loss_v, metrics_v, draw_v = _objective(architecture, scene, "velocity")
    _, _, loss_s, metrics_s, draw_s = _objective(architecture, scene, "sample")
    for key in ("timesteps", "x0", "x_t", "pose_x0"):
        assert torch.equal(draw_v[key], draw_s[key])
    time = (draw_s["timesteps"].float() / fm.num_train_timesteps)[:, :, None, None]
    clean_error = (draw_s["pose_prediction"] - draw_s["pose_target"]).float()
    velocity_error = (draw_v["pose_prediction"] - draw_v["pose_target"]).float()
    torch.testing.assert_close(velocity_error, -clean_error / time, rtol=2e-4, atol=1e-5)
    # The particle term and every other term are untouched: the losses differ by the pose terms alone.
    other_v = float(loss_v) - metrics_v["loss_pose_native"]
    other_s = float(loss_s) - metrics_s["loss_pose_native"]
    assert other_v == pytest.approx(other_s, rel=1e-5, abs=1e-6)
    assert metrics_v["loss_pose_native"] > metrics_s["loss_pose_native"]   # 1/t^2 >= 1 on every frame


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
def test_default_is_the_velocity_space_and_a_scene_without_rigid_bodies_ignores_the_option(architecture):
    _, _, loss_default, _, _ = _objective(architecture, "mixed", None)
    _, _, loss_velocity, _, _ = _objective(architecture, "mixed", "velocity")
    assert torch.equal(loss_default, loss_velocity)
    batch = make_batch(rigid=(), sidecar=False)
    batch.mat_type.zero_()
    losses = []
    for space in ("velocity", "sample"):
        torch.manual_seed(7)
        model = _construct(architecture, False)
        torch.manual_seed(4711)
        losses.append(trajectory_objective(model, batch, TrajectoryFlowMatching(prediction_type="sample"),
                                           diffusion_forcing=True, clip_deltas=False, pose_loss_space=space)[0])
    assert torch.equal(losses[0], losses[1])


def test_sample_space_needs_a_sample_parameterized_model():
    with pytest.raises(ValueError, match="requires flow_matching.prediction_type=sample"):
        _objective("unet", "mixed", "sample", prediction_type="velocity")
    with pytest.raises(ValueError, match="must be velocity or sample"):
        _objective("unet", "mixed", "epsilon")


def test_config_carries_and_guards_the_option():
    base = OmegaConf.load("configs/unet.yaml")
    chosen = resolve_config(OmegaConf.merge(base, OmegaConf.create(
        dict(flow_matching=dict(prediction_type="sample"), train=dict(pose_loss_space="sample")))))
    assert trajectory_loss_options(chosen)["pose_loss_space"] == "sample"
    assert trajectory_loss_options(resolve_config(base))["pose_loss_space"] == "velocity"
    with pytest.raises(ValueError, match="requires flow_matching.prediction_type=sample"):
        resolve_config(OmegaConf.merge(base, OmegaConf.create(dict(train=dict(pose_loss_space="sample")))))
    with pytest.raises(ValueError, match="must be velocity or sample"):
        resolve_config(OmegaConf.merge(base, OmegaConf.create(dict(train=dict(pose_loss_space="x0")))))
