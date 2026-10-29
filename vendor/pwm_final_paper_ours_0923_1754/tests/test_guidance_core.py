"""Guidance Layer 0: the differentiable sampler equals the production sampler,
actuator plans reproduce the batch they came from and carry gradients, and the
objectives have finite, nonzero gradients on both backbones."""

import pytest
import torch

from diffusion.flow_matching import TrajectoryFlowMatching
from guidance import objectives, plan as planlib
from guidance.sampler import draw_noise, sample_window
from inference import run_diffusion_inference_df
from models.config import ModelConfig
from models.transformer import TrajectoryTransformer
from models.unet_std import UNetWrapper
from tests.test_unet_std_pose_native import _model_cfg, make_batch


def _build(architecture):
    if architecture == "transformer":
        cfg = ModelConfig(dim=32, depth=2, heads=4, num_latents=24, k_neighbors=8, dropout=0,
                          context_len=3, all_relative=True, num_mat_types=6, pose_native=True,
                          use_stratified_encoding=True, use_stratified_decoding=True,
                          bypass_compression=False, per_frame_knn=True, concat_anchor_z=True,
                          embed_actuatable=True, causal_temporal=True)
        model = TrajectoryTransformer(cfg)
    else:
        model = UNetWrapper(_model_cfg())
    torch.manual_seed(29417)
    model = model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.025)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
def test_sampler_matches_production_sampler_bitwise(architecture):
    batch, model, diffusion = make_batch(), _build(architecture), TrajectoryFlowMatching()
    torch.manual_seed(1234)
    reference = run_diffusion_inference_df(model, batch, diffusion, num_inference_steps=4)
    torch.manual_seed(1234)
    with torch.no_grad():
        ours = sample_window(model, diffusion, batch, draw_noise(model, batch), steps=4)
    torch.testing.assert_close(ours[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(ours[1], reference[1], rtol=0, atol=0)


def test_plan_from_batch_reproduces_the_batch():
    batch = make_batch()
    same = planlib.apply_plan(batch, planlib.plan_from_batch(batch))
    aidx = planlib.actuated_bodies(batch)
    assert aidx.tolist() == [1, 2]     # the two gripper links of the test layout (body 3 = passive object)
    torch.testing.assert_close(same.pose_actions, batch.pose_actions)
    torch.testing.assert_close(same.body_poses, batch.body_poses, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(same.positions, batch.positions, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(same.actions, batch.actions, rtol=1e-4, atol=1e-4)
    # passive rows are untouched exactly
    passive = ~batch.action_mask
    assert torch.equal(same.positions[:, :, passive[0]], batch.positions[:, :, passive[0]])


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
def test_objectives_have_gradients_to_noise_and_plan(architecture):
    batch, model, diffusion = make_batch(), _build(architecture), TrajectoryFlowMatching()
    deltas = planlib.plan_from_batch(batch).requires_grad_()
    guided = planlib.apply_plan(batch, deltas)
    torch.manual_seed(7)
    x_t, x_pose = draw_noise(model, guided)
    x_t.requires_grad_(); x_pose.requires_grad_()
    x0p, x0pose = sample_window(model, diffusion, guided, (x_t, x_pose), steps=3)
    pred = objectives.materialize(guided, x0p, x0pose)
    C = batch.context_len
    passive = 3                                                   # the rigid object body
    target_t = batch.body_poses[:, C - 1, passive, :3, 3] + 0.05
    objective = objectives.weighted(
        (1.0, objectives.body_target(passive, target_t, batch.body_poses[:, C - 1, passive, :3, :3], rotation_weight=0.1)),
        (1.0, objectives.particle_target(batch.positions[:, -1])),
        (100.0, objectives.phantom_motion(passive, contact_radius=0.02)),
        (10.0, objectives.floor_penetration(0.0)))
    loss, diag = objective(pred, guided)
    assert torch.isfinite(loss)
    assert {"target_distance_mm", "target_rotation_deg", "particle_target_mm", "phantom_mm", "penetration_mm"} <= set(diag)
    loss.backward()
    for name, g in (("noise", x_t.grad), ("pose noise", x_pose.grad), ("plan", deltas.grad)):
        assert g is not None and torch.isfinite(g).all() and torch.count_nonzero(g) > 0, name


def test_zero_plan_keeps_the_actuator_still():
    batch = make_batch()
    still = planlib.apply_plan(batch, torch.zeros_like(planlib.plan_from_batch(batch)))
    C = batch.context_len
    for body in planlib.actuated_bodies(batch).tolist():
        rows = planlib.body_rows(batch, body)
        for f in range(C, batch.num_frames):
            torch.testing.assert_close(still.positions[0, f, rows[0]], batch.positions[0, C - 1, rows[0]], rtol=1e-5, atol=1e-6)
    assert still.actions[:, :, batch.action_mask[0]].abs().max() < 1e-6   # anchor vs frame einsum rounding


@pytest.mark.parametrize("degrees", [179.99, 180, 180.01])
def test_rotation_guidance_cost_and_gradient_near_half_turn(degrees):
    from utils.pose_ops import so3_exp

    angle = torch.tensor(degrees * torch.pi / 180, dtype=torch.float64, requires_grad=True)
    rotation = so3_exp(angle * torch.tensor([1.0, 0.0, 0.0], dtype=angle.dtype))
    if degrees == 180:
        rotation = torch.diag(torch.tensor([1.0, -1.0, -1.0], dtype=angle.dtype))
    translation = torch.zeros(1, 3, dtype=angle.dtype)
    pred = dict(rigid_idx=torch.tensor([0]), R=rotation.reshape(1, 1, 1, 3, 3),
                t=translation.reshape(1, 1, 1, 3))
    objective = objectives.body_target(0, translation, torch.eye(3, dtype=angle.dtype),
                                       rotation_weight=0.3)
    loss, diag = objective(pred, None)
    expected_degrees = min(degrees, 360 - degrees)
    assert diag["target_rotation_deg"] == pytest.approx(expected_degrees, abs=1e-10)
    assert loss.item() == pytest.approx(0.3 * expected_degrees * torch.pi / 180, abs=1e-12)
    if degrees != 180:  # No unique descent direction at the exact half-turn.
        gradient, = torch.autograd.grad(loss, angle)
        assert gradient.item() == pytest.approx(0.3 if degrees < 180 else -0.3, abs=1e-10)
