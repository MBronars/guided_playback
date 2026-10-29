"""Shared SO(3) arithmetic and the active unified world-frame pose contract."""
import math

import pytest
import torch

from utils.pose_ops import so3_exp, so3_log, rotation_geodesic_error


@pytest.mark.parametrize("magnitude", [0.0, 1e-7, 1e-3, 0.2, 1.0])
def test_so3_exp_and_gradients_match_frozen_implementation(magnitude):
    # The logarithm and metric intentionally depart from the frozen half-turn bug.
    from tests.test_model_scope import _reference_module
    reference = _reference_module("utils/pose_ops.py")
    generator = torch.Generator().manual_seed(72)
    rotvec = torch.randn(2, 3, 3, generator=generator) * magnitude
    values = []
    for implementation in (reference, __import__("utils.pose_ops", fromlist=["so3_exp"])):
        value = rotvec.clone().requires_grad_(True)
        rotation = implementation.so3_exp(value)
        weights = torch.arange(1, 10, dtype=rotation.dtype).reshape(3, 3)
        scalar = (rotation * weights).sum()
        scalar.backward()
        values.append((rotation.detach(), value.grad))
    for before, after in zip(*values):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
        assert torch.isfinite(after).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_half_turn_has_pi_error_and_reconstructs_its_rotation(dtype):
    axes = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [1, 2, 3], [1, -1, 0], [-3, 1, 2]], dtype=dtype)
    axes = axes / axes.norm(dim=-1, keepdim=True)
    identity = torch.eye(3, dtype=dtype)
    # A symmetric half-turn has an exactly zero antisymmetric part: the old
    # logarithm returned zero. Rodrigues at floating-point pi can hide this case.
    rotations = 2 * axes[..., :, None] * axes[..., None, :] - identity
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    expected = torch.full((len(axes),), math.pi, dtype=dtype)
    torch.testing.assert_close(rotation_geodesic_error(rotations, identity), expected,
                               rtol=0, atol=tolerance)
    recovered = so3_log(rotations)
    torch.testing.assert_close(recovered.norm(dim=-1), expected, rtol=0, atol=tolerance)
    # At exactly pi the axis sign is ambiguous; the reconstructed matrix is not.
    torch.testing.assert_close(so3_exp(recovered), rotations, rtol=0, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_so3_log_and_error_cover_principal_angles_and_reference_frames(dtype):
    generator = torch.Generator().manual_seed(1818)
    axes = torch.cat([torch.eye(3, dtype=dtype),
                      torch.randn(64, 3, dtype=dtype, generator=generator)])
    axes = axes / axes.norm(dim=-1, keepdim=True)
    angles = torch.tensor([0, 1e-8, 1e-6, 0.01, 0.2, 1, math.pi / 2,
                           math.pi - 1e-3, math.pi - 1e-6, math.pi,
                           math.pi + 1e-6, math.pi + 1e-3, 1.5 * math.pi,
                           2 * math.pi - 1e-6, 2 * math.pi], dtype=dtype)
    rotations = so3_exp(axes[:, None] * angles[None, :, None])
    expected = torch.minimum(angles, 2 * math.pi - angles).expand(len(axes), -1)
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    identity = torch.eye(3, dtype=dtype)
    frame = so3_exp(torch.tensor([0.4, -0.2, 0.1], dtype=dtype))
    for pred, target in ((rotations, identity), (identity, rotations),
                         (frame @ rotations, frame), (rotations @ frame, frame)):
        error = rotation_geodesic_error(pred, target)
        assert error.dtype == dtype and torch.isfinite(error).all()
        assert (error >= 0).all() and (error <= math.pi + tolerance).all()
        torch.testing.assert_close(error, expected, rtol=0, atol=tolerance)
    recovered = so3_log(rotations)
    torch.testing.assert_close(recovered.norm(dim=-1), expected, rtol=0, atol=tolerance)
    torch.testing.assert_close(so3_exp(recovered), rotations, rtol=0, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("degrees", [0.001, 45, 179.99, 180.01, 359.99])
def test_rotation_angle_gradient_on_both_sides_of_pi(dtype, degrees):
    angle = torch.tensor(math.radians(degrees), dtype=dtype, requires_grad=True)
    axis = torch.tensor([1, -2, 3], dtype=dtype)
    axis = axis / axis.norm()
    rotation = so3_exp(angle * axis)
    error = rotation_geodesic_error(rotation, torch.eye(3, dtype=dtype))
    error_grad, = torch.autograd.grad(error, angle, retain_graph=True)
    log_grad, = torch.autograd.grad((so3_log(rotation) * axis).sum(), angle)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    expected = torch.tensor(1 if degrees < 180 else -1, dtype=dtype)
    torch.testing.assert_close(error_grad, expected, rtol=0, atol=tolerance)
    torch.testing.assert_close(log_grad, torch.ones_like(angle), rtol=0, atol=tolerance)


def test_so3_log_and_squared_error_gradcheck_at_identity_and_near_pi():
    rotvec = torch.tensor([[0, 0, 0], [1e-7, -2e-7, 3e-7], [0.2, -0.1, 0.3],
                           [math.pi - 1e-3, 0, 0], [math.pi + 1e-3, 0, 0]],
                          dtype=torch.float64, requires_grad=True)
    def maps(value):
        rotation = so3_exp(value)
        # The squared distance is differentiable at identity. The principal
        # logarithm has a branch cut at pi, so check points on either side.
        error = rotation_geodesic_error(rotation, torch.eye(3, dtype=value.dtype))
        return so3_log(rotation), error.square()
    assert torch.autograd.gradcheck(maps, (rotvec,), eps=1e-6, atol=1e-5, rtol=1e-4)


def test_half_turn_survives_context_and_future_pose_channel_construction():
    from types import SimpleNamespace
    from data.unified_dataset import _augment_pose_channels
    from tests.test_unet_std_pose_native import _pose_stats
    from utils.pose_native import context_pose_deltas

    half_turn = torch.diag(torch.tensor([1.0, -1.0, -1.0]))
    poses = torch.eye(4).repeat(1, 3, 1, 1, 1)
    poses[0, 1, 0, :3, :3] = half_turn
    sidecar = dict(body_poses=poses, particle_body=torch.zeros(1, 1, dtype=torch.long),
                   body_type=torch.ones(1, 1, dtype=torch.long))
    channels = _augment_pose_channels(sidecar, torch.ones(1, 1, dtype=torch.bool),
                                      context_len=2, pose_stats=_pose_stats())
    context = context_pose_deltas(SimpleNamespace(context_len=2, body_poses=poses),
                                 torch.tensor([0]))
    for delta in (context, channels["pose_deltas_gt"], channels["pose_actions"]):
        torch.testing.assert_close(delta[..., :3], torch.zeros(1, 1, 1, 3))
        torch.testing.assert_close(so3_exp(delta[..., 3:]), half_turn.reshape(1, 1, 1, 3, 3),
                                   rtol=0, atol=2e-6)


def test_so3_roundtrip_and_rotation_geometry_for_small_native_increments():
    rotvec = torch.tensor([[0.2, -0.1, 0.3], [0.0, 0.0, 0.1]])
    rotation = so3_exp(rotvec)
    torch.testing.assert_close(so3_log(rotation), rotvec, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rotation.transpose(-1, -2) @ rotation,
                               torch.eye(3).expand_as(rotation), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(torch.linalg.det(rotation), torch.ones(2), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rotation_geodesic_error(rotation, torch.eye(3).expand_as(rotation)),
                               rotvec.norm(dim=-1), rtol=1e-5, atol=1e-6)


def test_native_context_translation_is_world_frame_not_rotated_body_frame():
    from types import SimpleNamespace
    from utils.pose_native import context_pose_deltas
    rotations = so3_exp(torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.1]]))
    poses = torch.eye(4).repeat(1, 2, 1, 1, 1)
    poses[0, :, 0, :3, :3] = rotations
    poses[0, 1, 0, :3, 3] = torch.tensor([0.05, 0.0, 0.0])
    deltas = context_pose_deltas(SimpleNamespace(context_len=2, body_poses=poses), torch.tensor([0]))
    torch.testing.assert_close(deltas[0, 0, 0], torch.tensor([0.05, 0.0, 0.0, 0.0, 0.0, 0.1]),
                               rtol=1e-5, atol=1e-6)
    body_translation = rotations[0].T @ torch.tensor([0.05, 0.0, 0.0])
    assert not torch.allclose(deltas[0, 0, 0, :3], body_translation)
