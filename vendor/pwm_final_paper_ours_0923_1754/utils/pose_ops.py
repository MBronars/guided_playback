"""SO(3) maps and rotation error for unified world-frame six-channel poses.

Only shared production geometry belongs here. Rigid alignment is separately
implemented in utils.rigid_ops; its arithmetic is not interchangeable with
historical standalone object-pose fitting helpers.
"""
import torch


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Return the skew-symmetric matrix for vectors shaped [..., 3]."""
    zero = torch.zeros_like(v[..., 0])
    x, y, z = v.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(rotvec: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exponential map from axis-angle vectors [..., 3] to rotations [..., 3, 3]."""
    theta = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    theta2 = theta * theta
    K = _skew(rotvec)
    eye = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    eye = eye.expand(*rotvec.shape[:-1], 3, 3)

    small = theta < eps
    sin_over_theta = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta.clamp_min(eps),
    )
    one_minus_cos_over_theta2 = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(eps * eps),
    )
    return eye + sin_over_theta[..., None] * K + one_minus_cos_over_theta2[..., None] * (K @ K)


def so3_log(rotation: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Principal logarithm from rotations [..., 3, 3] to rotation vectors [..., 3].

    Quaternion conversion handles the small-angle limit and half-turns. At
    exactly pi the axis sign is non-unique and the principal log has a branch
    cut. eps is retained for call compatibility; conversion handles its own
    small-angle threshold.
    """
    from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
    return quaternion_to_axis_angle(matrix_to_quaternion(rotation))


def quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Rotation matrices [..., 3, 3] from (w, x, y, z) quaternions [..., 4], normalized here."""
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
    w, x, y, z = quat.unbind(dim=-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def rotation_geodesic_error(pred_rotation: torch.Tensor, target_rotation: torch.Tensor) -> torch.Tensor:
    """Principal rotation error in [0, pi] radians for rotations [..., 3, 3]."""
    from pytorch3d.transforms import matrix_to_quaternion
    delta = pred_rotation.transpose(-1, -2) @ target_rotation
    q = matrix_to_quaternion(delta)
    return 2.0 * torch.atan2(torch.linalg.norm(q[..., 1:], dim=-1), q[..., 0].abs())
