"""Drive a block-pushing (push0418) rollout from one of the paper's PushBoundary scenarios.

  /data/user_data/mbronars/packages/PointWorld/scenarios/{hard,random}/seed_{0,42}/ep_NNN.npz
  keys: block_pos (3,) block_quat (4, wxyz) gripper_xy (2,) target_xy (2,)   -- all WORLD coordinates

The 04_18 data is centred per episode on the boundary centre, so model frame = world - (BOUNDARY_CENTER, 0).
A scenario is applied by RE-POSING a template episode's own frame-0 clouds, never by sampling new ones: the cube
cloud is de-yawed (the yaw that minimises its xy bounding box; a 0.05 m cube's yaw is only defined mod 90 deg)
and re-yawed to the scenario's yaw, then translated; the rod is a z-cylinder, so it is only translated. The
point distribution the model sees therefore stays exactly the one it was trained on.

Convention kept from gp/replay/sapien_push.py: body_poses0 rotations are identity and the cube's yaw lives in
the point cloud, so the replayed sim spawns the block axis-aligned with the cloud riding along."""
import numpy as np
import torch

BOUNDARY_CENTER = np.array([-0.135, 0.0])
ROD_Z = 0.085


def load(path):
    z = np.load(path)
    return dict(block_pos=z["block_pos"].astype(np.float64), block_quat=z["block_quat"].astype(np.float64),
                gripper_xy=z["gripper_xy"].astype(np.float64), target_xy=z["target_xy"].astype(np.float64))


def quat_yaw(q):
    """Yaw (about z) of a wxyz quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def cloud_yaw(xy, n=9001):
    """The yaw in [0, pi/2) whose removal minimises the xy bounding-box area — the cube's baked-in yaw mod 90 deg."""
    c = xy - xy.mean(0)
    a = np.linspace(0, np.pi / 2, n, endpoint=False)
    ca, sa = np.cos(-a), np.sin(-a)
    rx = c[:, 0][None] * ca[:, None] - c[:, 1][None] * sa[:, None]
    ry = c[:, 0][None] * sa[:, None] + c[:, 1][None] * ca[:, None]
    area = (rx.max(1) - rx.min(1)) * (ry.max(1) - ry.min(1))
    return float(a[int(area.argmin())])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def apply(batch, adapter, scenarios):
    """Re-pose every member of `batch` (frame 0, repeated over T) onto `scenarios[b]`. Returns the target [B, 3].

    The batch must come from adapter.episodes(): positions and body_poses are already frame 0 broadcast over T."""
    dev = batch.positions.device
    cube = adapter.object_mask(batch).cpu().numpy()                     # [B, N] passive object particles
    roles = adapter.roles(batch)                                        # 0 actuator, 2 object, -1 pad
    pos = batch.positions.detach().cpu().numpy().astype(np.float64)     # [B, T, N, 3]
    poses = batch.body_poses.detach().cpu().numpy().astype(np.float64)  # [B, T, K, 4, 4]
    act_body = batch.pose_action_mask.cpu().numpy()                     # [B, K] True = rod
    pbody = batch.particle_body.cpu().numpy()                           # [B, N] body index per particle
    tpl = batch.body_templates.detach().cpu().numpy().astype(np.float64)        # [B, K, P, 3] body-frame points
    tmask = batch.body_template_mask.cpu().numpy()                              # [B, K, P]
    B, T = pos.shape[0], pos.shape[1]
    targets = np.zeros((B, 3))
    for b in range(B):
        s = scenarios[b]
        cm, rm = cube[b], roles[b] == 0
        p0 = pos[b, 0]
        c_tpl = p0[cm].mean(0)
        dyaw = quat_yaw(s["block_quat"]) - cloud_yaw(p0[cm][:, :2])
        cz = float(s["block_pos"][2])
        c_new = np.array([s["block_pos"][0] - BOUNDARY_CENTER[0], s["block_pos"][1] - BOUNDARY_CENTER[1], cz])
        r_new = np.array([s["gripper_xy"][0] - BOUNDARY_CENTER[0], s["gripper_xy"][1] - BOUNDARY_CENTER[1], ROD_Z])
        new = p0.copy()
        new[cm] = (p0[cm] - c_tpl) @ _rot_z(dyaw).T + c_new
        new[rm] = p0[rm] - p0[rm].mean(0) + r_new
        pos[b, :] = new[None]
        for k in range(poses.shape[2]):
            poses[b, :, k, :3, :3] = np.eye(3)
            poses[b, :, k, :3, 3] = r_new if act_body[b, k] else c_new
            # A rigid body is rebuilt as body_pose @ body_template, and the loader's invariant is
            # template == the body's frame-0 cloud minus its centroid (pose identity at frame 0).
            # Re-posing the cloud without this leaves every PREDICTED frame at the template
            # episode's orientation, so the cube visibly snaps on frame 1.
            idx = np.nonzero(pbody[b] == k)[0]
            if idx.size:
                pts = new[idx] - new[idx].mean(0)
                n = min(idx.size, tpl.shape[2])
                tpl[b, k, :n] = pts[:n]
                tpl[b, k, n:] = 0.0
                tmask[b, k, :n] = True
                tmask[b, k, n:] = False
        targets[b] = [s["target_xy"][0] - BOUNDARY_CENTER[0], s["target_xy"][1] - BOUNDARY_CENTER[1], cz]
    batch.positions = torch.tensor(pos, dtype=batch.positions.dtype, device=dev)
    batch.body_poses = torch.tensor(poses, dtype=batch.body_poses.dtype, device=dev)
    batch.body_templates = torch.tensor(tpl, dtype=batch.body_templates.dtype, device=dev)
    batch.body_template_mask = torch.tensor(tmask, dtype=batch.body_template_mask.dtype, device=dev)
    return torch.tensor(targets, dtype=torch.float32, device=dev)
