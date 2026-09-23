"""Style 3: the Franka arm carrying the gripper. Numpy FK + damped-least-squares IK on the 7 revolute joints of the local
franka_gripper URDF (all axes 0 0 1; render_rollout_scene.py's solver), warm-started frame to frame, joint limits clamped.

pickup / rl_push : the hand pose is the Kabsch-fitted panda_hand frame of the generated / simulated gripper rows; the base is
                   the data's robot base (model frame (-0.45, 0, -0.10): panda_link0 at the pool-frame origin, 0.10 m below
                   the table top as recorded).
mould / pushing  : the tool (paddle box / rod cylinder) is HELD by the gripper: hand frame = tool frame @ GRASP (approach axis
                   straight down, fingertips 13 mm below the tool's top edge, fingers closed on its thickness); the base stands
                   on the table 0.50 m from the workspace centre on the side away from the camera, yawed toward it.
The arm is drawn where IK puts it; the max hand tracking error per clip is recorded (renders.json) and printed."""
import os, xml.etree.ElementTree as ET
import numpy as np, trimesh

AD = "/data/user_data/mbronars/packages/IsaacLabEnvs/assets/franka_gripper"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
ARM_LINKS = [f"panda_link{i}" for i in range(8)]
Q_READY = np.array([0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8])


def rpy(r, p_, y):
    return trimesh.transformations.euler_matrix(r, p_, y, "sxyz")


def _T(o):
    T = rpy(*[float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()])
    T[:3, 3] = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    return T


class Arm:
    def __init__(self):
        urdf = ET.parse(os.path.join(AD, "robots/franka_gripper.urdf")).getroot(); self.J = {}
        for j in urdf.findall("joint"):
            ax = j.find("axis"); lim = j.find("limit")
            self.J[j.get("name")] = dict(origin=_T(j.find("origin")), axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None,
                                         limits=(float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None)
        self.LIM = np.array([self.J[j]["limits"] for j in ARM_JOINTS]); self.HAND_T = self.J["panda_hand_joint"]["origin"]
        self.F1, self.F2 = self.J["panda_finger_joint1"], self.J["panda_finger_joint2"]

    def fk(self, q):
        T = np.eye(4); out = [T.copy()]
        for i, jn in enumerate(ARM_JOINTS):
            j = self.J[jn]; T = T @ j["origin"] @ trimesh.transformations.rotation_matrix(q[i], j["axis"]); out.append(T.copy())
        return out, T @ self.HAND_T

    def ik(self, q, T_target, iters=150, lam=0.08):
        for _ in range(iters):
            links, T_h = self.fk(q)
            ep = T_target[:3, 3] - T_h[:3, 3]; Rerr = T_target[:3, :3] @ T_h[:3, :3].T
            ang = np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1))
            er = np.zeros(3) if ang < 1e-9 else ang / (2 * np.sin(ang)) * np.array([Rerr[2, 1] - Rerr[1, 2], Rerr[0, 2] - Rerr[2, 0], Rerr[1, 0] - Rerr[0, 1]])
            if np.linalg.norm(ep) < 1e-5 and abs(ang) < 1e-4:
                break
            Jm = np.zeros((6, 7))
            for i in range(7):
                Ti = links[i] @ self.J[ARM_JOINTS[i]]["origin"]; z = Ti[:3, :3] @ self.J[ARM_JOINTS[i]]["axis"]
                Jm[:3, i] = np.cross(z, T_h[:3, 3] - Ti[:3, 3]); Jm[3:, i] = z
            dq = Jm.T @ np.linalg.solve(Jm @ Jm.T + lam ** 2 * np.eye(6), np.concatenate([ep, er]))
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.LIM[:, 0], self.LIM[:, 1])
        links, T_h = self.fk(q)
        Rerr = T_target[:3, :3] @ T_h[:3, :3].T
        return q, float(np.linalg.norm(T_target[:3, 3] - T_h[:3, 3])), float(np.arccos(np.clip((np.trace(Rerr) - 1) / 2, -1, 1)))

    def solve_first(self, T_target):
        """Frame 0: several seeds, keep the best."""
        best = None
        for seed in (Q_READY, np.array([0.0, 0.2, 0.0, -1.6, 0.0, 1.9, 0.8]), np.array([0.6, -0.4, -0.6, -2.0, 0.3, 1.6, 0.2]), np.array([-0.6, -0.4, 0.6, -2.0, -0.3, 1.6, 1.4]),
                     np.array([0.0, -0.9, 0.0, -2.6, 0.0, 1.8, 0.8])):
            q, ep, er = self.ik(seed.copy(), T_target, iters=400)
            if best is None or ep + 0.1 * er < best[1] + 0.1 * best[2]:
                best = (q, ep, er)
        return best

    def finger_poses(self, T_hand, q_f):
        """world poses of the two finger links for opening q_f (per finger, m)."""
        L = T_hand @ self.F1["origin"] @ trimesh.transformations.translation_matrix(self.F1["axis"] * q_f)
        R = T_hand @ self.F2["origin"] @ trimesh.transformations.translation_matrix(self.F2["axis"] * q_f)
        return L, R


def base_pose(domain, centre, top, away_dir, span=0.0, reach_proj=0.0):
    """world <- panda_link0. Tool domains: the base stands on the table on the side away from the camera, at the distance that
    keeps the farthest tool position (reach_proj = its offset from the centre away from the base) within ~0.62 m of the base."""
    if domain in ("pickup", "rl_push"):
        T = np.eye(4); T[:3, 3] = [-0.45, 0.0, -0.10]; return T
    d = min(0.65, max(0.30, 0.62 - reach_proj))
    p = np.array([centre[0] + d * away_dir[0], centre[1] + d * away_dir[1], top])
    yaw = np.arctan2(centre[1] - p[1], centre[0] - p[0])
    T = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1]); T[:3, 3] = p; return T


def grasp_offset(domain, tool_half_height):
    """tool frame -> panda_hand frame: hand z straight down, origin 0.09 m above the tool's top (fingertips at 0.1034 reach 13 mm down the tool)."""
    G = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]); G[:3, 3] = [0, 0, tool_half_height + 0.09]
    return G
