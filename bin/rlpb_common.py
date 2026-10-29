"""Shared pieces for the rl_push baselines (GNS, PointWorld) in the open/closed-loop comparison. numpy only, so the
PointWorld / GNS environments can import it too.

Frame-0 problem state (the SAME start our planner used): an npz with positions0 [N, 3] model frame, roles, object_mask,
particle_body, body_poses0, pose_action_mask, joint_type, body_type, object_ids, target [3] (keys as in a guided_playback
rollout.npz; a rollout.npz is accepted too: member 0, frame 0).
Gripper <-> hand pose: the rl_push data's own construction (reports/rlpush_limits/make_rollout.gripper_points,
link npz pwm_final_paper/data/embodiments/panda_gripper_pickup.npz), fitted by Kabsch on the 241 hand rows plus one
scalar per finger along the hand y axis (as gns-pytorch fit_hand)."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/data/user_data/mbronars/reports/rlpush_limits")
from make_rollout import gripper_points, qmat, LINKS, FINGER_Z          # noqa: E402

KEYS = ("roles", "object_mask", "particle_body", "pose_action_mask", "joint_type", "body_type", "object_ids")
_L = np.load(LINKS); HAND, LEFT, RIGHT = (_L[k].astype(np.float64) for k in ("panda_hand", "panda_leftfinger", "panda_rightfinger"))


def load_state(path, target=None):
    z = np.load(path, allow_pickle=True)
    if "positions0" in z.files:
        s = dict(P0=z["positions0"].astype(np.float64), body_poses0=z["body_poses0"], **{k: z[k] for k in KEYS if k in z.files})
        s["target"] = z["target"].astype(np.float64) if "target" in z.files else None
    else:                                                                    # a rollout.npz: member 0, frame 0
        s = dict(P0=z["predicted"][0, 0].astype(np.float64), body_poses0=z["body_poses0"][0], **{k: z[k][0] for k in KEYS})
        s["target"] = z["target"][0].astype(np.float64)
    if target is not None: s["target"] = np.asarray(target, np.float64)
    if "object_ids" not in s: s["object_ids"] = (np.asarray(s["roles"]) != 2).astype(np.int64)
    return s


def kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0); H = (src - cs).T @ (dst - cd); U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; return R, cd - R @ cs


def mat_to_quat(R):
    m = np.asarray(R, np.float64); t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2; return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def fit_hand(P, pbody):
    """one frame [N, 3] -> (R [3,3], t [3], quat wxyz [4], finger_q [2]); pbody: 1 hand, 2 left finger, 3 right finger."""
    P = np.asarray(P, np.float64); R, t = kabsch(HAND, P[pbody == 1])
    ql = float(np.mean((((P[pbody == 2] - t) @ R) - (LEFT + [0, 0, FINGER_Z]))[:, 1]))
    qr = -float(np.mean((((P[pbody == 3] - t) @ R) - (RIGHT + [0, 0, FINGER_Z]))[:, 1]))
    return R, t, mat_to_quat(R), np.array([ql, qr])


def cmd_frames(P0, roles, hand_pos, hand_quat, finger_q):
    """commanded clouds [T, N, 3]: gripper rows from the hand track (make_rollout order: hand, left, right = our row order),
    object rows held at frame 0 (the replay reads only frame 0 of them)."""
    T = hand_pos.shape[0]; out = np.repeat(np.asarray(P0, np.float64)[None], T, 0)
    out[:, np.asarray(roles) != 2] = gripper_points(hand_pos, hand_quat, finger_q)
    return out


def check_layout(s):
    """frame-0 gripper rows rebuilt from the fitted hand pose, max error mm (validates row order + fit)."""
    R, t, q, fq = fit_hand(s["P0"], np.asarray(s["particle_body"]))
    rebuilt = gripper_points(t, q, fq); return float(np.abs(rebuilt - s["P0"][np.asarray(s["roles"]) != 2]).max() * 1e3), (R, t, q, fq)
