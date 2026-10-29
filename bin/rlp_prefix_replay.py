"""Closed-loop rl_push execution helper: replay committed COMMANDED gripper trajectories of several problems in one Isaac
launch (gp/replay/isaac_float.py via bin/replay.py), and turn the executed states into model-frame history.

pack(out_dir, items): items = list of dict(loader, target [3], cmd [c+1, N, 3] commanded clouds (gripper rows matter; object
rows only at frame 0), roles, object_mask, particle_body, pose_action_mask, joint_type, body_type, object_ids, body_poses0 [K,4,4]).
Writes a run folder with B = len(items) members; replay(out_dir) runs Isaac; executed(out_dir) -> [B, c+1, N, 3] sim clouds.
history(sim [c+1, N, 3], item) -> dict(positions [c+1, N, 3], body_poses [c+1, K, 4, 4]) in the model frame: body pose at t =
Kabsch(frame-0 rows of the body -> its rows at t) @ body_poses0; frame 0 is the commanded (= scenario) frame 0 exactly."""
import json, os, subprocess, sys
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
KEYS = ("roles", "object_mask", "particle_body", "pose_action_mask", "joint_type", "body_type", "object_ids")


def pack(out, items, name=None):
    os.makedirs(out, exist_ok=True); B = len(items); T = items[0]["cmd"].shape[0]
    assert all(it["cmd"].shape[0] == T for it in items), "lockstep: every problem must have the same committed length"
    P = np.stack([it["cmd"] for it in items]).astype(np.float32)
    for b, it in enumerate(items):                                            # object rows: frame 0 held (only frame 0 is used by the replay)
        om = np.asarray(it["object_mask"]).astype(bool); P[b][:, om] = P[b][0, om][None]
    arr = {k: np.stack([np.asarray(it[k]) for it in items]) for k in KEYS}
    np.savez(os.path.join(out, "rollout.npz"), predicted=P, unguided=P, gt=P, gt_frames=np.full(B, T), target=np.stack([it["target"] for it in items]).astype(np.float32),
             offsets=np.zeros((B, 3), np.float32), boxes=np.zeros((0, 6), np.float32), trace_mm=np.zeros((1, B), np.float32), step=np.array(0),
             episodes=np.array([int(it["loader"]) for it in items]), body_poses0=np.stack([it["body_poses0"] for it in items]).astype(np.float32), **arr)
    json.dump(dict(domain="rl_push", task="push", sim="isaac", name=name or os.path.basename(out), best_mm=[0.0] * B, unguided_mm=[0.0] * B,
                   note="committed commanded trajectories packed for one Isaac launch (bin/rlp_prefix_replay.py)", items=[it.get("id") for it in items]),
              open(os.path.join(out, "result.json"), "w"), indent=1)


def replay(out, gpu="1"):
    log = open(os.path.join(out, "replay_stdout.log"), "w")
    subprocess.run([PY, "-u", os.path.join(HERE, "bin", "replay.py"), "--run", out, "--variants", "guided", "--gpu", str(gpu)], cwd=HERE, stdout=log, stderr=subprocess.STDOUT, check=True)


def executed(out):
    return np.load(os.path.join(out, "replay_isaac", "replay.npz"))["sim"][0].astype(np.float64)     # [B, T, N, 3]


def kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0); H = (src - cs).T @ (dst - cd); U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; return R, cd - R @ cs


def history(sim, item):
    pb = np.asarray(item["particle_body"]); K = item["body_poses0"].shape[0]; T = sim.shape[0]
    pos = sim.copy(); pos[0] = item["cmd"][0]
    poses = np.repeat(np.asarray(item["body_poses0"], np.float64)[None], T, 0); res = np.zeros((T, K))
    for k in range(K):
        rows = pb == k; x0 = pos[0][rows]
        for t in range(1, T):
            R, t_ = kabsch(x0, pos[t][rows]); M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t_
            poses[t, k] = M @ poses[0, k]; res[t, k] = np.sqrt((((x0 @ R.T + t_) - pos[t][rows]) ** 2).sum(-1).mean())
    return dict(positions=pos.astype(np.float32), body_poses=poses.astype(np.float32), fit_rms_mm_max=float(res.max() * 1e3))
