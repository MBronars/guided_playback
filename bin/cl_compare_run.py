"""Side-by-side render input for closed vs open loop on one pushing scenario: a run folder whose 'predicted' is the OPEN-loop
SAPIEN execution and whose 'sim' is the CLOSED-loop SAPIEN execution (bin/render.py draws left = predicted, right = sim), plus
one run per method with its own execution in both slots (for --keyframes). Clouds = the scenario's frame-0 generated cube and rod
(from the first plan's rollout.npz) posed by the simulated block pose / rod xy, fixed model frame (world - boundary centre).
  python bin/cl_compare_run.py --open runs/pushing/olpad_X --closed runs/pushing/clpad_X --out runs/pushing/cmp_X"""
import argparse, glob, json, os, shutil
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--open", required=True); ap.add_argument("--closed", required=True); ap.add_argument("--out", required=True)
A = ap.parse_args()


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def final_replay(run):
    rs = sorted(glob.glob(os.path.join(run, "plans", "r*")), key=lambda p: int(os.path.basename(p)[1:]))
    return np.load(glob.glob(os.path.join(rs[-1], "replay_set_pose", "*.npz"))[0])


r0 = A.open.rstrip("/") + "_r0"; z = np.load(os.path.join(r0, "rollout.npz"))
res0 = json.load(open(os.path.join(r0, "result.json"))); m = int(np.argmin(res0["best_mm"]))
P0 = z["predicted"][m, 0].astype(np.float64); cube = z["object_mask"][m].astype(bool); rod = z["roles"][m] == 0; target = z["target"][m]
c0 = P0[cube].mean(0)


def cloud(S):
    bp, bq, g = S["block_pos"].astype(np.float64), S["block_quat"].astype(np.float64), S["gripper_xy"].astype(np.float64)
    T = bp.shape[0]; Q = np.zeros((T,) + P0.shape); R0 = qmat(bq[0])
    for t in range(T):
        dR = qmat(bq[t]) @ R0.T
        Q[t][cube] = (P0[cube] - c0) @ dR.T + c0 + (bp[t] - bp[0]); Q[t][rod] = P0[rod] + np.array([g[t, 0] - g[0, 0], g[t, 1] - g[0, 1], 0.0])
        Q[t][~(cube | rod)] = P0[~(cube | rod)]
    return Q.astype(np.float32), float(np.linalg.norm(bp[-1, :2] - S["target_xy"]) * 1e3)


Qo, do = cloud(final_replay(A.open)); Qc, dc = cloud(final_replay(A.closed))
keys = {k: z[k][m:m + 1] for k in ("roles", "object_mask", "particle_body", "pose_action_mask", "joint_type", "body_type", "object_ids", "body_poses0")}


def write(out, left, right, label):
    os.makedirs(os.path.join(out, "replay_sapien"), exist_ok=True)
    np.savez(os.path.join(out, "rollout.npz"), predicted=left[None], unguided=left[None], gt=left[None], gt_frames=np.array([left.shape[0]]), target=target[None],
             offsets=np.zeros((1, 3), np.float32), boxes=np.zeros((0, 6), np.float32), trace_mm=np.zeros((1, 1), np.float32), episodes=np.array([0]), step=np.array(0), **keys)
    np.savez(os.path.join(out, "replay_sapien", "replay.npz"), sim=right[None, None], sim_roles=keys["roles"], variants=np.array(json.dumps(["guided"])))
    json.dump(dict(domain="pushing", task="push", sim="sapien", name=os.path.basename(out), layout=label, open_final_mm=round(do, 1), closed_final_mm=round(dc, 1),
                   open_run=os.path.abspath(A.open), closed_run=os.path.abspath(A.closed)), open(os.path.join(out, "result.json"), "w"), indent=1)


write(A.out, Qo, Qc, "left = OPEN-loop execution, right = CLOSED-loop execution (both SAPIEN, same camera)")
write(A.out + "_open", Qo, Qo, "open-loop execution (SAPIEN)"); write(A.out + "_closed", Qc, Qc, "closed-loop execution (SAPIEN)")
print(f"{A.out}: open {do:.1f} mm, closed {dc:.1f} mm")
