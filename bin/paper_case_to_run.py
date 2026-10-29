"""Turn one May-2026 block-pushing paper episode (positions.npz + replay_set_pose/sim_trajectory.npz) into a
guided_playback run folder (rollout.npz + result.json + replay_sapien/replay.npz) so bin/render.py can draw it.

  python bin/paper_case_to_run.py --episode <ep dir> --out runs/pushing/<name> --method ours|gns|pointworld

Frame 0 of the run = the last context frame of the plan (= step 0 of the SAPIEN replay). World frame (offset_world added).
Simulated cloud = frame-0 cube particles under the simulated block pose, frame-0 rod particles + simulated rod xy shift
(the same contract gp/replay/sapien_push.py writes). PointWorld saved no full imagined cloud: its "predicted" is the
frame-0 template posed by the imagined block xy (frame-0 yaw kept), flagged in result.json.
"""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--episode", required=True); ap.add_argument("--out", required=True); ap.add_argument("--method", default="ours")
A = ap.parse_args()


def qmat_wxyz(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


S = np.load(os.path.join(A.episode, "replay_set_pose", "sim_trajectory.npz"))
act = np.load(os.path.join(A.episode, "actions.npz"))
bp, bq, gxy = S["block_pos"].astype(np.float64), S["block_quat"].astype(np.float64), S["gripper_xy"].astype(np.float64)
T = bp.shape[0]
R0 = qmat_wxyz(act["init_block_quat"].astype(np.float64))
if A.method in ("ours", "gns"):
    z = np.load(os.path.join(A.episode, "positions.npz"))
    ctx = int(z["context_len"]); P = z["predicted_centered"].astype(np.float64) + z["offset_world"].astype(np.float64)
    P = P[ctx - 1:ctx - 1 + T]; ids = z["object_ids"]
    cube, rod = ids == 0, ids == 1
    template_note = None
else:                                                                    # pointworld: no imagined cloud saved; pose a 128-pt cube + 128-pt rod template
    from itertools import product
    g = (np.arange(4) - 1.5) / 3 * 0.05 * 0.8 + 0.0
    cube_loc = np.array(list(product(g, g, g)))[:64]; cube_loc = np.concatenate([cube_loc, cube_loc + np.array([0.004, 0.004, 0.0])])[:128]; cube_loc = cube_loc - cube_loc.mean(0)
    rod_loc = np.stack([np.zeros(128), np.zeros(128), np.linspace(0.0, 0.12, 128)], 1)
    ids = np.array([0] * 128 + [1] * 128); cube, rod = ids == 0, ids == 1
    ib = act["imagined_block_xy"].astype(np.float64); ig = act["imagined_gripper_xy"].astype(np.float64)
    ib, ig = ib[-T:], ig[-T:]
    ib = ib - ib[0] + bp[0, :2]; ig = ig - ig[0] + gxy[0]   # anchor the template at the simulator start pose (frame 0 identical by construction)
    P = np.zeros((T, 256, 3))
    P[:, cube] = (cube_loc @ R0.T)[None] + np.concatenate([ib, np.full((T, 1), 0.026)], 1)[:, None]
    P[:, rod] = rod_loc[None] + np.concatenate([ig, np.zeros((T, 1))], 1)[:, None]
    template_note = "PointWorld stored no imagined particle cloud; predicted = template cube/rod posed by imagined_block_xy / imagined_gripper_xy"
N = P.shape[1]
c0 = P[0, cube].mean(0); loc = (P[0, cube] - c0) @ R0                   # cube-local coordinates
Rt = np.stack([qmat_wxyz(q) for q in bq]); cube_sim = np.einsum("tij,pj->tpi", Rt, loc) + bp[:, None]
cube_sim[:, :, 2] += c0[2] - bp[0, 2]                                    # keep the frame-0 height of the generated cube (same resting height)
rod_sim = P[0, rod][None] + np.concatenate([gxy - gxy[0], np.zeros((T, 1))], 1)[:, None]
sim = np.zeros((1, 1, T, N, 3), np.float32); sim[0, 0][:, cube] = cube_sim; sim[0, 0][:, rod] = rod_sim
assert np.abs(sim[0, 0, 0] - P[0]).max() < 2e-3, f"frame-0 mismatch {np.abs(sim[0, 0, 0] - P[0]).max()}"
roles = np.where(cube, 2, 0).astype(np.int8)[None]; obj_mask = cube[None]
target = np.array([[act["target_xy"][0], act["target_xy"][1], c0[2]]], np.float32)
body_poses0 = np.repeat(np.eye(4)[None, None], 2, 1); body_poses0[0, 0, :3, :3] = R0; body_poses0[0, 0, :3, 3] = c0; body_poses0[0, 1, :3, 3] = P[0, rod].mean(0)
os.makedirs(os.path.join(A.out, "replay_sapien"), exist_ok=True)
np.savez(os.path.join(A.out, "rollout.npz"), predicted=P[None].astype(np.float32), unguided=P[None].astype(np.float32), gt=P[None].astype(np.float32), gt_frames=np.array([T]),
         roles=roles, object_mask=obj_mask, object_ids=ids[None], particle_body=np.where(cube, 0, 1)[None], pose_action_mask=np.array([[False, True]]),
         joint_type=np.zeros((1, 2), np.int64), body_type=np.array([[0, 1]]), body_poses0=body_poses0.astype(np.float32), target=target, offsets=np.zeros((1, 3), np.float32),
         boxes=np.zeros((0, 6), np.float32), trace_mm=np.zeros((1, 1), np.float32), episodes=np.array([os.path.basename(A.episode)]), step=np.array(0))
np.savez(os.path.join(A.out, "replay_sapien", "replay.npz"), sim=sim, sim_roles=roles, variants=np.array(json.dumps(["guided"])))
gen_mm = float(np.linalg.norm(P[-1, cube].mean(0)[:2] - target[0, :2]) * 1e3); sim_mm = float(np.linalg.norm(cube_sim[-1].mean(0)[:2] - target[0, :2]) * 1e3)
json.dump(dict(rows=[dict(member=0, variant="guided", gen_to_target_mm=gen_mm, sim_to_target_mm=sim_mm, sim_final_dist_file_mm=float(S["sim_final_dist"]) * 1e3)], variants=["guided"]),
          open(os.path.join(A.out, "replay_sapien", "analysis.json"), "w"), indent=1)
json.dump(dict(domain="pushing", task="push", sim="sapien", name=os.path.basename(A.out), method=A.method, source_episode=os.path.abspath(A.episode), frames=T, note=template_note,
               gen_to_target_mm=gen_mm, sim_to_target_mm=sim_mm), open(os.path.join(A.out, "result.json"), "w"), indent=1)
print(f"wrote {A.out}: T={T} N={N} gen {gen_mm:.1f} mm sim {sim_mm:.1f} mm (file says {float(S['sim_final_dist'])*1e3:.1f})")
