"""SAPIEN PushBoundary replay of a block-pushing (push0418) rollout: PointWorld/replay_actions.py in the
planning_wrapper env, --use_set_pose (the rod is teleported to the generated rod xy each step), env-default block
physics (user decision 2026-09-22).

Frame: the 04_18 data is centred per episode; the sim places the cube at the boundary centre with the recorded
frame-0 yaw, and everything (rod, goal) is shifted by the same xy translation. Output contract: cube = frame-0
generated cube particles under the simulated block pose; rod = frame-0 generated rod particles + simulated rod xy shift."""
import glob, json, os, subprocess
import numpy as np
from . import common as C

PW = "/data/user_data/mbronars/packages/PointWorld"
PY = "/data/user_data/mbronars/.miniconda3/envs/planning_wrapper/bin/python"
BOUNDARY_CENTER = np.array([-0.135, 0.0]); CUBE_Z_SPAWN = 0.025 + 1e-3


def run(run_dir, out, variants=("guided", "unguided"), gpu="0"):
    z, rj = C.load_run(run_dir); os.makedirs(out, exist_ok=True)
    B, T, N = z["predicted"].shape[:3]; V = len(variants); roles = z["roles"]; obj_mask = z["object_mask"]
    plans = os.path.abspath(os.path.join(out, "plans")); os.makedirs(plans, exist_ok=True)
    meta = {}
    for m in range(B):
        cube = obj_mask[m]; rod = roles[m] == 0
        for v, var in enumerate(variants):
            P = z[C.VARIANT_KEY[var]][m].astype(np.float64)
            c0 = P[0, cube].mean(0); shift = np.array([BOUNDARY_CENTER[0] - c0[0], BOUNDARY_CENTER[1] - c0[1], 0.0])
            ridx = int(np.nonzero(z["pose_action_mask"][m] == False)[0][0])                          # the cube's body index
            R0 = z["body_poses0"][m, ridx, :3, :3]
            rod_xy = P[:, rod].mean(1)[:, :2] + shift[:2]; cube_xy = P[:, cube].mean(1)[:, :2] + shift[:2]
            target_xy = z["target"][m][:2] + shift[:2]
            pd = os.path.join(plans, f"m{m}_{var}"); os.makedirs(pd, exist_ok=True)
            np.savez(os.path.join(pd, "actions.npz"), actions=np.diff(rod_xy, axis=0).astype(np.float32),
                     init_block_pos=np.array([cube_xy[0, 0], cube_xy[0, 1], CUBE_Z_SPAWN], np.float32), init_block_quat=C.mat_to_quat(R0).astype(np.float32),
                     init_gripper_xy=rod_xy[0].astype(np.float32), target_xy=target_xy.astype(np.float32), imagined_block_xy=cube_xy.astype(np.float32),
                     imagined_gripper_xy=rod_xy.astype(np.float32), final_dist=np.float32(np.linalg.norm(cube_xy[-1] - target_xy)), wall_time_sec=np.float32(0), action_max_norm=np.float32(0.015))
            meta[(m, var)] = dict(shift=shift, c0=c0, dir=pd)
    log = open(os.path.join(out, "replay.log"), "w")
    print(f"[sapien] replaying {B} members x {V} variants ({plans})", flush=True)
    if os.environ.get("GP_REUSE_SIM") and glob.glob(os.path.join(plans, "m*", "replay_set_pose", "*.npz")):
        print("[sapien] reusing existing replays", flush=True)
    else:
        subprocess.run([PY, os.path.join(PW, "replay_actions.py"), plans, "--glob", "m*", "--use_set_pose", "--no_video"], check=True, stdout=log, stderr=subprocess.STDOUT,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)))
    sim = np.zeros((V, B, T, N, 3), np.float32); rows = []
    for m in range(B):
        cube = obj_mask[m]; rod = roles[m] == 0
        for v, var in enumerate(variants):
            info = meta[(m, var)]; P = z[C.VARIANT_KEY[var]][m].astype(np.float64); shift = info["shift"]
            f = glob.glob(os.path.join(info["dir"], "replay_set_pose", "*.npz"))
            assert f, f"no replay npz under {info['dir']}/replay_set_pose"
            S = np.load(f[0]); bp, bq, gxy = S["block_pos"], S["block_quat"], S["gripper_xy"]              # [T', ...] step 0 = start
            Tk = min(T, bp.shape[0])
            ridx = int(np.nonzero(z["pose_action_mask"][m] == False)[0][0]); R0 = z["body_poses0"][m, ridx, :3, :3]
            loc = (P[0, cube] - P[0, cube].mean(0)) @ R0                                                    # cube-local
            Rt = C.qmat(bq[:Tk]); cube_sim = np.einsum("tij,pj->tpi", Rt, loc) + bp[:Tk][:, None] - shift   # back to the model frame
            rod_sim = P[0, rod][None] + np.concatenate([gxy[:Tk] - gxy[0], np.zeros((Tk, 1))], 1)[:, None]
            dst = sim[v, m]; dst[:Tk, cube] = cube_sim; dst[:Tk, rod] = rod_sim
            if Tk < T: sim[v, m, Tk:] = sim[v, m, Tk - 1]
            row = C.analysis_row(P, sim[v, m], roles[m], cube, z["target"][m], m, var); row.update(sim_steps=int(bp.shape[0]))
            rows.append(row)
    C.write(out, sim, roles, list(variants), rows, extra=dict(sim="sapien PushBoundary, set_pose rod, env-default block physics"))
