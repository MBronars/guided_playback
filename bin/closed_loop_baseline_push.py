"""Open vs closed loop for the block-pushing BASELINES (GNS best-of-N, PointWorld imagined MPPI) on a paper PushBoundary
scenario, SAPIEN set-pose execution. commit = total -> open loop (one plan, all executed). commit < total -> closed loop:
plan the REMAINING actions from the executed state, commit `commit` actions, execute everything committed from the start
in SAPIEN (deterministic set-pose), read the executed block pose / rod xy back, replan. Target unchanged.
  GNS: bin/gns_push_plan_from_history.py (context = last 6 executed frames, front-padded with frame 0; 256 candidates)
  PW : PointWorld/imagined_mppi.py from the current executed state (paper May settings: K 128, dynamic horizon base 10 cap 4,
       replan every 5 inside the model, action cap 6 mm, noise 0.004, temperature 0.3, seed 0)
  python bin/closed_loop_baseline_push.py --method gns|pw --ep 11 --commit 10 --name gns_cl_hard42_ep011"""
import argparse, glob, json, os, subprocess, time
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PW = "/data/user_data/mbronars/packages/PointWorld"
SIM_PY = "/data/user_data/mbronars/.miniconda3/envs/planning_wrapper/bin/python"; GNS_PY = "/data/user_data/mbronars/.miniconda3/envs/gns_torch/bin/python"
PW_PY = "/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python"
ap = argparse.ArgumentParser(); ap.add_argument("--method", required=True, choices=("gns", "pw", "pw60")); ap.add_argument("--ep", type=int, required=True)
ap.add_argument("--commit", type=int, default=10); ap.add_argument("--total", type=int, default=60); ap.add_argument("--name", required=True)
ap.add_argument("--difficulty", default="hard"); ap.add_argument("--seed_dir", default="seed_42"); ap.add_argument("--gpu", default="0"); ap.add_argument("--pw_url", default="http://127.0.0.1:5190")
A = ap.parse_args()
OUT = os.path.join(HERE, "runs", "pushing_baselines", A.name); os.makedirs(os.path.join(OUT, "plans"), exist_ok=True); os.makedirs(os.path.join(OUT, "exec"), exist_ok=True)
ALL = os.path.join(PW, "scenarios", A.difficulty, A.seed_dir, "all.npz"); z = np.load(ALL)
init = {k: z[k][A.ep].astype(np.float32) for k in ("block_pos", "block_quat", "gripper_xy", "target_xy")}
CAP = 0.015 if A.method == "gns" else 0.006
WCAP, PFD = ("6", "1e-9") if A.method == "pw60" else ("4", "0.015")   # pw60: every replan rolls out to task frame 60 and scores it (N = 6 windows always, H_active = remaining)
env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(A.gpu))


def execute(actions, tag):
    d = os.path.join(OUT, "exec", tag); os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "actions.npz"), actions=actions.astype(np.float32), init_block_pos=init["block_pos"], init_block_quat=init["block_quat"], init_gripper_xy=init["gripper_xy"],
             target_xy=init["target_xy"], imagined_block_xy=np.zeros((len(actions) + 1, 2), np.float32), imagined_gripper_xy=np.zeros((len(actions) + 1, 2), np.float32),
             final_dist=np.float32(0), wall_time_sec=np.float32(0), action_max_norm=np.float32(CAP))
    with open(os.path.join(d, "replay.log"), "w") as log:
        subprocess.run([SIM_PY, os.path.join(PW, "replay_actions.py"), os.path.join(OUT, "exec"), "--glob", tag, "--use_set_pose", "--no_video"], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return glob.glob(os.path.join(d, "replay_set_pose", "*.npz"))[0]


def plan(c, hist_path, tag):
    d = os.path.join(OUT, "plans", tag); t0 = time.time()
    if A.method == "gns":
        cmd = [GNS_PY, os.path.join(HERE, "bin", "gns_push_plan_from_history.py"), "--state_npz", ALL, "--scenario_idx", str(A.ep), "--history", hist_path or "none",
               "--total_frames", str(A.total), "--out", d]
    else:
        os.makedirs(d, exist_ok=True); st = os.path.join(d, "state.npz")
        if hist_path is None: np.savez(st, **init)
        else:
            h = np.load(hist_path); np.savez(st, block_pos=h["block_pos"][-1], block_quat=h["block_quat"][-1], gripper_xy=h["gripper_xy"][-1], target_xy=init["target_xy"])
        cmd = [PW_PY, "-u", os.path.join(PW, "imagined_mppi.py"), "--state_npz", st, "--server_url", A.pw_url, "--output_dir", d, "--total_steps", str(A.total - c),
               "--population_size", "128", "--replan_every", "5", "--dynamic_horizon", "--horizon_base", "10", "--window_cap", WCAP, "--per_frame_dist", PFD, "--noise_sigma", "0.004",
               "--temperature", "0.3", "--action_max_norm", "0.006", "--seed", "0", "--quiet"]
    with open(d + ".log", "w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=PW)
    secs = time.time() - t0
    P = np.load(os.path.join(d, "plan.npz" if A.method == "gns" else "actions.npz"))
    return P["actions"].astype(np.float64), float(P["final_dist"]), secs, P


committed = np.zeros((0, 2)); rows = []; hist = None; t_all = time.time(); r = 0
while committed.shape[0] < A.total:
    c = committed.shape[0]; hp = None
    if hist is not None:
        hp = os.path.join(OUT, "plans", f"history_c{c}.npz"); np.savez(hp, **hist)
    before = float(np.linalg.norm((hist["block_pos"][-1] if hist is not None else init["block_pos"])[:2] - init["target_xy"]) * 1e3)
    acts, imag, secs, P = plan(c, hp, f"r{r}")
    assert acts.shape[0] == A.total - c, f"plan has {acts.shape[0]} actions for {A.total - c} remaining"
    new = acts[:min(A.commit, A.total - c)]; committed = np.concatenate([committed, new])
    t0 = time.time(); f = execute(committed, f"c{committed.shape[0]}"); t_exec = time.time() - t0
    S = np.load(f); hist = dict(block_pos=S["block_pos"].astype(np.float32), block_quat=S["block_quat"].astype(np.float32), gripper_xy=S["gripper_xy"].astype(np.float32))
    after = float(np.linalg.norm(S["block_pos"][-1, :2] - init["target_xy"]) * 1e3)
    rows.append(dict(replan=r, executed_before=c, executed_after=int(committed.shape[0]), planned_mm=round(imag * 1e3, 1), block_before_mm=round(before, 1), block_after_mm=round(after, 1),
                     plan_seconds=round(secs, 1), exec_seconds=round(t_exec, 1), plan_dir=os.path.join(OUT, "plans", f"r{r}"), sim=f))
    print(json.dumps(rows[-1]), flush=True); r += 1
summ = dict(name=A.name, method=A.method, mode="open" if A.commit >= A.total else "closed", scenario=f"{A.difficulty} {A.seed_dir} ep {A.ep}", commit=A.commit, total=A.total,
            final_mm=rows[-1]["block_after_mm"], planned_mm=rows[0]["planned_mm"], start_mm=rows[0]["block_before_mm"], plan_seconds_total=round(sum(x["plan_seconds"] for x in rows), 1),
            wall_seconds=round(time.time() - t_all, 1), final_sim=rows[-1]["sim"], rows=rows)
json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1); print("FINAL", json.dumps({k: v for k, v in summ.items() if k != "rows"}), flush=True)
