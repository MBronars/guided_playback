"""Receding-horizon (closed-loop) guided block pushing on a paper PushBoundary scenario.

Receding horizon: frame 0 is always the episode start (the scenario); the executed frames 0..c from SAPIEN enter the
sampler as clean rows (gp/adapters/clean_core_prefix.py) and only frames c+1..--total are generated, so every plan ends
at the episode's last frame. Windows never shrink: near the end the window slides back over executed frames.
Loop: guide a population, pick the member with the best imagined objective, commit its next --commit rod actions,
execute ALL committed actions from the start in SAPIEN (deterministic set-pose rod, so re-simulating the prefix ==
running on), read the executed block poses and rod xy back as the new history, replan. The reported number is the
SIMULATED block-centroid distance to the target after the last committed action.

  python bin/closed_loop_push.py --url http://localhost:8797/guide --scenario <ep npz> --name cl_rand42_ep011 [--commit 10]
"""
import argparse, glob, json, os, subprocess, time, urllib.request
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PW = "/data/user_data/mbronars/packages/PointWorld"; SIM_PY = "/data/user_data/mbronars/.miniconda3/envs/planning_wrapper/bin/python"
BC = np.array([-0.135, 0.0])                                        # boundary centre (gp/push_scenes.py)
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--scenario", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--commit", type=int, default=10); ap.add_argument("--total", type=int, default=60); ap.add_argument("--transitions", type=int, default=60)
ap.add_argument("--members", type=int, default=3); ap.add_argument("--template", default="0"); ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--iterations", type=int, default=6); ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--t_freeze", type=float, default=0.5)
ap.add_argument("--context", type=int, default=3, help="executed transitions given as clean context (at most)"); ap.add_argument("--min_plan", type=int, default=26, help="generated transitions per plan never fewer than this (the trained window)")
ap.add_argument("--pad_tail", action="store_true", help="every model call reads the full window: schedule --min_plan extra noise rows past the plan and stop at the plan end; frame 0 goes through the same sampler")
ap.add_argument("--optimizer", default="unitgrad"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--replay_gpu", default="0")
ap.add_argument("--no_warm", action="store_true"); ap.add_argument("--runs", default=os.path.join(HERE, "runs")); ap.add_argument("--extra", default="{}", help="JSON merged into every /guide body (e.g. prefilter)")
A = ap.parse_args()
OUT = os.path.join(A.runs, "pushing", A.name); os.makedirs(os.path.join(OUT, "states"), exist_ok=True); os.makedirs(os.path.join(OUT, "plans"), exist_ok=True)
ep = np.load(A.scenario); init = {k: ep[k].astype(np.float64) for k in ("block_pos", "block_quat", "gripper_xy", "target_xy")}
state = dict(init); committed = np.zeros((0, 2)); rows = []; t_start = time.time(); hist = None


def post(body):
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def replay(actions, tag):
    """execute `actions` (rod xy deltas, m) from the episode start in SAPIEN; returns the replay npz."""
    pd = os.path.join(OUT, "plans", tag); os.makedirs(pd, exist_ok=True)
    np.savez(os.path.join(pd, "actions.npz"), actions=actions.astype(np.float32), init_block_pos=init["block_pos"].astype(np.float32), init_block_quat=init["block_quat"].astype(np.float32),
             init_gripper_xy=init["gripper_xy"].astype(np.float32), target_xy=init["target_xy"].astype(np.float32),
             imagined_block_xy=np.zeros((len(actions) + 1, 2), np.float32), imagined_gripper_xy=np.zeros((len(actions) + 1, 2), np.float32),
             final_dist=np.float32(0), wall_time_sec=np.float32(0), action_max_norm=np.float32(0.015))
    with open(os.path.join(pd, "replay.log"), "w") as log:
        subprocess.run([SIM_PY, os.path.join(PW, "replay_actions.py"), os.path.join(OUT, "plans"), "--glob", tag, "--use_set_pose", "--no_video"], check=True, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(A.replay_gpu)))
    f = glob.glob(os.path.join(pd, "replay_set_pose", "*.npz")); assert f, f"no replay under {pd}"
    return np.load(f[0])


r = 0; warm = None
while committed.shape[0] < A.total:
    # 1. the executed history 0..c (world frame) -- the scenario itself stays frame 0
    c = committed.shape[0]; hp = None
    if c > 0 or A.pad_tail:
        hp = os.path.join(OUT, "states", f"history_c{c}.npz")
        H = hist if hist is not None else dict(block_pos=init["block_pos"][None].astype(np.float32), block_quat=init["block_quat"][None].astype(np.float32), gripper_xy=init["gripper_xy"][None].astype(np.float32))
        np.savez(hp, block_pos=H["block_pos"], block_quat=H["block_quat"], gripper_xy=H["gripper_xy"])
    dist_now = float(np.linalg.norm(state["block_pos"][:2] - state["target_xy"]) * 1e3)
    k = min(c, A.context); n_new = max(A.total - c, A.min_plan); n_plan = k + n_new; goal = k + (A.total - c)   # rollout frame 0 = executed frame c - k
    # 2. guided plan from that state
    name = f"{A.name}_r{r}"; t0 = time.time()
    res = post(dict(domain="pushing", task="push", episodes=",".join([A.template] * A.members), transitions=n_plan + (A.min_plan if A.pad_tail else 0), history_stop_at=(n_plan if A.pad_tail else None), steps=A.steps, churn=True,
                    optimizer=A.optimizer, shell=True, t_freeze=A.t_freeze, scenario=A.scenario, history=hp, history_context=A.context, history_full_window=False, goal_frame=goal, iterations=A.iterations, lr=A.lr, lr_final=None, seed=A.seed,
                    name=name, runs=A.runs, save_best_noise=True, **({"init_noise": warm} if (warm and not A.no_warm) else {}), **json.loads(A.extra), desc=f"closed loop replan {r}: {c} executed, context {k}, {n_new} generated (task frames {c + 1}..{c + n_new}), reward at task frame {A.total} = rollout frame {goal}, block {dist_now:.0f} mm from target"))
    t_guide = time.time() - t0
    pick = int(np.argmin(res["best_mm"]))
    bn = os.path.join(A.runs, "pushing", name, "best_noise.npz")                # warm start: plan row j <-> task transition (c - k) + j
    if os.path.exists(bn):
        m_ = min(A.commit, A.total - c); c_next = c + m_; k_next = min(c_next, A.context)
        v = np.load(bn)["v0"][pick]; shift = (c_next - k_next) - (c - k)          # rows of this plan that fall at or after the next anchor
        warm = os.path.join(A.runs, "pushing", name, "warm_next.npz"); np.savez(warm, P0=v[shift + k_next:], start=k_next)
    z = np.load(os.path.join(A.runs, "pushing", name, "rollout.npz")); rod = z["roles"][pick] == 0
    rod_xy = z["predicted"][pick][:, rod].mean(1)[:, :2].astype(np.float64)          # rollout frames 0..n_plan; frame k = executed frame c
    new = np.diff(rod_xy, axis=0)[k:k + min(A.commit, A.total - c)]
    committed = np.concatenate([committed, new])
    # 3. execute everything committed so far from the start, read the state back
    S = replay(committed, f"r{r}")
    hist = dict(block_pos=S["block_pos"].astype(np.float32), block_quat=S["block_quat"].astype(np.float32), gripper_xy=S["gripper_xy"].astype(np.float32))
    state = dict(block_pos=S["block_pos"][-1].astype(np.float64), block_quat=S["block_quat"][-1].astype(np.float64), gripper_xy=S["gripper_xy"][-1].astype(np.float64), target_xy=init["target_xy"])
    dist_after = float(np.linalg.norm(state["block_pos"][:2] - state["target_xy"]) * 1e3)
    rows.append(dict(replan=r, run=name, context=k, generated=n_new, goal_rollout_frame=goal, committed_before=int(committed.shape[0] - len(new)), committed_after=int(committed.shape[0]), block_to_target_before_mm=round(dist_now, 1),
                     imagined_best_mm=[round(v, 1) for v in res["best_mm"]], picked_member=pick, block_to_target_after_mm=round(dist_after, 1), guide_seconds=round(t_guide, 1), plan_seconds=round(float(sum(res.get("iteration_seconds", []))) + float(json.load(open(os.path.join(A.runs, "pushing", name, "result.json"))).get("prefilter_seconds", 0) or 0), 2),
                     rod_step_mm_max=round(float(np.linalg.norm(new, axis=1).max() * 1e3), 1)))
    print(json.dumps(rows[-1]), flush=True)
    r += 1
final = dict(name=A.name, scenario=A.scenario, commit=A.commit, total=A.total, mode=("tail padded, " if A.pad_tail else "") + f"closed loop: context min(c,{A.context}) executed transitions, plan max(remaining,{A.min_plan}), reward at task frame {A.total}", planner=dict(members=A.members, steps=A.steps, iterations=A.iterations, lr=A.lr, t_freeze=A.t_freeze, optimizer=A.optimizer, shell=True, churn=True, seed=A.seed),
             sim_final_mm=rows[-1]["block_to_target_after_mm"], target_mm=round(float(np.linalg.norm(init["block_pos"][:2] - init["target_xy"]) * 1e3), 1),
             replans=len(rows), guide_seconds_total=round(sum(x["guide_seconds"] for x in rows), 1), plan_seconds_total=round(sum(x["plan_seconds"] for x in rows), 1), wall_seconds=round(time.time() - t_start, 1), rows=rows)
json.dump(final, open(os.path.join(OUT, "summary.json"), "w"), indent=1); np.save(os.path.join(OUT, "committed_actions.npy"), committed)
print("FINAL", json.dumps({k: v for k, v in final.items() if k != "rows"}), flush=True)
