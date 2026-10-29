"""No-guidance baseline on a paper PushBoundary scenario: MPPI with argmin selection, planned entirely in the world model.

Per replan (c transitions committed so far):
  draw K candidates of the remaining transitions, K/B chunks of B through the warm server's /guide with iterations 0
  (the same churn staircase sampler as the guided runs, no gradient step), each chunk with its own seed; the committed
  frames enter as clean context rows through the prefix sampler (gp/adapters/clean_core_prefix.py) with full_window on,
  so every model call reads the trained window (Cm1 + window rows) -- near the end the window slides back over committed
  frames instead of shrinking. Score = the objective at frame 60 (the episode's last frame). Keep the argmin candidate,
  commit its next --commit transitions, repeat until 60 are committed.
The committed plan is replayed ONCE in SAPIEN from the scenario's own start pose (position and yaw); the reported
number is that simulated block-centroid distance to the target. The imagined history passed back as context is the
winner's own predicted cube (rigid fit) and rod motion, relative to the scenario start.

  python bin/mppi_push_scenario.py --url http://host:8812/guide --scenario <ep npz> --name <run> --K 26 --members 13
"""
import argparse, glob, json, os, shutil, subprocess, sys, time, urllib.request
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from gp.replay.common import mat_to_quat, qmat
PW = "/data/user_data/mbronars/packages/PointWorld"; SIM_PY = "/data/user_data/mbronars/.miniconda3/envs/planning_wrapper/bin/python"
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--scenario", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--K0", type=int, default=0, help="first replan only: this many candidates (rounded up to chunks of --members)"); ap.add_argument("--K", type=int, required=True, help="candidates per replan (multiple of --members)"); ap.add_argument("--members", type=int, default=13)
ap.add_argument("--commit", type=int, default=5); ap.add_argument("--total", type=int, default=60); ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--t_freeze", type=float, default=0.5); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--template", default="0")
ap.add_argument("--replay_gpu", default="0"); ap.add_argument("--no_replay", action="store_true")
ap.add_argument("--ode", action="store_true", help="deterministic Euler sampler (same as guided --ode runs)")
ap.add_argument("--iterations", type=int, default=0, help="guided MPPI: gradient steps per candidate chunk at every replan (0 = plain MPPI)")
ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--optimizer", default="unitgrad")
ap.add_argument("--runs", default=os.path.join(HERE, "runs"), help="root for run folders and the temporary chunk folders (server writes <runs>/pushing/<chunk>)")
A = ap.parse_args()
assert A.K % A.members == 0, "K must be a multiple of --members"
OUT = os.path.join(A.runs, "pushing_mppi", A.name); os.makedirs(os.path.join(OUT, "states"), exist_ok=True)
sc = np.load(A.scenario); init = {k: sc[k].astype(np.float64) for k in ("block_pos", "block_quat", "gripper_xy", "target_xy")}
TMP = os.path.join(A.runs, "pushing")


def post(body):
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def kabsch(P, Q):
    Pc, Qc = P - P.mean(0), Q - Q.mean(0); U, _, Vt = np.linalg.svd(Pc.T @ Qc); d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def world_history(pred, cube, rod, upto):
    """frames 0..upto of one imagined rollout [T, N, 3] (model frame) -> world block_pos/quat, gripper_xy, relative to the scenario start."""
    R0 = qmat(init["block_quat"]); c0, g0 = pred[0, cube].mean(0), pred[0, rod].mean(0)
    bp, bq, gx = [], [], []
    for t in range(upto + 1):
        dR = kabsch(pred[0, cube], pred[t, cube])
        bp.append(init["block_pos"] + (pred[t, cube].mean(0) - c0)); bq.append(mat_to_quat(dR @ R0)); gx.append(init["gripper_xy"] + (pred[t, rod].mean(0) - g0)[:2])
    return dict(block_pos=np.array(bp, np.float32), block_quat=np.array(bq, np.float32), gripper_xy=np.array(gx, np.float32))


hist = None; c = 0; r = 0; log = []; plan_seconds = 0.0; t_wall = time.time(); winner_pred = None
TRASH = []
while c < A.total:
    if c == 0:                                                         # the prefix sampler needs a history: frame 0 = the scenario start
        hist = dict(block_pos=init["block_pos"][None].astype(np.float32), block_quat=init["block_quat"][None].astype(np.float32), gripper_xy=init["gripper_xy"][None].astype(np.float32))
    hp = os.path.join(OUT, "states", f"history_c{c}.npz"); np.savez(hp, **hist)
    best = None; cand = []
    for ch in range(-(-max(A.K0, A.K) // A.members) if c == 0 else A.K // A.members):   # --K0: initial pool at the first replan only
        sd = int(A.seed) * 1_000_003 + r * 1009 + ch; name = f"_mppi_{A.name}_r{r:02d}_c{ch}"
        body = dict(domain="pushing", task="push", episodes=",".join([A.template] * A.members), transitions=A.total, steps=A.steps, churn=not A.ode,
                    optimizer=A.optimizer, shell=True, t_freeze=A.t_freeze, scenario=A.scenario, history=hp, history_context=None, history_full_window=True,
                    iterations=A.iterations, lr=A.lr if A.iterations else 0.0, lr_final=None, seed=sd, name=name, runs=A.runs, desc="MPPI argmin candidate chunk (no guidance)")
        t0 = time.time(); post(body); dt = time.time() - t0
        d = os.path.join(TMP, name); rj = json.load(open(os.path.join(d, "result.json")))
        secs = float(sum(rj.get("iteration_seconds", [])) or dt); plan_seconds += secs
        z = np.load(os.path.join(d, "rollout.npz")); mm = np.array(rj["best_mm"], float)
        k = int(np.argmin(mm)); cand += mm.round(2).tolist()
        if best is None or mm[k] < best[0]:
            best = (float(mm[k]), ch, k, z["predicted"][k].astype(np.float64), z["object_mask"][k], z["roles"][k] == 0, sd)
        shutil.rmtree(d, ignore_errors=True); TRASH.append(d)             # chunk folders are temporary (disk); NFS may hold a file open briefly
    mm, ch, k, pred, cube, rod, sd = best; m = min(A.commit, A.total - c)
    hist = world_history(pred, cube, rod, c + m); winner_pred = pred
    log.append(dict(replan=r, committed_before=c, K=A.K, best_mm=round(mm, 2), chunk=ch, member=k, seed=sd, candidates_mm=cand, seconds_so_far=round(plan_seconds, 2)))
    print(f"[{A.name}] replan {r:2d}  c={c:2d}  best imagined {mm:6.1f} mm of {len(cand)} candidates  (chunk {ch} member {k})  plan s {plan_seconds:6.1f}", flush=True)
    c += m; r += 1

for d in TRASH:
    shutil.rmtree(d, ignore_errors=True)
imag_final = float(np.linalg.norm(hist["block_pos"][-1][:2] - init["target_xy"]) * 1e3)
actions = np.diff(hist["gripper_xy"].astype(np.float64), axis=0)
res = dict(name=A.name, scenario=A.scenario, iterations=A.iterations, lr=A.lr, t_freeze=A.t_freeze, K=A.K, members=A.members, commit=A.commit, total=A.total, steps=A.steps, seed=A.seed, replans=r,
           plan_seconds=round(plan_seconds, 2), wall_seconds=round(time.time() - t_wall, 1), imagined_final_mm=round(imag_final, 2), log=log)
np.savez(os.path.join(OUT, "plan.npz"), actions=actions.astype(np.float32), **{f"imagined_{k}": v for k, v in hist.items()}, winner_last_rollout=winner_pred.astype(np.float32))
if not A.no_replay:
    pd = os.path.join(OUT, "replay"); os.makedirs(pd, exist_ok=True)
    np.savez(os.path.join(pd, "actions.npz"), actions=actions.astype(np.float32), init_block_pos=init["block_pos"].astype(np.float32), init_block_quat=init["block_quat"].astype(np.float32),
             init_gripper_xy=init["gripper_xy"].astype(np.float32), target_xy=init["target_xy"].astype(np.float32),
             imagined_block_xy=hist["block_pos"][:, :2].astype(np.float32), imagined_gripper_xy=hist["gripper_xy"].astype(np.float32),
             final_dist=np.float32(imag_final / 1e3), wall_time_sec=np.float32(plan_seconds), action_max_norm=np.float32(0.015))
    with open(os.path.join(pd, "replay.log"), "w") as lf:
        subprocess.run([SIM_PY, os.path.join(PW, "replay_actions.py"), OUT, "--glob", "replay", "--use_set_pose", "--no_video"], check=True, stdout=lf, stderr=subprocess.STDOUT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(A.replay_gpu)))
    S = np.load(glob.glob(os.path.join(pd, "replay_set_pose", "*.npz"))[0])
    res.update(sim_final_mm=round(float(S["sim_final_dist"]) * 1e3, 2), target_mm=round(float(np.linalg.norm(init["block_pos"][:2] - init["target_xy"]) * 1e3), 2))
json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
print(f"[{A.name}] DONE  plan {plan_seconds:.1f} s  imagined {imag_final:.1f} mm" + (f"  SIM {res['sim_final_mm']:.1f} mm (start {res['target_mm']:.1f})" if "sim_final_mm" in res else ""), flush=True)
