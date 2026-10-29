"""Closed-loop pick (25 benchmark problems, one planner setting) through the gp.pool workers, Isaac re-simulation per round.

Per round (c = frames executed so far, commit --commit, task length 60): one pool unit per problem = the planner setting of
sweeps/noise_vs_action/settings_main.json (--tag) with the executed history (gp/adapters/paper_prefix.py: last
min(c, 3) executed transitions as clean context rows, plan max(60 - c, 16) rows + 16 tail rows, reward at task frame 60).
When all 25 are planned: append each problem's next --commit planned frames to its committed trajectory, replay every
problem's committed frames 0..c+commit from the start in ONE Isaac launch (1/480 s, normal gripper close; GPU via the
--gpu_srun prefix), read back the executed clouds -> next history (gp/closed_loop_hist.history_from_clouds). After the last
round the Isaac result of the full 60 frames is the closed-loop score. Resumable: finished rounds are skipped.
  python bin/closed_loop_pick_pool.py --pool <root> --tag dno_pool32_I5_P4 --gpu_srun "<srun prefix>"
Output: <pool>/closed_loop/<tag>/{exec_rNN/, history/, summary.json}"""
import argparse, json, os, subprocess, sys, time
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from gp.pool import add_units, _atomic_json
from gp.closed_loop_hist import history_from_clouds
ap = argparse.ArgumentParser(); ap.add_argument("--pool", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--commit", type=int, default=5); ap.add_argument("--total", type=int, default=60); ap.add_argument("--context", type=int, default=3)
ap.add_argument("--window_future", type=int, default=16); ap.add_argument("--gpu_srun", default=""); ap.add_argument("--settings", default=os.path.join(HERE, "sweeps/noise_vs_action/settings_main.json"))
ap.add_argument("--problems", type=int, default=25, help="first N problems (smoke tests)")
A = ap.parse_args()
CKPT = "/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_pickup_drop50_100m_unet_bidir/checkpoints/final.pt"
ASSETS = [("milk_carton", 11), ("peanut_butter_jar", 1), ("soda_can", 13), ("pasta_box", 9), ("water_bottle", 16)]
TARGETS = [l.split() for l in open(f"{HERE}/sweeps/cyl_targets.txt") if l.strip()]
probs = [(f"{a}_t{t}", i, o) for a, i in ASSETS for t, o in TARGETS][:A.problems]
S = next(s for s in json.load(open(A.settings)) if s["tag"] == A.tag); Bm = int(S.get("B", 4))
OUT = os.path.join(A.pool, "closed_loop", A.tag + (f"_c{A.commit}" if A.commit != 5 else "")); os.makedirs(os.path.join(OUT, "history"), exist_ok=True)
T, F = A.total, A.window_future
log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
comp, meta, Q = {}, {}, {}
r, c = 0, 0
while c < T:
    k = min(c, A.context); n_plan = k + max(T - c, F); n_int = n_plan + F; goal = k + (T - c); n_new = min(A.commit, T - c)
    ids = []
    units = []
    for pid, idx, off in probs:
        if c == 0: hp = "frame0"
        else:
            hp = os.path.join(OUT, "history", f"{pid}_c{c}.npz")
            if not os.path.exists(hp):
                H = history_from_clouds(Q[pid][:c + 1], meta[pid]["body_poses0"], meta[pid]["particle_body"], meta[pid]["body_type"])
                np.savez(hp, positions=H["positions"], body_poses=H["body_poses"])
        uid = f"cl_{A.tag}" + (f"_c{A.commit}" if A.commit != 5 else "") + f"_r{r:02d}__{pid}"; ids.append((uid, pid))
        req = dict(ckpt=CKPT, domain="pickup", task="lift", select=f"{HERE}/sweeps/assetsweep_select.json", episodes=",".join([str(idx)] * Bm),
                   transitions=n_int, steps=10, offset=off, seed=0, churn=True, t_freeze=0.0, fix_fps=True, history=hp, history_context=A.context,
                   history_stop_at=n_plan, goal_frame=goal, desc=f"closed loop {A.tag} {pid} round {r}: {c} executed")
        req.update(S["knobs"])
        units.append(dict(id=uid, shape=S["shape"], tag=f"cl_{A.tag}", problem=pid, round=r, req=req))
    add_units(A.pool, units)
    log(f"round {r}: c={c}, {len(units)} units issued (plan {n_plan - k} rows, goal rollout frame {goal})")
    while True:                                                                    # wait for every problem's plan
        done = [os.path.exists(os.path.join(A.pool, "done", u + ".json")) for u, _ in ids]
        failed = [u for u, _ in ids if os.path.exists(os.path.join(A.pool, "failed", u + ".json"))]
        if failed: log(f"FAILED units {failed[:3]}...; stopping"); sys.exit(2)
        if all(done): break
        time.sleep(30)
    rows = []
    for uid, pid in ids:
        d = json.load(open(os.path.join(A.pool, "done", uid + ".json"))); z = np.load(os.path.join(d["plan"], "rollout.npz")); m = d["member"]
        P = z["predicted"][m]                                                      # rollout frame i = task frame c - k + i
        if pid not in comp:
            meta[pid] = {key: z[key][m] for key in ("roles", "object_mask", "particle_body", "pose_action_mask", "joint_type", "body_type", "object_ids", "body_poses0", "target", "offsets")}
            comp[pid] = P[:1].copy()
        if comp[pid].shape[0] == c + 1:                                            # not yet appended this round (resume-safe)
            comp[pid] = np.concatenate([comp[pid], P[k + 1:k + 1 + n_new]], 0)
        rows.append(dict(problem=pid, imagined_mm=d["imagined_mm"], request_seconds=d["request_seconds"]))
    ed = os.path.join(OUT, f"exec_r{r:02d}")
    if not os.path.exists(os.path.join(ed, "replay_isaac", "analysis.json")):
        os.makedirs(ed, exist_ok=True)
        pids = [p for p, _, _ in probs]
        np.savez(os.path.join(ed, "rollout.npz"), predicted=np.stack([comp[p] for p in pids]).astype(np.float32), unguided=np.stack([comp[p] for p in pids]).astype(np.float32),
                 episodes=np.array(pids), step=np.array(0), boxes=np.zeros((0, 6), np.float32), trace_mm=np.zeros((1, len(pids)), np.float32),
                 **{key: np.stack([meta[p][key] for p in pids]) for key in meta[pids[0]]})
        json.dump(dict(domain="pickup", task="lift", sim="isaac", name=os.path.basename(ed), joint_limits_mm=[13.5, 53.5], best_mm=[0.0] * len(pids), unguided_mm=[0.0] * len(pids)),
                  open(os.path.join(ed, "result.json"), "w"))
        t1 = time.time()
        rc = subprocess.run(["bash", "-c", f"{A.gpu_srun} {sys.executable} -u {HERE}/bin/replay.py --run {ed} --sim isaac --variants guided --gpu 0"], cwd=HERE,
                            stdout=open(ed + ".log", "w"), stderr=subprocess.STDOUT).returncode
        log(f"round {r}: Isaac rc={rc} in {time.time() - t1:.0f} s")
    sim = np.load(os.path.join(ed, "replay_isaac", "replay.npz"))["sim"][0]
    an = {x["member"]: x for x in json.load(open(os.path.join(ed, "replay_isaac", "analysis.json")))["rows"]}
    for i, (pid, _, _) in enumerate(probs):
        Q[pid] = sim[i].astype(np.float64)
    c += n_new
    _atomic_json(os.path.join(OUT, f"round_{r:02d}.json"), dict(round=r, executed=c, plans=rows, isaac_mm=[round(an[i]["sim_to_target_mm"], 1) for i in range(len(probs))]))
    log(f"round {r} done: executed {c}; Isaac mean distance {np.mean([an[i]['sim_to_target_mm'] for i in range(len(probs))]):.1f} mm")
    r += 1
fin = json.load(open(os.path.join(OUT, f"round_{r - 1:02d}.json")))
plan_s = [np.mean([p["request_seconds"] for p in json.load(open(os.path.join(OUT, f"round_{i:02d}.json")))["plans"]]) for i in range(r)]
_atomic_json(os.path.join(OUT, "summary.json"), dict(tag=A.tag, commit=A.commit, rounds=r, final_isaac_mm=fin["isaac_mm"], problems=[p for p, _, _ in probs],
             mean_plan_seconds_per_replan=round(float(np.mean(plan_s)), 1), plan_seconds_per_round=[round(float(x), 1) for x in plan_s]))
log(f"DONE {A.tag}: mean final Isaac distance {np.mean(fin['isaac_mm']):.1f} mm")
