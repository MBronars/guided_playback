"""PointWorld on the 25-problem pick benchmark, open vs closed loop, scored by the benchmark's own Isaac playback
(gp/replay/isaac_float.py, pickup path: floating Franka hand, binary gripper below 35 mm, object = convex hull of its frame-0
particles with the Isaac pool's density / friction). Open loop = the round-0 plan (60 frames) executed in full. Closed loop:
commit 10 frames, replay every problem's committed commanded clouds from the start in ONE Isaac launch (25 problems in fixed
slot order, same GPU every launch), read back the executed hand pose (Kabsch of the hand template onto the executed hand rows),
finger separation (executed finger-centroid distance) and object points, and replan the frames left from that state.
Planning: bin/pw_plan_pick_cem.py (CEM 128 x 3 over the remaining horizon) on this job's GPU; Isaac via --gpu_srun (CEM copy, 23 Oct).
  python bin/pick_pw_closed_loop.py --tag pwpick --ckpt <frozen> [--problems a_t0,...] [--det_check] [--open_only]"""
import argparse, json, os, subprocess, sys, time
import numpy as np
GP = "/data/user_data/mbronars/packages/guided_playback"; PWM = "/data/user_data/mbronars/reports/pw_mpc"; sys.path.insert(0, PWM)
from gripper import Gripper                                                   # noqa: E402  (numpy only)
PYPW = "/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--problems", default=None)
ap.add_argument("--commit", type=int, default=10); ap.add_argument("--horizon", type=int, default=60); ap.add_argument("--isaac_gpu", default="1")
ap.add_argument("--plan_job", default=""); ap.add_argument("--gpu_srun", default=""); ap.add_argument("--population", type=int, default=128); ap.add_argument("--plan_gpus", default="0,1"); ap.add_argument("--stats", default=None); ap.add_argument("--det_check", action="store_true"); ap.add_argument("--open_only", action="store_true")
A = ap.parse_args()
Z = np.load(f"{GP}/sweeps/pick_benchmark_5x5_scenes.npz"); assets = sorted({k.split("__")[0] for k in Z.files})
probs = [dict(id=f"{a}_t{t}", asset=a, tid=t) for a in assets for t in range(5)]
if A.problems: probs = [p for p in probs if p["id"] in A.problems.split(",")]
OUT = f"{GP}/runs/pickup_baselines"; ST = f"{OUT}/{A.tag}_state"; os.makedirs(ST, exist_ok=True)
LOG = open(f"{ST}/driver_log.jsonl", "a")
def log(r): LOG.write(json.dumps(r) + "\n"); LOG.flush(); print(json.dumps(r), flush=True)

G, S = {}, {}
for p in probs:
    a = p["asset"]; P0 = Z[f"{a}__points_model"].astype(np.float64)
    g = Gripper.from_scene(P0, Z[f"{a}__roles"], Z[f"{a}__particle_body"], Z[f"{a}__body_poses0_model"]); G[p["id"]] = g
    S[p["id"]] = dict(P0=P0, roles=Z[f"{a}__roles"].astype(np.int8), pb=Z[f"{a}__particle_body"], target=Z[f"{a}__targets_model"][p["tid"]].astype(np.float64), cmd=P0[None].copy())


def readback(g, Q):
    """executed cloud [N, 3] -> (R, t, separation, cloud with actuator rows rebuilt from them, rebuild error mm)."""
    h = g.rows[g.hand]; T = g.templates[g.hand]; cs, cd = T.mean(0), Q[h].mean(0); H_ = (T - cs).T @ (Q[h] - cd); U, _, Vt = np.linalg.svd(H_)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T; t = cd - R @ cs
    c = [Q[g.rows[k]].mean(0) for k in g.fingers]; s = float(np.linalg.norm(c[0] - c[1]))
    act = g.points(R[None, None], t[None, None], np.array([[s]]))[0, 0]; P = Q.copy(); P[g.act_rows] = act
    return R, t, s, P, float(np.abs(act - Q[g.act_rows]).max() * 1e3)


def plan(tag, horizon, states):
    """states: id -> (points, R, t, s). Two planner processes on babel-q9-28 GPUs 0 and 1."""
    ids = [p["id"] for p in probs]; gpus = A.plan_gpus.split(","); halves = [ids[i::len(gpus)] for i in range(len(gpus))]; procs = []
    for gi, (gpu, part) in enumerate(zip(gpus, halves)):
        if not part: continue
        inp, out = f"{ST}/{tag}_g{gi}_in.npz", f"{ST}/{tag}_g{gi}_out.npz"
        if os.path.exists(out): continue
        np.savez(inp, assets=np.array([S[i] and next(p['asset'] for p in probs if p['id'] == i) for i in part]), points=np.stack([states[i][0] for i in part]),
                 R0=np.stack([states[i][1] for i in part]), t0=np.stack([states[i][2] for i in part]), sep0=np.array([states[i][3] for i in part]),
                 target=np.stack([S[i]["target"] for i in part]), horizon=horizon, seed=0)
        cmd = (f"cd /data/user_data/mbronars/packages/pwm_final_paper && {PYPW} -u {GP}/bin/pw_plan_pick_cem.py "
               f"--inp {inp} --out {out} --ckpt {A.ckpt} --population {A.population}" + (f" --stats {A.stats}" if A.stats else "") + f" > {ST}/{tag}_g{gi}_planner.log 2>&1")
        procs.append(subprocess.Popen(["bash", "-c", cmd]))
    for pr in procs:
        pr.wait(); assert pr.returncode == 0, f"planner failed: see {ST}/{tag}_g*_planner.log"
    res = {}
    for gi, part in enumerate(halves):
        if not part: continue
        out = f"{ST}/{tag}_g{gi}_out.npz"
        for _ in range(300):
            if os.path.exists(out) and os.path.exists(out[:-4] + ".json"): break
            time.sleep(1); os.listdir(ST)
        o = np.load(out); j = json.load(open(out[:-4] + ".json"))
        for b, i in enumerate(part): res[i] = dict(imagined=o["imagined"][b].astype(np.float64), row=j["rows"][b])
    return res


def launch(name, frames):
    """frames: id -> commanded clouds [T, N, 3]; one Isaac launch, fixed slot order (probs order)."""
    rp = f"{OUT}/{name}"; t0 = time.time()
    if not os.path.exists(f"{rp}/replay_isaac/analysis.json"):
        os.makedirs(rp, exist_ok=True); ids = [p["id"] for p in probs]; P = np.stack([frames[i] for i in ids]).astype(np.float32)
        for b, i in enumerate(ids): om = S[i]["roles"] == 2; P[b][:, om] = S[i]["P0"][om][None]      # object rows: frame 0 held (only frame 0 is read)
        np.savez(f"{rp}/rollout.npz", predicted=P, unguided=P, roles=np.stack([S[i]["roles"] for i in ids]), object_mask=np.stack([S[i]["roles"] == 2 for i in ids]),
                 particle_body=np.stack([S[i]["pb"] for i in ids]), episodes=np.array(ids), target=np.stack([S[i]["target"] for i in ids]).astype(np.float32),
                 offsets=np.zeros((len(ids), 3), np.float32), boxes=np.zeros((0, 6), np.float32), trace_mm=np.zeros((1, len(ids)), np.float32), step=np.array(0))
        json.dump(dict(adapter="pointworld_mppi", domain="pickup", sim="isaac", task="lift", frame_dt=1 / 20, episodes=ids, joint_limits_mm=[13.5, 53.5],
                       best_mm=[0.0] * len(ids), unguided_mm=[0.0] * len(ids), note="committed PointWorld command clouds (bin/pick_pw_closed_loop.py)"), open(f"{rp}/result.json", "w"), indent=1)
        with open(f"{rp}/replay_stdout.log", "w") as lf:
            subprocess.run(["bash", "-c", f"{A.gpu_srun} {PY} -u {GP}/bin/replay.py --run {rp} --variants guided --gpu 0"], cwd=GP, stdout=lf, stderr=subprocess.STDOUT, check=True)
    an = json.load(open(f"{rp}/replay_isaac/analysis.json")); sim = np.load(f"{rp}/replay_isaac/replay.npz")["sim"][0].astype(np.float64)
    ids = [p["id"] for p in probs]; rows = {ids[r["member"]]: r for r in an["rows"] if r["variant"] == "guided"}
    return rp, sim, rows, time.time() - t0


res = dict(tag=A.tag, checkpoint=A.ckpt, horizon=A.horizon, commit=A.commit, problems=[p["id"] for p in probs], open={}, closed={})
st0 = {}
for p in probs:
    g = G[p["id"]]; st0[p["id"]] = (S[p["id"]]["P0"], g.R0, g.t0, g.sep0)
t0 = time.time(); pl0 = plan("r0", A.horizon, st0); t_p0 = time.time() - t0
rp, sim, rows, t_is = launch(f"{A.tag}_ol_exec", {i: pl0[i]["imagined"] for i in pl0})
for b, p in enumerate(probs):
    i = p["id"]; r = rows[i]; res["open"][i] = dict(final_mm=round(r["sim_to_target_mm"], 1), planned_mm=pl0[i]["row"]["pred_mm"], planned_lift_mm=pl0[i]["row"]["pred_lift_mm"],
        lift_mm=round(r["sim_lift_mm"], 1), plan_seconds=pl0[i]["row"]["seconds"], replay=f"{rp}/replay_isaac/replay.npz", slot=b)
log(dict(open_loop=rp, plan_wall_s=round(t_p0, 1), isaac_s=round(t_is, 1), final_mm={i: v["final_mm"] for i, v in res["open"].items()}))
if A.det_check:
    rp2, sim2, _, _ = launch(f"{A.tag}_ol_exec_dup", {i: pl0[i]["imagined"] for i in pl0}); res["determinism_max_mm"] = round(float(np.abs(sim2 - sim).max() * 1e3), 4)
    log(dict(determinism_max_mm=res["determinism_max_mm"]))
if not A.open_only:
    per = {p["id"]: [] for p in probs}; state = st0
    for r in range(-(-A.horizon // A.commit)):
        c = r * A.commit; n = min(A.commit, A.horizon - c)
        if r == 0: pl, t_p = pl0, t_p0
        else: t0 = time.time(); pl = plan(f"r{r}", A.horizon - c, state); t_p = time.time() - t0
        for p in probs:
            i = p["id"]; S[i]["cmd"] = np.concatenate([S[i]["cmd"], pl[i]["imagined"][1:1 + n]])
            per[i].append(dict(round=r, executed_before=c, planned_mm=pl[i]["row"]["pred_mm"], planned_lift_mm=pl[i]["row"]["pred_lift_mm"], plan_seconds=pl[i]["row"]["seconds"]))
        rp, sim, rows, t_is = launch(f"{A.tag}_cl_exec_c{c + n}", {p["id"]: S[p["id"]]["cmd"] for p in probs})
        state, errs = {}, []
        for b, p in enumerate(probs):
            i = p["id"]; R, t, s, P, e = readback(G[i], sim[b][c + n]); state[i] = (P, R, t, s); errs.append(e)
            per[i][-1].update(obj_to_target_after_mm=round(rows[i]["sim_to_target_mm"], 1), lift_after_mm=round(rows[i]["sim_lift_mm"], 1), readback_rebuild_mm=round(e, 2))
        log(dict(round=r, exec=rp, plan_wall_s=round(t_p, 1), isaac_s=round(t_is, 1), readback_rebuild_mm_max=round(max(errs), 2),
                 obj_to_target_mm={i: round(v["sim_to_target_mm"], 1) for i, v in rows.items()}))
    for b, p in enumerate(probs):
        i = p["id"]; r = rows[i]; res["closed"][i] = dict(final_mm=round(r["sim_to_target_mm"], 1), lift_mm=round(r["sim_lift_mm"], 1), planned_mm=per[i][-1]["planned_mm"],
            plan_seconds=round(sum(x["plan_seconds"] for x in per[i]), 1), rounds=per[i], replay=f"{rp}/replay_isaac/replay.npz", slot=b)
json.dump(res, open(f"{OUT}/{A.tag}_results.json", "w"), indent=1); print("DONE", f"{OUT}/{A.tag}_results.json", flush=True)
