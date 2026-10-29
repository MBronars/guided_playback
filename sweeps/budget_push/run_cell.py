"""One cell of the block-pushing time-budget study (docs/BUDGET_PUSH_PLAN.md).

  curve A: iterations scale, population 3        (adjusts I)
  curve B: population scales, 6 iterations       (adjusts P)
  curve C: population = iterations = n           (adjusts n)

1. compile: a 1-iteration request at population P (cached on the server afterwards).
2. calibrate on held-out hard seed 0, scenarios 0 and 1: planner time = sum(iteration_seconds). If the mean is not within
   max(10 %, 2 s) of the budget, move the knob by the measured cost model and re-check (at most 4 rounds).
3. run the 50 hard seed-42 test scenarios; per scenario pick the member with the best IMAGINED distance and replay it in
   SAPIEN (bin/replay.py, background, at most --replays at once). Resumable: finished scenarios are skipped.
Writes <root>/_cells/<cell>.json (settings, calibration rounds, rows).
Base recipe (user, 2026-09-27): S 6, 60 transitions, full churn, unitgrad lr 0.1 constant, t_freeze 0.8, shell, fixed FPS.
"""
import argparse, json, os, subprocess, time, urllib.request
import numpy as np

HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
SCEN = "/data/user_data/mbronars/packages/PointWorld/scenarios/hard"
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--curve", required=True, choices="ABC"); ap.add_argument("--budget", type=float, required=True)
ap.add_argument("--P", type=int, required=True); ap.add_argument("--I", type=int, required=True)
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget")
ap.add_argument("--n", type=int, default=50); ap.add_argument("--replays", type=int, default=3); ap.add_argument("--replay_gpu", default="0")
ap.add_argument("--calib_only", action="store_true")
A = ap.parse_args()
CELL = f"{A.curve}{int(A.budget):03d}"; RUNS = os.path.join(A.root, "runs"); PUSH = os.path.join(RUNS, "pushing")
os.makedirs(os.path.join(A.root, "_cells"), exist_ok=True); os.makedirs(PUSH, exist_ok=True)
REC_PATH = os.path.join(A.root, "_cells", f"{CELL}.json")
rec = json.load(open(REC_PATH)) if os.path.exists(REC_PATH) else dict(cell=CELL, curve=A.curve, budget=A.budget, calibration=[], rows=[])
log = lambda *a: print(f"[{CELL}]", *a, flush=True)


def save():
    json.dump(rec, open(REC_PATH + ".tmp", "w"), indent=1); os.replace(REC_PATH + ".tmp", REC_PATH)


def post(body):
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def body(P, I, scen, name):
    return dict(domain="pushing", task="push", episodes=",".join(["0"] * P), transitions=60, steps=6, churn=True, optimizer="unitgrad",
                shell=True, t_freeze=0.8, scenario=scen, iterations=I, lr=0.1, lr_final=None, free_fps=False, name=name, runs=RUNS,
                desc=f"budget study cell {CELL}: P {P}, I {I}")


def plan_seconds(res):
    return float(sum(res["iteration_seconds"]))


# ---------------------------------------------------------------- calibration
def measure(P, I):
    post(body(P, 1, f"{SCEN}/seed_0/ep_000.npz", f"_bcal_{CELL}_warm_P{P}"))                        # compile / warm for this P
    ts, its = [], []
    for e in (0, 1):
        res = post(body(P, I, f"{SCEN}/seed_0/ep_{e:03d}.npz", f"_bcal_{CELL}_P{P}_I{I}_s0ep{e}"))
        ts.append(plan_seconds(res)); its.append(res["iteration_seconds"])
    it = [s for r in its for s in r[:-1]]; fin = float(np.mean([r[-1] for r in its]))
    g = float(np.mean(it)) if it else 0.0
    r = dict(P=P, I=I, seconds=round(float(np.mean(ts)), 2), each=[round(t, 2) for t in ts], grad_iter_s=round(g, 3), final_s=round(fin, 3))
    rec["calibration"].append(r); save(); log("calib", r); return r


def ok(t):
    return abs(t - A.budget) <= max(0.10 * A.budget, 2.0)


if rec.get("chosen") is None:
    P, I = A.P, A.I; tried = {}
    for rnd in range(5):
        r = measure(P, I); tried[(P, I)] = r["seconds"]
        if ok(r["seconds"]) or rnd == 4:
            break
        g, f = max(r["grad_iter_s"], 0.5), r["final_s"]
        if A.curve == "A":
            I2 = max(1, int(round((A.budget - f) / g)))
            P2 = P
        elif A.curve == "B":                    # time ~ I*g(P) + f(P); g grows ~0.35 s per member (calibration table)
            P2 = max(1, P + int(round((A.budget - r["seconds"]) / (I * 0.35 + 0.1))))
            I2 = I
        else:                                   # P = I = n; time ~ n*g(n)
            n = I; step = 1 if r["seconds"] < A.budget else -1
            P2 = I2 = max(1, n + step)
        if (P2, I2) in tried:                   # oscillating between neighbours: stop, keep the nearest
            break
        P, I = P2, I2
    (P, I), t = min(tried.items(), key=lambda kv: abs(kv[1] - A.budget))
    rec["chosen"] = dict(P=P, I=I, calib_seconds=t); save(); log("CHOSEN", rec["chosen"])
if A.calib_only:
    raise SystemExit
P, I = rec["chosen"]["P"], rec["chosen"]["I"]

# ---------------------------------------------------------------- test scenarios (hard seed 42)
done = {r["scenario"] for r in rec["rows"]}; pending = []


def collect(job, block):
    proc, name, ep, pick, res, target_mm = job
    if not block and proc.poll() is None:
        return False
    proc.wait()
    try:
        a = json.load(open(os.path.join(PUSH, name, "replay_sapien", "analysis.json")))
        row = next(x for x in a["rows"] if x["member"] == pick and x["variant"] == "guided")
        sim = round(row["sim_to_target_mm"], 1); gap = round(row["sim_vs_gen_final_mm"], 1)
    except Exception as e:                                                  # keep going, mark the failure
        sim = gap = None; log("replay failed", name, e)
    rec["rows"].append(dict(scenario=ep, target_mm=round(target_mm, 1), picked_member=pick, imagined_mm=round(res["best_mm"][pick], 1),
                            sim_mm=sim, gap_mm=gap, plan_seconds=round(plan_seconds(res), 2), iteration_seconds=res["iteration_seconds"],
                            peak_gib=max(res["peak_gib"]), unguided_imagined_mm=round(res["unguided_mm"][pick], 1), name=name))
    rec["rows"].sort(key=lambda x: x["scenario"]); save()
    sims = [x["sim_mm"] for x in rec["rows"] if x["sim_mm"] is not None]
    log(f"ep {ep:3d} sim {sim} mm (imagined {res['best_mm'][pick]:.1f}) plan {plan_seconds(res):.1f} s | running mean {np.mean(sims):.1f} mm over {len(sims)}")
    return True


rec["settings"] = dict(P=P, I=I, steps=6, transitions=60, lr=0.1, t_freeze=0.8, shell=True, churn=True, optimizer="unitgrad", fix_fps=True,
                       ckpt="pwm_final_clean/runs/push0418_rigid_x0_rope_regpass_mix_cont200k_stream/checkpoints/final.pt"); save()
post(body(P, 1, f"{SCEN}/seed_0/ep_000.npz", f"_bcal_{CELL}_warm_P{P}"))           # make sure the graphs for P are warm
for ep in range(A.n):
    if ep in done:
        continue
    path = f"{SCEN}/seed_42/ep_{ep:03d}.npz"; z = np.load(path)
    target_mm = float(np.linalg.norm(z["target_xy"] - z["block_pos"][:2]) * 1e3)
    name = f"budget_{CELL}_P{P}_I{I}_hard42_ep{ep:03d}"
    res = post(body(P, I, path, name)); pick = int(np.argmin(res["best_mm"]))
    while len(pending) >= A.replays:                                       # bounded replay pool
        pending = [j for j in pending if not collect(j, False)]
        if len(pending) >= A.replays:
            time.sleep(2)
    proc = subprocess.Popen([PY, os.path.join(HERE, "bin", "replay.py"), "--run", os.path.join(PUSH, name), "--variants", "guided",
                             "--members", str(pick), "--gpu", A.replay_gpu], cwd=HERE,
                            stdout=open(os.path.join(PUSH, name, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT)
    pending.append((proc, name, ep, pick, res, target_mm))
for j in pending:
    collect(j, True)
sims = np.array([x["sim_mm"] for x in rec["rows"] if x["sim_mm"] is not None]); secs = np.array([x["plan_seconds"] for x in rec["rows"]])
rec["summary"] = dict(n=int(sims.size), sim_mean_mm=round(float(sims.mean()), 2), sim_std_mm=round(float(sims.std()), 2),
                      sim_median_mm=round(float(np.median(sims)), 2), plan_seconds_mean=round(float(secs.mean()), 2),
                      plan_seconds_std=round(float(secs.std()), 2)); save()
log("CELL_DONE", rec["summary"])
