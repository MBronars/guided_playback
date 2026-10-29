"""MPPI-argmin (no guidance) cell of the block-pushing budget study: same scenarios, server, sampler and replay as the LNO cells.
Knob: K candidates per replan, in chunks of `members` (4 for budgets <= 30 s, 8 above). Calibrate K on hard seed 0 eps 0-1
(plan_seconds = sum of generation time), then run the 50 hard seed-42 scenarios. Resumable. Writes <root>/_cells/M<budget>.json."""
import argparse, json, os, subprocess, time
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
SCEN = "/data/user_data/mbronars/packages/PointWorld/scenarios/hard"
ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--budget", type=float, required=True)
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget"); ap.add_argument("--n", type=int, default=50)
ap.add_argument("--replay_gpu", default="0"); ap.add_argument("--jobs", type=int, default=1)
A = ap.parse_args(); CELL = f"M{int(A.budget):03d}"; RUNS = os.path.join(A.root, "runs")
MEM = 4 if A.budget <= 30 else 8
REC = os.path.join(A.root, "_cells", f"{CELL}.json"); os.makedirs(os.path.dirname(REC), exist_ok=True)
rec = json.load(open(REC)) if os.path.exists(REC) else dict(cell=CELL, curve="M", budget=A.budget, members=MEM, calibration=[], rows=[])
save = lambda: (json.dump(rec, open(REC + ".tmp", "w"), indent=1), os.replace(REC + ".tmp", REC))
log = lambda *a: print(f"[{CELL}]", *a, flush=True)


def plan(K, scen, name, replay):
    cmd = [PY, "-u", os.path.join(HERE, "bin", "mppi_push_scenario.py"), "--url", A.url, "--scenario", scen, "--name", name, "--K", str(K), "--members", str(MEM),
           "--runs", RUNS, "--replay_gpu", A.replay_gpu] + ([] if replay else ["--no_replay"])
    lf = os.path.join(RUNS, "pushing_mppi", name + ".log"); os.makedirs(os.path.dirname(lf), exist_ok=True)
    subprocess.run(cmd, check=True, cwd=HERE, stdout=open(lf, "w"), stderr=subprocess.STDOUT)
    return json.load(open(os.path.join(RUNS, "pushing_mppi", name, "result.json")))


plan(MEM, f"{SCEN}/seed_0/ep_000.npz", f"_mwarm_{CELL}", False)          # compile this server's sampler paths before anything is timed
if rec.get("chosen") is None:
    K = MEM * max(1, int(round(A.budget / (12 * (0.8 if MEM == 4 else 1.2)))) ); tried = {}
    for rnd in range(5):
        ts = [plan(K, f"{SCEN}/seed_0/ep_{e:03d}.npz", f"_mcal_{CELL}_K{K}_s0ep{e}", False)["plan_seconds"] for e in (0, 1)]
        t = float(np.mean(ts)); tried[K] = t; rec["calibration"].append(dict(K=K, seconds=round(t, 2), each=ts)); save(); log("calib K", K, round(t, 1), "s")
        if abs(t - A.budget) <= max(0.1 * A.budget, 2.0): break
        K2 = max(MEM, MEM * int(round(K * A.budget / t / MEM)))
        if K2 in tried: break
        K = K2
    K, t = min(tried.items(), key=lambda kv: abs(kv[1] - A.budget)); rec["chosen"] = dict(K=K, calib_seconds=round(t, 2)); save(); log("CHOSEN", rec["chosen"])
K = rec["chosen"]["K"]; done = {r["scenario"] for r in rec["rows"]}
for ep in range(A.n):
    if ep in done: continue
    name = f"budget_{CELL}_K{K}_m{MEM}_hard42_ep{ep:03d}"
    try:
        r = plan(K, f"{SCEN}/seed_42/ep_{ep:03d}.npz", name, True)
    except subprocess.CalledProcessError as e:
        log("FAILED", name, e); continue
    rec["rows"].append(dict(scenario=ep, target_mm=r["target_mm"], imagined_mm=r["imagined_final_mm"], sim_mm=r["sim_final_mm"], plan_seconds=r["plan_seconds"], name=name))
    rec["rows"].sort(key=lambda x: x["scenario"]); save()
    s = [x["sim_mm"] for x in rec["rows"]]; log(f"ep {ep:3d} sim {r['sim_final_mm']} mm plan {r['plan_seconds']} s | mean {np.mean(s):.1f} over {len(s)}")
s = np.array([x["sim_mm"] for x in rec["rows"]]); t = np.array([x["plan_seconds"] for x in rec["rows"]])
rec["summary"] = dict(n=len(s), sim_mean_mm=round(float(s.mean()), 2), sim_std_mm=round(float(s.std()), 2), plan_seconds_mean=round(float(t.mean()), 2)); save(); log("CELL_DONE", rec["summary"])
