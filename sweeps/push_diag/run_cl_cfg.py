"""Closed-loop (receding-horizon, SAPIEN feedback) guided block pushing on a scenario set: bin/closed_loop_push.py per scenario.
Full windows everywhere (--pad_tail): every model call reads Cm1 + window rows. Record <root>/_cfgs/<name>__<set>.json. Resumable.
  python run_cl_cfg.py --url URL --name CL_c10_ctx3 --commit 10 --context 3 --P 8 --I 6 [--extra '{"prefilter": 15}'] --set dev"""
import argparse, json, os, subprocess
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--commit", type=int, default=10); ap.add_argument("--context", type=int, default=3); ap.add_argument("--min_plan", type=int, default=26)
ap.add_argument("--P", type=int, default=8); ap.add_argument("--I", type=int, default=6); ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--t_freeze", type=float, default=0.0)
ap.add_argument("--extra", default="{}"); ap.add_argument("--no_warm", action="store_true"); ap.add_argument("--set", default="dev", choices=("dev", "test")); ap.add_argument("--n", type=int, default=50); ap.add_argument("--replay_gpu", default="0")
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget")
A = ap.parse_args(); RUNS = os.path.join(A.root, "runs")
SCEN = f"/data/user_data/mbronars/packages/PointWorld/scenarios/hard/{'seed_0' if A.set == 'dev' else 'seed_42'}"
REC = os.path.join(A.root, "_cfgs", f"{A.name}__{A.set}.json"); os.makedirs(os.path.dirname(REC), exist_ok=True)
rec = json.load(open(REC)) if os.path.exists(REC) else dict(name=A.name, set=A.set, args=vars(A), rows=[])
save = lambda: (json.dump(rec, open(REC + ".tmp", "w"), indent=1), os.replace(REC + ".tmp", REC))
done = {r["scenario"] for r in rec["rows"]}
for ep in range(A.n):
    if ep in done: continue
    name = f"cl_{A.name}_{A.set}_ep{ep:03d}"; lf = os.path.join(RUNS, "pushing", name + ".log"); os.makedirs(os.path.dirname(lf), exist_ok=True)
    cmd = [PY, "-u", os.path.join(HERE, "bin", "closed_loop_push.py"), "--url", A.url, "--scenario", f"{SCEN}/ep_{ep:03d}.npz", "--name", name, "--commit", str(A.commit),
           "--context", str(A.context), "--min_plan", str(A.min_plan), "--pad_tail", "--members", str(A.P), "--iterations", str(A.I), "--lr", str(A.lr),
           "--t_freeze", str(A.t_freeze), "--runs", RUNS, "--replay_gpu", A.replay_gpu, "--extra", A.extra] + (["--no_warm"] if A.no_warm else [])
    if subprocess.run(cmd, cwd=HERE, stdout=open(lf, "w"), stderr=subprocess.STDOUT).returncode != 0:
        print("FAILED", name, flush=True); continue
    s = json.load(open(os.path.join(RUNS, "pushing", name, "summary.json")))
    rec["rows"].append(dict(scenario=ep, sim_mm=s["sim_final_mm"], target_mm=s["target_mm"], plan_seconds=s.get("plan_seconds_total"), replans=s["replans"],
                            imagined_mm=s["rows"][0]["imagined_best_mm"][s["rows"][0]["picked_member"]], first_plan_seconds=s["rows"][0].get("plan_seconds"), name=name))
    rec["rows"].sort(key=lambda x: x["scenario"]); save()
    v = [x["sim_mm"] for x in rec["rows"]]; print(f"[{A.name}] ep {ep} sim {s['sim_final_mm']} plan {s.get('plan_seconds_total')} s | mean {np.mean(v):.1f} over {len(v)}", flush=True)
s = np.array([x["sim_mm"] for x in rec["rows"]]); tg = np.array([x["target_mm"] for x in rec["rows"]]); t = np.array([x["plan_seconds"] for x in rec["rows"]])
rec["summary"] = dict(n=len(s), sim_mm=round(float(s.mean()), 2), closed_sim_mm=round(float((tg - s).mean()), 2), plan_seconds=round(float(t.mean()), 1)); save()
print("CFG_DONE", A.name, rec["summary"], flush=True)
