"""Fixed MPPI (or guided MPPI, --iterations > 0) configuration on a scenario set; SAPIEN replay inside bin/mppi_push_scenario.py.
Record: <root>/_cfgs/<name>__<set>.json (same row format as run_cfg.py). Resumable."""
import argparse, json, os, subprocess
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--K", type=int, required=True); ap.add_argument("--members", type=int, default=4); ap.add_argument("--iterations", type=int, default=0)
ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--t_freeze", type=float, default=0.8); ap.add_argument("--commit", type=int, default=5); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--set", default="dev", choices=("dev", "test")); ap.add_argument("--n", type=int, default=50); ap.add_argument("--replay_gpu", default="0")
ap.add_argument("--no_replay", action="store_true", help="timing pass: plan only"); ap.add_argument("--K0", type=int, default=0, help="initial pool at the first replan only"); ap.add_argument("--ode", action="store_true", help="deterministic Euler sampler for every candidate")
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget")
A = ap.parse_args(); RUNS = os.path.join(A.root, "runs")
SCEN = f"/data/user_data/mbronars/packages/PointWorld/scenarios/hard/{'seed_0' if A.set == 'dev' else 'seed_42'}"
REC = os.path.join(A.root, "_cfgs", f"{A.name}__{A.set}.json"); os.makedirs(os.path.dirname(REC), exist_ok=True)
rec = json.load(open(REC)) if os.path.exists(REC) else dict(name=A.name, set=A.set, args=vars(A), rows=[])
save = lambda: (json.dump(rec, open(REC + ".tmp", "w"), indent=1), os.replace(REC + ".tmp", REC))
done = {r["scenario"] for r in rec["rows"]}
wn = f"_warm_{A.name}"                                                    # compile the sampler paths on this server before any timed plan
subprocess.run([PY, "-u", os.path.join(HERE, "bin", "mppi_push_scenario.py"), "--url", A.url, "--scenario", f"{SCEN}/ep_000.npz", "--name", wn, "--K", str(A.members),
                "--members", str(A.members), "--iterations", str(A.iterations), "--lr", str(A.lr), "--t_freeze", str(A.t_freeze), "--commit", str(A.commit),
                "--runs", RUNS, "--no_replay"] + (["--ode"] if A.ode else []), cwd=HERE, stdout=open(os.path.join(RUNS, "pushing_mppi", wn + ".log"), "w"), stderr=subprocess.STDOUT)
for ep in range(A.n):
    if ep in done: continue
    name = f"diag_{A.name}_{A.set}_ep{ep:03d}"; lf = os.path.join(RUNS, "pushing_mppi", name + ".log"); os.makedirs(os.path.dirname(lf), exist_ok=True)
    cmd = [PY, "-u", os.path.join(HERE, "bin", "mppi_push_scenario.py"), "--url", A.url, "--scenario", f"{SCEN}/ep_{ep:03d}.npz", "--name", name, "--K", str(A.K),
           "--members", str(A.members), "--iterations", str(A.iterations), "--lr", str(A.lr), "--t_freeze", str(A.t_freeze), "--commit", str(A.commit),
           "--runs", RUNS, "--replay_gpu", A.replay_gpu, "--seed", str(A.seed), "--K0", str(A.K0)] + (["--no_replay"] if A.no_replay else []) + (["--ode"] if A.ode else [])
    if subprocess.run(cmd, cwd=HERE, stdout=open(lf, "w"), stderr=subprocess.STDOUT).returncode != 0:
        print("FAILED", name, flush=True); continue
    r = json.load(open(os.path.join(RUNS, "pushing_mppi", name, "result.json")))
    rec["rows"].append(dict(scenario=ep, K=A.K, iterations=A.iterations, imagined_mm=r["imagined_final_mm"], sim_mm=r.get("sim_final_mm", float("nan")), target_mm=r.get("target_mm", float("nan")),
                            plan_seconds=r["plan_seconds"], name=name)); rec["rows"].sort(key=lambda x: x["scenario"]); save()
    s = [x["sim_mm"] for x in rec["rows"]]; print(f"[{A.name}] ep {ep} sim {r.get('sim_final_mm')} plan {r['plan_seconds']} s | mean {np.mean(s):.1f} over {len(s)}", flush=True)
s = np.array([x["sim_mm"] for x in rec["rows"]]); t = np.array([x["plan_seconds"] for x in rec["rows"]]); tg = np.array([x["target_mm"] for x in rec["rows"]])
rec["summary"] = dict(n=len(s), plan_seconds=round(float(t.mean()), 2), imagined_mm=round(float(np.mean([x["imagined_mm"] for x in rec["rows"]])), 2),
                      sim_mm=round(float(s.mean()), 2), closed_sim_mm=round(float((tg - s).mean()), 2)); save(); print("CFG_DONE", A.name, rec["summary"], flush=True)
