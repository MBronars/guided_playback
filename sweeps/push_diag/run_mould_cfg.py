"""One planner configuration on the pile (mould cloud-centroid) benchmark: the paper's seed-42 scenarios
(pwm_efficient/scenarios/cloud_centroid/seed_42, first --n), Genesis replay of the planner's pick (lowest imagined distance; never the
simulator). Same formulation as the push / pick tables unless --churn: deterministic sampler (initial pose + particle draws), Adam, fixed FPS, shell.
  guided / plain sampling (--I 0):  POST /guide  (--P population, --I iterations, --extra JSON e.g. {"prefilter": 7})
  MPPI argmin (--mode plan):        POST /plan   (--K candidates per replan in chunks of --P, commit 5, full windows, same sampler)
A warm-up request with the same batch shape compiles the graphs first, so recorded plan times are warm. Replays run on --replay_gpu after
each plan (synchronous). Record: <root>/_cfgs/<name>__mould.json (rows: scenario, target, imagined, sim, closed, gap, plan s). Resumable."""
import argparse, json, os, subprocess, urllib.request
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
SCEN_ROOT = "/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid"
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--P", type=int, default=4); ap.add_argument("--I", type=int, default=5); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--optimizer", default="adam"); ap.add_argument("--steps", type=int, default=6); ap.add_argument("--transitions", type=int, default=150)
ap.add_argument("--churn", action="store_true", help="the old paper recipe's full-churn sampler (with --t_freeze)"); ap.add_argument("--t_freeze", type=float, default=0.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=50); ap.add_argument("--start", type=int, default=0)
ap.add_argument("--mode", default="guide", choices=("guide", "plan")); ap.add_argument("--K", type=int, default=8)
ap.add_argument("--extra", default="{}"); ap.add_argument("--no_replay", action="store_true"); ap.add_argument("--replay_gpu", default="0")
ap.add_argument("--scen_seed", type=int, default=42, help="scenario set: pwm_efficient/scenarios/cloud_centroid/seed_<N> (paper = 42)")
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget")
A = ap.parse_args(); SCEN = f"{SCEN_ROOT}/seed_{A.scen_seed}"; RUNS = os.path.join(A.root, "runs"); OUTD = os.path.join(RUNS, "mould")
REC = os.path.join(A.root, "_cfgs", f"{A.name}__mould.json"); os.makedirs(os.path.dirname(REC), exist_ok=True)
rec = json.load(open(REC)) if os.path.exists(REC) else dict(name=A.name, set=f"mould_seed{A.scen_seed}", args=vars(A), rows=[])
save = lambda: (json.dump(rec, open(REC + ".tmp", "w"), indent=1), os.replace(REC + ".tmp", REC))
URL = A.url if A.mode == "guide" else A.url.replace("/guide", "/plan")


def post(body):
    r = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def body(ep, name, it, warm=False):
    z = np.load(f"{SCEN}/ep_{ep:03d}.npz"); b = int(z["batch_idx"]); off = z["target_offset_xy"].astype(float)
    d = dict(domain="mould", task="move", episodes=",".join([str(b)] * A.P), transitions=A.transitions, steps=A.steps, churn=A.churn, fix_fps=True,
             optimizer=A.optimizer, shell=True, t_freeze=A.t_freeze, offset=f"{off[0]},{off[1]},0", iterations=it, lr=A.lr, seed=A.seed, name=name, runs=RUNS,
             desc=f"pile table {A.name}: scenario {ep}, episode {b}")
    if A.mode == "plan": d.update(K=A.P if warm else A.K, commit=5, keep=3)
    d.update(json.loads(A.extra)); return d, float(np.linalg.norm(off) * 1e3), b


post(body(A.start, f"_mwarm_{A.name}", min(A.I, 1), warm=True)[0])
done = {r["scenario"] for r in rec["rows"]}
for ep in range(A.start, A.start + A.n):
    if ep in done: continue
    name = f"pile_{A.name}_ep{ep:03d}"; bd, target, episode = body(ep, name, A.I)
    res = post(bd); rj = json.load(open(os.path.join(OUTD, name, "result.json")))
    plan_s = float(rj.get("plan_seconds") or sum(rj["iteration_seconds"])) + float(rj.get("prefilter_seconds", 0) or 0)
    pick = int(np.argmin(res["best_mm"]))                                # plan mode: one committed plan, member 0
    cmd = [PY, "-u", "bin/replay.py", "--run", os.path.join(OUTD, name), "--variants", "guided", "--members", str(pick), "--gpu", A.replay_gpu]
    if A.no_replay: cmd = ["false"]
    rc = subprocess.run(cmd, cwd=HERE, stdout=open(os.path.join(OUTD, name, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT).returncode
    row = dict(scenario=ep, episode=episode, target_mm=round(target, 1), picked=pick, imagined_mm=round(float(res["best_mm"][pick]), 1),
               closed_imagined_mm=round(target - float(res["best_mm"][pick]), 1), plan_seconds=round(plan_s, 1), peak_gib=max(rj.get("peak_gib") or [0]),
               unguided_min_mm=round(float(min(rj.get("unguided_mm") or [np.nan])), 1), name=name)
    try:
        a = next(x for x in json.load(open(os.path.join(OUTD, name, "replay_genesis", "analysis.json")))["rows"] if x["member"] == pick and x["variant"] == "guided")
        row.update(sim_mm=round(a["sim_to_target_mm"], 1), closed_sim_mm=round(target - a["sim_to_target_mm"], 1), gap_mm=round(a["sim_vs_gen_final_mm"], 1),
                   gen_moved_mm=round(a["gen_moved_mm"], 1), sim_moved_mm=round(a["sim_moved_mm"], 1))
    except Exception as e:
        row.update(sim_mm=None); print("replay failed", name, rc, e, flush=True)
    rec["rows"].append(row); save()
    s = [x["closed_sim_mm"] for x in rec["rows"] if x.get("sim_mm") is not None]
    print(f"[{A.name}] scenario {ep}: imagined closed {row['closed_imagined_mm']} sim closed {row.get('closed_sim_mm')} gap {row.get('gap_mm')} plan {plan_s:.0f} s | mean sim closed {np.mean(s) if s else float('nan'):.1f} over {len(s)}", flush=True)
r = [x for x in rec["rows"] if x.get("sim_mm") is not None]
rec["summary"] = dict(n=len(r), closed_sim_mm=round(float(np.mean([x["closed_sim_mm"] for x in r])), 1), closed_imagined_mm=round(float(np.mean([x["closed_imagined_mm"] for x in r])), 1),
                      gap_mm=round(float(np.mean([x["gap_mm"] for x in r])), 1), plan_seconds=round(float(np.mean([x["plan_seconds"] for x in rec["rows"]])), 1)); save()
print("PILE_DONE", A.name, rec["summary"], flush=True)
