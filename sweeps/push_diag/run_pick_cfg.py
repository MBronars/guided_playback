"""One planner configuration on the 25-problem pick-to-location benchmark (docs/BASELINE_PICK_BENCHMARK.md: 5 assets x the 5
seed-0 targets of sweeps/cyl_targets.txt), pickup-only cont200k checkpoint (adapter clean_pickup), Isaac floating-hand replay of
the planner's pick (lowest imagined distance; never the simulator). Isaac launches are serialized per node with flock.
  guided / plain sampling (--I 0):  POST /guide   (--P population, --I iterations, --lr, --optimizer, --extra JSON e.g. prefilter)
  MPPI argmin (--mode plan):        POST /plan    (--K candidates per replan, commit 5, full windows)
Record: <root>/_cfgs/<name>__pick.json. Resumable."""
import argparse, json, os, subprocess, time, urllib.request
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
SEL = f"{HERE}/sweeps/assetsweep_select.json"
ASSETS = [("milk_carton", 11), ("peanut_butter_jar", 1), ("soda_can", 13), ("pasta_box", 9), ("water_bottle", 16)]
TARGETS = [l.split() for l in open(f"{HERE}/sweeps/cyl_targets.txt") if l.strip()]
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--name", required=True); ap.add_argument("--mode", default="guide", choices=("guide", "plan"))
ap.add_argument("--P", type=int, default=4); ap.add_argument("--I", type=int, default=10); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--optimizer", default="adam"); ap.add_argument("--no_shell", action="store_true"); ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--K", type=int, default=20); ap.add_argument("--members", type=int, default=4, help="plan mode: samples per batch (K must be a multiple)"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--extra", default="{}"); ap.add_argument("--n", type=int, default=0, help="timing pass: stop after this many rows"); ap.add_argument("--no_replay", action="store_true")
ap.add_argument("--pad_tail", type=int, default=0, help="full windows to the end: schedule 60 + PAD transitions, commit frames 0..60"); ap.add_argument("--replay_gpu", default="0"); ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget")
A = ap.parse_args(); RUNS = os.path.join(A.root, "runs"); OUTD = os.path.join(RUNS, "pickup")
REC = os.path.join(A.root, "_cfgs", f"{A.name}__pick.json"); os.makedirs(os.path.dirname(REC), exist_ok=True)
rec = json.load(open(REC)) if os.path.exists(REC) else dict(name=A.name, set="pick", args=vars(A), rows=[])
save = lambda: (json.dump(rec, open(REC + ".tmp", "w"), indent=1), os.replace(REC + ".tmp", REC))
url = A.url if A.mode == "guide" else A.url.replace("/guide", "/plan")


def post(u, body):
    r = urllib.request.Request(u, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def body(idx, off, name, it):
    b = dict(domain="pickup", task="lift", select=SEL, episodes=",".join([str(idx)] * (A.P if A.mode == "guide" else A.members)), transitions=60, steps=A.steps,
             offset=off, optimizer=A.optimizer, lr=A.lr, iterations=it, shell=not A.no_shell, seed=A.seed, name=name, runs=RUNS, desc=f"pick study {A.name}")
    if A.mode == "plan": b.update(K=A.K, commit=5, full_window=True)
    if A.pad_tail and A.mode == "guide": b.update(transitions=60 + A.pad_tail, stop_at=60)
    b.update(json.loads(A.extra)); return b


a0, i0 = ASSETS[0]; post(url, body(i0, TARGETS[0][1], f"_pwarm_{A.name}", min(A.I, 1)) | ({"K": A.members} if A.mode == "plan" else {}))   # compile / warm
done = {(r["asset"], r["target"]) for r in rec["rows"]}
for asset, idx in ASSETS:
    for tid, off in TARGETS:
        if (asset, int(tid)) in done: continue
        if A.n and len(rec["rows"]) >= A.n: break
        name = f"pick_{A.name}_t{tid}_{asset}"
        res = post(url, body(idx, off, name, A.I)); pick = int(np.argmin(res["best_mm"])) if A.mode == "guide" else 0
        rj = json.load(open(os.path.join(OUTD, name, "result.json")))
        plan_s = float(rj.get("plan_seconds") or sum(rj["iteration_seconds"])) + float(rj.get("prefilter_seconds", 0) or 0)
        cmd = f"flock /tmp/gp_isaac_$(hostname).lock {PY} -u bin/replay.py --run {os.path.join(OUTD, name)} --variants guided --members {pick} --gpu {A.replay_gpu}"
        if A.no_replay: cmd = "false"
        rc = subprocess.run(["bash", "-c", cmd], cwd=HERE, stdout=open(os.path.join(OUTD, name, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT).returncode
        row = dict(asset=asset, target=int(tid), picked=pick, imagined_mm=round(float(res["best_mm"][pick]), 1), plan_seconds=round(plan_s, 1), name=name,
                   unguided_min_mm=round(float(min(rj.get("unguided_mm") or [np.nan])), 1))
        try:
            an = json.load(open(os.path.join(OUTD, name, "replay_isaac", "analysis.json")))
            x = next(r for r in an["rows"] if r["member"] == pick and r["variant"] == "guided")
            row.update(sim_mm=round(x["sim_to_target_mm"], 1), sim_lift_mm=round(x["sim_lift_mm"], 1), gen_lift_mm=round(x["gen_lift_mm"], 1))
        except Exception as e:
            row.update(sim_mm=None); print("replay failed", name, rc, e, flush=True)
        rec["rows"].append(row); save()
        s = [r["sim_mm"] for r in rec["rows"] if r.get("sim_mm") is not None]
        print(f"[{A.name}] {asset} t{tid}: imagined {row['imagined_mm']} Isaac {row.get('sim_mm')} plan {row['plan_seconds']} s | median {np.median(s) if s else float('nan'):.1f} over {len(s)}", flush=True)
s = np.array([r["sim_mm"] for r in rec["rows"] if r.get("sim_mm") is not None]); t = np.array([r["plan_seconds"] for r in rec["rows"]])
rec["summary"] = dict(n=len(s), median_mm=round(float(np.median(s)), 1), within50=int((s <= 50).sum()), within60=int((s <= 60).sum()),
                      plan_seconds=round(float(t.mean()), 1), imagined_median=round(float(np.median([r["imagined_mm"] for r in rec["rows"]])), 1))
save(); print("PICK_DONE", A.name, rec["summary"], flush=True)
