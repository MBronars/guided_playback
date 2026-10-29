"""Pile (mould cloud-centroid) scaling probe: plan time, peak memory and IMAGINED distance closed for one planner cell, no Genesis replay.
Same formulation as the push / pick tables: deterministic sampler (initial noise only: pose draws + initial particle draws), Adam,
fixed FPS, shell. Scenarios = the paper's cloud_centroid seed-42 set (pwm_efficient/scenarios/cloud_centroid/seed_42), first --n.
  guided / plain sampling (--I 0):  POST /guide  (--P population, --I iterations, --extra JSON e.g. {"prefilter": 7})
A warm-up request with the same batch shape compiles the graphs first, so every recorded time is warm.
Record: <root>/_cfgs/<name>__mould.json (rows: scenario, plan_seconds, peak_gib, imagined_mm, unguided_mm, target_mm). Resumable."""
import argparse, json, os, urllib.request
import numpy as np
SCEN = "/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid/seed_42"
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--P", type=int, default=4); ap.add_argument("--I", type=int, default=5); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--optimizer", default="adam"); ap.add_argument("--steps", type=int, default=6); ap.add_argument("--transitions", type=int, default=150)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--n", type=int, default=5); ap.add_argument("--extra", default="{}")
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget")
A = ap.parse_args(); RUNS = os.path.join(A.root, "runs")
REC = os.path.join(A.root, "_cfgs", f"{A.name}__mould.json"); os.makedirs(os.path.dirname(REC), exist_ok=True)
rec = json.load(open(REC)) if os.path.exists(REC) else dict(name=A.name, set="mould_seed42", args=vars(A), rows=[])
save = lambda: (json.dump(rec, open(REC + ".tmp", "w"), indent=1), os.replace(REC + ".tmp", REC))


def post(body):
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def body(ep, name, it):
    z = np.load(f"{SCEN}/ep_{ep:03d}.npz"); b = int(z["batch_idx"]); off = z["target_offset_xy"].astype(float)
    d = dict(domain="mould", task="move", episodes=",".join([str(b)] * A.P), transitions=A.transitions, steps=A.steps, churn=False, fix_fps=True,
             optimizer=A.optimizer, shell=True, t_freeze=0.0, offset=f"{off[0]},{off[1]},0", iterations=it, lr=A.lr, seed=A.seed, name=name, runs=RUNS,
             desc=f"pile scaling probe {A.name}")
    d.update(json.loads(A.extra)); return d, float(np.linalg.norm(off) * 1e3)


post(body(0, f"_mwarm_{A.name}", min(A.I, 1))[0])                       # compile this batch shape (and the grad path when I > 0)
done = {r["scenario"] for r in rec["rows"]}
for ep in range(A.n):
    if ep in done: continue
    name = f"mscale_{A.name}_ep{ep:03d}"; b, target = body(ep, name, A.I)
    res = post(b); rj = json.load(open(os.path.join(RUNS, "mould", name, "result.json")))
    plan_s = float(sum(rj["iteration_seconds"])) + float(rj.get("prefilter_seconds", 0) or 0)
    k = int(np.argmin(res["best_mm"]))
    row = dict(scenario=ep, target_mm=round(target, 1), plan_seconds=round(plan_s, 1), peak_gib=max(rj.get("peak_gib") or [0]),
               imagined_mm=round(float(res["best_mm"][k]), 1), closed_imagined_mm=round(target - float(res["best_mm"][k]), 1),
               unguided_mm=round(float(min(rj.get("unguided_mm") or [np.nan])), 1), name=name)
    rec["rows"].append(row); save()
    print(f"[{A.name}] scenario {ep}: plan {plan_s:.1f} s peak {row['peak_gib']} GiB imagined {row['imagined_mm']} mm (target {target:.0f})", flush=True)
r = rec["rows"]
rec["summary"] = dict(n=len(r), plan_seconds=round(float(np.mean([x["plan_seconds"] for x in r])), 1), peak_gib=max(x["peak_gib"] for x in r),
                      closed_imagined_mm=round(float(np.mean([x["closed_imagined_mm"] for x in r])), 1)); save()
print("MOULD_DONE", A.name, rec["summary"], flush=True)
