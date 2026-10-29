"""Closed-loop guided picking on the 25-problem benchmark (pickup-only cont200k, adapter clean_pickup, Isaac floating hand).

Round r (c frames executed so far, target fixed in the model frame):
  plan    guided (/guide) from the EXECUTED state: the executed cloud at frame c becomes frame 0 (gp/scene_edit.set_frame0_points,
          every body rigidly fitted), the remaining n = 60 - c frames are generated with full windows (noise for n + PAD
          transitions, commit 0..n: every model call reads the trained window), objective = object centroid at the plan's
          last frame = task frame 60. The planner picks the member with the lowest imagined objective.
  commit  that member's next `commit` generated frames are appended to the commanded cloud sequence.
  execute the whole commanded sequence 0..c+commit is replayed from the start in Isaac (bin/replay.py, flock per node); the
          executed cloud at its last frame is the next round's frame 0.
Context: this checkpoint reads only frame 0 at inference (it generates its own context), so re-anchoring on the executed frame is
the model's own input contract. Record: <root>/_cfgs/<name>__pick.json (rows with sim_mm, start/closed, plan seconds). Resumable."""
import argparse, json, os, shutil, subprocess, time, urllib.request
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
SEL = f"{HERE}/sweeps/assetsweep_select.json"
ASSETS = [("milk_carton", 11), ("peanut_butter_jar", 1), ("soda_can", 13), ("pasta_box", 9), ("water_bottle", 16)]
TARGETS = [l.split() for l in open(f"{HERE}/sweeps/cyl_targets.txt") if l.strip()]
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--name", required=True); ap.add_argument("--commit", type=int, default=10); ap.add_argument("--pad", type=int, default=16)
ap.add_argument("--P", type=int, default=4); ap.add_argument("--I", type=int, default=5); ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--optimizer", default="adam")
ap.add_argument("--steps", type=int, default=10); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--extra", default="{}"); ap.add_argument("--replay_gpu", default="0")
ap.add_argument("--problems", default=None, help="comma list asset_tN (default: all 25)")
ap.add_argument("--no_warm", action="store_true", help="every replan draws fresh noise (default: member 0 continues the previous plan)")
ap.add_argument("--mode", default="history", choices=("history", "frame0"), help="history: executed frames 0..c as clean prefix rows (full windows); frame0: executed frame c re-anchored as frame 0")
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget")
A = ap.parse_args(); RUNS = os.path.join(A.root, "runs"); OUTD = os.path.join(RUNS, "pickup")
REC = os.path.join(A.root, "_cfgs", f"{A.name}__pick.json"); os.makedirs(os.path.dirname(REC), exist_ok=True)
rec = json.load(open(REC)) if os.path.exists(REC) else dict(name=A.name, set="pick", args=vars(A), rows=[])
save = lambda: (json.dump(rec, open(REC + ".tmp", "w"), indent=1), os.replace(REC + ".tmp", REC))


def post(body):
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def plan(idx, target_abs, n, name, f0, warm=None):
    if A.mode == "history":                                           # task-indexed rollout 0..60; f0 = executed clouds 0..c (npz 'clouds') or None at c = 0
        b = dict(domain="pickup", task="lift", select=SEL, episodes=",".join([str(idx)] * A.P), transitions=60, steps=A.steps,
                 offset="0,0,0.15", target_abs=",".join(f"{v:.6f}" for v in target_abs), optimizer=A.optimizer, lr=A.lr, iterations=A.I, shell=True,
                 seed=A.seed, name=name, runs=RUNS, desc=f"pick closed loop {A.name}: executed history, {n} frames left")
        if f0: b["history_clouds"] = f0
        b["save_best_noise"] = True
        if warm: b["init_noise"] = warm
        b.update(json.loads(A.extra)); return post(b)
    b = dict(domain="pickup", task="lift", select=SEL, episodes=",".join([str(idx)] * A.P), transitions=n + A.pad, stop_at=n, steps=A.steps,
             offset="0,0,0.15", target_abs=",".join(f"{v:.6f}" for v in target_abs), optimizer=A.optimizer, lr=A.lr, iterations=A.I, shell=True,
             seed=A.seed, name=name, runs=RUNS, desc=f"pick closed loop {A.name}: {n} frames left")
    if f0: b["frame0_points"] = f0
    b.update(json.loads(A.extra)); return post(b)


def execute(plan_dir, pick, cmd, name):
    """replay the commanded cloud sequence cmd [c+1, N, 3] from the start; returns (executed cloud at the last frame, analysis row)."""
    d = os.path.join(OUTD, name); os.makedirs(d, exist_ok=True)
    z = dict(np.load(os.path.join(plan_dir, "rollout.npz"))); B = z["predicted"].shape[0]
    out = {k: (v[pick:pick + 1] if (isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == B and k not in ("boxes", "trace_mm")) else v) for k, v in z.items()}
    out["predicted"] = cmd[None].astype(np.float32); out["unguided"] = cmd[None].astype(np.float32)
    np.savez(os.path.join(d, "rollout.npz"), **out)
    rj = json.load(open(os.path.join(plan_dir, "result.json"))); rj["episodes"] = [rj["episodes"][pick]]; json.dump(rj, open(os.path.join(d, "result.json"), "w"))
    cmdl = f"flock /tmp/gp_isaac_$(hostname).lock {PY} -u bin/replay.py --run {d} --variants guided --members 0 --gpu {A.replay_gpu}"
    subprocess.run(["bash", "-c", cmdl], cwd=HERE, stdout=open(os.path.join(d, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT)
    sim = np.load(os.path.join(d, "replay_isaac", "replay.npz"))["sim"][0, 0].astype(np.float64)
    an = next(x for x in json.load(open(os.path.join(d, "replay_isaac", "analysis.json")))["rows"] if x["member"] == 0 and x["variant"] == "guided")
    return sim[-1], an


Z = np.load(f"{HERE}/sweeps/pick_benchmark_5x5_scenes.npz"); SCN = {k.split("__")[0]: Z[k].astype(np.float64) for k in Z.files if k.endswith("__targets_model")}
want = None if A.problems is None else set(A.problems.split(","))
done = {(r["asset"], r["target"]) for r in rec["rows"]}
for asset, idx in ASSETS:
    for tid, off in TARGETS:
        if (asset, int(tid)) in done or (want and f"{asset}_t{tid}" not in want): continue
        c, r, cmd, f0, rounds, plan_s, t0, warm = 0, 0, None, None, [], 0.0, time.time(), None
        key = next(k for k in SCN if k.startswith(asset)); target_abs = SCN[key][int(tid)]
        while c < 60:
            n = 60 - c; name = f"pickcl_{A.name}_t{tid}_{asset}_r{r}"
            res = plan(idx, target_abs, n, name, f0, None if A.no_warm else warm)
            rj = json.load(open(os.path.join(OUTD, name, "result.json"))); pick = int(np.argmin(res["best_mm"]))
            plan_s += float(sum(rj["iteration_seconds"])) + float(rj.get("prefilter_seconds", 0) or 0)
            pred = np.load(os.path.join(OUTD, name, "rollout.npz"))["predicted"][pick].astype(np.float64)        # [n + 1, N, 3], frame 0 = executed state
            m = min(A.commit, n); o = c if A.mode == "history" else 0           # history mode: rollout frames are task frames
            bn = os.path.join(OUTD, name, "best_noise.npz")
            if os.path.exists(bn):                                            # warm start for the next replan: the committed member's noise, minus the m executed rows
                wv = np.load(bn)["v0"][pick][m:]; warm = os.path.join(OUTD, name, "warm_next.npz"); np.savez(warm, P0=wv)
            cmd = pred[: m + 1] if cmd is None else np.concatenate([cmd, pred[o + 1: o + m + 1]])
            c += m
            ex, an = execute(os.path.join(OUTD, name), pick, cmd, f"pickcl_{A.name}_t{tid}_{asset}_exec{r}")
            ed = os.path.join(OUTD, f"pickcl_{A.name}_t{tid}_{asset}_exec{r}")
            if A.mode == "history":
                clouds = np.load(os.path.join(ed, "replay_isaac", "replay.npz"))["sim"][0, 0].astype(np.float32)   # executed frames 0..c
                fp = os.path.join(ed, "executed_clouds.npz"); np.savez(fp, clouds=clouds)
            else:
                fp = os.path.join(ed, "executed_last.npz"); np.savez(fp, points=ex.astype(np.float32))
            f0 = fp
            rounds.append(dict(round=r, executed=c, imagined_mm=round(float(res["best_mm"][pick]), 1), sim_mm_now=round(an["sim_to_target_mm"], 1), sim_lift_mm=round(an["sim_lift_mm"], 1)))
            print(f"[{A.name}] {asset} t{tid} round {r}: executed {c}, imagined {res['best_mm'][pick]:.1f}, Isaac now {an['sim_to_target_mm']:.1f} mm", flush=True)
            r += 1
        start = float(np.linalg.norm([float(v) for v in off.split(",")]) * 1e3)
        row = dict(asset=asset, target=int(tid), sim_mm=rounds[-1]["sim_mm_now"], start_mm=round(start, 1), closed_mm=round(start - rounds[-1]["sim_mm_now"], 1),
                   sim_lift_mm=rounds[-1]["sim_lift_mm"], imagined_mm=rounds[0]["imagined_mm"], plan_seconds=round(plan_s, 1), wall_seconds=round(time.time() - t0, 1), rounds=rounds)
        rec["rows"].append(row); save()
        s = [x["closed_mm"] for x in rec["rows"]]; print(f"[{A.name}] {asset} t{tid}: Isaac {row['sim_mm']} mm, closed {row['closed_mm']} | mean closed {np.mean(s):.1f} over {len(s)}", flush=True)
cl = np.array([x["closed_mm"] for x in rec["rows"]]); sm = np.array([x["sim_mm"] for x in rec["rows"]])
rec["summary"] = dict(n=len(cl), closed_mean_mm=round(float(cl.mean()), 1), median_mm=round(float(np.median(sm)), 1), within50=int((sm <= 50).sum()),
                      plan_seconds=round(float(np.mean([x["plan_seconds"] for x in rec["rows"]])), 1)); save()
print("PICKCL_DONE", A.name, rec["summary"], flush=True)
