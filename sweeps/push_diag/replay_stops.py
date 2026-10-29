"""Simulated quality vs number of guidance steps, from ONE guided run per problem (saved with save_iterates).
For stop s in STOPS the planner's plan is the (member, iteration <= s) with the lowest imagined objective. Each distinct plan is written
as a sibling run folder <run>__stop{s} (rollout.npz with that member's trajectory = that iterate) and replayed in Isaac (flock per node).
Appends rows to <root>/_cfgs/<name>__stops.json: problem, stop, member, iteration, imagined_mm, sim_mm.
  python replay_stops.py --cfg <_cfgs/X__pick.json> --gpu G [--stops 0,1,2,3,5,7,10]"""
import argparse, json, os, shutil, subprocess
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
ap = argparse.ArgumentParser(); ap.add_argument("--cfg", required=True); ap.add_argument("--gpu", default="0"); ap.add_argument("--stops", default="0,1,2,3,5,7,10")
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget")
A = ap.parse_args(); cfg = json.load(open(A.cfg)); STOPS = [int(v) for v in A.stops.split(",")]
OUTD = os.path.join(A.root, "runs", "pickup"); REC = os.path.join(A.root, "_cfgs", f"{cfg['name']}__stops.json")
rec = json.load(open(REC)) if os.path.exists(REC) else dict(name=cfg["name"], rows=[])
have = {(r["asset"], r["target"], r["stop"]) for r in rec["rows"]}
for r in cfg["rows"]:
    run = os.path.join(OUTD, r["name"]); it = np.load(os.path.join(run, "iterates.npz")); obj = it["objective"][:-1] if len(it["objective"]) > max(STOPS) + 1 else it["objective"]
    base = dict(np.load(os.path.join(run, "rollout.npz")))
    for s in STOPS:
        if (r["asset"], r["target"], s) in have or s >= len(it["objective"]): continue
        o = it["objective"][: s + 1]; i, m = np.unravel_index(int(np.argmin(o)), o.shape)
        d = f"{run}__stop{s}"; os.makedirs(d, exist_ok=True)
        z = dict(base); pred = z["predicted"].copy(); pred[m] = it["positions"][i, m].astype(np.float32); z["predicted"] = pred
        np.savez(os.path.join(d, "rollout.npz"), **z); shutil.copy(os.path.join(run, "result.json"), os.path.join(d, "result.json"))
        cmd = f"flock /tmp/gp_isaac_$(hostname).lock {PY} -u bin/replay.py --run {d} --variants guided --members {m} --gpu {A.gpu}"
        subprocess.run(["bash", "-c", cmd], cwd=HERE, stdout=open(os.path.join(d, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT)
        try:
            an = json.load(open(os.path.join(d, "replay_isaac", "analysis.json"))); x = next(q for q in an["rows"] if q["member"] == m and q["variant"] == "guided")
            sim = round(x["sim_to_target_mm"], 1)
        except Exception as e:
            sim = None; print("replay failed", d, e, flush=True)
        rec["rows"].append(dict(asset=r["asset"], target=r["target"], stop=s, member=int(m), iteration=int(i), imagined_mm=round(float(o[i, m]) * 1e3, 1), sim_mm=sim))
        json.dump(rec, open(REC, "w"), indent=1)
        os.remove(os.path.join(d, "rollout.npz"))                               # disk: the replay keeps its own outputs
        print(f"{r['asset']} t{r['target']} stop {s}: iter {i} member {m} imagined {o[i, m]*1e3:.1f} sim {sim}", flush=True)
print("STOPS_DONE", flush=True)
