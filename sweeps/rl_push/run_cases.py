"""rl_push guided cases through the warm server + Isaac floating-hand replay, planner view (member with the lowest
GENERATED distance; its SIM distance is the number). Runs ON babel-l9-16 (replay GPU must be local).
  python sweeps/rl_push/run_cases.py --tag <tag> --cases 2,8,11 [--goal own|offset:dx,dy] [knobs...]
Writes runs/rl_push/<tag>_ep<NNN>/ and runs/_paper_eval/<tag>_summary.json."""
import argparse, json, os, subprocess, time, sys
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; sys.path.insert(0, os.path.join(HERE, "sweeps", "rl_push"))
PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
INDEX = os.path.join(HERE, "sweeps", "rl_push", "test_index.json")
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8797/guide"); ap.add_argument("--tag", required=True); ap.add_argument("--cases", required=True)
ap.add_argument("--members", type=int, default=12); ap.add_argument("--transitions", type=int, default=60); ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--iterations", type=int, default=10); ap.add_argument("--lr", type=float, default=0.2); ap.add_argument("--lr_final", type=float, default=0.02)
ap.add_argument("--t_freeze", type=float, default=0.0); ap.add_argument("--optimizer", default="adam"); ap.add_argument("--shell", action="store_true")
ap.add_argument("--no_churn", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--goal", default="own", help="own | offset:dx,dy (m) | dist:D (own direction, D m)")
ap.add_argument("--upright_weight", type=float, default=0.0); ap.add_argument("--upright_frames", type=int, default=5); ap.add_argument("--floor_weight", type=float, default=0.0)
ap.add_argument("--replay_gpu", default="1"); ap.add_argument("--no_replay", action="store_true"); ap.add_argument("--replay_variants", default="guided")
ap.add_argument("--extra", default="{}", help="JSON of extra knobs merged into the request")
A = ap.parse_args()
IDX = {r["loader"]: r for r in json.load(open(INDEX))}
OUT = os.path.join(HERE, "runs", "rl_push"); os.makedirs(os.path.join(HERE, "runs", "_paper_eval"), exist_ok=True)
SUMMARY = os.path.join(HERE, "runs", "_paper_eval", f"{A.tag}_summary.json")
from tilt import report as tilt_report

def post(body):
    import urllib.request
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())

def body_for(ep, name):
    t = IDX[ep]; b = dict(domain="rl_push", task="push", episodes=",".join([str(ep)] * A.members), transitions=A.transitions, steps=A.steps, churn=not A.no_churn,
                          fix_fps=True, t_freeze=A.t_freeze, optimizer=A.optimizer, shell=A.shell, lr=A.lr, lr_final=(A.lr_final if A.lr_final >= 0 else None), iterations=A.iterations, seed=A.seed,
                          upright_weight=A.upright_weight, upright_frames=A.upright_frames, floor_weight=A.floor_weight, name=name,
                          desc=f"{t['asset']} ep{t['e']} {t['cell']} goal {A.goal} | pop {A.members} S{A.steps} {A.optimizer} lr {A.lr}->{A.lr_final} tf {A.t_freeze} shell {A.shell} churn {not A.no_churn} up {A.upright_weight}")
    if A.goal == "own": b["own_goal"] = INDEX
    elif A.goal.startswith("dist:"): b["own_goal"] = INDEX; b["goal_dist_m"] = float(A.goal[5:])
    elif A.goal.startswith("offset:"): dx, dy = [float(v) for v in A.goal[7:].split(",")]; b["offset"] = f"{dx},{dy},0"
    else: raise SystemExit(A.goal)
    b.update(json.loads(A.extra)); return b

rows = []; pending = None
def collect(job):
    proc, name, ep, pick, res, t_guide, t0r = job
    if proc is not None: proc.wait()
    rec = dict(case=ep, asset=IDX[ep]["asset"], cell=IDX[ep]["cell"], picked_member=pick, imagined_mm=round(res["best_mm"][pick], 1), unguided_imagined_mm=round(res["unguided_mm"][pick], 1),
               guide_seconds=res["seconds"], guide_wall_seconds=round(t_guide, 1), name=name, best_imagined_all=[round(v, 1) for v in res["best_mm"]])
    try:
        tr = tilt_report(os.path.join(OUT, name)); rec["tilt"] = tr; tp = tr[pick]
        rec.update(gen_tilt_final=tp["gen_tilt_final"], gen_tilt_max=tp["gen_tilt_max"])
        if "sim_mm" in tp:
            rec.update(sim_mm=tp["sim_mm"], gap_mm=tp["gap_mm"], sim_tilt_final=tp["sim_tilt_final"], sim_tilt_max=tp["sim_tilt_max"], sim_lift_mm=tp["sim_lift_mm"], replay_seconds=round(time.time() - t0r, 1))
            sims = [r["sim_mm"] for r in tr if "sim_mm" in r]; rec.update(sim_mm_min_all=round(min(sims), 1), sim_mm_median_all=round(float(np.median(sims)), 1))
    except Exception as e:
        rec["error"] = repr(e)
    rows.append(rec)
    json.dump(dict(tag=A.tag, settings=vars(A), rows=rows), open(SUMMARY, "w"), indent=1)
    print(f"[{len(rows)}] case {ep} {IDX[ep]['asset']} {IDX[ep]['cell']} | imagined {rec['imagined_mm']} (unguided {rec['unguided_imagined_mm']}) -> SIM {rec.get('sim_mm')} mm gap {rec.get('gap_mm')} | tilt gen {rec.get('gen_tilt_final')} sim {rec.get('sim_tilt_final')} (max {rec.get('sim_tilt_max')}) | lift {rec.get('sim_lift_mm')} | {rec['guide_seconds']} s", flush=True)

for ep in [int(v) for v in A.cases.split(",")]:
    name = f"{A.tag}_ep{ep:03d}"
    if os.path.exists(os.path.join(OUT, name, "result.json")):
        res = json.load(open(os.path.join(OUT, name, "result.json"))); res["seconds"] = round(sum(res["iteration_seconds"]), 1); t_guide = 0.0
        print(f"  reuse {name}", flush=True)
    else:
        t0 = time.time(); res = post(body_for(ep, name)); t_guide = time.time() - t0
    pick = int(np.argmin(res["best_mm"]))
    print(f"  guided case {ep} in {res['seconds']:.0f} s; imagined {[round(v, 1) for v in res['best_mm']]} -> member {pick}", flush=True)
    if pending is not None: collect(pending)
    proc = None
    if not A.no_replay and not os.path.exists(os.path.join(OUT, name, "replay_isaac", "analysis.json")):
        proc = subprocess.Popen([PY, "-u", os.path.join(HERE, "bin", "replay.py"), "--run", os.path.join(OUT, name), "--variants", A.replay_variants, "--gpu", A.replay_gpu],
                                cwd=HERE, stdout=open(os.path.join(OUT, name, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT)
    pending = (proc, name, ep, pick, res, t_guide, time.time())
if pending is not None: collect(pending)
print("summary:", SUMMARY)
