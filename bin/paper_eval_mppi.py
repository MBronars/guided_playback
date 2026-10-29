"""No-guidance (MPPI argmin) twin of bin/paper_eval.py on the paper's cloud_centroid scenarios: per scenario one /plan
request (K fresh candidates per replan, commit `commit`, argmin), the committed plan replayed in Genesis on --replay_gpu in
the background while the next scenario plans. Writes runs/_paper_eval/<tag>_summary.json in the SAME row format as
paper_eval.py (imagined_mm = the plan's own last-frame objective, unguided_imagined_mm = the first plain sample,
guide_seconds = plan seconds), so bin/paper_report.py --summary <that file> aggregates it unchanged.

  python bin/paper_eval_mppi.py --url http://host:8790/plan --seed 42 --n 50 --K 6 --commit 5 --replay_gpu 0 --tag mppi42_K6
"""
import argparse, json, os, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = "/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid"
PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n", type=int, default=50); ap.add_argument("--members", type=int, default=3, help="batch = compiled B; K must be a multiple")
ap.add_argument("--transitions", type=int, default=150); ap.add_argument("--steps", type=int, default=6); ap.add_argument("--keep", type=int, default=3)
ap.add_argument("--K", type=int, default=6); ap.add_argument("--commit", type=int, default=5); ap.add_argument("--plan_seed", type=int, default=0)
ap.add_argument("--replay_gpu", default="0"); ap.add_argument("--no_replay", action="store_true", help="plan only; replay later with a second pass (plans are skipped when result.json exists)"); ap.add_argument("--tag", default=None); ap.add_argument("--start", type=int, default=0)
A = ap.parse_args()
TAG = A.tag or f"mppi{A.seed}_K{A.K}"
OUT = os.path.join(HERE, "runs", "mould"); os.makedirs(os.path.join(HERE, "runs", "_paper_eval"), exist_ok=True)
SUMMARY = os.path.join(HERE, "runs", "_paper_eval", f"{TAG}_summary.json")
rows, pending, t_replay0 = [], None, [time.time()]


def post(body):
    import urllib.request
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def collect(job):
    proc, name, ep, res, target_mm, episode, t_plan = job
    proc.wait()
    a = json.load(open(os.path.join(OUT, name, "replay_genesis", "analysis.json")))
    row = next(r for r in a["rows"] if r["member"] == 0 and r["variant"] == "guided")
    rec = dict(scenario=ep, episode=episode, target_mm=round(target_mm, 1), picked_member=0, imagined_mm=round(res["best_mm"][0], 1),
               sim_mm=round(row["sim_to_target_mm"], 1), gap_mm=round(row["sim_vs_gen_final_mm"], 1), gen_moved_mm=round(row["gen_moved_mm"], 1),
               sim_moved_mm=round(row["sim_moved_mm"], 1), unguided_imagined_mm=round(res["unguided_mm"][0], 1), guide_seconds=res["seconds"],
               guide_wall_seconds=round(t_plan, 1), replay_seconds=round(time.time() - t_replay0[0], 1), name=name)
    rows.append(rec)
    json.dump(dict(tag=TAG, seed=A.seed, settings=dict(method="mppi_argmin", K=A.K, commit=A.commit, members=A.members, transitions=A.transitions,
              steps=A.steps, keep=A.keep, churn=True, fix_fps=True, plan_seed=A.plan_seed, full_window=True), rows=rows), open(SUMMARY, "w"), indent=1)
    sims = [r["sim_mm"] for r in rows]
    print(f"[{len(rows):2d}/{A.n}] scenario {ep:3d} ep {rec['episode']} target {rec['target_mm']:5.1f} mm | plan {rec['imagined_mm']:5.1f} -> SIM {rec['sim_mm']:5.1f} mm "
          f"(gap {rec['gap_mm']:4.1f}) | {rec['guide_seconds']:.0f} s | running mean sim {np.mean(sims):5.1f} mm over {len(sims)}", flush=True)


for ep in range(A.start, A.start + A.n):
    z = np.load(f"{SCEN}/seed_{A.seed}/ep_{ep:03d}.npz")
    b = int(z["batch_idx"]); off = z["target_offset_xy"].astype(float)
    name = f"{TAG}_ep{ep:03d}"
    if not os.path.exists(os.path.join(OUT, name, "result.json")):
        body = dict(domain="mould", task="move", episodes=",".join([str(b)] * A.members), transitions=A.transitions, steps=A.steps, keep=A.keep,
                    churn=True, fix_fps=True, seed=A.plan_seed, offset=f"{off[0]},{off[1]},0", K=A.K, commit=A.commit, name=name,
                    desc=f"MPPI argmin K {A.K} commit {A.commit}; paper seed {A.seed} scenario {ep}: episode {b}, target {np.linalg.norm(off)*1e3:.0f} mm")
        t0 = time.time(); res = post(body); t_plan = time.time() - t0
    else:
        rj = json.load(open(os.path.join(OUT, name, "result.json"))); res = dict(best_mm=rj["best_mm"], unguided_mm=rj["unguided_mm"], seconds=rj["plan_seconds"]); t_plan = rj["plan_seconds"]
    print(f"  planned scenario {ep} (episode {b}) in {res['seconds']:.0f} s; plan {res['best_mm'][0]:.1f} mm, first plain sample {res['unguided_mm'][0]:.1f} mm", flush=True)
    if A.no_replay:
        continue
    if pending is not None:
        collect(pending)
    if os.path.exists(os.path.join(OUT, name, "replay_genesis", "analysis.json")):
        proc = subprocess.Popen(["true"])
    else:
        proc = subprocess.Popen([PY, os.path.join(HERE, "bin", "replay.py"), "--run", os.path.join(OUT, name), "--variants", "guided", "--members", "0",
                                 "--gpu", A.replay_gpu], cwd=HERE, stdout=open(os.path.join(OUT, name, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT)
    t_replay0[0] = time.time()
    pending = (proc, name, ep, res, float(np.linalg.norm(off) * 1e3), b, t_plan)
if pending is not None:
    collect(pending)
print(f"summary: {SUMMARY}  -> python bin/paper_report.py --summary {SUMMARY}")
