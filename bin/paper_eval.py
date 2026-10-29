"""The previous paper's cloud-centroid evaluation set, run through this package.

Scenarios: pwm_efficient/scenarios/cloud_centroid/seed_<S>/ep_NNN.npz -- each pins the dataset episode (batch_idx)
and the in-plane target offset (target_offset_xy, magnitude 7.5-15 cm, random angle), sampled once with
np.random.default_rng(S). The paper run `cloud_centroid_diff_paper_seed42` used seed 42, 50 scenarios.

Per scenario: one POPULATION of --members rollouts of that episode with that target, guided through the warm server;
the member with the best IMAGINED objective is picked (what you can do at deployment) and replayed in Genesis; the
reported number is that member's simulated dough-centroid distance to the target. Replays run in the background on
another GPU while the next scenario is guided.

  python bin/paper_eval.py --url http://host:8790/guide --seed 42 --n 50 [--members 3] [--replay_gpu 0]
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = "/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid"
PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--n", type=int, default=50); ap.add_argument("--members", type=int, default=3)
ap.add_argument("--transitions", type=int, default=150); ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--iterations", type=int, default=6); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--t_freeze", type=float, default=0.8); ap.add_argument("--replay_gpu", default="0")
ap.add_argument("--tag", default=None); ap.add_argument("--start", type=int, default=0)
A = ap.parse_args()
TAG = A.tag or f"paper{A.seed}"
OUT = os.path.join(HERE, "runs", "mould")
os.makedirs(os.path.join(HERE, "runs", "_paper_eval"), exist_ok=True)
SUMMARY = os.path.join(HERE, "runs", "_paper_eval", f"{TAG}_summary.json")
rows, pending, t_replay0 = [], None, [time.time()]


def post(body):
    import urllib.request
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def collect(job):
    """Finish a backgrounded replay and record its row."""
    proc, name, ep, pick, res, target_mm, episode, t_guide = job
    proc.wait()
    a = json.load(open(os.path.join(OUT, name, "replay_genesis", "analysis.json")))
    row = next(r for r in a["rows"] if r["member"] == pick and r["variant"] == "guided")
    rec = dict(scenario=ep, episode=episode, target_mm=round(target_mm, 1), picked_member=pick,
               imagined_mm=round(res["best_mm"][pick], 1), sim_mm=round(row["sim_to_target_mm"], 1),
               gap_mm=round(row["sim_vs_gen_final_mm"], 1), gen_moved_mm=round(row["gen_moved_mm"], 1),
               sim_moved_mm=round(row["sim_moved_mm"], 1), unguided_imagined_mm=round(res["unguided_mm"][pick], 1),
               guide_seconds=res["seconds"], guide_wall_seconds=round(t_guide, 1),
               replay_seconds=round(time.time() - t_replay0[0], 1), name=name)
    rows.append(rec)
    json.dump(dict(tag=TAG, seed=A.seed, settings=dict(members=A.members, transitions=A.transitions, steps=A.steps,
              iterations=A.iterations, lr=A.lr, t_freeze=A.t_freeze, shell=True, churn=True, optimizer="unitgrad"),
              rows=rows), open(SUMMARY, "w"), indent=1)
    sims = [r["sim_mm"] for r in rows]
    print(f"[{len(rows):2d}/{A.n}] scenario {ep:3d} ep {rec['episode']} target {rec['target_mm']:5.1f} mm | "
          f"imagined {rec['imagined_mm']:5.1f} -> SIM {rec['sim_mm']:5.1f} mm (gap {rec['gap_mm']:4.1f}) | "
          f"running mean sim {np.mean(sims):5.1f} mm over {len(sims)}", flush=True)


for ep in range(A.start, A.start + A.n):
    z = np.load(f"{SCEN}/seed_{A.seed}/ep_{ep:03d}.npz")
    b = int(z["batch_idx"]); off = z["target_offset_xy"].astype(float)
    name = f"{TAG}_ep{ep:03d}"
    body = dict(domain="mould", task="move", episodes=",".join([str(b)] * A.members), transitions=A.transitions,
                steps=A.steps, churn=True, fix_fps=True, optimizer="unitgrad", shell=True, t_freeze=A.t_freeze,
                offset=f"{off[0]},{off[1]},0", iterations=A.iterations, lr=A.lr, name=name,
                desc=f"paper seed {A.seed} scenario {ep}: episode {b}, target {np.linalg.norm(off)*1e3:.0f} mm")
    t0 = time.time(); res = post(body); t_guide = time.time() - t0
    pick = int(np.argmin(res["best_mm"]))
    print(f"  guided scenario {ep} (episode {b}) in {res['seconds']:.0f} s; imagined {[round(v,1) for v in res['best_mm']]} -> member {pick}", flush=True)
    if pending is not None:
        collect(pending)                                   # the previous scenario's replay
    proc = subprocess.Popen([PY, os.path.join(HERE, "bin", "replay.py"), "--run", os.path.join(OUT, name),
                             "--variants", "guided", "--members", str(pick), "--gpu", A.replay_gpu],
                            cwd=HERE, stdout=open(os.path.join(OUT, name, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT)
    t_replay0[0] = time.time()
    pending = (proc, name, ep, pick, res, float(np.linalg.norm(off) * 1e3), b, t_guide)
if pending is not None:
    collect(pending)

sims = np.array([r["sim_mm"] for r in rows]); gaps = np.array([r["gap_mm"] for r in rows])
ung = np.array([r["unguided_imagined_mm"] for r in rows]); tgt = np.array([r["target_mm"] for r in rows])
print(f"\n== {TAG}: {len(rows)} scenarios, seed {A.seed}")
print(f"  target distance           mean {tgt.mean():5.1f} mm")
print(f"  SIM centroid to target    mean {sims.mean():5.1f} mm   median {np.median(sims):5.1f}   min {sims.min():5.1f}   max {sims.max():5.1f}")
print(f"  unguided (imagined)       mean {ung.mean():5.1f} mm")
print(f"  sim vs generated gap      mean {gaps.mean():5.1f} mm   max {gaps.max():5.1f}")
print(f"  summary: {SUMMARY}")
