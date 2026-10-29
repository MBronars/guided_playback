"""The paper's PushBoundary scenarios (block pushing) run through this package.

Scenarios: PointWorld/scenarios/<difficulty>/seed_<S>/ep_NNN.npz -- each pins the FULL initial state
(block_pos, block_quat, gripper_xy) and the target (target_xy), in world coordinates. gp/push_scenes.py re-poses a
template episode's frame-0 clouds onto that state, so the model sees the scenario, not a dataset clip.

Per scenario: one POPULATION of --members rollouts of that same scenario, guided through the warm server; the member
with the best IMAGINED objective is picked (what you can do at deployment) and replayed in SAPIEN; the reported number
is that member's SIMULATED cube-centroid DISTANCE TO TARGET. Replays run in the background on another GPU while the
next scenario is guided.

  python bin/paper_eval_push.py --url http://host:8795/guide --difficulty hard --seed 42 --n 50 [--members 3]
"""
import argparse, json, os, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCEN = "/data/user_data/mbronars/packages/PointWorld/scenarios"
PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--difficulty", default="hard"); ap.add_argument("--n", type=int, default=50)
ap.add_argument("--members", type=int, default=3); ap.add_argument("--template", default="0")
ap.add_argument("--transitions", type=int, default=60); ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--iterations", type=int, default=10); ap.add_argument("--lr", type=float, default=0.2)
ap.add_argument("--lr_final", type=float, default=0.02, help="negative = no decay (constant lr)"); ap.add_argument("--t_freeze", type=float, default=0.8)
ap.add_argument("--replay_gpu", default="0"); ap.add_argument("--tag", default=None)
ap.add_argument("--start", type=int, default=0); ap.add_argument("--no_warmup", action="store_true")
ap.add_argument("--scenarios", default=None, help="explicit comma list of scenario indices (overrides --start/--n)")
ap.add_argument("--no_shell", action="store_true"); ap.add_argument("--optimizer", default="unitgrad")
ap.add_argument("--free_fps", action="store_true", help="skip the per-rollout RNG reseed: FPS starts drawn fresh on every model call")
A = ap.parse_args()
TAG = A.tag or f"push_{A.difficulty}{A.seed}"
OUT = os.path.join(HERE, "runs", "pushing")
os.makedirs(os.path.join(HERE, "runs", "_paper_eval"), exist_ok=True)
SUMMARY = os.path.join(HERE, "runs", "_paper_eval", f"{TAG}_summary.json")
LR_FINAL = A.lr_final if A.lr_final >= 0 else None          # negative -> constant lr
rows, pending, t_replay0 = [], None, [time.time()]


def post(body):
    import urllib.request
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def body_for(path, name, iterations, desc):
    return dict(domain="pushing", task="push", episodes=",".join([A.template] * A.members), transitions=A.transitions,
                steps=A.steps, churn=True, optimizer=A.optimizer, shell=not A.no_shell, t_freeze=A.t_freeze, scenario=path,
                iterations=iterations, lr=A.lr, lr_final=LR_FINAL, name=name, desc=desc, free_fps=A.free_fps)


def collect(job):
    proc, name, ep, pick, res, target_mm, t_guide = job
    proc.wait()
    a = json.load(open(os.path.join(OUT, name, "replay_sapien", "analysis.json")))
    row = next(r for r in a["rows"] if r["member"] == pick and r["variant"] == "guided")
    rec = dict(scenario=ep, target_mm=round(target_mm, 1), picked_member=pick,
               imagined_mm=round(res["best_mm"][pick], 1), sim_mm=round(row["sim_to_target_mm"], 1),
               gap_mm=round(row["sim_vs_gen_final_mm"], 1), gen_moved_mm=round(row["gen_moved_mm"], 1),
               sim_moved_mm=round(row["sim_moved_mm"], 1), unguided_imagined_mm=round(res["unguided_mm"][pick], 1),
               guide_seconds=res["seconds"], guide_wall_seconds=round(t_guide, 1),
               replay_seconds=round(time.time() - t_replay0[0], 1), name=name)
    rows.append(rec)
    json.dump(dict(tag=TAG, seed=A.seed, difficulty=A.difficulty,
                   settings=dict(members=A.members, template_episode=A.template, transitions=A.transitions, steps=A.steps,
                                 iterations=A.iterations, lr=A.lr, lr_final=LR_FINAL, t_freeze=A.t_freeze,
                                 shell=not A.no_shell, churn=True, optimizer=A.optimizer, free_fps=A.free_fps), rows=rows), open(SUMMARY, "w"), indent=1)
    sims = [r["sim_mm"] for r in rows]
    print(f"[{len(rows):2d}/{A.n}] scenario {ep:3d} target {rec['target_mm']:5.1f} mm | imagined {rec['imagined_mm']:5.1f} "
          f"-> SIM {rec['sim_mm']:5.1f} mm (gap {rec['gap_mm']:4.1f}) | running mean sim {np.mean(sims):5.1f} mm over {len(sims)}", flush=True)


EPS = [int(v) for v in A.scenarios.split(",")] if A.scenarios else list(range(A.start, A.start + A.n))
A.n = len(EPS)
P0 = f"{SCEN}/{A.difficulty}/seed_{A.seed}/ep_{EPS[0]:03d}.npz"
if not A.no_warmup:
    t0 = time.time(); post(body_for(P0, "_warmup_push", 1, "compile the graphs")); print(f"warmup (compile) {time.time() - t0:.0f} s", flush=True)

for ep in EPS:
    path = f"{SCEN}/{A.difficulty}/seed_{A.seed}/ep_{ep:03d}.npz"
    z = np.load(path); target_mm = float(np.linalg.norm(z["target_xy"] - z["block_pos"][:2]) * 1e3)
    name = f"{TAG}_ep{ep:03d}"
    t0 = time.time(); res = post(body_for(path, name, A.iterations, f"paper {A.difficulty} seed {A.seed} scenario {ep}: target {target_mm:.0f} mm"))
    t_guide = time.time() - t0
    pick = int(np.argmin(res["best_mm"]))
    print(f"  guided scenario {ep} in {res['seconds']:.0f} s; imagined {[round(v, 1) for v in res['best_mm']]} -> member {pick}", flush=True)
    if pending is not None:
        collect(pending)
    proc = subprocess.Popen([PY, os.path.join(HERE, "bin", "replay.py"), "--run", os.path.join(OUT, name),
                             "--variants", "guided", "--members", str(pick), "--gpu", A.replay_gpu],
                            cwd=HERE, stdout=open(os.path.join(OUT, name, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT)
    t_replay0[0] = time.time()
    pending = (proc, name, ep, pick, res, target_mm, t_guide)
if pending is not None:
    collect(pending)

sims = np.array([r["sim_mm"] for r in rows]); tgt = np.array([r["target_mm"] for r in rows])
gaps = np.array([r["gap_mm"] for r in rows]); sec = np.array([r["guide_seconds"] for r in rows])
print(f"\n== {TAG}: {len(rows)} scenarios, {A.difficulty} seed {A.seed}")
print(f"  SIM cube to target   mean {sims.mean():5.1f} mm   median {np.median(sims):5.1f}   min {sims.min():5.1f}   max {sims.max():5.1f}")
print(f"  start distance       mean {tgt.mean():5.1f} mm")
print(f"  sim vs generated gap mean {gaps.mean():5.1f} mm   max {gaps.max():5.1f}")
print(f"  guide seconds (warm) mean {sec.mean():5.1f} s   median {np.median(sec):5.1f}   max {sec.max():5.1f}")
print(f"  summary: {SUMMARY}")
