"""One guided (or plain-sampling, --I 0) configuration on a scenario set of block pushing, with SAPIEN replay of the planner's pick
(lowest imagined objective). Everything the server's /guide accepts is a flag. Resumable. Record: <root>/_cfgs/<name>.json with per
scenario imagined / sim / seconds and the per-iteration trace summary (fraction of up-steps). Only result.json/rollout.npz of each
run are kept (no iterates), ~3 MB per scenario.
  python run_cfg.py --url URL --name G_base_P8_I6 --P 8 --I 6 [--lr 0.1 --t_freeze 0.8 --optimizer unitgrad --no_shell] --set dev --n 50"""
import argparse, json, os, subprocess, time, urllib.request
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--name", required=True); ap.add_argument("--P", type=int, required=True); ap.add_argument("--I", type=int, required=True)
ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--lr_final", type=float, default=None); ap.add_argument("--t_freeze", type=float, default=0.8)
ap.add_argument("--optimizer", default="unitgrad"); ap.add_argument("--no_shell", action="store_true"); ap.add_argument("--steps", type=int, default=6)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--set", default="dev", choices=("dev", "test")); ap.add_argument("--n", type=int, default=50)
ap.add_argument("--replay_gpu", default="0"); ap.add_argument("--replays", type=int, default=4); ap.add_argument("--no_replay", action="store_true")
ap.add_argument("--extra", default="{}", help="JSON merged into the /guide body (e.g. objective knobs)")
ap.add_argument("--pad_tail", type=int, default=0, help="full windows to the end: schedule 60 + PAD transitions, stop at 60 (frame-0 history, prefix sampler)")
ap.add_argument("--ode", action="store_true", help="deterministic Euler sampler, optimize only the initial noise P0 (DNO / ReNO)")
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget")
A = ap.parse_args()
SEED_DIR = "seed_0" if A.set == "dev" else "seed_42"; SCEN = f"/data/user_data/mbronars/packages/PointWorld/scenarios/hard/{SEED_DIR}"
RUNS = os.path.join(A.root, "runs"); PUSH = os.path.join(RUNS, "pushing"); os.makedirs(os.path.join(A.root, "_cfgs"), exist_ok=True)
REC = os.path.join(A.root, "_cfgs", f"{A.name}__{A.set}.json")
rec = json.load(open(REC)) if os.path.exists(REC) else dict(name=A.name, set=A.set, args=vars(A), rows=[])
save = lambda: (json.dump(rec, open(REC + ".tmp", "w"), indent=1), os.replace(REC + ".tmp", REC))


def post(body):
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def body(scen, name, it):
    b = dict(domain="pushing", task="push", episodes=",".join(["0"] * A.P), transitions=60, steps=A.steps, churn=not A.ode, optimizer=A.optimizer,
             shell=not A.no_shell, t_freeze=A.t_freeze, scenario=scen, iterations=it, lr=A.lr, lr_final=A.lr_final, free_fps=False, seed=A.seed,
             name=name, runs=RUNS, desc=f"push diag {A.name}")
    if A.pad_tail:                                                    # frame-0 "history" = the scenario start; every model call reads the trained window
        hd = os.path.join(A.root, "hist_frame0", A.set); os.makedirs(hd, exist_ok=True); hp = os.path.join(hd, os.path.basename(scen))
        if not os.path.exists(hp):
            z = np.load(scen); np.savez(hp, block_pos=z["block_pos"][None].astype(np.float32), block_quat=z["block_quat"][None].astype(np.float32), gripper_xy=z["gripper_xy"][None].astype(np.float32))
        b.update(transitions=60 + A.pad_tail, history=hp, history_context=0, history_full_window=False, history_stop_at=60, goal_frame=60)
    b.update(json.loads(A.extra)); return b


post(body(f"{SCEN}/ep_000.npz", f"_warm_{A.name}", min(A.I, 1)))                   # compile / warm this population size
done = {r["scenario"] for r in rec["rows"]}; pending = []


def collect(job, block):
    proc, name, ep, pick, res, rmem = job
    if proc is not None:
        if not block and proc.poll() is None: return False
        proc.wait()
    rj = json.load(open(os.path.join(PUSH, name, "result.json"))); tr = np.array(rj["trace_mm"])
    row = dict(scenario=ep, P=A.P, I=A.I, picked=pick, imagined_mm=rj["best_mm"][pick], best_iter0_mm=float(tr[0].min()),
               plan_seconds=round(float(sum(rj["iteration_seconds"])) + float(rj.get("prefilter_seconds", 0) or 0), 2), prefilter_seconds=rj.get("prefilter_seconds", 0), frac_up=float((np.diff(tr[:-1], axis=0) > 0).mean()) if A.I > 1 else None,
               target_mm=None, sim_mm=None, name=name)
    z = np.load(f"{SCEN}/ep_{ep:03d}.npz"); row["target_mm"] = round(float(np.linalg.norm(z["target_xy"] - z["block_pos"][:2]) * 1e3), 1)
    if proc is not None:
        try:
            a = json.load(open(os.path.join(PUSH, name, "replay_sapien", "analysis.json")))
            row["sim_mm"] = round(next(x for x in a["rows"] if x["member"] == rmem and x["variant"] == "guided")["sim_to_target_mm"], 1)
        except Exception as e:
            print("replay failed", name, e, flush=True)
    rec["rows"].append(row); rec["rows"].sort(key=lambda x: x["scenario"]); save()
    s = [x["sim_mm"] for x in rec["rows"] if x["sim_mm"] is not None]
    print(f"[{A.name}] ep {ep:3d} imagined {row['imagined_mm']:.1f} sim {row['sim_mm']} plan {row['plan_seconds']} s | sim mean {np.mean(s) if s else float('nan'):.1f} over {len(s)}", flush=True)
    return True


for ep in range(A.n):
    if ep in done: continue
    name = f"diag_{A.name}_{A.set}_ep{ep:03d}"
    res = post(body(f"{SCEN}/ep_{ep:03d}.npz", name, A.I)); pick = int(np.argmin(res["best_mm"]))
    for f in ("rollout.npz",):                                                        # disk: keep only the picked member's trajectories
        p = os.path.join(PUSH, name, f)
        if os.path.exists(p) and A.P > 4:
            z = dict(np.load(p)); z = {k: (v[[pick]] if k in ("predicted", "unguided") else v) for k, v in z.items()}; np.savez(p, **z); pick_local = 0
    proc = None
    if not A.no_replay:
        while len(pending) >= A.replays:
            pending = [j for j in pending if not collect(j, False)]
            if len(pending) >= A.replays: time.sleep(2)
        mem = "0" if A.P > 4 else str(pick)
        proc = subprocess.Popen([PY, os.path.join(HERE, "bin", "replay.py"), "--run", os.path.join(PUSH, name), "--variants", "guided", "--members", mem,
                                 "--gpu", A.replay_gpu], cwd=HERE, stdout=open(os.path.join(PUSH, name, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT)
    job = (proc, name, ep, pick, res, pick if A.P <= 4 else 0)
    if proc is None: collect(job, True)
    else: pending.append(job)
for j in pending: collect(j, True)
s = np.array([x["sim_mm"] for x in rec["rows"] if x["sim_mm"] is not None]); t = np.array([x["plan_seconds"] for x in rec["rows"]])
im = np.array([x["imagined_mm"] for x in rec["rows"]]); tg = np.array([x["target_mm"] for x in rec["rows"]])
rec["summary"] = dict(n=len(rec["rows"]), plan_seconds=round(float(t.mean()), 2), imagined_mm=round(float(im.mean()), 2),
                      sim_mm=round(float(s.mean()), 2) if len(s) else None, closed_sim_mm=round(float((tg[:len(s)] - s).mean()), 2) if len(s) else None)
save(); print("CFG_DONE", A.name, rec["summary"], flush=True)
