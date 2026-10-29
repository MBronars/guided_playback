"""Closed-loop (receding-horizon) guided pile manipulation on a paper cloud-centroid scenario, paper adapter + Genesis.

Per replan (c = frames executed): the last k = min(c, --context) executed transitions enter as clean context rows
(gp/adapters/paper_prefix.py), the plan is k + max(T - c, 16) rows, every model call reads the full 19-row window (the
tail is padded with 16 extra scheduled rows past the plan), the reward is the dough centroid at task frame T. The best
imagined member's next --commit paddle frames are appended to the committed paddle trajectory, which Genesis re-simulates
from the start (frame-0 dough injected); the executed clouds 0..c become the next history. --commit T = matched open loop.
  python bin/closed_loop_mould.py --url http://host:8810/guide --scenario <ep npz> --name clmould_s42_ep000_k25 [--commit 25]"""
import argparse, json, os, sys, time, urllib.request
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from gp.replay import genesis_mould
from gp.closed_loop_hist import history_from_clouds
ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--scenario", required=True); ap.add_argument("--name", required=True)
ap.add_argument("--commit", type=int, default=25); ap.add_argument("--total", type=int, default=150); ap.add_argument("--context", type=int, default=3)
ap.add_argument("--window_future", type=int, default=16, help="the checkpoint's future rows per window (NW - keep)")
ap.add_argument("--members", type=int, default=3); ap.add_argument("--steps", type=int, default=6); ap.add_argument("--iterations", type=int, default=6)
ap.add_argument("--lr", type=float, default=0.1); ap.add_argument("--t_freeze", type=float, default=0.8); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--replay_gpu", default="0")
A = ap.parse_args()
OUT = os.path.join(HERE, "runs", "mould", A.name); os.makedirs(os.path.join(OUT, "states"), exist_ok=True)
sc = np.load(A.scenario); b = int(sc["batch_idx"]); off = sc["target_offset_xy"].astype(float)
T, F = A.total, A.window_future


def post(body):
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


comp = None; meta0 = None; Q = None; rows = []; t_start = time.time(); r = 0; c = 0
while c < T:
    k = min(c, A.context); n_plan = k + max(T - c, F); n_int = n_plan + F; goal = k + (T - c)
    if c == 0:
        hp = "frame0"
    else:
        H = history_from_clouds(Q[:c + 1], meta0["body_poses0"], meta0["particle_body"], meta0["body_type"])
        hp = os.path.join(OUT, "states", f"history_c{c}.npz"); np.savez(hp, positions=H["positions"], body_poses=H["body_poses"])
    name = f"{A.name}_r{r}"; t0 = time.time()
    res = post(dict(domain="mould", task="move", episodes=",".join([str(b)] * A.members), transitions=n_int, steps=A.steps, churn=True, fix_fps=True,
                    optimizer="unitgrad", shell=True, t_freeze=A.t_freeze, offset=f"{off[0]},{off[1]},0", iterations=A.iterations, lr=A.lr, seed=A.seed,
                    history=hp, history_context=A.context, history_stop_at=n_plan, goal_frame=goal, name=name,
                    desc=f"mould closed loop replan {r}: {c} executed, context {k}, plan {n_plan - k} rows (task frames {c + 1}..{c - k + n_plan}), reward at task frame {T} = rollout frame {goal}; scenario {os.path.basename(A.scenario)}"))
    t_guide = time.time() - t0
    pick = int(np.argmin(res["best_mm"])); rd = os.path.join(HERE, "runs", "mould", name)
    z = np.load(os.path.join(rd, "rollout.npz")); P = z["predicted"][pick]                               # rollout frame i = task frame c - k + i
    if comp is None:
        meta0 = {key: z[key][pick] for key in ("roles", "object_mask", "particle_body", "pose_action_mask", "joint_type", "body_type", "object_ids", "body_poses0", "target", "offsets")}
        comp = P[:1].copy()                                                                               # task frame 0 (the episode start)
    n_new = min(A.commit, T - c)
    comp = np.concatenate([comp, P[k + 1:k + 1 + n_new]], 0)                                              # commanded frames c+1 .. c+n_new
    # execute everything committed from the start in Genesis
    ed = os.path.join(OUT, f"exec_r{r}"); os.makedirs(ed, exist_ok=True)
    np.savez(os.path.join(ed, "rollout.npz"), predicted=comp[None].astype(np.float32), unguided=comp[None].astype(np.float32), gt=comp[None].astype(np.float32),
             gt_frames=np.array([comp.shape[0]]), episodes=np.array([b]), step=np.array(0), boxes=np.zeros((0, 6), np.float32), trace_mm=np.zeros((1, 1), np.float32),
             **{key: v[None] for key, v in meta0.items()})
    json.dump(dict(domain="mould", task="move", sim="genesis", name=os.path.basename(ed)), open(os.path.join(ed, "result.json"), "w"))
    t1 = time.time(); genesis_mould.run(ed, os.path.join(ed, "replay_genesis"), variants=("guided",), gpu=A.replay_gpu, members=[0]); t_rep = time.time() - t1
    Q = np.load(os.path.join(ed, "replay_genesis", "replay.npz"))["sim"][0, 0].astype(np.float64)       # executed task frames 0..c+n_new
    obj = meta0["object_mask"].astype(bool); tgt = meta0["target"]
    d_before = float(np.linalg.norm(Q[c, obj].mean(0)[:2] - tgt[:2]) * 1e3) if c > 0 else float(np.linalg.norm(comp[0, obj].mean(0)[:2] - tgt[:2]) * 1e3)
    c += n_new
    d_after = float(np.linalg.norm(Q[c, obj].mean(0)[:2] - tgt[:2]) * 1e3)
    rows.append(dict(replan=r, run=name, context=k, plan_rows=n_plan - k, scheduled=n_int, goal_rollout_frame=goal, committed_before=c - n_new, committed_after=c,
                     imagined_best_mm=[round(v, 1) for v in res["best_mm"]], picked_member=pick, centroid_to_target_before_mm=round(d_before, 1),
                     centroid_to_target_after_mm=round(d_after, 1), guide_seconds=round(res["seconds"], 1), guide_wall_seconds=round(t_guide, 1), replay_seconds=round(t_rep, 1)))
    print(json.dumps(rows[-1]), flush=True); r += 1
obj = meta0["object_mask"].astype(bool); tgt = meta0["target"]
target_mm = float(np.linalg.norm(Q[0, obj].mean(0)[:2] - tgt[:2]) * 1e3); final_mm = float(np.linalg.norm(Q[T, obj].mean(0)[:2] - tgt[:2]) * 1e3)
final = dict(name=A.name, scenario=A.scenario, episode=b, commit=A.commit, total=T, mode=("open loop" if A.commit >= T else "closed loop") + f": context min(c,{A.context}), plan max(remaining,{F}), 19-row windows with tail padding, reward at task frame {T}",
             planner=dict(members=A.members, steps=A.steps, iterations=A.iterations, lr=A.lr, t_freeze=A.t_freeze, optimizer="unitgrad", shell=True, churn=True, fix_fps=True, seed=A.seed),
             target_mm=round(target_mm, 1), sim_final_mm=round(final_mm, 1), distance_closed_mm=round(target_mm - final_mm, 1),
             planned_first_mm=rows[0]["imagined_best_mm"][rows[0]["picked_member"]], sim_moved_mm=round(float(np.linalg.norm(Q[T, obj].mean(0) - Q[0, obj].mean(0)) * 1e3), 1),
             replans=len(rows), guide_seconds_total=round(sum(x["guide_seconds"] for x in rows), 1), replay_seconds_total=round(sum(x["replay_seconds"] for x in rows), 1),
             wall_seconds=round(time.time() - t_start, 1), rows=rows)
json.dump(final, open(os.path.join(OUT, "summary.json"), "w"), indent=1); np.save(os.path.join(OUT, "executed.npy"), Q.astype(np.float32)); np.save(os.path.join(OUT, "commanded.npy"), comp.astype(np.float32))
print("FINAL", json.dumps({kk: v for kk, v in final.items() if kk != "rows"}), flush=True)
