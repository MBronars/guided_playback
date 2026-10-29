"""Closed vs open loop guided planning on random-object pushing (rl_push), all problems in LOCKSTEP so each replan round is
one Isaac launch. Spec: sweeps/rlpush_cl_5x5.json (5 objects x 5 goals). Per round r (c = r * commit executed frames):
for every problem, POST a plan to the warm server with the executed history (context = min(c, 3) clean rows, tail padded,
plan max(remaining, window), reward on task frame H); commit the best-imagined member's next `commit` COMMANDED gripper
frames; then replay every problem's committed commanded trajectory from the start in ONE Isaac launch (deterministic:
checked 0.0 mm across launches and prefix lengths, fixed problem -> env slot order), convert the executed states to history.
Open loop = round 0's plan executed in full (it is the same request as the closed loop's first plan: same sampler, seed 0).
  python bin/closed_loop_rlpush.py --url http://babel-l9-20:8811/guide [--problems L053_t0,...] [--tag rlpcl]"""
import argparse, json, os, sys, time, urllib.request
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; sys.path.insert(0, os.path.join(HERE, "bin")); import rlp_prefix_replay as X
ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--spec", default=os.path.join(HERE, "sweeps", "rlpush_cl_5x5.json"))
ap.add_argument("--problems", default=None); ap.add_argument("--tag", default="rlpcl"); ap.add_argument("--window", type=int, default=16, help="trained future window of the checkpoint")
ap.add_argument("--r0_from", default=None, help="reuse round-0 plans of this tag (they depend only on frame 0)"); ap.add_argument("--replay_gpu", default="1"); ap.add_argument("--rounds", type=int, default=None)
A = ap.parse_args()
SPEC = json.load(open(A.spec)); PL = SPEC["planner"]; H = PL["transitions"]; COMMIT = SPEC["closed_loop"]["commit"]
probs = [p for p in SPEC["problems"] if A.problems is None or p["id"] in A.problems.split(",")]
OUT = os.path.join(HERE, "runs", "rl_push"); ST = os.path.join(OUT, f"{A.tag}_state"); os.makedirs(ST, exist_ok=True)
LOG = open(os.path.join(ST, "driver_log.jsonl"), "a")


def post(body):
    r = urllib.request.Request(A.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    return json.loads(urllib.request.urlopen(r, timeout=24 * 3600).read())


def request(p, name, c, hist_path):
    k = min(c, 3); n_new = max(H - c, A.window); n_plan = k + n_new; goal = k + (H - c)
    b = dict(domain="rl_push", task="push", episodes=",".join([str(p["loader"])] * PL["members"]), transitions=n_plan + A.window, steps=PL["steps"], churn=PL["churn"],
             fix_fps=PL["fix_fps"], t_freeze=PL["t_freeze"], optimizer=PL["optimizer"], shell=PL["shell"], lr=PL["lr"], lr_final=PL["lr_final"], iterations=PL["iterations"],
             seed=PL["seed"], upright_weight=PL["upright_weight"], upright_frames=min(PL["upright_frames"], goal),   # upright over the scored frames (= the whole task at round 0)
             contact_weight=PL["contact_weight"],
             offset=",".join(str(v) for v in p["offset_m"]), history=hist_path, history_context=3, history_full_window=False, history_stop_at=n_plan, goal_frame=goal, name=name,
             desc=f"rl_push closed loop {p['id']} {p['asset']}: {c} executed, context {k}, {n_new} generated (+{A.window} padding), reward at task frame {H} = rollout frame {goal}")
    b.update(SPEC.get("request_extra", {})); b.update(p.get("request_extra", {}))     # e.g. the frame-0 gripper placement of the near-start set
    return b, k, n_plan, goal


state = {}                                    # id -> dict(cmd [c+1, N, 3], item fields)
rounds = (H + COMMIT - 1) // COMMIT if A.rounds is None else A.rounds
for r in range(rounds):
    c = r * COMMIT; t_round = time.time()
    for p in probs:
        name = f"{A.r0_from}_{p['id']}_r0" if (r == 0 and A.r0_from) else f"{A.tag}_{p['id']}_r{r}"
        hp = "frame0" if r == 0 else os.path.join(ST, f"{p['id']}_history_c{c}.npz")      # frame 0: the episode's own start through the same padded sampler
        body, k, n_plan, goal = request(p, name, c, hp)
        t0 = time.time()
        if not os.path.exists(os.path.join(OUT, name, "result.json")):
            res = post(body)
        for _ in range(600):                                                   # the server writes on another node: wait out NFS attribute caching
            if os.path.exists(os.path.join(OUT, name, "result.json")) and os.path.exists(os.path.join(OUT, name, "rollout.npz")): break
            time.sleep(1); os.listdir(os.path.join(OUT))
        rj = json.load(open(os.path.join(OUT, name, "result.json"))); z = np.load(os.path.join(OUT, name, "rollout.npz")); m = int(np.argmin(rj["best_mm"]))
        if r == 0:
            state[p["id"]] = dict(id=p["id"], loader=p["loader"], target=z["target"][m].astype(np.float64), cmd=z["predicted"][m, :1].astype(np.float64),
                                  body_poses0=z["body_poses0"][m], plan0=z["predicted"][m].astype(np.float64), **{kk: z[kk][m] for kk in X.KEYS})
        s = state[p["id"]]; seg = z["predicted"][m, k + 1:k + 1 + min(COMMIT, H - c)].astype(np.float64)
        s["cmd"] = np.concatenate([s["cmd"], seg])
        rec = dict(round=r, id=p["id"], run=name, executed=c, context=k, plan_rows=n_plan, goal_rollout_frame=goal, picked=m, imagined_mm=[round(v, 1) for v in rj["best_mm"]],
                   guide_s=round(sum(rj["iteration_seconds"]), 1), wall_s=round(time.time() - t0, 1))
        LOG.write(json.dumps(rec) + "\n"); LOG.flush(); print(json.dumps(rec), flush=True)
    # one Isaac launch: every problem's committed commanded trajectory from the start (fixed problem -> env slot order)
    items = [state[p["id"]] for p in probs]; rp = os.path.join(OUT, f"{A.tag}_exec_c{c + COMMIT}")
    if not os.path.exists(os.path.join(rp, "replay_isaac", "analysis.json")):
        X.pack(rp, items); t0 = time.time(); X.replay(rp, gpu=A.replay_gpu); t_rep = time.time() - t0
    else:
        t_rep = 0.0
    sims = X.executed(rp)
    for b, p in enumerate(probs):
        h = X.history(sims[b], state[p["id"]])
        np.savez(os.path.join(ST, f"{p['id']}_history_c{c + COMMIT}.npz"), **{kk: v for kk, v in h.items() if kk != "fit_rms_mm_max"})
    an = json.load(open(os.path.join(rp, "replay_isaac", "analysis.json")))
    print(json.dumps(dict(round=r, exec_run=rp, replay_s=round(t_rep, 1), round_s=round(time.time() - t_round, 1),
                          obj_to_target_mm={probs[row["member"]]["id"]: round(row["sim_to_target_mm"], 1) for row in an["rows"]})), flush=True)
    if r == 0:                                                                  # open loop: round 0's plan executed in full
        items_ol = [dict(state[p["id"]], cmd=state[p["id"]]["plan0"][:H + 1]) for p in probs]; ro = os.path.join(OUT, f"{A.tag}_openloop_exec")
        if not os.path.exists(os.path.join(ro, "replay_isaac", "analysis.json")):
            X.pack(ro, items_ol); X.replay(ro, gpu=A.replay_gpu)
        an = json.load(open(os.path.join(ro, "replay_isaac", "analysis.json")))
        print(json.dumps(dict(open_loop=ro, obj_to_target_mm={probs[row["member"]]["id"]: round(row["sim_to_target_mm"], 1) for row in an["rows"]})), flush=True)
print("DONE", flush=True)
