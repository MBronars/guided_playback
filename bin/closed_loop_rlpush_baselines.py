"""Open vs closed loop for the rl_push BASELINES (PointWorld MPPI-argmin, GNS receding best-of-K), scored exactly like our
planner (bin/closed_loop_rlpush.py): every problem's committed COMMANDED gripper trajectory replayed from the start in ONE
Isaac launch per round (fixed problem -> env slot order, bin/rlp_prefix_replay.py), final object-centroid-to-goal distance.

Open loop: one plan from frame 0 over the whole horizon, all frames executed. Closed loop: plan, commit `commit` frames,
execute everything committed in Isaac, read the executed state back, replan the frames left. Round 0's closed-loop plan is the
open-loop plan (same inputs, same seed), so it is computed once. What each baseline is given as its observation:
  PointWorld: the executed object points at frame c (its anchor; context length 1) and the executed hand pose at frame c
              (fitted to the executed gripper rows); fingers held at the frame-0 opening (as in its benchmark).
  GNS:        the last 6 executed frames (context length 6), frame 0 repeated in front when fewer exist (its open-loop start
              is frame 0 repeated 6 times).
Planners run on the local L40S (their environments have no Blackwell kernels), Isaac on the same GPU between planner calls.

  python bin/closed_loop_rlpush_baselines.py --spec <json> --method pw|gns --tag <tag> [--problems id,id] [--det_check]
spec json: {"problems": [{"id", "loader", "state": <frame-0 state npz>, "target"?: [3] model frame}], "horizon"?: 40, "commit"?: 10}"""
import argparse, json, os, subprocess, sys, time
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; sys.path.insert(0, os.path.join(HERE, "bin"))
import rlp_prefix_replay as X                                              # noqa: E402
import rlpb_common as C                                                    # noqa: E402
ENV = dict(pw="/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python", gns="/data/user_data/mbronars/.miniconda3/envs/gns_torch/bin/python")
ap = argparse.ArgumentParser(); ap.add_argument("--spec", required=True); ap.add_argument("--method", choices=("pw", "gns"), required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--problems", default=None); ap.add_argument("--gpu", default="1")
ap.add_argument("--horizon", type=int, default=None); ap.add_argument("--commit", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--det_check", action="store_true", help="replay the open-loop launch a second time and report the max difference")
ap.add_argument("--open_only", action="store_true")
A = ap.parse_args()
SPEC = json.load(open(A.spec)); H = A.horizon or int(SPEC.get("horizon", 40)); COMMIT = A.commit or int(SPEC.get("commit", 10))
probs = [p for p in SPEC["problems"] if A.problems is None or p["id"] in A.problems.split(",")]
OUT = os.path.join(HERE, "runs", "rl_push_baselines"); ST = os.path.join(OUT, f"{A.tag}_{A.method}_state"); os.makedirs(ST, exist_ok=True)
LOG = open(os.path.join(ST, "driver_log.jsonl"), "a")


def log(rec):
    LOG.write(json.dumps(rec) + "\n"); LOG.flush(); print(json.dumps(rec), flush=True)


S = {}
for p in probs:
    s = C.load_state(p["state"], p.get("target")); err, (R, t, q, fq) = C.check_layout(s)
    assert err < 0.2, f"{p['id']}: frame-0 gripper rows not rebuilt by the hand fit ({err:.3f} mm): row order differs"
    S[p["id"]] = dict(s, fq0=fq, item=dict(id=p["id"], loader=int(p["loader"]), target=s["target"], body_poses0=s["body_poses0"], cmd=s["P0"][None].copy(),
                                            **{k: np.asarray(s[k]) for k in X.KEYS}))


def plan(tag, inputs, horizon):
    """one planner process for all problems; inputs: list of per-problem dicts at the current state."""
    inp, out = os.path.join(ST, f"{tag}_in.npz"), os.path.join(ST, f"{tag}_out.npz")
    if not os.path.exists(out):
        if A.method == "pw":
            np.savez(inp, obj=np.stack([d["obj"] for d in inputs]), R0=np.stack([d["R"] for d in inputs]), t0=np.stack([d["t"] for d in inputs]),
                     fq0=np.stack([d["fq0"] for d in inputs]), target=np.stack([d["target"] for d in inputs]), horizon=horizon, seed=A.seed)
            cmd = [ENV["pw"], "-u", os.path.join(HERE, "bin", "pw_plan_rlp.py"), "--inp", inp, "--out", out]; cwd = "/data/user_data/mbronars/packages/pwm_final_paper"
        else:
            np.savez(inp, hist=np.stack([d["hist"] for d in inputs]).astype(np.float32), body=np.stack([d["body"] for d in inputs]),
                     target=np.stack([d["target"] for d in inputs]), horizon=horizon, seed=A.seed)
            cmd = [ENV["gns"], "-u", os.path.join(HERE, "bin", "gns_plan_rlp.py"), "--inp", inp, "--out", out]; cwd = HERE
        with open(os.path.join(ST, f"{tag}_planner.log"), "w") as lf:
            subprocess.run(cmd, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT, check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(A.gpu)))
    return np.load(out), json.load(open(out[:-4] + ".json"))


def frames_from_plan(pl, b, s, n):
    """the next n commanded frames [n, N, 3] (gripper rows; object rows held at frame 0) + the planner's final predicted object distance."""
    if A.method == "pw":
        return C.cmd_frames(s["P0"], s["roles"], pl["hand_pos"][b][1:1 + n].astype(np.float64), pl["hand_quat"][b][1:1 + n].astype(np.float64),
                            pl["finger_q"][b][1:1 + n].astype(np.float64))
    F = pl["traj"][b][:n].astype(np.float64); om = np.asarray(s["object_mask"]).astype(bool); F[:, om] = s["P0"][om]; return F


def planned_mm(pl, b, s):
    if A.method == "pw": return float(np.linalg.norm(pl["object_pred"][b][-1].mean(0) - s["target"]) * 1e3)
    return float(pl["pred_mm"][b])


def state_inputs(sim_hist):
    """per problem: the planner's observation from the executed clouds 0..c (None = frame 0)."""
    ins = []
    for p in probs:
        s = S[p["id"]]; pb = np.asarray(s["particle_body"]); om = np.asarray(s["object_mask"]).astype(bool)
        pos = s["P0"][None] if sim_hist is None else sim_hist[p["id"]]
        cur = pos[-1]; R, t, q, fq = C.fit_hand(cur, pb)
        hist = np.concatenate([np.repeat(s["P0"][None], max(0, 6 - pos.shape[0]), 0), pos], 0)[-6:]
        ins.append(dict(obj=cur[om], R=R, t=t, fq0=s["fq0"], target=s["target"], hist=hist, body=pb))
    return ins


def launch(name, items):
    rp = os.path.join(OUT, name); t0 = time.time()
    if not os.path.exists(os.path.join(rp, "replay_isaac", "analysis.json")):
        X.pack(rp, items, name=name); X.replay(rp, gpu=A.gpu)
    an = json.load(open(os.path.join(rp, "replay_isaac", "analysis.json")))
    return rp, X.executed(rp), {probs[r["member"]]["id"]: r for r in an["rows"] if r["variant"] == "guided"}, time.time() - t0


res = dict(method=A.method, tag=A.tag, horizon=H, commit=COMMIT, spec=os.path.abspath(A.spec), problems=[p["id"] for p in probs], open={}, closed={})
# ---- round 0 plan = the open-loop plan
t0 = time.time(); pl0, pj0 = plan("r0", state_inputs(None), H); t_plan0 = time.time() - t0
items_ol = []
for b, p in enumerate(probs):
    s = S[p["id"]]; items_ol.append(dict(s["item"], cmd=np.concatenate([s["P0"][None], frames_from_plan(pl0, b, s, H)])))
rp_ol, sim_ol, an_ol, t_ol = launch(f"{A.tag}_{A.method}_ol_exec", items_ol)
for b, p in enumerate(probs):
    r = an_ol[p["id"]]; res["open"][p["id"]] = dict(final_mm=round(r["sim_to_target_mm"], 1), planned_mm=round(planned_mm(pl0, b, S[p["id"]]), 1),
        moved_mm=round(r["sim_moved_mm"], 1), plan_seconds=round(float(pj0["rows"][b]["seconds"]), 1), replay=os.path.join(rp_ol, "replay_isaac", "replay.npz"), slot=b)
log(dict(open_loop=rp_ol, isaac_s=round(t_ol, 1), plan_wall_s=round(t_plan0, 1), final_mm={k: v["final_mm"] for k, v in res["open"].items()}))
if A.det_check:
    rp2, sim2, _, _ = launch(f"{A.tag}_{A.method}_ol_exec_dup", items_ol)
    res["determinism_max_mm"] = round(float(np.abs(sim2 - sim_ol).max() * 1e3), 4); log(dict(determinism_max_mm=res["determinism_max_mm"], dup=rp2))
if not A.open_only:
    hist = None; rounds = (H + COMMIT - 1) // COMMIT; per = {p["id"]: [] for p in probs}
    for r in range(rounds):
        c = r * COMMIT; n = min(COMMIT, H - c)
        if r == 0: pl, pj, t_plan = pl0, pj0, t_plan0
        else: t0 = time.time(); pl, pj = plan(f"r{r}", state_inputs(hist), H - c); t_plan = time.time() - t0
        for b, p in enumerate(probs):
            s = S[p["id"]]; s["item"]["cmd"] = np.concatenate([s["item"]["cmd"], frames_from_plan(pl, b, s, n)])
            per[p["id"]].append(dict(round=r, executed_before=c, planned_mm=round(planned_mm(pl, b, s), 1), plan_seconds=round(float(pj["rows"][b]["seconds"]), 1)))
        rp, sims, an, t_is = launch(f"{A.tag}_{A.method}_cl_exec_c{c + n}", [S[p["id"]]["item"] for p in probs])
        hist = {}
        for b, p in enumerate(probs):
            hist[p["id"]] = X.history(sims[b], S[p["id"]]["item"])["positions"].astype(np.float64)
            per[p["id"]][-1].update(obj_to_target_after_mm=round(an[p["id"]]["sim_to_target_mm"], 1), isaac_s=round(t_is, 1))
        log(dict(round=r, exec=rp, plan_wall_s=round(t_plan, 1), isaac_s=round(t_is, 1), obj_to_target_mm={k: round(v["sim_to_target_mm"], 1) for k, v in an.items()}))
    for b, p in enumerate(probs):
        r = an[p["id"]]; res["closed"][p["id"]] = dict(final_mm=round(r["sim_to_target_mm"], 1), moved_mm=round(r["sim_moved_mm"], 1),
            plan_seconds=round(sum(x["plan_seconds"] for x in per[p["id"]]), 1), rounds=per[p["id"]], replay=os.path.join(rp, "replay_isaac", "replay.npz"), slot=b)
json.dump(res, open(os.path.join(OUT, f"{A.tag}_{A.method}_results.json"), "w"), indent=1)
print("DONE", os.path.join(OUT, f"{A.tag}_{A.method}_results.json"), flush=True)
