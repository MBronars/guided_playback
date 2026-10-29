"""Causal probe: is the fake dough motion of all-noise guidance on pile caused by the drift of the dough-noise MEAN?

Per scenario (paper cloud_centroid seed 42), P4, all noise optimized (pose + dough particle draws), Adam lr 0.1, I5, fixed FPS, shell:
  initial         the sampled noise, no optimization
  optimized       the guided plan's best noise (what the planner commits)
  opt-no-drift    optimized noise, but the dough noise's mean over dough points (per transition, per axis) reset to its sampled value
  init+drift      sampled noise (paddle untouched) + ONLY that mean shift added to the dough noise
  opt-paddle      optimized paddle noise, sampled dough noise
Each variant is rolled out by the model (no optimization) and the planner's member is replayed in Genesis. Writes
<out>/probe.json: per scenario and variant the model's and Genesis's dough-centroid paths, the paddle path, the paddle-dough distance.
Runs in-process on CUDA_VISIBLE_DEVICES (queue job, no server)."""
import argparse, json, os, subprocess, sys
import numpy as np, torch
HERE = "/data/user_data/mbronars/packages/guided_playback"; sys.path.insert(0, HERE)
from gp import session as SESSION
from gp.session import Args, DEFAULTS, load_adapter
PY = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
SCEN = "/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid/seed_42"
CKPT = "/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mould_ctxgen_100m_uncond_unet_bidir/checkpoints/step_60000.pt"
ap = argparse.ArgumentParser(); ap.add_argument("--scenarios", default="0,3,5"); ap.add_argument("--I", type=int, default=5); ap.add_argument("--lr", type=float, default=0.1)
ap.add_argument("--root", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget"); ap.add_argument("--tag", default="probe")
A0 = ap.parse_args(); RUNS = os.path.join(A0.root, "runs"); OUT = os.path.join(A0.root, "diag", f"pile_noise_{A0.tag}"); os.makedirs(OUT, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True
base = Args(DEFAULTS, adapter="paper", ckpt=CKPT, runs=RUNS, eager=False)
AD = load_adapter(base); ORIG_PREPARE = AD.prepare


def request(ep, name, iterations, noise=None, **over):
    z = np.load(f"{SCEN}/ep_{ep:03d}.npz"); b = int(z["batch_idx"]); off = z["target_offset_xy"].astype(float)
    A = Args(base, domain="mould", task="move", episodes=",".join([str(b)] * 4), transitions=150, steps=6, churn=False, fix_fps=True, optimizer="adam",
             shell=True, t_freeze=0.0, offset=f"{off[0]},{off[1]},0", iterations=iterations, lr=A0.lr, seed=0, name=name, runs=RUNS, save_best_noise=True,
             desc=f"pile noise probe scenario {ep}")
    A.update(over)
    if "P" in over: A["episodes"] = ",".join([str(b)] * int(over.pop("P"))); A.pop("P", None)
    if noise is not None:                                            # roll out exactly these draws: swap them into the freshly prepared state
        def prep(*a, **k):
            st = ORIG_PREPARE(*a, **k)
            for v, n in zip(st.vars, noise): v.data.copy_(torch.as_tensor(n, device=v.device, dtype=v.dtype))
            return st
        AD.prepare = prep
    try:
        out, res = SESSION.guided_run(AD, A)
    finally:
        AD.prepare = ORIG_PREPARE
    return out, res, float(np.linalg.norm(off) * 1e3)


def replay(run, member):
    subprocess.run([PY, "-u", "bin/replay.py", "--run", run, "--variants", "guided", "--members", str(member), "--gpu", "0"], cwd=HERE,
                   stdout=open(os.path.join(run, "replay_stdout.log"), "w"), stderr=subprocess.STDOUT)
    a = next(x for x in json.load(open(os.path.join(run, "replay_genesis", "analysis.json")))["rows"] if x["member"] == member and x["variant"] == "guided")
    sim = np.load(os.path.join(run, "replay_genesis", "replay.npz"))["sim"][0]          # [members, T, P, 3]; only replayed slots are filled
    nz = [i for i in range(sim.shape[0]) if np.abs(sim[i]).sum() > 0]
    sim = sim[member if member in nz else nz[0]]
    return a, sim


res_all = {}
for ep in [int(v) for v in A0.scenarios.split(",")]:
    o_init, r_init, target = request(ep, f"probe_{A0.tag}_ep{ep:03d}_initial", 0)
    X0 = np.load(os.path.join(o_init, "best_noise.npz")); init = [X0["v0"], X0["v1"]]
    o_opt, r_opt, _ = request(ep, f"probe_{A0.tag}_ep{ep:03d}_optimized", A0.I)
    Xs = np.load(os.path.join(o_opt, "best_noise.npz")); opt = [Xs["v0"], Xs["v1"]]
    k = int(np.argmin(r_opt["best_mm"]))                               # the planner's member (lowest imagined objective)
    roll = np.load(os.path.join(o_opt, "rollout.npz")); m = roll["object_mask"][0].astype(bool)
    drift = opt[1][:, :, m].mean(2, keepdims=True) - init[1][:, :, m].mean(2, keepdims=True)    # [B, N, 1, 3]
    no_drift = opt[1].copy(); no_drift[:, :, m] -= drift
    plus_drift = init[1].copy(); plus_drift[:, :, m] += drift
    variants = {"initial": None, "optimized": None, "opt-no-drift": [opt[0], no_drift], "init+drift": [init[0], plus_drift], "opt-paddle": [opt[0], init[1]]}
    rows = {}
    for vname, noise in variants.items():
        if vname == "initial": run = o_init
        elif vname == "optimized": run = None
        else: run, _, _ = request(ep, f"probe_{A0.tag}_ep{ep:03d}_{vname}", 0, noise)
        if vname == "optimized":                                       # the committed plan = the guided run's best iterate
            run = o_opt
        pred = np.load(os.path.join(run, "rollout.npz"))["predicted"][k]            # [T, P, 3]
        a, sim = replay(run, k)
        pad = ~m
        dist = [float(np.min(np.linalg.norm(pred[t, pad][:, None] - pred[t, m][None], axis=-1))) for t in range(0, pred.shape[0], 3)]
        rows[vname] = dict(model_centroid=pred[:, m].mean(1)[:, :2].round(4).tolist(), sim_centroid=sim[:, m].mean(1)[:, :2].round(4).tolist() if sim.shape[1] == m.size else None,
                           paddle=pred[:, pad].mean(1)[:, :2].round(4).tolist(), paddle_dough_min_m=[round(d, 4) for d in dist],
                           target_mm=round(target, 1), model_to_target_mm=round(a["gen_to_target_mm"], 1), sim_to_target_mm=round(a["sim_to_target_mm"], 1),
                           model_moved_mm=round(a["gen_moved_mm"], 1), sim_moved_mm=round(a["sim_moved_mm"], 1),
                           drift_rms=round(float(np.sqrt(m.sum()) * np.sqrt((drift[k] ** 2).mean())), 2), run=run)
        print(f"ep {ep} {vname:13s} model moved {a['gen_moved_mm']:6.1f} sim moved {a['sim_moved_mm']:6.1f} | model to target {a['gen_to_target_mm']:6.1f} sim {a['sim_to_target_mm']:6.1f}", flush=True)
    # the OLD paper recipe on the same scenario: churn sampler, per-draw step lr 0.1, t_freeze 0.8, P3 I6 (its own planner pick)
    OLD = dict(churn=True, t_freeze=0.8, optimizer="unitgrad", lr=0.1, P=3)
    o_oi, _, _ = request(ep, f"probe_{A0.tag}_ep{ep:03d}_old-initial", 0, **OLD)
    o_old, r_old, _ = request(ep, f"probe_{A0.tag}_ep{ep:03d}_old-recipe", 6, **OLD)
    ko = int(np.argmin(r_old["best_mm"])); E0 = np.load(os.path.join(o_oi, "best_noise.npz")); E1 = np.load(os.path.join(o_old, "best_noise.npz"))
    po = np.load(os.path.join(o_old, "rollout.npz"))["predicted"][ko]; a, sim = replay(o_old, ko)
    dX = (E1["v1"] - E0["v1"])[ko][:, :, m]                             # [N, L, Nd, 3] change of the dough churn draws (only levels >= 0.8 move)
    lv_moved = np.abs(dX).reshape(dX.shape[0], dX.shape[1], -1).max(-1) > 0
    old_drift = np.sqrt(m.sum()) * dX.mean(2)                          # mean shift over dough points, same z-scale as above
    rows["old-recipe"] = dict(model_centroid=po[:, m].mean(1)[:, :2].round(4).tolist(), sim_centroid=sim[:, m].mean(1)[:, :2].round(4).tolist(),
                              paddle=po[:, ~m].mean(1)[:, :2].round(4).tolist(),
                              paddle_dough_min_m=[round(float(np.min(np.linalg.norm(po[t, ~m][:, None] - po[t, m][None], axis=-1))), 4) for t in range(0, po.shape[0], 3)],
                              target_mm=round(target, 1), model_to_target_mm=round(a["gen_to_target_mm"], 1), sim_to_target_mm=round(a["sim_to_target_mm"], 1),
                              model_moved_mm=round(a["gen_moved_mm"], 1), sim_moved_mm=round(a["sim_moved_mm"], 1),
                              drift_rms=round(float(np.sqrt((old_drift[lv_moved] ** 2).mean())) if lv_moved.any() else 0.0, 2), run=o_old)
    print(f"ep {ep} old-recipe    model moved {a['gen_moved_mm']:6.1f} sim moved {a['sim_moved_mm']:6.1f} | drift rms (moved draws) {rows['old-recipe']['drift_rms']}", flush=True)
    t0 = np.array(roll["target"][k]) if "target" in roll.files else None
    res_all[str(ep)] = dict(member=k, target_xy=None if t0 is None else t0[:2].round(4).tolist(), dough0=roll["predicted"][k, 0][m][::4, :2].round(4).tolist(), variants=rows)
    json.dump(res_all, open(os.path.join(OUT, "probe.json"), "w"))
print("PROBE_DONE", OUT, flush=True)
