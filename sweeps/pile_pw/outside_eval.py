"""Pile: centroid reward vs whole-pile rewards (footprint = sparse, outside = dense), per planner, on the same scenarios, in
the model's prediction and in Genesis.
  centroid   (d0 - |centroid_T - target|) / d0                                   (the paper metric)
  outside    1 - D_T / D_0, D = mean over particles of the xy distance outside the target disc (1.25 x p95 footprint radius)
  in1.0      share of the pile ending inside the disc of 1.0 x the p95 frame-0 footprint radius (the footprint run's disc)
  in1.25     same with the 1.25 x disc (the outside run's disc)"""
import json, os, sys, numpy as np
B = "/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action"; SC = "/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid/seed_42/ep_{:03d}.npz"
REW = [("centroid", "pool_pile_gpwm_200k", ""), ("footprint", "pool_pile_footprint", "_fp"), ("outside", "pool_pile_outside", "_out")]
PL = [("LNO", "a_dno_pool32_I5_P4"), ("Noise CEM", "b_cem_noise_K32_R3")]
EPS = sorted(set(list(range(0, 50, 3)) + [25, 33]))


def metrics(P0, PT, tgt):
    c0 = P0[:, :2].mean(0); d0 = np.linalg.norm(c0 - tgt); r = np.quantile(np.linalg.norm(P0[:, :2] - c0, axis=1), 0.95)
    out = lambda P: np.maximum(np.linalg.norm(P[:, :2] - tgt, axis=1) - 1.25 * r, 0).mean()
    dT = np.linalg.norm(PT[:, :2] - tgt, axis=1)
    return dict(centroid=(d0 - np.linalg.norm(PT[:, :2].mean(0) - tgt)) / d0, outside=1 - out(PT) / out(P0),
                in10=float((dT < r).mean()), in125=float((dT < 1.25 * r).mean()))


def load(pool, tag, ep):
    u = f"{tag}__ep{ep:03d}"; dj = f"{B}/{pool}/done/{u}.json"; rp = f"{B}/{pool}/plans/{u}/replay_genesis_bigbounds/replay.npz"
    if not (os.path.exists(dj) and os.path.exists(rp)): return None
    m = json.load(open(dj))["member"]; z = np.load(f"{B}/{pool}/plans/{u}/rollout.npz"); om = z["object_mask"][m].astype(bool)
    S = np.load(rp)["sim"][0, m][:, om]; Pd = z["predicted"][m][:, om]
    tgt = Pd[0, :, :2].mean(0) + np.load(SC.format(ep))["target_offset_xy"]
    return dict(pred=metrics(Pd[0], Pd[-1], tgt), sim=metrics(S[0], S[-1], tgt))


res = {}
for pname, tag in PL:
    for ep in EPS:
        r = {k: load(pool, tag + suf, ep) for k, pool, suf in REW}
        if all(r.values()): res.setdefault(pname, []).append((ep, r))
for pname, R in res.items():
    print(f"\n== {pname}: {len(R)} scenarios with all three rewards replayed   (cells: Genesis  [model prediction])")
    print(f"{'':12s}" + "".join(f"{k:>20s}" for k, _, _ in REW))
    for m, lab in (("centroid", "centroid"), ("outside", "outside"), ("in10", "inside 1.0x"), ("in125", "inside 1.25x")):
        cells = [f"{np.mean([r[k]['sim'][m] for _, r in R]):8.3f} [{np.mean([r[k]['pred'][m] for _, r in R]):6.3f}]" for k, _, _ in REW]
        print(f"{lab:12s}" + "".join(f"{c:>20s}" for c in cells))
    if "-v" in sys.argv:
        for ep, r in R: print(f"   ep{ep:03d} inside1.25 sim  " + "  ".join(f"{k} {r[k]['sim']['in125']:.2f}" for k, _, _ in REW) + "   outside sim  " + "  ".join(f"{k} {r[k]['sim']['outside']:.2f}" for k, _, _ in REW))
json.dump(res, open("/home/mbronars/.claude/jobs/51ca2775/tmp/outside_eval.json", "w"), default=float)
