"""Pile, centroid reward vs whole-pile (footprint) reward: per scenario and planner, in the model's prediction and in Genesis,
  centroid progress   (d0 - |centroid_T - target|) / d0                         (the paper metric)
  footprint progress  same, with particles outside the moved-footprint disc counted at their start (hard edge)
  inside              share of the pile ending inside the moved-footprint disc (radius = p95 of the frame-0 footprint)."""
import glob, json, os, sys, numpy as np
B = "/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action"; SC = "/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid/seed_42/ep_{:03d}.npz"
PAIRS = [("LNO", "pool_pile_gpwm_200k", "a_dno_pool32_I5_P4", "pool_pile_footprint", "a_dno_pool32_I5_P4_fp"),
         ("Noise CEM", "pool_pile_gpwm_200k", "b_cem_noise_K32_R3", "pool_pile_footprint", "b_cem_noise_K32_R3_fp")]


def metrics(P0, PT, tgt):
    c0 = P0[:, :2].mean(0); d0 = np.linalg.norm(c0 - tgt); R = np.quantile(np.linalg.norm(P0[:, :2] - c0, axis=1), 0.95)
    inside = np.linalg.norm(PT[:, :2] - tgt, axis=1) < R
    mixed = np.where(inside[:, None], PT[:, :2], P0[:, :2]).mean(0)
    return dict(centroid=(d0 - np.linalg.norm(PT[:, :2].mean(0) - tgt)) / d0, footprint=(d0 - np.linalg.norm(mixed - tgt)) / d0, inside=float(inside.mean()))


def load(pool, tag, ep):
    u = f"{tag}__ep{ep:03d}"; dj = f"{B}/{pool}/done/{u}.json"; rp = f"{B}/{pool}/plans/{u}/replay_genesis_bigbounds/replay.npz"
    if not (os.path.exists(dj) and os.path.exists(rp)): return None
    m = json.load(open(dj))["member"]; z = np.load(f"{B}/{pool}/plans/{u}/rollout.npz"); om = z["object_mask"][m].astype(bool)
    S = np.load(rp)["sim"][0, m][:, om]; Pd = z["predicted"][m][:, om]
    tgt = Pd[0, :, :2].mean(0) + np.load(SC.format(ep))["target_offset_xy"]
    return dict(pred=metrics(Pd[0], Pd[-1], tgt), sim=metrics(S[0], S[-1], tgt))


eps = sorted({int(os.path.basename(f).split("__ep")[1][:3]) for f in glob.glob(f"{B}/pool_pile_footprint/units/*.json")})
rows = {}
for name, p0, t0, p1, t1 in PAIRS:
    for ep in eps:
        a, b = load(p0, t0, ep), load(p1, t1, ep)
        if a and b: rows.setdefault(name, []).append((ep, a, b))
for name, R in rows.items():
    print(f"\n== {name}: {len(R)} scenarios with both plans replayed (old = centroid reward, new = whole-pile reward)")
    print(f"{'':22s}{'old pred':>9s}{'old sim':>9s}{'new pred':>9s}{'new sim':>9s}")
    for k, lab in (("centroid", "centroid progress"), ("footprint", "whole-pile progress"), ("inside", "share inside circle")):
        v = lambda which, src: np.mean([r[1 if which == "old" else 2][src][k] for r in R])
        print(f"{lab:22s}{v('old', 'pred'):9.3f}{v('old', 'sim'):9.3f}{v('new', 'pred'):9.3f}{v('new', 'sim'):9.3f}")
    if "-v" in sys.argv:
        for ep, a, b in R: print(f"   ep{ep:03d}  inside sim old {a['sim']['inside']:.2f} new {b['sim']['inside']:.2f} | centroid sim old {a['sim']['centroid']:.2f} new {b['sim']['centroid']:.2f}")
json.dump({n: [(e, a, b) for e, a, b in R] for n, R in rows.items()}, default=float, fp=open("/home/mbronars/.claude/jobs/51ca2775/tmp/footprint_eval.json", "w"))
