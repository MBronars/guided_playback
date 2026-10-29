"""Object tilt (deg, Kabsch of the object particles from frame 0) per member for a run: generated final/max and, if a
replay_isaac exists, simulated final/max; plus planner pick and distances. python sweeps/rl_push/tilt.py runs/rl_push/<name> [...]"""
import json, os, sys, numpy as np

def kabsch_tilt(P0, Pt):
    A = P0 - P0.mean(0); B = Pt - Pt.mean(0); H = A.T @ B; U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

def tilts(P, mask):
    o = P[:, mask]; return np.array([kabsch_tilt(o[0], o[t]) for t in range(o.shape[0])])

def report(run):
    z = np.load(os.path.join(run, "rollout.npz")); rj = json.load(open(os.path.join(run, "result.json")))
    B = z["predicted"].shape[0]; pick = int(np.argmin(rj["best_mm"])); rows = []
    sim = None; an = None
    rp = os.path.join(run, "replay_isaac" + os.environ.get("GP_TILT_SUFFIX", ""))
    if os.path.exists(os.path.join(rp, "replay.npz")):
        R = np.load(os.path.join(rp, "replay.npz")); vars_ = json.loads(str(R["variants"])); sim = R["sim"][vars_.index("guided")] if "guided" in vars_ else None
        an = {r["member"]: r for r in json.load(open(os.path.join(rp, "analysis.json")))["rows"] if r["variant"] == "guided"}
    for m in range(B):
        mask = z["object_mask"][m]; g = tilts(z["predicted"][m], mask)
        row = dict(member=m, pick=(m == pick), gen_mm=round(float(rj["best_mm"][m]), 1), gen_tilt_final=round(float(g[-1]), 1), gen_tilt_max=round(float(g.max()), 1))
        if sim is not None:
            s = tilts(sim[m], mask); row.update(sim_tilt_final=round(float(s[-1]), 1), sim_tilt_max=round(float(s.max()), 1))
            if an and m in an: row.update(sim_mm=round(an[m]["sim_to_target_mm"], 1), gap_mm=round(an[m]["sim_vs_gen_final_mm"], 1), sim_lift_mm=round(an[m]["sim_lift_mm"], 1))
        rows.append(row)
    return rows

if __name__ == "__main__":
    for run in sys.argv[1:]:
        print("==", run)
        for r in report(run): print(("*" if r["pick"] else " "), {k: v for k, v in r.items() if k != "pick"})
