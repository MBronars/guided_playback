"""Asset screen (runs/pickup/asset_<label>): per member, generated + Isaac pre-contact tilt (deg), lift, goal distance, held.
Contact = first frame with finger-object min distance < 5 mm (sweeps/assetsweep_tools/measure.py definitions). Writes sweeps/asset_screen_summary.{json,md}."""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, "sweeps/assetsweep_tools")
from measure import rot_angle_deg, min_dist
rows = []
for d in sorted(glob.glob("runs/pickup/" + (sys.argv[1] if len(sys.argv) > 1 else "assetS_") + "*")):
    if not os.path.exists(d + "/replay_isaac/replay.npz"): continue
    z = np.load(d + "/rollout.npz"); S = np.load(d + "/replay_isaac/replay.npz"); var = json.loads(str(S["variants"]))
    label = os.path.basename(d).split("_", 1)[1]; mem = []
    for m in range(z["predicted"].shape[0]):
        o = z["object_mask"][m]; f = z["roles"][m] == 1; r = {}
        for name, P in (("gen", z["predicted"][m].astype(np.float64)), ("sim", S["sim"][var.index("guided"), m].astype(np.float64))):
            md = np.array([min_dist(P[t, f], P[t, o]) for t in range(P.shape[0])]); hit = np.nonzero(md < 0.005)[0]
            cf = int(hit[0]) if hit.size else P.shape[0] - 1
            tilts = [rot_angle_deg(P[0, o], P[t, o]) for t in range(0, max(cf, 1))]
            oc = P[:, o].mean(1)
            r[name] = dict(contact=cf, tilt=round(max(tilts), 1), disp=round(float(np.linalg.norm(oc[cf - 1 if cf else 0, :2] - oc[0, :2]) * 1e3), 1),
                           peak=round(float((oc[:, 2].max() - oc[0, 2]) * 1e3)), end=round(float((oc[-1, 2] - oc[0, 2]) * 1e3)),
                           goal=round(float(np.linalg.norm(oc[-1] - z["target"][m]) * 1e3)))
        r["held"] = r["sim"]["end"] > 50; r["stable"] = r["held"] and r["gen"]["tilt"] < 5 and r["sim"]["tilt"] < 5
        mem.append(r)
    ext = np.ptp(z["gt"][0, 0][z["object_mask"][0]], 0) * 1e3
    rows.append(dict(label=label, extent=[int(v) for v in ext], held=sum(r["held"] for r in mem), stable=sum(r["stable"] for r in mem), members=mem))
rows.sort(key=lambda r: (-r["stable"], -r["held"]))
json.dump(rows, open("sweeps/asset_screen_summary_" + (sys.argv[1] if len(sys.argv) > 1 else "assetS_") + ".json", "w"), indent=1)
L = ["| asset | extent mm | held in Isaac | held + tilt<5 deg (gen & Isaac) | pre-contact tilt gen / Isaac (deg, 4 members) | Isaac goal mm |", "|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['label']} | {'x'.join(map(str, r['extent']))} | {r['held']}/4 | {r['stable']}/4 | " + ", ".join(f"{m['gen']['tilt']}/{m['sim']['tilt']}" for m in r["members"]) + " | " + ", ".join(str(m['sim']['goal']) for m in r["members"]) + " |")
open("sweeps/asset_screen_summary_" + (sys.argv[1] if len(sys.argv) > 1 else "assetS_") + ".md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
