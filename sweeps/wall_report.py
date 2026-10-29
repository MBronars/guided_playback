"""Per-member generated vs Isaac numbers for a wall run: python sweeps/wall_report.py <run dir>..."""
import json, sys
import numpy as np
for d in sys.argv[1:]:
    z = np.load(d + "/rollout.npz"); rj = json.load(open(d + "/result.json")); c, hh = z["boxes"][0][:3], z["boxes"][0][3:] / 2
    S = np.load(d + "/replay_isaac/replay.npz"); var = json.loads(str(S["variants"]))
    print(d.split("/")[-1])
    for m in range(z["predicted"].shape[0]):
        o = z["object_mask"][m]; valid = z["roles"][m] >= 0; row = []
        for name, P in (("gen", z["predicted"][m]), ("sim", S["sim"][var.index("guided"), m])):
            q = np.abs(P - c) - hh; sd = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0); oc = P[:, o].mean(1)
            row.append(f"{name} goal {np.linalg.norm(oc[-1]-z['target'][m])*1e3:4.0f} over {'Y' if oc[-1,0] > c[0]+hh[0] else 'n'} clr obj {sd[1:, o].min()*1e3:4.0f} grip {sd[1:, valid & ~o].min()*1e3:4.0f} peak/end z {oc[:,2].max()*1e3:4.0f}/{oc[-1,2]*1e3:4.0f}")
        print(f"  m{m} unguided {rj['unguided_mm'][m]:4.0f} | " + " | ".join(row))
