"""Per-member generated vs Isaac: goal distance, over-wall (if any), clearance, peak/end height. python sweeps/p15_report.py <run>..."""
import json, sys
import numpy as np
for d in sys.argv[1:]:
    z = np.load(d + "/rollout.npz"); rj = json.load(open(d + "/result.json"))
    S = np.load(d + "/replay_isaac/replay.npz"); var = json.loads(str(S["variants"])); boxes = z["boxes"]
    print(d.split("/")[-1], "| s/it", rj["iteration_seconds"][1:3])
    for m in range(z["predicted"].shape[0]):
        o = z["object_mask"][m]; valid = z["roles"][m] >= 0; row = []
        for name, P in (("gen", z["predicted"][m]), ("sim", S["sim"][var.index("guided"), m])):
            oc = P[:, o].mean(1); s = f"{name} goal {np.linalg.norm(oc[-1]-z['target'][m])*1e3:4.0f} peak/end z {oc[:,2].max()*1e3:4.0f}/{oc[-1,2]*1e3:4.0f}"
            if len(boxes):
                c, hh = boxes[0][:3], boxes[0][3:] / 2; q = np.abs(P - c) - hh; sd = np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(q.max(-1), 0)
                s += f" over {'Y' if oc[-1,0] > c[0]+hh[0] else 'n'} clr obj/grip {sd[1:, o].min()*1e3:4.0f}/{sd[1:, valid & ~o].min()*1e3:4.0f}"
            row.append(s)
        print(f"  m{m} unguided {rj['unguided_mm'][m]:4.0f} | " + " | ".join(row))
