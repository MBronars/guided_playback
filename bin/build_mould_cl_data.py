"""Data for the closed-vs-open-loop pile-manipulation page: per scenario one js file, int16 clouds (model frame, metres),
every --fstride-th task frame, every 2nd dough particle and every 2nd paddle particle. Layers: open / closed executed
(Genesis), open plan (model), closed plans (each replan's model plan over its task frames), target."""
import base64, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(HERE, "runs", "mould")
OUT = os.path.join(HERE, "cl_mould_site", "data"); FS = 2; T = 150
EPS = ["000", "001", "002", "003", "004"]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
index = []
for e in EPS:
    ol, cl = os.path.join(R, f"olmould_s42_ep{e}"), os.path.join(R, f"clmould_s42_ep{e}_k25")
    ho, hc = os.path.exists(os.path.join(ol, "summary.json")), os.path.exists(os.path.join(cl, "summary.json"))
    if not (ho or hc):
        index.append(dict(id=f"ep{e}", scenario=f"seed 42 scenario {int(e)}", pending=True)); print("pending", e); continue
    so = json.load(open(os.path.join(ol, "summary.json"))) if ho else None; sc = json.load(open(os.path.join(cl, "summary.json"))) if hc else None
    ref = sc or so
    z0 = np.load(os.path.join(R, ref["rows"][0]["run"], "rollout.npz")); m0 = ref["rows"][0]["picked_member"]
    obj = z0["object_mask"][m0].astype(bool); pad = z0["roles"][m0] == 0; target = z0["target"][m0].astype(np.float64)
    di, pi = np.nonzero(obj)[0][::2], np.nonzero(pad)[0][::2]; keep = np.concatenate([di, pi]); isd = np.r_[np.ones(len(di)), np.zeros(len(pi))].astype(np.uint8)
    frames = np.arange(0, T + 1, FS)
    Qo = np.load(os.path.join(ol, "executed.npy"))[:, keep] if ho else None; Qc = np.load(os.path.join(cl, "executed.npy"))[:, keep] if hc else None
    zo = np.load(os.path.join(R, ref["rows"][0]["run"], "rollout.npz")); Po = zo["predicted"][ref["rows"][0]["picked_member"]][:T + 1, keep]
    plans = []
    for row in (sc["rows"] if hc else []):
        z = np.load(os.path.join(R, row["run"], "rollout.npz")); Pm = z["predicted"][row["picked_member"]][:, keep]
        start = row["committed_before"] - row["context"]; tf = np.arange(start, min(T, start + Pm.shape[0] - 1) + 1)
        sel = tf[(tf % FS) == 0]; plans.append(dict(start=int(start), commit_from=row["committed_before"], commit_to=row["committed_after"], frames=sel.tolist(), P=Pm[sel - start],
                                                   imagined_mm=row["imagined_best_mm"][row["picked_member"]], before=row["centroid_to_target_before_mm"], after=row["centroid_to_target_after_mm"]))
    allp = np.concatenate([x.reshape(-1, 3) for x in (Qo, Qc, Po) if x is not None] + [target[None]] + [p["P"].reshape(-1, 3) for p in plans])
    lo, hi = allp.min(0), allp.max(0); span = np.maximum(hi - lo, 1e-6)
    q = lambda x: np.clip(np.rint((x - lo) / span * 65534 - 32767), -32767, 32767).astype(np.int16)
    dist = lambda C: [round(float(np.linalg.norm(C[t][:len(di)].mean(0)[:2] - target[:2]) * 1e3), 1) for t in range(C.shape[0])]
    rec = dict(id=f"ep{e}", scenario=f"seed 42 scenario {int(e)} (episode {ref['episode']})", N=int(len(keep)), isd=b64(isd), lo=lo.tolist(), span=span.tolist(), target=target.tolist(),
               table_z=float(Po[0, :len(di), 2].min()), frames=frames.tolist(), open_exec=b64(q(Qo[frames])) if ho else None, closed_exec=b64(q(Qc[frames])) if hc else None, open_plan=b64(q(Po[frames])),
               plans=[dict({kk: v for kk, v in p.items() if kk != "P"}, P=b64(q(p["P"]))) for p in plans],
               target_mm=ref["target_mm"], open_final_mm=so and so["sim_final_mm"], closed_final_mm=sc and sc["sim_final_mm"], open_closed_mm=so and so["distance_closed_mm"], closed_closed_mm=sc and sc["distance_closed_mm"],
               open_planned_mm=ref["planned_first_mm"], open_guide_s=so and so["guide_seconds_total"], closed_guide_s=sc and sc["guide_seconds_total"], open_replay_s=so and so["replay_seconds_total"], closed_replay_s=sc and sc["replay_seconds_total"],
               series=dict(open_exec=dist(Qo) if ho else None, closed_exec=dist(Qc) if hc else None, open_plan=dist(Po)))
    fn = os.path.join(OUT, f"case_ep{e}.js"); open(fn, "w").write(f"window.CASES=window.CASES||{{}};window.CASES['ep{e}']={json.dumps(rec)};")
    index.append({kk: rec[kk] for kk in ("id", "scenario", "target_mm", "open_final_mm", "closed_final_mm", "open_closed_mm", "closed_closed_mm", "open_planned_mm", "open_guide_s", "closed_guide_s", "open_replay_s", "closed_replay_s")} | dict(replans=len(plans)))
    print(e, "open", so and so["sim_final_mm"], "closed", sc and sc["sim_final_mm"], os.path.getsize(fn) // 1024, "KB")
open(os.path.join(OUT, "index.js"), "w").write("window.INDEX=" + json.dumps(index) + ";")
