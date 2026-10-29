"""Data for the closed-vs-open-loop pushing page: per scenario one js file with int16 clouds (fixed model frame, metres):
open_exec / closed_exec = SAPIEN executions (frame-0 generated cube + rod posed by the simulated block pose / rod xy),
open_plan = the open-loop plan (model), closed_plans = each replan's plan (model) over its task frames, plus numbers."""
import base64, glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R = os.path.join(HERE, "runs", "pushing")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "cl_vs_ol_site", "data")
EPS = ["005", "011", "014", "029", "037"]


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def last_replay(run):
    rs = sorted(glob.glob(os.path.join(run, "plans", "r*")), key=lambda p: int(os.path.basename(p)[1:]))
    return np.load(glob.glob(os.path.join(rs[-1], "replay_set_pose", "*.npz"))[0])


def best(run):
    z = np.load(os.path.join(run, "rollout.npz")); rj = json.load(open(os.path.join(run, "result.json"))); m = int(np.argmin(rj["best_mm"]))
    return z, m, rj


def b64(a): return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


index = []
for e in EPS:
    ol, cl = os.path.join(R, f"olpad_hard42_ep{e}"), os.path.join(R, f"clpad_hard42_ep{e}_k10")
    if not (os.path.exists(os.path.join(ol, "summary.json")) and os.path.exists(os.path.join(cl, "summary.json"))):
        print("skip", e); continue
    z0, m0, rj0 = best(ol + "_r0"); P0 = z0["predicted"][m0, 0].astype(np.float64); cube = z0["object_mask"][m0].astype(bool); rod = z0["roles"][m0] == 0
    target = z0["target"][m0].astype(np.float64); c0 = P0[cube].mean(0); keep = cube | rod

    def exec_cloud(S):
        bp, bq, g = S["block_pos"].astype(np.float64), S["block_quat"].astype(np.float64), S["gripper_xy"].astype(np.float64); Rq0 = qmat(bq[0]); out = []
        for t in range(bp.shape[0]):
            dR = qmat(bq[t]) @ Rq0.T; Q = P0.copy()
            Q[cube] = (P0[cube] - c0) @ dR.T + c0 + (bp[t] - bp[0]); Q[rod] = P0[rod] + np.array([g[t, 0] - g[0, 0], g[t, 1] - g[0, 1], 0.0]); out.append(Q[keep])
        return np.stack(out)

    So, Sc = last_replay(ol), last_replay(cl)
    clouds = dict(open_exec=exec_cloud(So), closed_exec=exec_cloud(Sc), open_plan=z0["predicted"][m0][:, keep].astype(np.float64))
    c0run = os.path.join(R, f"clctx0_hard42_ep{e}_k10"); has0 = os.path.exists(os.path.join(c0run, "summary.json"))
    if has0: clouds["ctx0_exec"] = exec_cloud(last_replay(c0run)); s0 = json.load(open(os.path.join(c0run, "summary.json")))
    summ = json.load(open(os.path.join(cl, "summary.json"))); plans = []
    for row in summ["rows"]:
        z, m, rj = best(os.path.join(R, row["run"])); c, k = row["committed_before"], row["context"]
        n = min(z["predicted"].shape[1], 61 - (c - k))                       # task frames c-k .. 60
        plans.append(dict(start=c - k, commit_from=c, commit_to=row["committed_after"], P=z["predicted"][m][:n, keep].astype(np.float64),
                          imagined_mm=round(float(row["imagined_best_mm"][row["picked_member"]]), 1), before=row["block_to_target_before_mm"], after=row["block_to_target_after_mm"]))
    allp = np.concatenate([v.reshape(-1, 3) for v in clouds.values()] + [p["P"].reshape(-1, 3) for p in plans] + [target[None]])
    lo, hi = allp.min(0), allp.max(0); span = np.maximum(hi - lo, 1e-6)
    q = lambda x: np.clip(np.rint((x - lo) / span * 65534 - 32767), -32767, 32767).astype(np.int16)
    osum = json.load(open(os.path.join(ol, "summary.json")))
    dist = lambda C: [round(float(np.linalg.norm(C[t][cube[keep]].mean(0)[:2] - target[:2]) * 1e3), 1) for t in range(C.shape[0])]
    rec = dict(id=f"ep{e}", scenario=f"hard seed 42 ep {int(e)}", N=int(keep.sum()), cube=b64(cube[keep].astype(np.uint8)), lo=lo.tolist(), span=span.tolist(),
               target=target.tolist(), table_z=float(P0[cube][:, 2].min()), T=61,
               open_exec=b64(q(clouds["open_exec"])), closed_exec=b64(q(clouds["closed_exec"])), open_plan=b64(q(clouds["open_plan"])), ctx0_exec=b64(q(clouds["ctx0_exec"])) if has0 else None,
               ctx0_final_mm=s0["sim_final_mm"] if has0 else None, ctx0_guide_s=s0["guide_seconds_total"] if has0 else None,
               plans=[dict(start=p["start"], commit_from=p["commit_from"], commit_to=p["commit_to"], T=int(p["P"].shape[0]), P=b64(q(p["P"])), imagined_mm=p["imagined_mm"],
                           before=p["before"], after=p["after"]) for p in plans],
               open_final_mm=osum["sim_final_mm"], closed_final_mm=summ["sim_final_mm"], open_imagined_mm=round(float(rj0["best_mm"][m0]), 1),
               target_mm=summ["target_mm"], closed_guide_s=summ["guide_seconds_total"], open_guide_s=osum["guide_seconds_total"],
               series=dict({k: dist(v) for k, v in clouds.items()}))
    open(os.path.join(OUT, f"case_ep{e}.js"), "w").write(f"window.CASES=window.CASES||{{}};window.CASES['ep{e}']={json.dumps(rec)};")
    index.append(dict(id=f"ep{e}", scenario=rec["scenario"], target_mm=rec["target_mm"], open_final_mm=rec["open_final_mm"], closed_final_mm=rec["closed_final_mm"],
                      open_imagined_mm=rec["open_imagined_mm"], ctx0_final_mm=rec["ctx0_final_mm"], ctx0_guide_s=rec["ctx0_guide_s"], replans=len(plans), closed_guide_s=rec["closed_guide_s"], open_guide_s=rec["open_guide_s"]))
    print(e, "open", rec["open_final_mm"], "closed", rec["closed_final_mm"], "size", os.path.getsize(os.path.join(OUT, f"case_ep{e}.js")) // 1024, "KB")
open(os.path.join(OUT, "index.js"), "w").write("window.INDEX=" + json.dumps(index) + ";")
