"""Page data for rl_push closed vs open loop (runs/rl_push/<tag>_*), PARTIAL-tolerant: per problem whatever exists.
open_exec = Isaac execution of the round-0 plan (rlpcl25_openloop_exec); closed_exec = the latest lockstep Isaac execution
(rlpcl25_exec_c<c>, frames 0..c; final when c = 40); open_plan = round-0 plan; plans = each finished round's plan from its anchor.
Clouds int16-quantized, model frame, metres."""
import base64, glob, json, os, sys
import numpy as np
HERE = "/data/user_data/mbronars/packages/guided_playback"; R = os.path.join(HERE, "runs", "rl_push")
TAG = sys.argv[1] if len(sys.argv) > 1 else "rlpcl5"; OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "rlpcl_site", "data")
SPEC = json.load(open(os.environ.get("RLPCL_SPEC", os.path.join(HERE, "sweeps", "rlpush_cl_5.json")))); H = SPEC["planner"]["transitions"]; COMMIT = SPEC["closed_loop"]["commit"]
probs = SPEC["problems"]; os.makedirs(OUT, exist_ok=True)
logp = os.path.join(R, f"{TAG}_state", "driver_log.jsonl"); log = [json.loads(l) for l in open(logp)] if os.path.exists(logp) else []
execs = sorted([p for p in glob.glob(os.path.join(R, f"{TAG}_exec_c*")) if os.path.exists(os.path.join(p, "replay_isaac", "analysis.json"))], key=lambda p: int(p.rsplit("_c", 1)[1]))
last = execs[-1] if execs else None; c_exec = int(last.rsplit("_c", 1)[1]) if last else 0
Scl = np.load(os.path.join(last, "replay_isaac", "replay.npz"))["sim"][0] if last else None; Acl = json.load(open(os.path.join(last, "replay_isaac", "analysis.json"))) if last else None
olp = os.path.join(R, f"{TAG}_openloop_exec"); has_ol = os.path.exists(os.path.join(olp, "replay_isaac", "analysis.json"))
Sol = np.load(os.path.join(olp, "replay_isaac", "replay.npz"))["sim"][0] if has_ol else None; Aol = json.load(open(os.path.join(olp, "replay_isaac", "analysis.json"))) if has_ol else None
rounds_done = sorted({x["round"] for x in log if "exec_run" in x}) if False else None
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
index = []
for b, p in enumerate(probs):
    rows = {x["round"]: x for x in log if x.get("id") == p["id"]}
    ent = dict(id=p["id"], asset=p["asset"], plans_done=len(rows), executed_frames=c_exec if last else 0, open_mm=None, closed_mm=None, closed_final=bool(last and c_exec >= H),
               open_imagined_mm=None, start_mm=None, open_moved_mm=None, closed_moved_mm=None, open_guide_s=None, closed_guide_s=None, has_data=False)
    if 0 not in rows or not os.path.exists(os.path.join(R, rows[0]["run"], "rollout.npz")):
        index.append(ent); continue
    z0 = np.load(os.path.join(R, rows[0]["run"], "rollout.npz")); m0 = rows[0]["picked"]; roles = z0["roles"][m0]; keep = roles >= 0; tgt = z0["target"][m0].astype(np.float64)
    om = z0["object_mask"][m0][keep].astype(bool); op = z0["predicted"][m0][:H + 1][:, keep].astype(np.float64)
    ol = Sol[b][:, keep].astype(np.float64) if has_ol else None; cl = Scl[b][:c_exec + 1][:, keep].astype(np.float64) if last else None
    plans = []
    for r in sorted(rows):
        x = rows[r]; zp = os.path.join(R, x["run"], "rollout.npz")
        if not os.path.exists(zp): continue
        z = np.load(zp); c, k = x["executed"], x["context"]; P = z["predicted"][x["picked"]][:, keep].astype(np.float64); n = min(P.shape[0], H + 1 - (c - k))
        plans.append(dict(start=c - k, commit_from=c, commit_to=min(c + COMMIT, H), P=P[:n], imagined_mm=x["imagined_mm"][x["picked"]]))
    allp = np.concatenate([op.reshape(-1, 3), tgt[None]] + [a.reshape(-1, 3) for a in (ol, cl) if a is not None] + [q["P"].reshape(-1, 3) for q in plans])
    lo, hi = allp.min(0), allp.max(0); span = np.maximum(hi - lo, 1e-6)
    q = lambda x: np.clip(np.rint((x - lo) / span * 65534 - 32767), -32767, 32767).astype(np.int16)
    dist = lambda C: [round(float(np.linalg.norm(C[t][om].mean(0)[:2] - tgt[:2]) * 1e3), 1) for t in range(C.shape[0])]
    ro = next(x for x in Aol["rows"] if x["member"] == b) if has_ol else None; rc = next(x for x in Acl["rows"] if x["member"] == b) if last else None
    ent.update(has_data=True, start_mm=round(float(np.linalg.norm(op[0][om].mean(0)[:2] - tgt[:2]) * 1e3), 1), open_imagined_mm=rows[0]["imagined_mm"][m0], open_guide_s=rows[0]["guide_s"],
               closed_guide_s=round(sum(x["guide_s"] for x in rows.values()), 1),
               open_mm=round(ro["sim_to_target_mm"], 1) if ro else None, open_moved_mm=round(ro["sim_moved_mm"], 1) if ro else None,
               closed_mm=round(rc["sim_to_target_mm"], 1) if rc else None, closed_moved_mm=round(rc["sim_moved_mm"], 1) if rc else None)
    rec = dict(ent, N=int(keep.sum()), roles=b64(roles[keep].astype(np.int8)), lo=lo.tolist(), span=span.tolist(), target=tgt.tolist(), table_z=float(op[0][om][:, 2].min()), T=H + 1,
               open_exec=b64(q(ol)) if ol is not None else None, closed_exec=b64(q(cl)) if cl is not None else None, closed_T=int(cl.shape[0]) if cl is not None else 0, open_plan=b64(q(op)),
               plans=[dict(start=x["start"], commit_from=x["commit_from"], commit_to=x["commit_to"], T=int(x["P"].shape[0]), P=b64(q(x["P"])), imagined_mm=x["imagined_mm"]) for x in plans],
               series=dict(open_exec=dist(ol) if ol is not None else None, closed_exec=dist(cl) if cl is not None else None, open_plan=dist(op)))
    open(os.path.join(OUT, f"case_{p['id']}.js"), "w").write(f"window.CASES=window.CASES||{{}};window.CASES['{p['id']}']={json.dumps(rec)};")
    index.append(ent)
status = dict(tag=TAG, plans=sum(1 for x in log if "id" in x), executed_frames=c_exec, open_loop_done=has_ol, updated=__import__("time").strftime("%Y-%m-%d %H:%M"),
              exec_rounds=[dict(run=os.path.basename(e), frames=int(e.rsplit("_c", 1)[1])) for e in execs])
open(os.path.join(OUT, "index.js"), "w").write("window.INDEX=" + json.dumps(index) + ";window.STATUS=" + json.dumps(status) + ";")
print(json.dumps(status)); print(sum(e["has_data"] for e in index), "problems with data")
