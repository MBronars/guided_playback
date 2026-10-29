"""One page, three domains: re-packs the three closed-vs-open-loop page data sets (block pushing cl_vs_ol_site, random-object
pushing rlpcl_site, pile manipulation cl_mould_site) into one format for all_cl_site. Encoded clouds are passed through untouched
(int16 over each case's lo/span); only point roles, frame lists and summary columns are normalized.
Unified case: dom, id, label, N, roles (int8 b64: 0 actuator, 1 finger, 2 object, -1 pad), lo, span, target, table_z,
frames (task frame of each stored index), layers {open_exec, closed_exec, ctx0_exec?, open_plan} (b64 or null) with their stored
frame counts, plans [{frames, commit_from, commit_to, P, imagined_mm}], series {layer: [mm per stored index]}."""
import base64, glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(HERE, "all_cl_site", "data"); os.makedirs(OUT, exist_ok=True)
SRC = [("push", "Block pushing", "cl_vs_ol_site"), ("rlpush", "Random-object pushing", os.environ.get("ALLCL_RLPUSH_SITE", "rlpnear_site")), ("pile", "Pile manipulation", "cl_mould_site")]


def read_js(path, marker):
    s = open(path).read().strip().rstrip(";"); j = s.index(marker) + len(marker); return json.loads(s[j:])


def dec(b64, lo, span, N):
    a = np.frombuffer(base64.b64decode(b64), np.int16).astype(np.float64).reshape(-1, N, 3)
    return np.asarray(lo) + (a + 32767) / 65534 * np.asarray(span)


def enc(x, lo, span):
    q = np.clip(np.rint((x - lo) / span * 65534 - 32767), -32767, 32767).astype(np.int16); return base64.b64encode(q.tobytes()).decode()


BASE = {"rlpush": "runs/rl_push_baselines/results.json", "push": "runs/pushing_baselines/results.json"}
BASE_NAMES = {"pw": "PointWorld", "gns": "GNS"}


def baseline_layers(dom, pid, spec_order):
    """{layer: [T, N, 3] float} + {row key: mm} from a baseline agent's results.json (rl_push: Isaac replay clouds in our particle order)."""
    out, row = {}, {}
    if dom not in BASE: return out, row
    f = os.path.join(HERE, BASE[dom])
    if not os.path.isfile(f): return out, row
    if dom == "push": return push_baseline_layers(pid, json.load(open(f)))
    R = json.load(open(f))
    for m in ("pw", "gns"):
        if m not in R: continue
        for mode, tag in (("open", "open_exec"), ("closed", "closed_exec")):
            e = R[m].get(mode, {}).get(pid)
            if not e: continue
            rp = e.get("replay") or os.path.join(HERE, "runs", "rl_push_baselines", f"dir5_{m}_cl_exec_c40", "replay_isaac", "replay.npz")
            slot = e.get("slot", spec_order.index(pid))
            if os.path.exists(rp): out[f"{m}_{tag}"] = np.load(rp)["sim"][0, slot].astype(np.float64)
            row[f"{m}_{mode}_mm"] = e.get("final_mm")
    return out, row


def _qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def push_baseline_layers(pid, R):
    """Block pushing baselines (runs/pushing_baselines/results.json): the SAPIEN block pose / rod xy trajectory of each cell posed on OUR
    frame-0 cube + rod clouds (the same construction as our own executed layers, bin/build_cl_vs_ol_data.py). PointWorld = the frame-60 config."""
    e = R.get("scenarios", {}).get(pid); out, row = {}, {}
    if not e: return out, row
    ep = pid.replace("ep", "")
    r0 = os.path.join(HERE, "runs", "pushing", f"olpad_hard42_ep{ep}_r0"); z = np.load(os.path.join(r0, "rollout.npz")); rj = json.load(open(os.path.join(r0, "result.json")))
    m = int(np.argmin(rj["best_mm"])); P0 = z["predicted"][m, 0].astype(np.float64); cube = z["object_mask"][m].astype(bool); rod = z["roles"][m] == 0; keep = cube | rod; c0 = P0[cube].mean(0)
    for meth, key in (("gns", "gns"), ("pw", "pw60")):
        for mode in ("open", "closed"):
            c = e.get(f"{key}_{mode}")
            if not c or not c.get("final_sim") or not os.path.exists(c["final_sim"]): continue
            S = np.load(c["final_sim"]); bp, bq, g = S["block_pos"].astype(np.float64), S["block_quat"].astype(np.float64), S["gripper_xy"].astype(np.float64)
            R0 = _qmat(bq[0]); Q = []
            for t in range(bp.shape[0]):
                dR = _qmat(bq[t]) @ R0.T; X = P0.copy()
                X[cube] = (P0[cube] - c0) @ dR.T + c0 + (bp[t] - bp[0]); X[rod] = P0[rod] + np.array([g[t, 0] - g[0, 0], g[t, 1] - g[0, 1], 0.0]); Q.append(X[keep])
            out[f"{meth}_{mode}_exec"] = np.stack(Q); row[f"{meth}_{mode}_mm"] = c.get("final_mm")
    return out, row


def n_frames(b64, N):
    return 0 if not b64 else len(base64.b64decode(b64)) // (2 * 3 * N)




# ---- picking (5 objects x 5 targets): ours = the guided pick benchmark (runs/pickup/cyl_t*_it10, open loop only),
# PointWorld three-domain open (+ closed when present) and two-domain open (runs/pickup_baselines/results.json). Isaac executions.
def build_pick(index):
    rp = os.path.join(HERE, "runs", "pickup_baselines", "results.json")
    if not os.path.exists(rp): return
    R = json.load(open(rp)); PE = {(r["asset"], r["target"]): r for r in json.load(open(os.path.join(HERE, "sweeps", "planner_eval.json")))}
    SUBP, SUBF = 3, 2                                                    # page size: every 3rd point, every 2nd frame
    rows = []
    for pid in R["problems"]:
        asset, t = pid.rsplit("_t", 1); cat = asset.split("_asset_")[0]
        run = os.path.join(HERE, "runs", "pickup", f"cyl_t{t}_{cat}_it10")
        rj = json.load(open(os.path.join(run, "result.json"))); z = np.load(os.path.join(run, "rollout.npz")); m = int(np.argmin(rj["best_mm"]))
        S = np.load(os.path.join(run, "replay_isaac", "replay.npz")); var = json.loads(str(S["variants"])).index("guided")
        sel = np.arange(z["predicted"].shape[2])[::SUBP]; fr = list(range(0, z["predicted"].shape[1], SUBF))
        L = {"open_exec": S["sim"][var, m][fr][:, sel], "open_plan": z["predicted"][m][fr][:, sel]}
        pe = PE.get((asset, f"t{t}"), {})
        row = dict(start_mm=None, open_mm=pe.get("sim_goal_mm"), open_planned_mm=pe.get("gen_goal_mm"))
        for key, lay, col in (("open", "pw_open_exec", "pw_open_mm"), ("closed", "pw_closed_exec", "pw_closed_mm"), ("two_domain_open", "pw2_open_exec", "pw2_open_mm")):
            e = (R.get(key) or {}).get(pid)
            if not e or not e.get("replay") or not os.path.exists(e["replay"]): continue
            L[lay] = np.load(e["replay"])["sim"][0, e.get("slot", R["problems"].index(pid))][fr][:, sel]; row[col] = e.get("final_mm")
        tgt = z["target"][m].astype(np.float64); om = z["object_mask"][m][sel].astype(bool)
        row["start_mm"] = round(float(np.linalg.norm(z["predicted"][m, 0][z["object_mask"][m].astype(bool)].mean(0) - tgt) * 1e3), 1)
        allp = np.concatenate([v.reshape(-1, 3) for v in L.values()] + [tgt[None]]); lo = allp.min(0); span = np.maximum(allp.max(0) - lo, 1e-6)
        roles = z["roles"][m][sel].astype(np.int8)
        ser = {k: [round(float(np.linalg.norm(v[i][om].mean(0) - tgt) * 1e3), 1) for i in range(v.shape[0])] for k, v in L.items()}
        rec = dict(dom="pick", id=pid, label=f"{cat.replace('_', ' ')}, target {t}", N=int(len(sel)), roles=base64.b64encode(roles.tobytes()).decode(), lo=lo.tolist(),
                   span=span.tolist(), target=tgt.tolist(), table_z=float(z["predicted"][m, 0][z["object_mask"][m].astype(bool)][:, 2].min()), frames=[f for f in fr],
                   layers={k: enc(v, lo, span) for k, v in L.items()}, layer_frames={k: len(fr) for k in L}, plans=[], series=ser)
        open(os.path.join(OUT, f"case_pick_{pid}.js"), "w").write(f"window.CASES=window.CASES||{{}};window.CASES['pick_{pid}']={json.dumps(rec)};")
        rows.append(dict(id=pid, key=f"pick_{pid}", label=rec["label"], **row))
    index["pick"] = dict(label="Picking", rows=rows, status=None)


SPEC_ORDER = {}
try:
    SPEC_ORDER["rlpush"] = [p["id"] for p in json.load(open(os.path.join(HERE, "sweeps", "rlpush_cl_mid5_dir1_goalonly.json")))["problems"]]
except Exception: pass
index = {}
for dom, dlabel, site in SRC:
    idx_path = os.path.join(HERE, site, "data", "index.js")
    if not os.path.exists(idx_path): continue
    rows = []
    src_idx = open(idx_path).read(); listed = set(json.loads(src_idx[src_idx.index("window.INDEX=") + 13:].split(";")[0].strip()) and [r["id"] for r in json.loads(src_idx[src_idx.index("window.INDEX=") + 13:].split(";")[0].strip())])
    for f in sorted(glob.glob(os.path.join(HERE, site, "data", "case_*.js"))):
        if os.path.basename(f)[5:-3] not in listed: continue                 # only the cases the domain page currently lists
        rec = read_js(f, "]="); N = int(rec["N"])
        if dom == "push":
            roles = np.where(np.frombuffer(base64.b64decode(rec["cube"]), np.uint8) > 0, 2, 0).astype(np.int8)
            frames = list(range(int(rec["T"]))); label = rec["scenario"]
            plans = [dict(frames=list(range(p["start"], p["start"] + p["T"])), commit_from=p["commit_from"], commit_to=p["commit_to"], P=p["P"], imagined_mm=p["imagined_mm"]) for p in rec["plans"]]
            row = dict(start_mm=rec["target_mm"], open_planned_mm=rec["open_imagined_mm"], open_mm=rec["open_final_mm"], closed_mm=rec["closed_final_mm"], ctx0_mm=rec.get("ctx0_final_mm"))
        elif dom == "pile":
            roles = np.where(np.frombuffer(base64.b64decode(rec["isd"]), np.uint8) > 0, 2, 0).astype(np.int8)
            frames = list(rec["frames"]); label = rec["scenario"]
            plans = [dict(frames=list(p["frames"]), commit_from=p["commit_from"], commit_to=p["commit_to"], P=p["P"], imagined_mm=p["imagined_mm"]) for p in rec["plans"]]
            row = dict(start_mm=rec["target_mm"], open_planned_mm=rec.get("open_planned_mm"), open_mm=rec.get("open_final_mm"), closed_mm=rec.get("closed_final_mm"),
                       open_closed_mm=rec.get("open_closed_mm"), closed_closed_mm=rec.get("closed_closed_mm"))
        else:
            roles = np.frombuffer(base64.b64decode(rec["roles"]), np.int8).copy()
            frames = list(range(int(rec["T"]))); label = (f'{rec["id"].split("_")[0]} {rec["asset"].split("/")[-2].replace("_", " ")}' + (f', goal {rec["id"].split("_t")[-1]}' if "_t" in rec["id"] else ""))
            plans = [dict(frames=list(range(p["start"], p["start"] + p["T"])), commit_from=p["commit_from"], commit_to=p["commit_to"], P=p["P"], imagined_mm=p["imagined_mm"]) for p in rec["plans"]]
            row = dict(start_mm=rec.get("start_mm"), open_planned_mm=rec.get("open_imagined_mm"), open_mm=rec.get("open_mm"), closed_mm=rec.get("closed_mm") if rec.get("closed_final") else None,
                       closed_partial=None if rec.get("closed_final") else rec.get("executed_frames"))
        layers = {k: rec.get(k) for k in ("open_exec", "closed_exec", "ctx0_exec", "open_plan") if k in rec}
        extra, brow = baseline_layers(dom, rec["id"], SPEC_ORDER.get(dom, []))
        if extra:                                                          # re-quantize every cloud over the union range
            F = {k: dec(v, rec["lo"], rec["span"], N) for k, v in layers.items() if v}
            Pp = [dec(p["P"], rec["lo"], rec["span"], N) for p in plans]
            allp = np.concatenate([x.reshape(-1, 3) for x in list(F.values()) + list(extra.values()) + Pp])
            lo = allp.min(0); span = np.maximum(allp.max(0) - lo, 1e-6); rec["lo"], rec["span"] = lo.tolist(), span.tolist()
            layers = {k: (enc(F[k], lo, span) if k in F else None) for k in layers}
            for k, v in extra.items(): layers[k] = enc(v, lo, span)
            for p, x in zip(plans, Pp): p["P"] = enc(x, lo, span)
            tgt = np.asarray(rec["target"]); om = roles == 2
            ser = dict(rec.get("series", {}))
            for k, v in extra.items(): ser[k] = [round(float(np.linalg.norm(v[t][om].mean(0)[:2] - tgt[:2]) * 1e3), 1) for t in range(v.shape[0])]
            rec["series"] = ser
        row.update(brow)
        out = dict(dom=dom, id=rec["id"], label=label, N=N, roles=base64.b64encode(roles.tobytes()).decode(), lo=rec["lo"], span=rec["span"], target=rec["target"],
                   table_z=rec["table_z"], frames=frames, layers=layers, layer_frames={k: n_frames(v, N) for k, v in layers.items()}, plans=plans, series=rec.get("series", {}))
        open(os.path.join(OUT, f"case_{dom}_{rec['id']}.js"), "w").write(f"window.CASES=window.CASES||{{}};window.CASES['{dom}_{rec['id']}']={json.dumps(out)};")
        rows.append(dict(id=rec["id"], key=f"{dom}_{rec['id']}", label=label, **row))
    status = None
    if dom == "rlpush":
        s = open(idx_path).read(); status = json.loads(s[s.index("window.STATUS=") + 14:].split(";")[0]) if "window.STATUS=" in s else None
    index[dom] = dict(label=dlabel, rows=rows, status=status)
build_pick(index)
open(os.path.join(OUT, "index.js"), "w").write("window.ALL=" + json.dumps(index) + ";")
keep = {f"case_{d}_{r['id']}.js" for d, v in index.items() for r in v["rows"]}
for f in glob.glob(os.path.join(OUT, "case_*.js")):
    if os.path.basename(f) not in keep: os.rename(f, f + ".unlisted")      # page copies only (source data untouched)
print({d: len(v["rows"]) for d, v in index.items()}, "->", OUT)
