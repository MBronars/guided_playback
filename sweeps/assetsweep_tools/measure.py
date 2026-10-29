"""Measurements for the asset sweep: per member, generated (guided) and Isaac (guided variant), plus the recorded clip.
-> sweeps/assetsweep_summary.{json,md}"""
import glob, json, os
import numpy as np

ROOT = "/data/user_data/mbronars/packages/guided_playback"
SEL = json.load(open(os.path.join(ROOT, "sweeps/assetsweep_select.json")))
CONTACT_M = 0.005


def rot_angle_deg(P0, P):
    a, b = P0 - P0.mean(0), P - P.mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def min_dist(A, B):
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None] - 2 * A @ B.T
    return float(np.sqrt(max(d2.min(), 0.0)))


def measure(X, fing, obj, target=None):
    """X [T, N, 3] float64; fing / obj boolean row masks."""
    T = X.shape[0]
    O = X[:, obj]; c = O.mean(1)
    md = np.array([min_dist(X[t, fing], O[t]) for t in range(T)])
    hit = np.nonzero(md < CONTACT_M)[0]
    cf = int(hit[0]) if hit.size else None
    last = (cf - 1) if cf is not None else T - 1          # last pre-contact frame
    pre = range(0, max(last, 0) + 1)
    tilts = [rot_angle_deg(O[0], O[t]) for t in pre]
    disp = [float(np.linalg.norm(c[t] - c[0]) * 1e3) for t in pre]
    rise = (c[:, 2] - c[0, 2]) * 1e3
    r = dict(contact_frame=cf, precontact_last_frame=last, tilt_at_contact_deg=round(tilts[-1], 2), tilt_max_precontact_deg=round(max(tilts), 2),
             disp_at_contact_mm=round(disp[-1], 1), disp_max_precontact_mm=round(max(disp), 1), min_finger_obj_mm_f0=round(md[0] * 1e3, 1),
             peak_rise_mm=round(float(rise.max()), 1), final_rise_mm=round(float(rise[-1]), 1))
    if target is not None:
        r["goal_mm"] = round(float(np.linalg.norm(c[-1] - target) * 1e3), 1)
    return r


rows, fails = [], []
for e in SEL:
    lab = e["label"]; run = os.path.join(ROOT, "runs/pickup", f"assetsweep_{lab}")
    row = dict(label=lab, run=run, asset_id=e["asset_id"], asset=e["asset"].split("assets/")[1], asset_verified=e["asset_verified"],
               best_fit_asset=e["best_fit_asset"].split("assets/")[1], rho=round(e["rho"], 1), mu=round(e["mu"], 3), rec_lift_pool_mm=round(e["rec_lift_mm"], 1))
    if not os.path.exists(os.path.join(run, "rollout.npz")):
        row["error"] = "no rollout.npz"; rows.append(row); fails.append(lab); continue
    z = np.load(os.path.join(run, "rollout.npz"), allow_pickle=True)
    roles, om = z["roles"], z["object_mask"]
    Tgt = int(z["gt_frames"][0])
    G0 = z["gt"][0, :Tgt].astype(np.float64)
    ext = (G0[0, om[0]].max(0) - G0[0, om[0]].min(0)) * 1e3
    row["extent_mm"] = [round(float(v)) for v in ext]
    row["recorded"] = measure(G0, roles[0] == 1, om[0])
    rp = os.path.join(run, "replay_isaac", "replay.npz")
    R = np.load(rp, allow_pickle=True) if os.path.exists(rp) else None
    if R is None: row["error"] = "no replay_isaac/replay.npz"; fails.append(lab)
    else:
        vi = json.loads(str(R["variants"])).index("guided"); sroles = R["sim_roles"]
    mem = []
    for m in range(z["predicted"].shape[0]):
        d = dict(member=m, gen=measure(z["predicted"][m].astype(np.float64), roles[m] == 1, om[m], z["target"][m].astype(np.float64)))
        if R is not None:
            s = measure(R["sim"][vi, m].astype(np.float64), sroles[m] == 1, om[m], z["target"][m].astype(np.float64))
            s["held"] = bool(s["final_rise_mm"] > 50.0); d["isaac"] = s
        mem.append(d)
    row["members"] = mem
    if R is not None:
        row["n_held"] = sum(d["isaac"]["held"] for d in mem)
        row["n_held_isaac_tilt_lt5"] = sum(d["isaac"]["held"] and d["isaac"]["tilt_at_contact_deg"] < 5 for d in mem)
        row["n_held_gen_tilt_lt5"] = sum(d["isaac"]["held"] and d["gen"]["tilt_at_contact_deg"] < 5 for d in mem)
        row["n_held_both_tilt_lt5"] = sum(d["isaac"]["held"] and d["gen"]["tilt_at_contact_deg"] < 5 and d["isaac"]["tilt_at_contact_deg"] < 5 for d in mem)
    rows.append(row)

key = lambda r: (-r.get("n_held_both_tilt_lt5", -1), -r.get("n_held_isaac_tilt_lt5", -1), -r.get("n_held", -1),
                 np.median([d["isaac"]["tilt_at_contact_deg"] for d in r["members"]]) if r.get("n_held") is not None else 1e9)
rows.sort(key=key)
json.dump(dict(contact_threshold_mm=CONTACT_M * 1e3, held_threshold_mm=50, rows=rows, failed=fails), open(os.path.join(ROOT, "sweeps/assetsweep_summary.json"), "w"), indent=1)

f = lambda v: "-" if v is None else v
L = ["# Asset sweep: pick-to-location lift, pickup-only ckpt (cont200k final.pt)", "",
     "Recipe: 4 members, 39 transitions, S=10, target = f0 centroid + (0.15, 0, 0.15) m, Adam lr 0.1, 10 it, shell ON, seed 0, eager bf16; Isaac floating-hand replay of the guided rollout.",
     "Contact = first frame with min finger(roles==1)-object distance < 5 mm; tilt = Kabsch rotation angle of object points vs frame 0 at the last pre-contact frame (max over pre-contact frames in brackets); disp = pre-contact centroid displacement at that frame. Held = Isaac final rise > 50 mm.",
     "Sorted by #held with gen AND Isaac tilt < 5 deg, then #held with Isaac tilt < 5 deg, then #held.", "",
     "## Per episode", "",
     "| episode | asset (id) | extent mm | rec contact f / tilt deg | held | held & Isaac tilt<5 | held & gen tilt<5 | held & both<5 | gen tilt deg (4 members) | Isaac tilt deg (4 members) |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    if "members" not in r:
        L.append(f"| {r['label']} | {r['asset']} ({r['asset_id']}) | | | ERROR: {r.get('error')} | | | | | |"); continue
    rc = r["recorded"]; ms = r["members"]
    gt = ", ".join(f"{d['gen']['tilt_at_contact_deg']:.1f}" for d in ms)
    it = ", ".join(f"{d['isaac']['tilt_at_contact_deg']:.1f}" for d in ms) if "isaac" in ms[0] else "-"
    nm = r["asset"] + ("" if r["asset_verified"] else (" [libero id: sorted-order mapping not verified for libero; mesh fit agrees]" if r["best_fit_asset"] == r["asset"] else " [id unverified; best fit " + r["best_fit_asset"] + "]"))
    L.append(f"| {r['label']} | {nm} ({r['asset_id']}) | {'x'.join(map(str, r['extent_mm']))} | {f(rc['contact_frame'])} / {rc['tilt_at_contact_deg']:.1f} | "
             f"{f(r.get('n_held'))}/4 | {f(r.get('n_held_isaac_tilt_lt5'))} | {f(r.get('n_held_gen_tilt_lt5'))} | {f(r.get('n_held_both_tilt_lt5'))} | {gt} | {it} |")
L += ["", "## Per member", "",
      "| episode | m | gen contact f | gen tilt@c (max) deg | gen disp@c mm | gen peak/final rise mm | gen goal mm | Isaac contact f | Isaac tilt@c (max) deg | Isaac disp@c mm | Isaac peak/final rise mm | Isaac goal mm | held |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    for d in r.get("members", []):
        g = d["gen"]; s = d.get("isaac")
        sg = (f"{f(s['contact_frame'])} | {s['tilt_at_contact_deg']:.1f} ({s['tilt_max_precontact_deg']:.1f}) | {s['disp_at_contact_mm']:.1f} | {s['peak_rise_mm']:.0f} / {s['final_rise_mm']:.0f} | {s['goal_mm']:.0f} | {'Y' if s['held'] else 'n'}"
              if s else "- | - | - | - | - | -")
        L.append(f"| {r['label']} | {d['member']} | {f(g['contact_frame'])} | {g['tilt_at_contact_deg']:.1f} ({g['tilt_max_precontact_deg']:.1f}) | {g['disp_at_contact_mm']:.1f} | "
                 f"{g['peak_rise_mm']:.0f} / {g['final_rise_mm']:.0f} | {g['goal_mm']:.0f} | {sg} |")
L += ["", "## Recorded clips (gt)", "", "| episode | contact f | tilt@c (max) deg | disp@c mm | peak/final rise mm |", "|---|---|---|---|---|"]
for r in rows:
    if "recorded" in r:
        rc = r["recorded"]
        L.append(f"| {r['label']} | {f(rc['contact_frame'])} | {rc['tilt_at_contact_deg']:.1f} ({rc['tilt_max_precontact_deg']:.1f}) | {rc['disp_at_contact_mm']:.1f} | {rc['peak_rise_mm']:.0f} / {rc['final_rise_mm']:.0f} |")
if fails: L += ["", "Failed / incomplete: " + ", ".join(fails)]
open(os.path.join(ROOT, "sweeps/assetsweep_summary.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
