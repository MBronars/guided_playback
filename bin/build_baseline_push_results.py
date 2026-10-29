"""runs/pushing_baselines/results.json: 5 hard seed-42 scenarios x {gns, pw (old paper config), pw60 (frame-60 cost)} x {open, closed},
plus the stored May numbers. Per cell: final_mm, planned_mm, start_mm, plan_seconds_total, wall_seconds, per-replan rows, final SAPIEN sim_trajectory path."""
import glob, json, os
import numpy as np
R = "/data/user_data/mbronars/packages/guided_playback/runs/pushing_baselines"; PWR = "/data/user_data/mbronars/packages/PointWorld/runs"
out = dict(scenarios={}, notes=dict(
    gns="best of 256 template realizations, free rollout to task frame 60 in every plan (open loop rolls 60 frames; closed-loop replan after c executed rolls 60 - c); cost = block-centroid distance at task frame 60 only. Context = last 6 executed SAPIEN frames (front-padded with frame 0).",
    pw="old paper config: dynamic horizon base 10, cap 4 windows, per_frame_dist 0.015; each MPPI replan scores the LAST frame of its active horizon H_active = min(40, N*10, remaining), i.e. a frame earlier than 60 until remaining <= H_active.",
    pw60="frame-60 cost: window_cap 6, per_frame_dist 1e-9 -> N = 6 always, H_active = remaining steps, so every MPPI replan (5-step internal replans, and every closed-loop replan) rolls out to task frame 60 and scores it."))
for e in (5, 11, 14, 29, 37):
    E = f"{e:03d}"; s = out["scenarios"][f"ep{E}"] = {}
    for m in ("gns", "pw", "pw60"):
        for mode in ("ol", "cl"):
            f = os.path.join(R, f"{m}_{mode}_hard42_ep{E}", "summary.json")
            if os.path.exists(f):
                d = json.load(open(f)); s[f"{m}_{'open' if mode == 'ol' else 'closed'}"] = {k: d[k] for k in ("final_mm", "planned_mm", "start_mm", "plan_seconds_total", "wall_seconds", "final_sim", "rows")}
    g = os.path.join(PWR, "gns_bestofN_seed42_hard_K256", f"ep{E}", "replay_set_pose", "sim_trajectory.npz")
    p = os.path.join(PWR, "seed42_hard_k128", f"ep_{E}", "replay_set_pose", "sim_trajectory.npz")
    s["may"] = dict(gns_open_mm=round(float(np.load(g)["sim_final_dist"]) * 1e3, 1) if os.path.exists(g) else None,
                    pw_open_mm=round(float(np.load(p)["sim_final_dist"]) * 1e3, 1) if os.path.exists(p) else None)
json.dump(out, open(os.path.join(R, "results.json"), "w"), indent=1)
hdr = ["ep", "May GNS", "GNS open", "GNS closed", "May PW", "PW old open", "PW old closed", "PW60 open", "PW60 closed"]
print(" | ".join(hdr))
for k, s in out["scenarios"].items():
    g = lambda c: s.get(c, {}).get("final_mm")
    print(" | ".join(str(x) for x in [k, s["may"]["gns_open_mm"], g("gns_open"), g("gns_closed"), s["may"]["pw_open_mm"], g("pw_open"), g("pw_closed"), g("pw60_open"), g("pw60_closed")]))
for c in ("gns_open", "gns_closed", "pw_open", "pw_closed", "pw60_open", "pw60_closed"):
    v = [s[c]["final_mm"] for s in out["scenarios"].values() if c in s]; t = [s[c]["plan_seconds_total"] for s in out["scenarios"].values() if c in s]
    if v: print(f"{c}: n {len(v)} mean {np.mean(v):.1f} mm, plan s mean {np.mean(t):.1f} (min {min(t):.1f}, max {max(t):.1f})")
