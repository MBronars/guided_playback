"""Rebuild a pushing paper_eval summary from replay_sapien_yawfix/ (scenario-yaw spawn fix, 2026-09-25): same picked member
(chosen by the planner's imagined objective), simulator result from the corrected replay. Writes <tag>_yawfix_summary.json."""
import json, sys, os
import numpy as np
G = "/data/user_data/mbronars/packages/guided_playback/runs"
for tag in sys.argv[1:]:
    d = json.load(open(f"{G}/_paper_eval/{tag}_summary.json")); rows = []
    for r in d["rows"]:
        a = json.load(open(f"{G}/pushing/{r['name']}/replay_sapien_yawfix/analysis.json"))["rows"]
        new = [x for x in a if x["member"] == r["picked_member"] and x["variant"] == "guided"][0]["sim_to_target_mm"]
        rows.append(dict(r, sim_mm_yaw0=r["sim_mm"], sim_mm=round(new, 1)))
    d["rows"] = rows; d["note"] = "sim_mm from replay_sapien_yawfix (cube spawned at the scenario yaw); sim_mm_yaw0 = the earlier replay"
    json.dump(d, open(f"{G}/_paper_eval/{tag}_yawfix_summary.json", "w"), indent=1)
    s0 = np.array([x["sim_mm_yaw0"] for x in rows]); s = np.array([x["sim_mm"] for x in rows]); t = np.array([x["target_mm"] for x in rows]); c = t - s
    print(f"{tag:46s} n={len(rows)}  sim {s.mean()/10:.2f}±{s.std()/10:.2f} cm (was {s0.mean()/10:.2f})  closed {100*np.mean(c/t):.1f}% (was {100*np.mean((t-s0)/t):.1f}%)  <1.5 {(s<15).sum()} <3 {(s<30).sum()} <5 {(s<50).sum()}  | per-ep change {np.mean(s-s0)/10:+.2f}±{np.std(s-s0)/10:.2f} cm")
