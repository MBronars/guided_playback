"""No-guidance planner (MPPI argmin K 32) on the 25 benchmark picks, next to the guided planner view (sweeps/planner_eval.json).
Per pick: generated and Isaac goal distance of the committed plan, Isaac peak lift, warm plan time (sum of replan seconds).
-> sweeps/mppi_eval.{json,md}"""
import glob, json, os
import numpy as np
G = {r["run"].replace("cyl_", "").replace("_it10", ""): r for r in json.load(open("sweeps/planner_eval.json"))}
rows = []
for d in sorted(glob.glob("runs/pickup/mppiargmin_t*_K32")):
    if not os.path.exists(d + "/replay_isaac/analysis.json"): continue
    rj = json.load(open(d + "/result.json")); sim = next(r for r in json.load(open(d + "/replay_isaac/analysis.json"))["rows"] if r["variant"] == "guided")
    key = os.path.basename(d).replace("mppiargmin_", "").replace("_K32", ""); g = G.get(key, {})
    rows.append(dict(run=os.path.basename(d), asset=str(rj["episodes"][0]), target=key.split("_")[0], gen_goal_mm=round(rj["best_mm"][0], 1),
                     sim_goal_mm=round(sim["sim_to_target_mm"], 1), sim_peak_lift_mm=round(sim["sim_lift_mm"], 1), plan_s=round(sum(rj["iteration_seconds"]), 1),
                     guided_sim_goal_mm=g.get("sim_goal_mm"), guided_gen_s=g.get("gen_time_s")))
json.dump(rows, open("sweeps/mppi_eval.json", "w"), indent=1)
L = ["| asset | target | gen goal mm | Isaac goal mm | Isaac peak lift mm | plan s | guided Isaac goal mm |", "|---|---|---|---|---|---|---|"]
for r in rows: L.append(f"| {r['asset']} | {r['target']} | {r['gen_goal_mm']:.0f} | {r['sim_goal_mm']:.0f} | {r['sim_peak_lift_mm']:.0f} | {r['plan_s']:.1f} | {r['guided_sim_goal_mm']} |")
v = np.array([r["sim_goal_mm"] for r in rows]); t = np.array([r["plan_s"] for r in rows])
if len(v): L.append(f"\nN {len(v)}: Isaac goal median {np.median(v):.0f} mm, <=50 mm {(v <= 50).sum()}/{len(v)}, <=60 mm {(v <= 60).sum()}/{len(v)}; plan time {t.mean():.1f} +- {t.std(ddof=1) if len(t) > 1 else 0:.1f} s")
open("sweeps/mppi_eval.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
