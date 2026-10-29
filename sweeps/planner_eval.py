"""Planner view of the random-target picks: per run, pick the member with the lowest GENERATED objective (what the planner
knows), report its generated and Isaac goal distance, plus warm generation time (server-side, sum of iteration seconds;
no replay). -> sweeps/planner_eval.{json,md}"""
import glob, json, os
import numpy as np
rows = []
for d in sorted(glob.glob("runs/pickup/cyl_t*_it10")):
    if not os.path.exists(d + "/replay_isaac/analysis.json"): continue
    rj = json.load(open(d + "/result.json")); an = json.load(open(d + "/replay_isaac/analysis.json"))["rows"]
    obj = rj["best_mm"]; m = int(np.argmin(obj))
    sim = next(r for r in an if r["member"] == m and r["variant"] == "guided")
    rows.append(dict(run=os.path.basename(d), asset=str(rj["episodes"][0]), target=os.path.basename(d).split("_")[1], offset=rj["offset"], chosen_member=m,
                     gen_goal_mm=round(obj[m], 1), sim_goal_mm=round(sim["sim_to_target_mm"], 1), sim_peak_lift_mm=round(sim["sim_lift_mm"], 1),
                     gen_time_s=round(sum(rj["iteration_seconds"]), 1), iterations=rj["iterations"]))
json.dump(rows, open("sweeps/planner_eval.json", "w"), indent=1)
L = ["| asset | target | chosen member | gen goal mm | Isaac goal mm | gen time s (warm) |", "|---|---|---|---|---|---|"]
for r in rows: L.append(f"| {r['asset']} | {r['target']} | m{r['chosen_member']} | {r['gen_goal_mm']:.0f} | {r['sim_goal_mm']:.0f} | {r['gen_time_s']:.1f} |")
open("sweeps/planner_eval.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
