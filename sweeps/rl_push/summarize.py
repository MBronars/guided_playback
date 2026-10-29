"""Results table from disk for runs/rl_push/<tag>_ep*: planner view (member with the lowest imagined distance) in the floating-hand
and full-arm Isaac replays, plus the best-in-sim member (oracle) for reference.  python sweeps/rl_push/summarize.py <tag> [<tag2>...]"""
import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tilt import report
HERE = "/data/user_data/mbronars/packages/guided_playback"
IDX = {r["loader"]: r for r in json.load(open(f"{HERE}/sweeps/rl_push/test_index.json"))}
for tag in sys.argv[1:]:
    rows = []
    for run in sorted(glob.glob(f"{HERE}/runs/rl_push/{tag}_ep*")):
        if not os.path.exists(run + "/replay_isaac/analysis.json"): continue
        rj = json.load(open(run + "/result.json")); ep = int(rj["episodes"][0]); t = IDX[ep]; pick = int(np.argmin(rj["best_mm"]))
        fl = report(run); os.environ["GP_TILT_SUFFIX"] = "_arm"; arm = report(run) if os.path.exists(run + "/replay_isaac_arm/analysis.json") else None; os.environ.pop("GP_TILT_SUFFIX")
        a = {r["member"]: r for r in json.load(open(run + "/replay_isaac_arm/analysis.json"))["rows"]} if arm else {}
        p = fl[pick]; pa = arm[pick] if arm else {}
        sims = [r["sim_mm"] for r in fl]; ora = int(np.argmin(sims))
        rows.append(dict(case=ep, obj=t["asset"].split("/")[-2] + " " + t["asset"].split("/")[-1].replace("asset_", "#"), cell=t["cell"], goal_mm=round(t["goal_dist"] * 1e3),
                         imagined=p["gen_mm"], float_mm=p["sim_mm"], float_tilt=p["sim_tilt_max"], arm_mm=pa.get("sim_mm"), arm_tilt=pa.get("sim_tilt_max"),
                         arm_track_mm=round(a[pick]["arm_hand_err_mm_max"], 1) if a else None, n_float_ok=int(sum(s <= 50 for s in sims)), oracle_mm=round(sims[ora], 1), oracle_tilt=fl[ora]["sim_tilt_max"], name=os.path.basename(run)))
    print(f"== {tag}: {len(rows)} cases (planner view: pick = lowest imagined distance; success = <= 50 mm)")
    print(f"{'case':>4} {'object':28s} {'cell':22s} {'goal':>4} {'imag':>6} | {'float':>6} {'tilt':>5} | {'arm':>6} {'tilt':>5} {'track':>5} | {'ok/12':>5} {'oracle':>6} {'tilt':>5}")
    for r in rows:
        print(f"{r['case']:4d} {r['obj']:28s} {r['cell']:22s} {r['goal_mm']:4d} {r['imagined']:6.1f} | {r['float_mm']:6.1f} {r['float_tilt']:5.1f} | {str(r['arm_mm']):>6} {str(r['arm_tilt']):>5} {str(r['arm_track_mm']):>5} | {r['n_float_ok']:5d} {r['oracle_mm']:6.1f} {r['oracle_tilt']:5.1f}")
    f = np.array([r["float_mm"] for r in rows]); ft = np.array([r["float_tilt"] for r in rows]); o = np.array([r["oracle_mm"] for r in rows])
    am = np.array([r["arm_mm"] for r in rows if r["arm_mm"] is not None]); at = np.array([r["arm_tilt"] for r in rows if r["arm_mm"] is not None])
    print(f"  floating hand, planner pick: median {np.median(f):.0f} mm, {int((f <= 50).sum())}/{len(f)} <= 50 mm, {int(((f <= 50) & (ft < 20)).sum())} of those with tilt < 20 deg")
    if len(am): print(f"  full arm, planner pick:      median {np.median(am):.0f} mm, {int((am <= 50).sum())}/{len(am)} <= 50 mm, {int(((am <= 50) & (at < 20)).sum())} with tilt < 20 deg")
    print(f"  oracle (best member in sim): median {np.median(o):.0f} mm, {int((o <= 50).sum())}/{len(o)} <= 50 mm; cases with >= 1 member <= 50 mm: {int(sum(r['n_float_ok'] > 0 for r in rows))}/{len(rows)}")
    json.dump(rows, open(f"{HERE}/runs/_paper_eval/{tag}_table.json", "w"), indent=1)
