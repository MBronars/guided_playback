"""Pool units for the pick experiment (noise vs actions): every (planner setting, problem) is one unit.
  python sweeps/noise_vs_action/make_units.py <pool root> <settings.json> [--problems all|smoke]
settings.json: [{"tag": "cem_K32_R3", "shape": "B4_nograd", "knobs": {...}}, ...]"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from gp.pool import add_units
HERE = "/data/user_data/mbronars/packages/guided_playback"
CKPT = "/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_pickup_drop50_100m_unet_bidir/checkpoints/final.pt"
ASSETS = [("milk_carton", 11), ("peanut_butter_jar", 1), ("soda_can", 13), ("pasta_box", 9), ("water_bottle", 16)]
TARGETS = [l.split() for l in open(f"{HERE}/sweeps/cyl_targets.txt") if l.strip()]
root, settings = sys.argv[1], json.load(open(sys.argv[2])); smoke = "smoke" in sys.argv
probs = [(a, i, t, o) for a, i in ASSETS for t, o in TARGETS]
if smoke: probs = probs[:2]
units = []
for s in settings:
    B = int(s.get("B", 4))
    for a, i, t, o in probs:
        req = dict(ckpt=CKPT, domain="pickup", task="lift", select=f"{HERE}/sweeps/assetsweep_select.json", episodes=",".join([str(i)] * B),
                   transitions=60, steps=10, offset=o, seed=0, churn=True, t_freeze=0.0, fix_fps=True, desc=f"noise_vs_action {s['tag']} {a} t{t}")
        req.update(s["knobs"])
        units.append(dict(id=f"{s['tag']}__{a}_t{t}", shape=s["shape"], tag=s["tag"], problem=f"{a}_t{t}", req=req))
print("added", add_units(root, units), "of", len(units))
