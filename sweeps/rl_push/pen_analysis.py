"""Per member: planned hull penetration (mm), lowest finger z, floating vs arm replay outcome. python pen_analysis.py <run glob>"""
import sys, glob, os, json; sys.path.insert(0, "/data/user_data/mbronars/packages/guided_playback")
import numpy as np, torch
from gp.objectives import Objective
dev = torch.device("cpu")
for run in sorted(glob.glob(sys.argv[1])):
    if not os.path.exists(run + "/replay_isaac/analysis.json"): continue
    z = np.load(run + "/rollout.npz"); rj = json.load(open(run + "/result.json")); pick = int(np.argmin(rj["best_mm"]))
    pred = torch.tensor(z["predicted"], device=dev); om = torch.tensor(z["object_mask"], device=dev); valid = torch.tensor(z["roles"] >= 0, device=dev); tgt = torch.tensor(z["target"], device=dev)
    obj = Objective(tgt, om, valid)
    with torch.no_grad(): ct = obj.contact(pred).numpy() * 1e3
    fing = z["roles"][0] == 1
    fl = {r["member"]: r for r in json.load(open(run + "/replay_isaac/analysis.json"))["rows"]}
    arm = {r["member"]: r for r in json.load(open(run + "/replay_isaac_arm/analysis.json"))["rows"]} if os.path.exists(run + "/replay_isaac_arm/analysis.json") else {}
    ep = int(z["episodes"][0]); idx = {r["loader"]: r for r in json.load(open("/data/user_data/mbronars/packages/guided_playback/sweeps/rl_push/test_index.json"))}[ep]
    print(f"== {os.path.basename(run)} {idx['asset'].split('/')[-2]} {idx['cell']} bbox {idx['bbox_mm']}", flush=True)
    for m in range(z["predicted"].shape[0]):
        g = z["predicted"][m]; fz = g[:, fing, 2].min() * 1e3; a = arm.get(m, {})
        print(f"{'*' if m == pick else ' '} m{m:2d} gen {rj['best_mm'][m]:6.1f} pen {ct[m]:5.1f}mm fing_zmin {fz:4.0f} | float {fl[m]['sim_to_target_mm']:6.1f} lift {fl[m]['sim_lift_mm']:5.1f} | arm {a.get('sim_to_target_mm', float('nan')):6.1f} lift {a.get('sim_lift_mm', float('nan')):5.1f} jm {a.get('arm_min_joint_margin_rad', float('nan')):+.2f}", flush=True)
