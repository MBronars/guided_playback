"""Pool units for the pile experiment on the gpwm drop50 mould checkpoint: (setting, paper scenario seed 42) -> unit.
  python make_units_pile.py <pool> <settings.json> [--ckpt <pt>] [--eps 0-49]"""
import json, sys, numpy as np
sys.path.insert(0, "/data/user_data/mbronars/packages/guided_playback")
from gp.pool import add_units
CK = "/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mould_drop50_100m_unet_bidir/checkpoints/step_100000.pt"
SC = "/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid/seed_42"
AS = "/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action_dev/pile/action_stats_mould.json"
root, settings = sys.argv[1], json.load(open(sys.argv[2]))
CK = sys.argv[sys.argv.index("--ckpt") + 1] if "--ckpt" in sys.argv else CK                 # 27 Oct: rerun on final.pt (200k)
lo, hi = map(int, (sys.argv[sys.argv.index("--eps") + 1] if "--eps" in sys.argv else "0-49").split("-"))
units = []
for s in settings:
    B = int(s.get("B", 4))
    for ep in range(lo, hi + 1):
        d = np.load(f"{SC}/ep_{ep:03d}.npz"); bi = int(d["batch_idx"]); dx, dy = map(float, d["target_offset_xy"])
        req = dict(ckpt=CK, domain="mould", task="move", episodes=",".join([str(bi)] * B), transitions=150, steps=6, offset=f"{dx},{dy},0",
                   seed=0, churn=True, t_freeze=0.0, fix_fps=True, action_stats=AS, desc=f"pile gpwm {s['tag']} ep{ep:03d}")
        req.update(s["knobs"])
        pre = {"dno": "a", "cem_noise": "b", "cem_act": "c"}[next(k for k in ("dno", "cem_noise", "cem_act") if s["tag"].startswith(k))]   # claim order: guided first
        units.append(dict(id=f"{pre}_{s['tag']}__ep{ep:03d}", shape=s["shape"], tag=s["tag"], problem=f"ep{ep:03d}", req=dict(req, ep=ep)))
print("added", add_units(root, units), "of", len(units))
