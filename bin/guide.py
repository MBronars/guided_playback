"""Guided streaming rollout on a served or locally loaded checkpoint; writes runs/<domain>/<name>/{rollout.npz,result.json}.

  python bin/guide.py --adapter paper --ckpt <step_120000.pt> --domain pickup --task push --episodes 10,82,1,26 \
      --transitions 60 --steps 10 --churn --t_freeze 0 --fix_fps --joint_limits 13.5,53.5 --close_fingers \
      --hand_start natural --offset 0.15,0,0 --iterations 10 --lr 0.2 --lr_final 0.02 --name s120k_push_x
  python bin/guide.py --adapter clean_push --ckpt <push0418 final.pt> --task push --episodes 0,0,0,0 --transitions 60 --steps 8 --churn --t_freeze 0.8 --optimizer unitgrad --shell --offset 0.1,0,0 --name ...
  python bin/guide.py --adapter paper --ckpt <mould step_60000.pt> --domain mould --task move --episodes 0,1,2,3 --transitions 150 --steps 10 --churn --t_freeze 0.8 --optimizer unitgrad --shell --offset 0.1,0,0 --name ...
"""
import argparse, json, os, sys, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="paper", choices=("paper", "clean_push")); ap.add_argument("--ckpt", required=True); ap.add_argument("--domain", default=None)
ap.add_argument("--task", default="push", choices=("push", "lift", "wall", "move"), help="push/move: goal on the object's start plane; lift: 3-D offset; wall: push goal + wall box between start and goal")
ap.add_argument("--episodes", default="0,1,2,3", help="loader indices; repeats = population (the batch IS the population)")
ap.add_argument("--transitions", type=int, default=60); ap.add_argument("--steps", type=int, default=10); ap.add_argument("--keep", type=int, default=3)
ap.add_argument("--offset", default="0.15,0,0"); ap.add_argument("--random_offset", default=None, help="RMIN,RMAX metres: per-member random goal on the plane, seeded")
ap.add_argument("--own_goal", default=None, help="rl_push tags json: goal = the episode's recorded goal_pos (+ loader shift)"); ap.add_argument("--goal_dist_m", type=float, default=None)
ap.add_argument("--iterations", type=int, default=10); ap.add_argument("--lr", type=float, default=0.2); ap.add_argument("--lr_final", type=float, default=None)
ap.add_argument("--optimizer", default="adam", choices=("adam", "unitgrad")); ap.add_argument("--shell", action="store_true", help="re-project every draw onto its Gaussian shell after each update")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--fix_fps", action="store_true"); ap.add_argument("--churn", action="store_true"); ap.add_argument("--t_freeze", type=float, default=0.0)
ap.add_argument("--joint_limits", default=None, help="QMIN,QMAX mm (paper pickup fingers): 13.5,53.5")
ap.add_argument("--close_fingers", action="store_true"); ap.add_argument("--closed_sep_mm", type=float, default=27.0)
ap.add_argument("--hand_start", choices=("mid", "natural"), default=None); ap.add_argument("--hand_jitter_mm", type=float, default=30.0); ap.add_argument("--hand_z_mm", type=float, default=105.0)
ap.add_argument("--hand_bearing_deg", type=float, default=60.0); ap.add_argument("--hand_bearing_jitter_deg", type=float, default=15.0); ap.add_argument("--hand_dist_mm", type=float, default=250.0)
ap.add_argument("--hand_dist_jitter_mm", type=float, default=30.0); ap.add_argument("--hand_dz_mm", type=float, default=85.0); ap.add_argument("--hand_tilt_deg", type=float, default=0.0)
ap.add_argument("--wall_height", type=float, default=0.10); ap.add_argument("--wall_thickness", type=float, default=0.02); ap.add_argument("--wall_width", type=float, default=0.30)
ap.add_argument("--clearance_mm", type=float, default=5.0); ap.add_argument("--box_weight", type=float, default=5.0); ap.add_argument("--floor_weight", type=float, default=0.0)
ap.add_argument("--eager", action="store_true"); ap.add_argument("--name", required=True); ap.add_argument("--runs", default=os.path.join(os.path.dirname(HERE), "runs"))
ap.add_argument("--desc", default="", help="one line shown on the page")
ap.add_argument("--sampler", default="stream", choices=("stream", "chunked"), help="chunked: 3 clean context rows + 16 future rows denoised together per window (first context = recorded)")
A = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = True
from gp.session import load_adapter, guided_run
t0 = time.time(); ad = load_adapter(A); print(f"loaded in {time.time() - t0:.0f} s", flush=True)
guided_run(ad, A)
