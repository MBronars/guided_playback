"""Replay a run's generated rollouts in its simulator and write <run>/replay_<sim>/{replay.npz,analysis.json}.

  python bin/replay.py --run runs/pickup/<name> [--sim isaac|genesis|sapien] [--variants guided,unguided] [--gpu 0]
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--sim", default=None); ap.add_argument("--variants", default="guided,unguided")
ap.add_argument("--members", default=None, help="comma-separated member indices (default: all)"); ap.add_argument("--gpu", default="0"); ap.add_argument("--rate", type=int, default=20); ap.add_argument("--frame_skip", type=int, default=25); ap.add_argument("--suffix", default="", help="write to replay_<sim><suffix>/ (variant replays, e.g. a stiffer controller)")
A = ap.parse_args()
MEM = None if not A.members else [int(v) for v in A.members.split(",")]
rj = json.load(open(os.path.join(A.run, "result.json"))); sim = A.sim or rj.get("sim")
if not sim: raise SystemExit("no default simulator for this run; pass --sim")
out = os.path.join(A.run, f"replay_{sim}{A.suffix}"); variants = A.variants.split(",")
if sim == "isaac":
    from gp.replay import isaac_float; isaac_float.run(A.run, out, variants, rate=A.rate, gpu=A.gpu)
elif sim == "isaac_arm":                                   # the gripper on a full Franka (rl_push): replay_isaac_arm.py
    from gp.replay import isaac_float; isaac_float.run(A.run, out, variants, rate=A.rate, gpu=A.gpu, arm=True)
elif sim == "genesis":
    from gp.replay import genesis_mould; genesis_mould.run(A.run, out, variants, frame_skip=A.frame_skip, gpu=A.gpu, members=MEM)
elif sim == "sapien":
    from gp.replay import sapien_push; sapien_push.run(A.run, out, variants, gpu=A.gpu, members=MEM)
else:
    raise SystemExit(f"unknown sim {sim}")
