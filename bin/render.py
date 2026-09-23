"""Rendered frames (style 2) of a run + its replay: python bin/render.py --run runs/<domain>/<name> [--sim isaac] [--variants guided,unguided]
Run in the foundationpose env: PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 .../foundationpose/bin/python bin/render.py ..."""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--sim", default=None); ap.add_argument("--variants", default="guided")
ap.add_argument("--size", default="480x360"); ap.add_argument("--fps", type=float, default=10); ap.add_argument("--max_frames", type=int, default=40); ap.add_argument("--scale", type=float, default=None, help="render magnification (default per domain: pushing 2, mould 3, pickup 1)"); ap.add_argument("--arm", action="store_true", help="style 3: the Franka arm carries the gripper / holds the tool")
A = ap.parse_args(); rj = json.load(open(os.path.join(A.run, "result.json"))); sim = A.sim or rj.get("sim")
from gp.render.frames import render_run
render_run(A.run, sim, os.path.join(A.run, f"render_{sim}" + ("_arm" if A.arm else "")), A.variants.split(","), tuple(int(v) for v in A.size.split("x")), A.fps, A.max_frames, arm=A.arm, scale=A.scale)
