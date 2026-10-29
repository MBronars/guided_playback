"""PointWorld's own pick prediction (point cloud, no simulator) at the first and last frame, for the milk carton and the soda-can
candidates; same look as render_pick_pred_errors.py (v3, no ghosts). Writes figures/pick_pw_pred_start_end/.
  PYOPENGL_PLATFORM=egl .../foundationpose/bin/python bin/render_pick_pw_start_end.py [--cases milk_carton:0,soda_can:1] [--size 1400]"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import render_pick_pred_errors as E
ap = argparse.ArgumentParser(); ap.add_argument("--cases", default="milk_carton:0,soda_can:1,soda_can:3,soda_can:4"); ap.add_argument("--size", type=int, default=1400)
ap.add_argument("--out", default=os.path.join(E.ROOT, "figures", "pick_pw_pred_start_end")); A = ap.parse_args()
E.OUT = A.out; E.FRAMES = (0, 60); E.GHOSTS = False; E.SPAN_PAD = 0.05; E.FIT_PROJ = 0.86; os.makedirs(A.out, exist_ok=True); recs = []
for c in A.cases.split(","):
    obj, t = c.split(":"); rec = E.render_case("PointWorld", obj, int(t), W=A.size, H=A.size); recs.append(rec)
    print(f"PointWorld {obj} t{t}: object scale at frame 60 {rec['obj_scale'][-1]:.2f} (min {min(rec['obj_scale']):.2f})", flush=True)
json.dump(recs, open(os.path.join(A.out, "cases.json"), "w"), indent=1); print("wrote", A.out)
