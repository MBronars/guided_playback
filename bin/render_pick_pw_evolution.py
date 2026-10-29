"""PointWorld's own predicted cloud (no simulator) over time, object-centred: every panel looks at the predicted object's
centroid at that frame from the same direction and distance, so its apparent size is directly comparable across frames.
Hand grey-blue, fingers teal, object orange (as render_pick_pred_errors.py v3). Frame 0 = the initial (observed) object.
  PYOPENGL_PLATFORM=egl .../foundationpose/bin/python bin/render_pick_pw_evolution.py [--frames 0,15,30,45,60]"""
import argparse, json, os, sys
import numpy as np, pyrender
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import render_pick_pred_errors as E
from gp.render import frames as F
ap = argparse.ArgumentParser(); ap.add_argument("--cases", default="milk_carton:0,soda_can:1"); ap.add_argument("--frames", default="0,15,30,45,60")
ap.add_argument("--size", default="1600x1200"); ap.add_argument("--az", type=float, default=-62); ap.add_argument("--el", type=float, default=32)
ap.add_argument("--object_only", action="store_true"); ap.add_argument("--zoom", type=float, default=3.4, help="view height = zoom x the initial object's largest extent")
ap.add_argument("--out", default=os.path.join(E.ROOT, "figures", "pick_pw_evolution_v2")); A = ap.parse_args()
W, H = (int(v) for v in A.size.split("x")); FR = [int(f) for f in A.frames.split(",")]; os.makedirs(A.out, exist_ok=True); recs = []
yfov = np.pi / 6; az, el = np.radians(A.az), np.radians(A.el); d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
for c in A.cases.split(","):
    obj, t = c.split(":"); t = int(t); a = E.LONG[obj]; roles = E.SC[a + "__roles"]; X = E.load("PointWorld", obj, t)
    o = X[:, roles == 2]; top = float(o[0][:, 2].min()); ext0 = float((o[0].max(0) - o[0].min(0)).max())
    dist = A.zoom * ext0 / (2 * np.tan(yfov / 2)); files = []
    def rms(Y): return float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    for f in FR:
        focus = o[f].mean(0); cam = F.look_at(focus + dist * d, focus)
        sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.55, 0.55, 0.55]); wood = F.material_texture("Wood010_1K-JPG"); q = o[0].mean(0)
        sc.add(F.plane([[q[0] - 2.5, q[1] - 2.5, top], [q[0] + 2.5, q[1] - 2.5, top], [q[0] + 2.5, q[1] + 2.5, top], [q[0] - 2.5, q[1] + 2.5, top]], wood, (7.0, 6.6)))
        for role in ((2,) if A.object_only else (0, 1, 2)): sc.add(E.spheres(X[f][roles == role], E.COL[role], 0.0034))
        sc.add(pyrender.PerspectiveCamera(yfov=yfov), pose=cam); sc.add(pyrender.DirectionalLight(intensity=2.4), pose=cam)
        sc.add(pyrender.DirectionalLight(intensity=1.4), pose=F.look_at(focus + np.array([-0.4, -0.6, 1.2]), focus))
        r = pyrender.OffscreenRenderer(W, H); rgba, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
        al = rgba[..., 3:4] / 255.0; fn = f"pointworld_{obj}_t{t}_f{f:02d}.png"
        Image.fromarray((rgba[..., :3] * al + 255 * (1 - al)).astype(np.uint8)).save(os.path.join(A.out, fn)); files.append(fn)
    ext = o.max(1) - o.min(1)
    rec = dict(object=obj, target=f"t{t}", frames=FR, images=files, obj_scale=[round(rms(o[f]) / rms(o[0]), 3) for f in range(len(o))],
               obj_height_ratio=[round(float(ext[f, 2] / ext[0, 2]), 3) for f in range(len(o))], obj_lift_mm=[round(float((o[f, :, 2].mean() - o[0, :, 2].mean()) * 1e3)) for f in range(len(o))])
    recs.append(rec); print(obj, t, "scale", [rec["obj_scale"][f] for f in FR], "lift", [rec["obj_lift_mm"][f] for f in FR], flush=True)
json.dump(recs, open(os.path.join(A.out, "cases.json"), "w"), indent=1); print("wrote", A.out)
