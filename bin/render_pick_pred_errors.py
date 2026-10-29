"""Point-cloud views of the world models' own pick predictions (no simulator): GNS's gripper degenerates, PointWorld's object
shrinks. Each panel shows the predicted cloud at one frame (hand grey-blue, fingers teal, object orange) and, as a translucent
grey ghost, the rigid body it should be: each body's frame-0 point set Kabsch-fitted (rotation + translation, no scale) to the
prediction at that frame. Ours on the same problem is rendered the same way for reference.

  PYOPENGL_PLATFORM=egl .../foundationpose/bin/python bin/render_pick_pred_errors.py"""
import json, os, sys
import numpy as np, trimesh
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from gp.render import frames as F
import pyrender

RES = "/data/user_data/mbronars/packages/pwm_final_paper/results/picking"
OUT = os.path.join(ROOT, "figures", "pick_pred_errors_v3")
FIT_PROJ = None                                                                  # e.g. 0.85: every framed point within 85 % of the half-frame
SPAN_PAD = 0.05                                                                  # framing margin (m) for the non-GNS strips
GHOSTS = False                                                                   # v3: no grey rigid-body ghosts (the user's choice); v2 has them
SC = np.load(RES + "/benchmark_spec/pick_benchmark_5x5_scenes.npz")
LONG = {"milk_carton": "milk_carton_asset_0037", "peanut_butter_jar": "peanut_butter_jar_asset_0003", "soda_can": "soda_can_asset_0012", "pasta_box": "pasta_box_asset_0058", "water_bottle": "water_bottle_asset_0087"}
CASES = [("GNS", "milk_carton", 0), ("GNS", "pasta_box", 0), ("PointWorld", "soda_can", 2), ("PointWorld", "peanut_butter_jar", 1)]
FRAMES = (0, 20, 40, 60)
COL = {0: [92, 110, 150, 255], 1: [40, 150, 150, 255], 2: [226, 128, 60, 255]}          # role -> colour (hand, finger, object)


def load(method, obj, t):
    a = LONG[obj]
    if method == "GNS": return np.load(f"{RES}/gns/pick_benchmark_gns_bestof128/gns_t{t}_{a}_rollout.npz")["predicted"][0]
    if method == "PointWorld": return np.load(f"{RES}/pointworld/plans/pointworld_mppi_t{t}_{a}.npz")["predicted"][0]
    rj = json.load(open(f"{RES}/ours/runs/cyl_t{t}_{obj}_it10/result.json")); m = int(np.argmin(rj["best_mm"]))
    return np.load(f"{RES}/ours/runs/cyl_t{t}_{obj}_it10/rollout.npz")["predicted"][m]


def kabsch_fit(A, B):
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, _, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (A - ca) @ R.T + cb


def ghost_cloud(X0, Xf, pb):
    G = np.zeros_like(Xf)
    for b in np.unique(pb):
        i = pb == b; G[i] = kabsch_fit(X0[i].astype(np.float64), Xf[i].astype(np.float64))
    return G


def spheres(pts, rgba, r, alpha=None):
    s = trimesh.creation.icosphere(subdivisions=1, radius=r); poses = np.repeat(np.eye(4)[None], len(pts), 0); poses[:, :3, 3] = pts
    if alpha is None:
        s.visual = trimesh.visual.ColorVisuals(s, face_colors=rgba); return pyrender.Mesh.from_trimesh(s, poses=poses, smooth=False)
    mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[c / 255 for c in rgba[:3]] + [alpha], alphaMode="BLEND", metallicFactor=0.0, roughnessFactor=0.9)
    return pyrender.Mesh.from_trimesh(s, poses=poses, smooth=False, material=mat)


def render_case(method, obj, t, W=720, H=720):
    a = LONG[obj]; roles = SC[a + "__roles"]; pb = SC[a + "__particle_body"]; X = load(method, obj, t); top = float(X[0][roles == 2][:, 2].min())
    # one fixed camera for the whole strip: frame the ghosts (the true bodies) plus the dense 95 % of the prediction
    G = {f: ghost_cloud(X[0], X[f], pb) for f in FRAMES}
    if method == "GNS":                                                      # the gripper cloud scatters far: frame the starting layout with a fixed margin
        lo, hi = X[0].min(0) - 0.12, X[0].max(0) + 0.12; lo[2] = top; focus = (lo + hi) / 2; span = float(max(hi - lo))
    else:
        pts = np.concatenate([G[f] for f in FRAMES] + [np.percentile(X[f], [2.5, 97.5], axis=0) for f in FRAMES])
        lo, hi = pts.min(0), pts.max(0); focus = (lo + hi) / 2; span = float(max(hi - lo)) + SPAN_PAD
    yfov = np.pi / 5; dist = span / (2 * np.tan(yfov / 2)) * 1.05; az, el = np.radians(-62), np.radians(20)
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]); cam = F.look_at(focus + dist * d, focus)
    if FIT_PROJ and method != "GNS":                                          # back the camera off until every framed point projects inside the image
        Pm = pyrender.PerspectiveCamera(yfov=yfov).get_projection_matrix(W, H)
        for _ in range(40):
            q = (Pm @ np.linalg.inv(cam) @ np.c_[pts, np.ones(len(pts))].T); n = q[:2] / q[3]
            if np.abs(n).max() < FIT_PROJ: break
            dist *= 1.06; cam = F.look_at(focus + dist * d, focus)
    ims = []
    for f in FRAMES:
        sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.55, 0.55, 0.55])
        wood = F.material_texture("Wood010_1K-JPG"); c = focus
        sc.add(F.plane([[c[0] - 1.2, c[1] - 1.2, top], [c[0] + 1.2, c[1] - 1.2, top], [c[0] + 1.2, c[1] + 1.2, top], [c[0] - 1.2, c[1] + 1.2, top]], wood, (3.6, 3.4)))
        r_pt = 0.0048 if method == "GNS" else 0.0034
        for role in (0, 1, 2):
            sc.add(spheres(X[f][roles == role], COL[role], r_pt))
        if GHOSTS and method == "PointWorld":                                    # the object at its true size, rigidly fitted: shows the shrink
            sc.add(spheres(G[f][roles == 2], [140, 140, 140, 255], 0.0040, alpha=0.28))
        elif GHOSTS and method == "Ours":
            sc.add(spheres(G[f], [140, 140, 140, 255], 0.0040, alpha=0.28))
        sc.add(pyrender.PerspectiveCamera(yfov=yfov), pose=cam); sc.add(pyrender.DirectionalLight(intensity=2.4), pose=cam)
        sc.add(pyrender.DirectionalLight(intensity=1.4), pose=F.look_at(focus + np.array([-0.4, -0.6, 1.2]), focus))
        r = pyrender.OffscreenRenderer(W, H); rgba, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
        al = rgba[..., 3:4] / 255.0; im = Image.fromarray((rgba[..., :3] * al + 255 * (1 - al)).astype(np.uint8))
        name = f"{method.lower()}_{obj}_t{t}_f{f:02d}.png"; im.save(os.path.join(OUT, name)); ims.append(name)
    # per-frame measurements
    def rms_r(Y): return float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    grip = [float(np.mean([np.sqrt(((X[f][pb == b] - ghost_cloud(X[0], X[f], pb)[pb == b]) ** 2).sum(1).mean()) for b in np.unique(pb) if not (roles[pb == b] == 2).all()])) * 1e3 for f in range(61)]
    scale = [rms_r(X[f][roles == 2]) / rms_r(X[0][roles == 2]) for f in range(61)]
    return dict(method=method, object=obj, target=f"t{t}", images=ims, frames=list(FRAMES), grip_rigid_mm=[round(v, 1) for v in grip], obj_scale=[round(v, 3) for v in scale])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); recs = []
    for method, obj, t in CASES + [("Ours", o, t) for _, o, t in CASES]:
        rec = render_case(method, obj, t); recs.append(rec)
        print(f"{method} {obj} t{t}: gripper rigidity error at frame 60 {rec['grip_rigid_mm'][-1]:.1f} mm, object scale {rec['obj_scale'][-1]:.2f}", flush=True)
    json.dump(recs, open(os.path.join(OUT, "cases.json"), "w"), indent=1); print("wrote", OUT, flush=True)
