"""Start / end renders of UNGUIDED pick draws that lift in Isaac but miss the target, plus a gallery of the five benchmark objects.

Objects are the real textured visual meshes (Kitchen_Table/<cat>/asset_XXXX/asset_XXXX_visual.ply), registered once to the frame-0
object particles by ICP (24 axis-aligned starts x 8 yaws), then moved by the per-frame Kabsch pose of the simulated particles.
The hand and fingers are the Franka URDF visuals, as in gp/render/frames.py. Camera: for every case the azimuth / elevation is chosen
by rendering object-only and full-scene depth at low resolution over every lifted frame and maximising how often >= 60 % of the
object's pixels are unoccluded (ties: mean visible fraction). Target = green sphere; end image also draws the simulated object-centroid path.

  PYOPENGL_PLATFORM=egl .../foundationpose/bin/python bin/render_pick_misses.py [--out figures/pick_unguided_misses]"""
import argparse, json, os, sys, itertools
import numpy as np, trimesh, imageio
from PIL import Image, ImageDraw, ImageFont
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
from gp.render import frames as F
import pyrender

ASSETS = "/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table"
CASES = [  # (run, unguided member, asset path) -- picked from replay_isaac/analysis.json: lifted, still held at frame 60, misses the target
    ("cyl_t0_milk_carton_it10", 1, "milk_carton/asset_0037"),
    ("cyl_t0_peanut_butter_jar_it10", 1, "peanut_butter_jar/asset_0003"),
    ("cyl_t4_soda_can_it10", 0, "soda_can/asset_0012"),
    ("cyl_t2_pasta_box_it10", 2, "pasta_box/asset_0058"),
    ("cyl_t0_water_bottle_it10", 3, "water_bottle/asset_0087"),
]
BASELINE_CASES = [  # no-guidance baseline (MPPI argmin, K 20, commit 5, full 19-row windows): the plan lifts in Isaac, is still held at frame 60, misses the target
    ("mppiargmin_full19_t0_milk_carton_K20", 0, "milk_carton/asset_0037"),
    ("mppiargmin_full19_t1_peanut_butter_jar_K20", 0, "peanut_butter_jar/asset_0003"),
    ("mppiargmin_full19_t4_soda_can_K20", 0, "soda_can/asset_0012"),
    ("mppiargmin_full19_t0_pasta_box_K20", 0, "pasta_box/asset_0058"),
    ("mppiargmin_full19_t0_water_bottle_K20", 0, "water_bottle/asset_0087"),
]
GUIDED_CASES = [  # ours (guided) on the same five problems, the member the planner picked (lowest generated objective)
    ("cyl_t0_milk_carton_it10", 0, "milk_carton/asset_0037"),
    ("cyl_t1_peanut_butter_jar_it10", 1, "peanut_butter_jar/asset_0003"),
    ("cyl_t4_soda_can_it10", 1, "soda_can/asset_0012"),
    ("cyl_t0_pasta_box_it10", 3, "pasta_box/asset_0058"),
    ("cyl_t0_water_bottle_it10", 0, "water_bottle/asset_0087"),
]
CAM_FROM = None                                                                   # cases.json whose camera azimuth / elevation per object is reused
VAR = "unguided"                                                                 # replay variant rendered (cyl runs: the first unguided draw)
GUIDED_RUN = {"mppiargmin_full19_": "cyl_", "_K20": "_it10"}                    # the guided run on the same problem, for the comparison number
LABEL = {"milk_carton": "milk carton", "peanut_butter_jar": "peanut butter jar", "soda_can": "soda can", "pasta_box": "pasta box", "water_bottle": "water bottle"}


def load_visual(asset):
    n = asset.split("/")[1]
    m = trimesh.load(os.path.join(ASSETS, asset, f"{n}_visual.ply"), process=False)
    pass                                                                      # full-resolution mesh (no decimation module in this env)
    return m


def register(mesh, pts):
    """4x4 T with mesh.vertices @ T -> the frame-0 object particles (rigid, no scale)."""
    src = mesh.sample(4000); c = src.mean(0); tc = pts.mean(0)
    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            R = np.zeros((3, 3)); R[range(3), perm] = signs
            if np.linalg.det(R) > 0: rots.append(R)
    best = (np.inf, None)
    for R in rots:
        for yaw in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            Rz = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])[:3, :3]
            T0 = np.eye(4); T0[:3, :3] = Rz @ R; T0[:3, 3] = tc - (Rz @ R) @ c
            T, _, cost = trimesh.registration.icp(src, pts, initial=T0, threshold=1e-6, max_iterations=40, reflection=False, scale=False)
            d = np.sqrt(((trimesh.transform_points(src, T)[:, None] - pts[None]) ** 2).sum(-1)).min(0).mean()   # target -> mesh
            if d < best[0]: best = (d, T)
    return best[1], best[0]


def scene_for(z, S, m, run_rj, W, H, obj_mesh, T_reg, extra_pts=None, arm=False):
    roles = z["roles"][m]; obj = z["object_mask"][m]; ui = json.loads(str(S["variants"])).index(VAR)
    Q = S["sim"][ui, m]; P0 = Q[0]; target = z["target"][m]; top = float(P0[obj][:, 2].min())
    wall_img = Image.open(os.path.join(F.BG, "walls/textures", F.WALL["pickup"] + "_tex.png")).convert("RGB")
    wa = np.asarray(wall_img.convert("L"), dtype=float).mean(1); counter_frac = 1.0 - float(np.argmax(np.abs(np.diff(wa))) + 1) / len(wa)
    upper_img = Image.new("RGB", (8, 8), tuple(np.asarray(wall_img).reshape(-1, 3)[:wall_img.size[0] * 40].mean(0).astype(int)))
    pts = np.concatenate([Q[:, roles >= 0].reshape(-1, 3), target[None]] + ([extra_pts] if extra_pts is not None else []))
    Tb = F.base_pose("pickup", None, top, None) if arm else None                  # the data's Franka base (model frame -0.45, 0, -0.10)
    table_centre = (Tb[:3, 3] + np.array([0.42, 0.0, 0.0])) if arm else None       # the kitchen recipe: slab centred 0.42 m in front of the base
    scene = F.build_scene("pickup", top, pts, W, H, wall_img, upper_img, counter_frac, F.material_texture("Wood010_1K-JPG"), F.material_texture("Wood048_1K-JPG"), table_centre=table_centre, big=arm)
    bodies = F.add_bodies(scene, "pickup", P0, roles, obj, z, m)
    hull_node = bodies[-1].nodes[0][0]; scene.remove_node(hull_node)                   # replace the convex hull by the textured mesh
    obj_node = scene.add(pyrender.Mesh.from_trimesh(obj_mesh, smooth=False))
    bodies[-1] = F.Body(obj, [(obj_node, np.eye(4))], T_reg)
    tg = trimesh.creation.icosphere(subdivisions=3, radius=0.012); tp = np.eye(4); tp[:3, 3] = target
    tnode = scene.add(F.solid(tg, [30, 170, 120, 255]), pose=tp)
    armstate = None
    if arm:                                                                        # cosmetic arm: IK to the simulated hand pose each frame, warm-started
        A_ = F.Arm(); visuals = F._urdf_visuals(); nodes = []
        for li, name in enumerate(F.ARM_LINKS):
            for objp, o in visuals.get(name, []):
                for mm in F.link_meshes(objp): nodes.append((li, scene.add(mm), o))
        qs, errs, elbows = [], [], []; q = None
        for f in range(Q.shape[0]):
            T_h = bodies[0].place(scene, P0, Q[f]); T_local = np.linalg.inv(Tb) @ T_h
            if q is None: q, ep, er = A_.solve_first(T_local)
            else:
                q, ep, er = A_.ik(q, T_local)
                if ep > 0.005 or er > 0.09:
                    q2, ep2, er2 = A_.solve_first(T_local)
                    if ep2 + 0.1 * er2 < ep + 0.1 * er: q, ep, er = q2, ep2, er2
            qs.append(q.copy()); errs.append((ep, er)); links, _ = A_.fk(q); elbows.append(np.array([(Tb @ L)[:3, 3] for L in links]))
        armstate = dict(arm=A_, nodes=nodes, Tb=Tb, q=qs, err_mm=max(e[0] for e in errs) * 1e3, err_deg=float(np.degrees(max(e[1] for e in errs))), pts=np.concatenate(elbows))
    return scene, bodies, Q, P0, obj, target, top, pts, obj_node, tnode, armstate


def pose_arm(scene, armstate, f):
    if armstate is None: return
    links, _ = armstate["arm"].fk(armstate["q"][f])
    for li, node, o in armstate["nodes"]: scene.set_pose(node, armstate["Tb"] @ links[li] @ o)


def camera(focus, az, el, dist):
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return F.look_at(focus + dist * d, focus), focus + dist * d


def visible_fraction(r, scene, obj_node):
    """fraction of the object's pixels not hidden by anything else."""
    full = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    others = [n for n in scene.mesh_nodes if n is not obj_node]
    for n in others: n.mesh.is_visible = False
    only = r.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    for n in others: n.mesh.is_visible = True
    px = only > 0
    if px.sum() < 15: return 0.0
    return float((np.abs(full[px] - only[px]) < 0.004).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=os.path.join(ROOT, "figures", "pick_unguided_misses")); ap.add_argument("--size", default="960x720")
    ap.add_argument("--only", default=None, help="comma-separated objects to render (e.g. water_bottle,pasta_box)")
    ap.add_argument("--no_gallery", action="store_true")
    ap.add_argument("--opposite", action="store_true", help="view from the opposite side (camera azimuth + 180 deg), camera kept inside the back wall, a mirrored kitchen wall added on the -y side; output folders get an _opposite suffix")
    ap.add_argument("--cameras", default=None, help="cases.json whose per-object camera azimuth / elevation is reused without a search")
    ap.add_argument("--arm", action="store_true", help="draw the Franka arm (cosmetic: IK to the simulated hand, base at the data's robot base); output folders get an _arm suffix")
    ap.add_argument("--guided", action="store_true", help="render our guided plans on the baseline's problems, reusing the baseline's camera angles")
    ap.add_argument("--baseline", action="store_true", help="render the MPPI-argmin baseline plans (runs/pickup/mppiargmin_full19_*_K20, replay variant 'guided' = the committed plan)")
    A = ap.parse_args()
    global CASES, VAR, CAM_FROM
    if A.guided:
        CASES = GUIDED_CASES; VAR = "guided"; CAM_FROM = os.path.join(ROOT, "figures", "pick_baseline_mppi_misses", "cases.json")
        if A.out.endswith("pick_unguided_misses"): A.out = os.path.join(ROOT, "figures", "pick_guided_pairs")
    if A.baseline:
        CASES = BASELINE_CASES; VAR = "guided"
        if A.out.endswith("pick_unguided_misses"): A.out = os.path.join(ROOT, "figures", "pick_baseline_mppi_misses")
    if A.arm:
        A.out = A.out.rstrip("/") + "_arm"
        if CAM_FROM: CAM_FROM = CAM_FROM.replace("pick_baseline_mppi_misses/", "pick_baseline_mppi_misses_arm/")
    if A.opposite: A.out = A.out.rstrip("/") + "_opposite"
    if A.cameras: CAM_FROM = A.cameras
    if A.only: CASES = [c for c in CASES if c[2].split("/")[0] in A.only.split(",")]
    os.makedirs(A.out, exist_ok=True); W, H = (int(v) for v in A.size.split("x"))
    font = ImageFont.load_default(size=26) if hasattr(ImageFont, "load_default") else None
    prev = os.path.join(A.out, "cases.json"); summary = json.load(open(prev)) if os.path.exists(prev) else []; gallery = []
    for run, m, asset in CASES:
        name0 = f"{asset.split('/')[0]}__{run}__m{m}"
        done = os.path.exists(os.path.join(A.out, f"{name0}_start_end.png")) and any(r["run"] == run for r in summary)
        rd = os.path.join(ROOT, "runs", "pickup", run); z = np.load(os.path.join(rd, "rollout.npz")); S = np.load(os.path.join(rd, "replay_isaac", "replay.npz"))
        rj = json.load(open(os.path.join(rd, "result.json"))); an = json.load(open(os.path.join(rd, "replay_isaac", "analysis.json")))
        row = next(r for r in an["rows"] if r["member"] == m and r["variant"] == VAR)
        grd = rd
        if VAR == "guided":                                                            # baseline run: the comparison is the guided run on the same problem
            gname = run
            for a_, b_ in GUIDED_RUN.items(): gname = gname.replace(a_, b_)
            grd = os.path.join(ROOT, "runs", "pickup", gname)
        grj = json.load(open(os.path.join(grd, "result.json"))); gan = json.load(open(os.path.join(grd, "replay_isaac", "analysis.json")))
        gpick = int(np.argmin(grj["best_mm"])); grow = next(r for r in gan["rows"] if r["member"] == gpick and r["variant"] == "guided")
        cat = asset.split("/")[0]; mesh = load_visual(asset)
        obj = z["object_mask"][m]; P0 = S["sim"][json.loads(str(S["variants"])).index(VAR), m][0]
        T_reg, fit = register(mesh, P0[obj].astype(np.float64)); print(f"{cat}: mesh registered to frame-0 particles, mean target->mesh {fit*1e3:.1f} mm", flush=True)
        if done:
            gallery.append((cat, mesh, T_reg, P0[obj])); print(f"{cat}: renders exist, skipped", flush=True); continue
        scene, bodies, Q, P0, obj, target, top, pts, obj_node, tnode, armst = scene_for(z, S, m, rj, W, H, mesh, T_reg, arm=A.arm)
        if armst: print(f"{cat}: arm IK tracks the simulated hand to {armst['err_mm']:.1f} mm / {armst['err_deg']:.1f} deg (max over frames)", flush=True)
        c = Q[:, obj].mean(1); lifted = np.nonzero(c[:, 2] - c[0, 2] > 0.03)[0]
        lo, hi = pts.min(0), pts.max(0); focus = (lo + hi) / 2; focus[2] = max(focus[2], top + 0.04); span = max(float((hi - lo).max()), 0.15)
        dist = 0.32 + 1.55 * span
        if armst:                                                                  # frame the object, the hand and the whole arm
            ap_ = np.concatenate([pts, armst["pts"]]); lo, hi = ap_.min(0), ap_.max(0); focus = (lo + hi) / 2; focus[2] = max(focus[2], top + 0.10)
            span = max(float((hi - lo).max()), 0.15); dist = 0.30 + 1.30 * span
        # camera search at low resolution over lifted frames (and frames 0 / T-1)
        ref_cam = None
        if CAM_FROM and os.path.exists(CAM_FROM):
            ref_cam = next((r for r in json.load(open(CAM_FROM)) if r["object"] == cat), None)
        r_small = pyrender.OffscreenRenderer(240, 180); best = None
        cam_node = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.5, aspectRatio=W / H), pose=np.eye(4))
        chk = np.unique(np.concatenate([[0, len(c) - 1], lifted[:: max(1, len(lifted) // 12)]]))
        cy_ = armst["Tb"][1, 3] if armst else pts[:, 1].mean(); cx_ = (armst["Tb"][0, 3] + 0.42) if armst else pts[:, 0].mean()
        wall_y = cy_ + 1.35; wall_x = cx_ - 1.5
        for az in ([] if ref_cam else np.radians(np.arange(-105, 1, 7.5))):                         # eye on the +x / -y side: the camera faces the walled corner
            for el in np.radians([18, 28, 38] + ([48] if A.arm else [])):
                cam, eye = camera(focus, az, el, dist)
                if eye[1] > wall_y - 0.1 or eye[0] < wall_x + 0.1: continue
                scene.set_pose(cam_node, cam); fr = []
                for f in chk:
                    F.place(scene, bodies, P0, Q[f]); pose_arm(scene, armst, f); fr.append(visible_fraction(r_small, scene, obj_node))
                fr = np.array(fr)
                F.place(scene, bodies, P0, Q[chk[1]]); pose_arm(scene, armst, chk[1]); tvis = visible_fraction(r_small, scene, tnode)      # target sphere unoccluded in the END frame
                score = (float((fr[2:] >= 0.6).mean()) if len(fr) > 2 else 0.0, round(min(tvis, 0.8), 1), float(min(fr[0], fr[1])), float(fr.mean()))
                if best is None or score > best[0]: best = (score, az, el, fr)
        r_small.delete()
        if best is None: best = ((ref_cam["lifted_frames_object_60pct_visible"], None, None, None), 0.0, 0.0, None)
        (frac_ok, tvis_end, endpoints, mean_vis), az, el, fr = best
        if CAM_FROM and os.path.exists(CAM_FROM):
            ref = next((r for r in json.load(open(CAM_FROM)) if r["object"] == cat), None)
            if ref: az, el = np.radians(ref["camera_azimuth_deg"]), np.radians(ref["camera_elevation_deg"])
        if A.opposite:
            az = az + np.pi
            for _ in range(40):                                                    # keep the eye inside the room (back wall at +y, side wall at -x)
                _, eye_ = camera(focus, az, el, dist)
                if eye_[1] < wall_y - 0.15 and eye_[0] > wall_x + 0.15: break
                dist *= 0.95
            wall_img = Image.open(os.path.join(F.BG, "walls/textures", F.WALL["pickup"] + "_tex.png")).convert("RGB")
            wa = np.asarray(wall_img.convert("L"), dtype=float).mean(1); counter_frac = 1.0 - float(np.argmax(np.abs(np.diff(wa))) + 1) / len(wa)
            upper_img = Image.new("RGB", (8, 8), tuple(np.asarray(wall_img).reshape(-1, 3)[:wall_img.size[0] * 40].mean(0).astype(int)))
            floor_z = top - 0.74; ph = 0.90 / counter_frac; ptop = floor_z + ph; wtop = floor_z + 3.4; fy = cy_ - 1.35
            scene.add(F.plane([[cx_ + 4.0, fy, floor_z], [cx_ - 3.0, fy, floor_z], [cx_ - 3.0, fy, ptop], [cx_ + 4.0, fy, ptop]], wall_img, (7.0 / ph, 1)))
            scene.add(F.plane([[cx_ + 4.0, fy, ptop], [cx_ - 3.0, fy, ptop], [cx_ - 3.0, fy, wtop], [cx_ + 4.0, fy, wtop]], upper_img, (1, 1)))
        cam, eye = camera(focus, az, el, dist); scene.set_pose(cam_node, cam)
        scene.add(pyrender.DirectionalLight(intensity=3.2), pose=cam); scene.add(pyrender.DirectionalLight(intensity=2.0), pose=F.look_at(focus + np.array([-0.5, -0.8, 1.9]), focus))
        r = pyrender.OffscreenRenderer(W, H); imgs = []
        Pm = pyrender.PerspectiveCamera(yfov=np.pi / 4.5, aspectRatio=W / H).get_projection_matrix(W, H); Vi = np.linalg.inv(cam)
        def px(p):
            q = Pm @ (Vi @ np.append(p, 1.0)); n = q[:3] / q[3]; return ((n[0] + 1) / 2 * W, (1 - n[1]) / 2 * H)
        name = f"{cat}__{run}__m{m}"
        for f, tag in ((0, "start"), (len(c) - 1, "end")):
            F.place(scene, bodies, P0, Q[f]); pose_arm(scene, armst, f)
            rgba, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            al = rgba[..., 3:4] / 255.0; im = Image.fromarray((rgba[..., :3] * al + 255 * (1 - al)).astype(np.uint8)); d = ImageDraw.Draw(im)
            # height cues: a dashed vertical drop line from the target (all frames) and from the object's final centroid (end frame) down to the table top
            def stem(p3, col):
                a = px(p3); b = px(np.array([p3[0], p3[1], top])); n = 14
                for k in range(n):
                    if k % 2: continue
                    t0, t1 = k / n, (k + 1) / n
                    d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0), (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)], fill=col, width=3)
                d.ellipse([b[0] - 9, b[1] - 4, b[0] + 9, b[1] + 4], outline=col, width=3)
            stem(target, (20, 150, 105))
            if tag == "end":
                stem(c[-1], (214, 40, 130))
                path = [px(p) for p in c]; d.line(path, fill=(214, 40, 130), width=4, joint="curve")
                for (x, y) in (path[0], path[-1]): d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(214, 40, 130), outline=(255, 255, 255), width=2)
            tx, ty = px(target)
            if A.arm:                                                              # the arm often stands in front of the target: draw it on top, filled
                ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); do = ImageDraw.Draw(ov)
                do.ellipse([tx - 15, ty - 15, tx + 15, ty + 15], fill=(20, 170, 120, 215), outline=(255, 255, 255, 255), width=3)
                im = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB"); d = ImageDraw.Draw(im)
            else:
                d.ellipse([tx - 16, ty - 16, tx + 16, ty + 16], outline=(20, 160, 110), width=4)
            im.save(os.path.join(A.out, f"{name}_{tag}.png")); imgs.append(np.asarray(im))
        r.delete()
        pair = np.concatenate([imgs[0], np.full((H, 10, 3), 255, np.uint8), imgs[1]], 1); Image.fromarray(pair).save(os.path.join(A.out, f"{name}_start_end.png"))
        rec = dict(object=cat, asset=asset, run=run, variant=VAR, unguided_member=m, target_id=run.split("_")[1], lift_peak_mm=round(row["sim_lift_mm"]),
                   end_height_above_start_mm=round((c[-1, 2] - c[0, 2]) * 1e3), sim_to_target_mm=round(row["sim_to_target_mm"]),
                   miss_horizontal_mm=round(float(np.linalg.norm((c[-1] - target)[:2])) * 1e3), miss_vertical_mm=round(float((c[-1] - target)[2]) * 1e3),
                   guided_member=gpick, guided_sim_to_target_mm=round(grow["sim_to_target_mm"]), camera_azimuth_deg=round(float(np.degrees(az))), camera_elevation_deg=round(float(np.degrees(el))),
                   lifted_frames_object_60pct_visible=round(frac_ok, 2), mesh_fit_mm=round(fit * 1e3, 2), arm_track_mm=(round(armst["err_mm"], 1) if armst else None), arm_track_deg=(round(armst["err_deg"], 1) if armst else None), images=[f"{name}_start.png", f"{name}_end.png", f"{name}_start_end.png"])
        summary = [r for r in summary if r["run"] != run] + [rec]; json.dump(summary, open(prev, "w"), indent=1); print(json.dumps(rec), flush=True)
        gallery.append((cat, mesh, T_reg, P0[obj]))
    if A.no_gallery:
        print("wrote", A.out, flush=True); return
    # object gallery: each object alone, same camera recipe, true relative scale (5 panels + one row with all five)
    panels = []
    for cat, mesh, T_reg, p0 in gallery:
        sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.45, 0.45, 0.45])
        mm = mesh.copy(); mm.apply_transform(T_reg); mm.apply_translation(-np.array([*mm.centroid[:2], mm.bounds[0, 2]]))
        sc.add(pyrender.Mesh.from_trimesh(mm, smooth=False))
        disk = trimesh.creation.cylinder(radius=0.11, height=0.004); dp = np.eye(4); dp[2, 3] = -0.002; sc.add(F.solid(disk, [225, 222, 215, 255]), pose=dp)
        foc = np.array([0, 0, 0.075]); cam, _ = camera(foc, np.radians(-60), np.radians(22), 0.42)
        sc.add(pyrender.PerspectiveCamera(yfov=np.pi / 5), pose=cam); sc.add(pyrender.DirectionalLight(intensity=3.0), pose=cam); sc.add(pyrender.DirectionalLight(intensity=1.5), pose=F.look_at(np.array([-0.4, -0.5, 1.0]), foc))
        r = pyrender.OffscreenRenderer(480, 560); rgba, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
        al = rgba[..., 3:4] / 255.0; im = Image.fromarray((rgba[..., :3] * al + 255 * (1 - al)).astype(np.uint8)); d = ImageDraw.Draw(im); ext = np.round(mm.extents * 1e3).astype(int)
        d.text((16, 14), LABEL[cat], fill=(30, 30, 30), font=font); d.text((16, 48), f"{ext[0]} x {ext[1]} x {ext[2]} mm", fill=(90, 90, 90), font=ImageFont.load_default(size=20) if font else None)
        im.save(os.path.join(A.out, f"object_{cat}.png")); panels.append(np.asarray(im))
    Image.fromarray(np.concatenate(panels, 1)).save(os.path.join(A.out, "objects_all.png"))
    json.dump(summary, open(os.path.join(A.out, "cases.json"), "w"), indent=1); print("wrote", A.out, flush=True)


if __name__ == "__main__":
    main()
