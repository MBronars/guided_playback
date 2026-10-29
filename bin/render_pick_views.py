"""Multi-view start / peak / end renders of pick replays (Franka arm, textured object), for choosing figure cases.
Reuses the scene recipe of render_pick_misses.py. A case is RUN:MEMBER:ASSET (member -1 = the guided run's planner pick, argmin best_mm).
Views are AZ:EL in degrees; for views whose eye sits on the +y side (looking toward -y) a mirrored kitchen wall is added behind the scene,
and the camera is pulled in until it is inside the room.
  PYOPENGL_PLATFORM=egl .../foundationpose/bin/python bin/render_pick_views.py --cases ... --views -105:8,-60:18 --out figures/pick_can_candidates"""
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import pyrender
import render_pick_misses as R
from gp.render import frames as F

ap = argparse.ArgumentParser(); ap.add_argument("--cases", required=True); ap.add_argument("--views", default="-105:8,-60:18,-150:14,75:8")
ap.add_argument("--out", required=True); ap.add_argument("--size", default="1200x900"); ap.add_argument("--no_stems", action="store_true", help="no dashed height lines to the table"); ap.add_argument("--frames", default="start,peak,end"); ap.add_argument("--cam_from", default=None, help="cases.json whose stored bounds are unioned for the shared camera (default <out>/cases.json)"); ap.add_argument("--pw_cloud", action="store_true", help="draw the run's own PREDICTED particle cloud (hand, fingers, object) instead of the robot and object meshes"); ap.add_argument("--cloud_frames", default="0,15,30,45,60"); ap.add_argument("--one_gripper", action="store_true", help="fingers drawn in the hand colour"); ap.add_argument("--pt_r", type=float, default=0.004); ap.add_argument("--shared_cam", action="store_true", help="frame every case of one object with the union of the bounds stored in <out>/cases.json by an earlier run")
A = ap.parse_args(); W, H = (int(v) for v in A.size.split("x")); os.makedirs(A.out, exist_ok=True); k_ = W / 960
R.VAR = "guided"                                                                  # baseline runs: the committed plan; guided runs: the guided plans
recs = []
for spec in A.cases.split(","):
    run, m, asset = spec.split(":"); m = int(m)
    rd = os.path.join(ROOT, "runs", "pickup", run); z = np.load(os.path.join(rd, "rollout.npz")); S = np.load(os.path.join(rd, "replay_isaac", "replay.npz"))
    rj = json.load(open(os.path.join(rd, "result.json"))); an = json.load(open(os.path.join(rd, "replay_isaac", "analysis.json")))
    if m < 0: m = int(np.argmin(rj["best_mm"]))
    row = next(r for r in an["rows"] if r["member"] == m and r["variant"] == "guided")
    mesh = R.load_visual(asset); obj = z["object_mask"][m]; P0 = S["sim"][json.loads(str(S["variants"])).index("guided"), m][0]
    T_reg, fit = R.register(mesh, P0[obj].astype(np.float64))
    scene, bodies, Q, P0, obj, target, top, pts, obj_node, tnode, armst = R.scene_for(z, S, m, rj, W, H, mesh, T_reg, arm=True)
    c = Q[:, obj].mean(1); peak = int(np.argmax(c[:, 2]))
    ap_ = np.concatenate([pts, armst["pts"]]); lo, hi = ap_.min(0), ap_.max(0); lo0, hi0 = lo.copy(), hi.copy()
    if A.shared_cam:                                                              # one camera for every method on this object
        cf = A.cam_from or os.path.join(A.out, "cases.json")
        prev = [o for o in (json.load(open(cf)) if os.path.exists(cf) else []) if o["object"] == asset.split("/")[0] and "lo" in o]
        for o in prev: lo = np.minimum(lo, o["lo"]); hi = np.maximum(hi, o["hi"])
        print(f"{asset}: shared camera over {len(prev)} stored cases", flush=True)
    focus = (lo + hi) / 2; focus[2] = max(focus[2], top + 0.10)
    span = max(float((hi - lo).max()), 0.15); dist0 = 0.30 + 1.30 * span
    cy_ = armst["Tb"][1, 3]; cx_ = armst["Tb"][0, 3] + 0.42; wall_y = cy_ + 1.35; wall_x = cx_ - 1.5; fy = cy_ - 1.35
    # mirrored wall on the -y side (visible only from views looking toward -y)
    wall_img = Image.open(os.path.join(F.BG, "walls/textures", F.WALL["pickup"] + "_tex.png")).convert("RGB")
    wa = np.asarray(wall_img.convert("L"), dtype=float).mean(1); counter_frac = 1.0 - float(np.argmax(np.abs(np.diff(wa))) + 1) / len(wa)
    upper_img = Image.new("RGB", (8, 8), tuple(np.asarray(wall_img).reshape(-1, 3)[:wall_img.size[0] * 40].mean(0).astype(int)))
    floor_z = top - 0.74; ph = 0.90 / counter_frac; ptop = floor_z + ph; wtop = floor_z + 3.4
    scene.add(F.plane([[cx_ + 4.0, fy, floor_z], [cx_ - 3.0, fy, floor_z], [cx_ - 3.0, fy, ptop], [cx_ + 4.0, fy, ptop]], wall_img, (7.0 / ph, 1)))
    scene.add(F.plane([[cx_ + 4.0, fy, ptop], [cx_ - 3.0, fy, ptop], [cx_ - 3.0, fy, wtop], [cx_ + 4.0, fy, wtop]], upper_img, (1, 1)))
    cam_node = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.5, aspectRatio=W / H), pose=np.eye(4))
    l1 = scene.add(pyrender.DirectionalLight(intensity=3.2), pose=np.eye(4)); scene.add(pyrender.DirectionalLight(intensity=2.0), pose=F.look_at(focus + np.array([-0.5, -0.8, 1.9]), focus))
    r = pyrender.OffscreenRenderer(W, H); Pm = pyrender.PerspectiveCamera(yfov=np.pi / 4.5, aspectRatio=W / H).get_projection_matrix(W, H)
    cat = asset.split("/")[0]; files = []
    for v in A.views.split(","):
        az_d, el_d = (float(x) for x in v.split(":")); az, el = np.radians(az_d), np.radians(el_d); dist = dist0
        for _ in range(60):                                                        # keep the eye inside the room
            _, eye = R.camera(focus, az, el, dist)
            if wall_x + 0.15 < eye[0] and fy + 0.15 < eye[1] < wall_y - 0.15: break
            dist *= 0.95
        cam, eye = R.camera(focus, az, el, dist); scene.set_pose(cam_node, cam); scene.set_pose(l1, cam); Vi = np.linalg.inv(cam)
        def px(p):
            q = Pm @ (Vi @ np.append(p, 1.0)); n = q[:3] / q[3]; return ((n[0] + 1) / 2 * W, (1 - n[1]) / 2 * H)
        if A.pw_cloud:                                                             # hide robot + object meshes; the prediction is drawn as spheres
            import render_pick_pred_errors as E
            for b in bodies:
                for nd, _ in b.nodes: nd.mesh.is_visible = False
            for _, nd, _ in armst["nodes"]: nd.mesh.is_visible = False
            Xp = z["predicted"][m]; rl = z["roles"][m]
            todo = [(int(f), f"f{int(f):02d}") for f in A.cloud_frames.split(",")]
        else:
            todo = [ft for ft in ((0, "start"), (peak, "peak"), (len(c) - 1, "end")) if ft[1] in A.frames.split(",")]
        for f, tag in todo:
            cloud = []
            if A.pw_cloud:
                for role in (0, 1, 2): cloud.append(scene.add(E.spheres(Xp[f][rl == role], E.COL[0 if (A.one_gripper and role == 1) else role], A.pt_r)))
            else:
                F.place(scene, bodies, P0, Q[f]); R.pose_arm(scene, armst, f)
            rgba, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            al = rgba[..., 3:4] / 255.0; im = Image.fromarray((rgba[..., :3] * al + 255 * (1 - al)).astype(np.uint8)).convert("RGBA")
            ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
            def stem(p3, col):                                                     # dashed drop line to the table: shows height
                a = px(p3); b = px(np.array([p3[0], p3[1], top])); n = 14
                for kk in range(0, n, 2):
                    t0, t1 = kk / n, (kk + 1) / n
                    d.line([(a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0), (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1)], fill=col + (255,), width=round(3 * k_))
                d.ellipse([b[0] - 9 * k_, b[1] - 4 * k_, b[0] + 9 * k_, b[1] + 4 * k_], outline=col + (255,), width=round(3 * k_))
            if not A.no_stems: stem(target, (20, 150, 105))
            for nd in cloud: scene.remove_node(nd)
            if tag not in ("start",) and not A.pw_cloud:
                if not A.no_stems: stem(c[f], (214, 40, 130))
                path = [px(p) for p in c[: f + 1]]; d.line(path, fill=(214, 40, 130, 255), width=round(4 * k_), joint="curve")
            tx, ty = px(target); d.ellipse([tx - 15 * k_, ty - 15 * k_, tx + 15 * k_, ty + 15 * k_], fill=(20, 170, 120, 215), outline=(255, 255, 255, 255), width=round(3 * k_))
            fn = f"{cat}__{run}{'__pwcloud' if A.pw_cloud else ''}__m{m}__az{az_d:+.0f}_el{el_d:.0f}_{tag}.png"; Image.alpha_composite(im, ov).convert("RGB").save(os.path.join(A.out, fn)); files.append(fn)
    r.delete()
    rec = dict(run=run, member=m, object=cat, lift_peak_mm=round(row["sim_lift_mm"]), sim_to_target_mm=round(row["sim_to_target_mm"]),
               miss_vertical_mm=round(float((c[-1] - target)[2]) * 1e3), miss_horizontal_mm=round(float(np.linalg.norm((c[-1] - target)[:2])) * 1e3),
               arm_track_mm=round(armst["err_mm"], 1), lo=[float(v) for v in lo0], hi=[float(v) for v in hi0], mesh_fit_mm=round(fit * 1e3, 2), views=A.views.split(","), files=files)
    recs.append(rec); print(json.dumps({k: v for k, v in rec.items() if k != "files"}), flush=True)
fn = os.path.join(A.out, "cases.json"); old = json.load(open(fn)) if os.path.exists(fn) else []
keep = [o for o in old if (o["run"], o["member"]) not in {(r_["run"], r_["member"]) for r_ in recs}]; json.dump(keep + recs, open(fn, "w"), indent=1); print("wrote", fn)
