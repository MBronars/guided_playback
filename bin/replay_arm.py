"""Attached-arm Isaac replay (+ real-Franka render) of a package run: the floating-hand specs the run already has
(runs/<domain>/<name>/replay_isaac/specs.npz, MODEL frame) -> BASE-frame arm specs (+0.45, 0, +0.10 m: the Franka base
sits at the raw-data origin, which the pickup canonical shift moved to (-0.45, 0, -0.10) in the model frame) with the
pool index of the recorded episode (for add_arm_q0.py: the recorded frame-0 arm joints seed the arm) -> replay_isaac_arm.py
(isaaclab env; --obstacle = the run's wall box, if any, as a real kinematic cuboid) -> render_arm_replay.py (foundationpose
env, pyrender/EGL) with the wall drawn.

  python bin/replay_arm.py --run runs/pickup/<name> --members 3 [--gpu 1] [--eye x,y,z --look x,y,z] [--no_obstacle]
Outputs: runs/pickup/<name>/replay_isaac_arm/{specs.npz,specs.json,arm.npz,isaac/replay.{npz,json},render/*.gif|mp4}
"""
import argparse, json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = "/data/user_data/mbronars/packages/pickup_replay_isaac/scripts"
POOL_DIR = "/data/user_data/mbronars/packages/pickup_replay_isaac/out/pool_pick_val"
PY_ISAAC = "/data/user_data/mbronars/.miniconda3/envs/isaaclab/bin/python"
PY_PWM = "/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python"
PY_RENDER = "/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python"
SHIFT = np.array([0.45, 0.0, 0.10])          # model frame -> base (raw) frame

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--members", default="0"); ap.add_argument("--variants", default="guided")
ap.add_argument("--gpu", default="1"); ap.add_argument("--rate", type=int, default=20)
ap.add_argument("--eye", default="0.95,-1.15,0.80"); ap.add_argument("--look", default="0.42,0.20,0.22"); ap.add_argument("--size", default="960x720")
ap.add_argument("--asset", default="auto", help="visual mesh for the object; auto = sim_scene_gen <category>/asset_NNNN/asset_NNNN_visual.ply from the episode label; none = convex hull")
ap.add_argument("--render_dir", default=None, help="render subfolder (default: render, or render_asset when an object mesh is used); never overwrites another style")
ap.add_argument("--no_target", action="store_true", help="no green X at the target");
ap.add_argument("--obstacle_asset", default="auto", help="sim_scene_gen asset dir used as the obstacle (auto = result.json obstacle_asset; none = plain box); its convex hull is the collider, its _visual.ply the render")
ap.add_argument("--obstacle_jitter", type=float, default=None, help="0 = perfectly aligned units; 1 (default) = +-6 deg yaw and +-8 mm per unit for rows/stacks");
ap.add_argument("--obstacle_yaw", type=float, default=None, help="yaw of the obstacle asset about z (deg); default: turn its thinner horizontal side along x")
ap.add_argument("--auto_camera", action="store_true", help="camera framed on the start-target midpoint (same offset as the carton case)");
ap.add_argument("--obstacle_size", default=None, help="override the wall box size sx,sy,sz (m) in the SIM and the render, e.g. 0.02,0.40,0.12");
ap.add_argument("--no_obstacle", action="store_true"); ap.add_argument("--skip_sim", action="store_true"); ap.add_argument("--skip_render", action="store_true")
ap.add_argument("--target_color", default=None, help="passed to the renderer: target X/ball colour r,g,b")
ap.add_argument("--table_color", default=None, help="passed to the renderer: table colour r,g,b 0-255")
ap.add_argument("--object_tint", default=None, help="passed to the renderer: colour for the near-white parts of the object mesh")
ap.add_argument("--render_envs", default=None, help="comma-separated env indices of the existing sim to render (default: all envs)")
A = ap.parse_args()
run = A.run.rstrip("/"); out = os.path.join(run, "replay_isaac_arm"); os.makedirs(out, exist_ok=True)
src = os.path.join(run, "replay_isaac", "specs.npz"); S = np.load(src); meta = json.load(open(src[:-4] + ".json"))
z = np.load(os.path.join(run, "rollout.npz")); pool = np.load(os.path.join(POOL_DIR, "specs.npz"))
members = [int(v) for v in A.members.split(",")]
# base-frame copies of the spec arrays (positions shifted, orientations / finger q unchanged)
arrays = {k: S[k] for k in S.files}
for k in ("hand_pos", "obj_centroid", "target"):
    arrays[k] = S[k] + SHIFT
arrays["obj_pts0"] = S["obj_pts0"] + SHIFT
checks = []
rj0 = json.load(open(os.path.join(run, "result.json")))
sel_rows = None
if rj0.get("object_mesh") and rj0.get("select"):                       # synthetic object: the arm start comes from the scene's ORIGINAL episode (select -> pool_index)
    sel = json.load(open(rj0["select"])); eps_str = str(rj0.get("scene_episodes") or rj0["episodes"])
    sel_rows = [int(sel[int(i)]["pool_index"]) for i in eps_str.strip("[]").replace("'", "").split(",")]
for c in meta["checks"]:
    m = c["member"]; o = arrays["obj_pts0"][m]
    d = np.abs(pool["obj_pts0"] - o[None]).max(axis=(1, 2)); pi = int(np.argmin(d))
    if d[pi] > 0.002:
        if sel_rows is None:
            raise SystemExit(f"member {m}: no pool row matches the frame-0 object in the base frame (best {d[pi]*1e3:.1f} mm)")
        pi = sel_rows[m]; print(f"[arm] member {m}: synthetic object ({os.path.basename(os.path.dirname(rj0['object_mesh']))}); arm start from pool row {pi} of the original scene", flush=True)
    checks.append(dict(c, pool_index=pi, pool_match_base_mm=round(float(d[pi]) * 1e3, 2)))
np.savez(os.path.join(out, "specs.npz"), **arrays)
pool_eps = json.load(open(os.path.join(POOL_DIR, "specs.json")))["episodes"]          # raw (tar, key, element) per pool row
json.dump(dict(meta, episodes=[list(pool_eps[c["pool_index"]]) for c in checks], run_members=meta["episodes"], checks=checks, shift=(-SHIFT).tolist(),
               frame="base (raw data) frame: Franka base at the origin", source=os.path.abspath(src)),
          open(os.path.join(out, "specs.json"), "w"), indent=1)
subprocess.run([PY_PWM, os.path.join(SCRIPTS, "add_arm_q0.py"), os.path.abspath(out), "--pool", POOL_DIR], check=True)
obstacle = None; obstacle_mesh = None; obstacle_pose = None
rj = json.load(open(os.path.join(run, "result.json")))
oa = rj.get("obstacle_asset") if A.obstacle_asset == "auto" else (None if A.obstacle_asset == "none" else A.obstacle_asset)
if oa and len(z["boxes"]) and not A.no_obstacle:
    b = z["boxes"][0]; base = os.path.basename(oa.rstrip("/")); vis = os.path.join(oa, base + "_visual.ply"); col = os.path.join(oa, base + "_simplified.obj")
    if not os.path.exists(vis) or os.path.getsize(vis) == 0: vis = os.path.join(oa, base + ".ply")          # some assets ship an empty _visual.ply
    V = np.array([[float(t) for t in l.split()[1:4]] for l in open(col) if l.startswith("v ")])
    rpy = [float(v) for v in str(rj.get("obstacle_rpy", "0,0,0")).split(",")]; layout = str(rj.get("obstacle_layout", "single"))
    def rotm(r, p_, y):
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p_), np.sin(p_), np.cos(y), np.sin(y)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]); Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]]); Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]); return Rz @ Ry @ Rx
    Vr = V @ rotm(*np.radians(rpy)).T; lo, hi = Vr.min(0), Vr.max(0); ext = hi - lo                     # one unit, after roll/pitch/yaw
    lowest = float(z["gt"][members[0], 0][z["roles"][members[0]] == 2][:, 2].min()); table_z = lowest + SHIFT[2]
    cx, cy = b[0] + SHIFT[0], b[1] + SHIFT[1]; poses = []
    parts = layout.split(":"); n = int(parts[1]) if len(parts) > 1 else 1; n2 = int(parts[2]) if len(parts) > 2 else 1
    x0, y0, z0 = cx - (lo[0] + hi[0]) / 2, cy - (lo[1] + hi[1]) / 2, table_z - lo[2]
    if layout.startswith("stack"): poses = [(x0, y0, z0 + i * ext[2]) for i in range(n)]
    elif layout.startswith("rowstack"): poses = [(x0, y0 + (r - (n - 1) / 2) * ext[1], z0 + k * ext[2]) for r in range(n) for k in range(n2)]
    elif layout.startswith("row"): poses = [(x0, y0 + (i - (n - 1) / 2) * ext[1], z0) for i in range(n)]
    else: poses = [(x0, y0, z0)]
    jit = float(rj.get("obstacle_jitter", 1.0)) if A.obstacle_jitter is None else A.obstacle_jitter          # row/stack realism: per-unit yaw +-6 deg, xy +-8 mm (deterministic)
    rng = np.random.RandomState(7); pj = []
    for i, (x, y, zz) in enumerate(poses):
        dy = rpy[2] + (jit * rng.uniform(-6, 6) if len(poses) > 1 else 0.0); dx_, dy_ = (jit * rng.uniform(-0.008, 0.008, 2) if len(poses) > 1 else (0.0, 0.0))
        pj.append(f"{x + dx_:.4f},{y + dy_:.4f},{zz:.4f},{dy:.2f},{rpy[0]:.2f},{rpy[1]:.2f}")
    obstacle_pose = ";".join(pj); obstacle_mesh = (col, vis)
    print(f"[arm] obstacle asset {base}: layout {layout}, rpy {rpy}, unit extents {ext.round(3).tolist()} m -> {len(poses)} pose(s); planning box centre {b[:3].round(3).tolist()} size {b[3:].round(3).tolist()} (model frame)", flush=True)
elif len(z["boxes"]) and not A.no_obstacle:
    b = z["boxes"][0]; size = np.array([float(v) for v in A.obstacle_size.split(",")]) if A.obstacle_size else b[3:]
    obstacle = ",".join(f"{v:.4f}" for v in np.concatenate([b[:3] + SHIFT, size]))
    print("[arm] obstacle (base frame, centre + size):", obstacle, flush=True)
sim_dir = os.path.join(out, ("isaac" if A.variants == "guided" else "isaac_" + A.variants.replace(",", "_")) + (("_wall" + A.obstacle_size.replace(",", "x")) if A.obstacle_size else ""))
if not A.skip_sim:
    cmd = [PY_ISAAC, "-u", os.path.join(SCRIPTS, "replay_isaac_arm.py"), "--specs", os.path.abspath(os.path.join(out, "specs.npz")), "--out", os.path.abspath(sim_dir),
           "--rates", str(A.rate), "--variants", A.variants, "--episodes", A.members] + (["--obstacle", obstacle] if obstacle else []) + (["--obstacle_mesh", obstacle_mesh[0], "--obstacle_pose", obstacle_pose] if obstacle_mesh else [])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(A.gpu), OMNI_KIT_ACCEPT_EULA="YES")
    with open(os.path.join(out, "replay.log"), "w") as log:
        print("[arm] running Isaac (log", log.name + ")", flush=True)
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    for r in json.load(open(os.path.join(sim_dir, "replay.json")))["results"]:
        print(f"[arm] member {r['episode']} {r['variant']}: hand err mean {r['hand_pos_err_mm_mean']:.1f} max {r['hand_pos_err_mm_max']:.1f} mm, rot max {r['hand_rot_err_deg_max']:.1f} deg, "
              f"joint margin {r['min_joint_margin_rad']:+.3f} rad, object lift at the end {r['sim_lift_mm']:.0f} mm", flush=True)
if not A.skip_render:
    rd = os.path.join(out, A.render_dir or ("render_asset" if A.asset != "none" else "render") + ("" if A.variants == "guided" else "_" + A.variants.replace(",", "_"))); os.makedirs(rd, exist_ok=True)
    env = dict(os.environ, PYOPENGL_PLATFORM="egl", EGL_DEVICE_ID=str(A.gpu), CUDA_VISIBLE_DEVICES=str(A.gpu))
    eye, look = A.eye, A.look
    if A.auto_camera:
        c0 = z["gt"][members[0], 0][z["roles"][members[0]] == 2].mean(0) + SHIFT; tg = arrays["target"][members[0], 0]; mid = (c0 + tg) / 2
        dxy = (tg - c0)[:2]; dxy = dxy / max(1e-6, np.linalg.norm(dxy))
        nrm = np.array([0.0, -1.0]) if abs(dxy[0]) >= abs(dxy[1]) else np.array([np.sign(mid[0]) or 1.0, 0.0])   # side view: -y for x-carries (as before), the far (+x) side for y-carries
        eye_xy = mid[:2] + 1.22 * nrm + 0.26 * dxy
        look = ",".join(f"{v:.3f}" for v in [mid[0], mid[1], mid[2] + 0.12]); eye = ",".join(f"{v:.3f}" for v in [eye_xy[0], eye_xy[1], mid[2] + 0.42])   # far enough for the arm's peak to stay in frame
        print("[arm] camera eye", eye, "look", look, flush=True)
    cmd = [PY_RENDER, os.path.join(SCRIPTS, "render_arm_replay.py"), os.path.abspath(sim_dir), os.path.abspath(rd), "--size", A.size, f"--eye={eye}", f"--look={look}", "--mp4"] + (["--envs", A.render_envs] if A.render_envs else []) + (["--obstacle_mesh", obstacle_mesh[1], "--obstacle_pose", obstacle_pose, "--obstacle_collider", obstacle_mesh[0]] if obstacle_mesh else [])
    tgt = arrays["target"][members[0], 0]
    if not A.no_target: cmd += ["--target", ",".join(f"{v:.4f}" for v in tgt)]
    asset = A.asset
    if asset == "auto":
        lab = str(z["episodes"][members[0]]); asset = None
        if "_asset_" in lab:
            cat, num = lab.split("_asset_"); asset = None
            for scene in ("Kitchen_Table", "libero", "Bathroom_Counter", "real_world", "simpler"):
                cand = f"/data/group_data/katefgroup-ssd/sim_scene_gen/assets/{scene}/{cat}/asset_{num}/asset_{num}_visual.ply"
                if os.path.exists(cand) and os.path.getsize(cand) > 0: asset = cand; break
        print("[arm] asset mesh:", asset or "none (convex hull)", flush=True)
    if asset and asset != "none": cmd += ["--object_mesh", asset]
    if asset and asset != "none" and A.object_tint: cmd += ["--object_tint", A.object_tint]
    if A.table_color: cmd += ["--table_color", A.table_color]
    if A.target_color: cmd += ["--target_color", A.target_color]
    with open(os.path.join(out, "render.log"), "w") as log:
        print("[arm] rendering (log", log.name + ")", flush=True)
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    print("[arm] renders:", sorted(os.listdir(rd)), flush=True)
