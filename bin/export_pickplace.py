"""Export the successful pick/carry/place cases (world-model prediction + simulated 3D states + videos) to a shared folder.

usage: python bin/export_pickplace.py [--out /data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_pickplace]

A case is exported when the commanded Franka arm, replaying the chosen plan in Isaac Sim, lifted the object more than 100 mm
and left it within 100 mm of the target (air targets), or placed it within 100 mm (ground / cabinet targets).
Per case it writes <out>/cases/<case>/ with states.npz, case.json, and the rendered videos/keyframes.
"""
import argparse, glob, json, os, re, shutil
import numpy as np

ROOT = "/data/user_data/mbronars/packages/guided_playback"; os.chdir(ROOT)
SHIFT = np.array([0.45, 0.0, 0.10])
ALIGN_MM = 30.0                                              # "good alignment": world-model vs simulator object centroid within this at EVERY frame                        # model frame -> robot base frame (= raw data frame)
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_pickplace"); A = ap.parse_args()


def qmat(q):
    q = np.asarray(q, np.float64); w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def cases():
    """(case name, run dir, member, arm sim dir, media dir or None, kind, extra note)"""
    out = []
    for c in json.load(open("sweeps/gallery_cases.json")):
        b = c["base"]
        if c.get("failed") or b == "gal_jar_side30": continue          # jar_side30: best member lands on its side
        pj = f"runs/pickup/{b}_pick.json"
        if os.path.exists(pj):
            p = json.load(open(pj))
            if not (p["arm_peak_lift"] > 100 and p["arm_final_mm"] <= 100): continue
            out.append((b.replace("gal_", ""), f"runs/pickup/{p['run']}", int(p["member"]), "isaac", f"site_gallery/{b}", c["title"], c["object"]))
        else:                                                             # early chain (no pick.json): the pick is in the chain log
            m = re.search(r"gallery pick (\S+) m(\d+)", open(f"logs/{b}_chain.log").read()) if os.path.exists(f"logs/{b}_chain.log") else None
            if m: out.append((b.replace("gal_", ""), f"runs/pickup/{m.group(1)}", int(m.group(2)), "isaac", f"site_gallery/{b}", c["title"], c["object"]))
    # the four-panel figure cases (milk carton, ground target 30 cm away; with and without a 12 cm wall; the unconditional sample)
    out.append(("figure_guided_nowall", "runs/pickup/ground30_nowall_milk_carton_asset_0037", 0, "isaac", None, "Figure: guided, ground target, no obstacle", "milk carton"))
    out.append(("figure_guided_wall", "runs/pickup/wall12w20ground_pickonly_it40_lr0.1_shell_bw10_clr10", 3, "isaac_wall0.02x0.55x0.12", None, "Figure: guided over a 12 cm wall", "milk carton"))
    out.append(("figure_unconditional", "runs/pickup/ground30_nowall_milk_carton_asset_0037", 2, "isaac_unguided", None, "Figure: unconditional sample (no guidance)", "milk carton"))
    return out


os.makedirs(os.path.join(A.out, "cases"), exist_ok=True); index = []
for name, run, mem, simdir, media, title, obj in cases():
    rj = json.load(open(f"{run}/result.json")); z = np.load(f"{run}/rollout.npz", allow_pickle=True)
    variant = "unguided" if simdir == "isaac_unguided" else "guided"
    Zf = f"{run}/replay_isaac_arm/{simdir}/replay.npz"
    if not os.path.exists(Zf): print("SKIP (no arm replay)", name); continue
    Z = np.load(Zf, allow_pickle=True)
    ks = [k for k in range(len(Z["env_episode"])) if int(Z["env_episode"][k]) == mem]
    if not ks: print("SKIP (member not in arm replay)", name, mem); continue
    k = ks[0]
    pred = (z["unguided"] if variant == "unguided" else z["predicted"])[mem].astype(np.float32) + SHIFT.astype(np.float32)
    roles = z["roles"][mem]; objm = z["object_mask"][mem]
    o0 = Z["obj_pts0"][k]; c0 = o0.mean(0)
    sim_obj = (np.einsum("tij,pj->tpi", qmat(Z["obj_quat"][k]), o0 - c0) + Z["obj_pos"][k][:, None]).astype(np.float32)
    tgt = (z["target"][mem] + SHIFT).astype(np.float32)
    boxes = z["boxes"].copy() if len(z["boxes"]) else np.zeros((0, 6), np.float32)
    if len(boxes): boxes[:, :3] += SHIFT
    d = os.path.join(A.out, "cases", name); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "states.npz"),
        pred_points=pred, pred_roles=roles.astype(np.int8), pred_object_mask=objm, pred_particle_body=z["particle_body"][mem].astype(np.int16),
        target=tgt, obstacle_boxes=boxes.astype(np.float32),
        sim_object_points=sim_obj, sim_object_pos=Z["obj_pos"][k], sim_object_quat_wxyz=Z["obj_quat"][k],
        sim_hand_pos=Z["hand_pos"][k], sim_hand_quat_wxyz=Z["hand_quat"][k], sim_finger_q=Z["finger_q"][k], sim_gripper_closed=Z["cmd_closed"][k],
        sim_arm_q=Z["arm_q"][k], sim_link_pos=Z["link_pos"][k], sim_link_quat_wxyz=Z["link_quat"][k], sim_link_names=Z["link_names"],
        cmd_hand_pos=Z["target_pos"][k], cmd_hand_quat_wxyz=Z["target_quat"][k],
        table_x=Z["table_x"][k], table_y=Z["table_y"][k], table_top_z=Z["table_top_z"][k], fps=np.float32(Z["env_rate"][k]))
    op = Z["obj_pos"][k]; pc = pred[:, objm].mean(1)
    gap = np.linalg.norm(pc - sim_obj.mean(1), axis=-1) * 1e3                  # world-model vs simulator object centroid, per frame (mm)
    meta = dict(case=name, title=title, object=obj, source_run=os.path.abspath(run), member=mem, variant=variant, source_arm_replay=os.path.abspath(Zf),
                episode=str(z["episodes"][mem]), checkpoint=rj.get("ckpt"),
                settings={kk: rj.get(kk) for kk in ("transitions", "steps", "offset", "optimizer", "lr", "iterations", "shell", "seed", "freeze_frames", "still_weight",
                                                     "still_frames", "box_weight", "clearance_mm", "upright_weight", "obstacle_asset", "obstacle_rpy", "obstacle_layout",
                                                     "hand_start", "hand_above_mm", "desc") if rj.get(kk) is not None},
                measured_mm=dict(pred_final_to_target=round(float(np.linalg.norm(pc[-1] - tgt) * 1e3), 1),
                                 sim_final_to_target=round(float(np.linalg.norm(op[-1] - tgt) * 1e3), 1),
                                 sim_peak_lift=round(float((op[:, 2].max() - op[0, 2]) * 1e3), 1),
                                 sim_end_height_above_start=round(float((op[-1, 2] - op[0, 2]) * 1e3), 1),
                                 pred_vs_sim_mean=round(float(gap.mean()), 1), pred_vs_sim_max=round(float(gap.max()), 1), pred_vs_sim_final=round(float(gap[-1]), 1)))
    meta["good_alignment"] = bool(gap.max() <= ALIGN_MM)
    json.dump(meta, open(os.path.join(d, "case.json"), "w"), indent=1)
    if media and os.path.isdir(media):
        for f in ("arm.mp4", "hull.mp4", "keyframes.png"):
            if os.path.exists(os.path.join(media, f)): shutil.copy(os.path.join(media, f), os.path.join(d, {"arm.mp4": "video_rendered.mp4", "hull.mp4": "video_hull.mp4", "keyframes.png": "keyframes.png"}[f]))
    index.append(dict(case=name, object=obj, title=title, good_alignment=meta["good_alignment"], **meta["measured_mm"]))
    print("exported", name, meta["measured_mm"])
json.dump(index, open(os.path.join(A.out, "index_arm.json"), "w"), indent=1)
fig = os.path.join(A.out, "figures"); os.makedirs(fig, exist_ok=True)
for f in glob.glob("site_armwall/panel/four_panel_rendered_tan_walnut*") + glob.glob("site_armwall/panel/four_panel.p*") + glob.glob("site_armwall/panel/four_panel_tight.*"):
    if "low" not in f: shutil.copy(f, fig)
print("arm cases:", len(index))

# ---- floating-hand replays (no arm) where the world model and the simulator agree: object lifted > 100 mm and the
#      predicted vs simulated object centroid within ALIGN_MM at every frame. Guided members only, all pickup runs.
fl = []; fdir = os.path.join(A.out, "floating"); os.makedirs(fdir, exist_ok=True)
for f in sorted(glob.glob("runs/pickup/*/replay_isaac/analysis.json")):
    run = os.path.dirname(os.path.dirname(f))
    try: a = json.load(open(f)); R = np.load(f"{run}/replay_isaac/isaac/replay.npz", allow_pickle=True); S = np.load(f"{run}/replay_isaac/specs.npz"); z = np.load(f"{run}/rollout.npz", allow_pickle=True); rj = json.load(open(f"{run}/result.json"))
    except Exception as e: print("skip", run, e); continue
    variants = a.get("variants", ["guided", "unguided"])
    for r in a["rows"]:
        if r["variant"] != "guided" or r["sim_lift_mm"] <= 100 or r["sim_vs_gen_max_mm"] > ALIGN_MM: continue
        m = int(r["member"]); v = variants.index("guided")
        ks = [k for k in range(len(R["env_episode"])) if int(R["env_episode"][k]) == m and int(R["env_variant"][k]) == v]
        if not ks: continue
        k = ks[0]; o0 = S["obj_pts0"][m]; c0 = o0.mean(0)
        sim_obj = (np.einsum("tij,pj->tpi", qmat(R["obj_quat"][k]), o0 - c0) + R["obj_pos"][k][:, None] + SHIFT).astype(np.float32)
        pred = z["predicted"][m].astype(np.float32) + SHIFT.astype(np.float32); objm = z["object_mask"][m]
        name = f"{os.path.basename(run)}__m{m}"; d = os.path.join(fdir, name); os.makedirs(d, exist_ok=True)
        tgt = (z["target"][m] + SHIFT).astype(np.float32)
        np.savez_compressed(os.path.join(d, "states.npz"), pred_points=pred, pred_roles=z["roles"][m].astype(np.int8), pred_object_mask=objm,
            pred_particle_body=z["particle_body"][m].astype(np.int16), target=tgt,
            sim_object_points=sim_obj, sim_object_pos=(R["obj_pos"][k] + SHIFT).astype(np.float32), sim_object_quat_wxyz=R["obj_quat"][k],
            sim_hand_pos=(R["hand_pos"][k] + SHIFT).astype(np.float32), sim_hand_quat_wxyz=R["hand_quat"][k], sim_finger_q=R["finger_q"][k],
            sim_gripper_body_pos=(R["body_pos"][k] + SHIFT).astype(np.float32), sim_gripper_body_quat_wxyz=R["body_quat"][k], sim_gripper_closed=R["cmd_closed"][k],
            fps=np.float32(R["env_rate"][k]))
        meta = dict(case=name, source_run=os.path.abspath(run), member=m, episode=str(z["episodes"][m]), checkpoint=rj.get("ckpt"), desc=rj.get("desc", ""),
                    object=str(z["episodes"][m]).split("_asset_")[0].replace("_", " "),
                    measured_mm=dict(pred_final_to_target=round(r.get("gen_to_target_mm", float("nan")), 1), sim_final_to_target=round(r.get("sim_to_target_mm", float("nan")), 1),
                                     sim_peak_lift=round(r["sim_lift_mm"], 1), pred_vs_sim_max=round(r["sim_vs_gen_max_mm"], 1), pred_vs_sim_final=round(r["sim_vs_gen_final_mm"], 1)))
        json.dump(meta, open(os.path.join(d, "case.json"), "w"), indent=1); fl.append(dict(case=name, object=meta["object"], **meta["measured_mm"]))
json.dump(fl, open(os.path.join(A.out, "index_floating.json"), "w"), indent=1)
print("floating cases:", len(fl), "->", A.out)
