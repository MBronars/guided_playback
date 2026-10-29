"""Export the pick failure-case episodes behind the paper pick figures to
/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_pickplace/baseline_failure_cases/ (same states.npz keys and
robot-base frame as export_pickplace.py, plus every particle of the simulated scene), with the renders and figure PDFs."""
import glob, json, os, shutil
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RUNS = os.path.join(ROOT, "runs", "pickup"); FIG = os.path.join(ROOT, "figures")
OUT = "/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_pickplace/baseline_failure_cases/picking"
SHIFT = np.array([0.45, 0.0, 0.10])
def qmat(q):
    w, x, y, z = [q[..., i] for i in range(4)]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1), np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
EPIS = {"milk_carton_t0": ("milk_carton", "milk carton", 0, {"ours_unguided": ("mppiargmin_full19_t0_milk_carton_K20", 0), "pointworld": ("pw_benchmark", 0), "ours": ("cyl_t0_milk_carton_it10", None)}),
        "soda_can_t1": ("soda_can", "soda can", 1, {"ours_unguided": ("mppiargmin_full19_t1_soda_can_K20", 0), "pointworld": ("pw_benchmark", 8), "ours": ("cyl_t1_soda_can_it10", None)})}
METHOD = {"ours_unguided": "Ours Unguided: our world model, MPPI argmin (20 fresh samples per replan, keep the one whose predicted object centroid ends closest to the target, commit 5 frames, replan to frame 60), no guidance",
          "pointworld": "PointWorld: PointWorld world model, MPPI argmin (population 88), same replanning scheme",
          "ours": "Ours: our world model with guidance (noise optimisation, 10 iterations, population 4); the member the planner picked (lowest generated objective)"}
index = []
os.makedirs(OUT, exist_ok=True)
for ep, (cat, oname, t, runs) in EPIS.items():
    for meth, (run, m) in runs.items():
        rd = os.path.join(RUNS, run); z = np.load(f"{rd}/rollout.npz", allow_pickle=True); rj = json.load(open(f"{rd}/result.json"))
        if m is None: m = int(np.argmin(rj["best_mm"]))
        a = json.load(open(f"{rd}/replay_isaac/analysis.json")); row = next(r for r in a["rows"] if r["member"] == m and r["variant"] == "guided")
        variants = a.get("variants", ["guided", "unguided"]); v = variants.index("guided")
        P = np.load(f"{rd}/replay_isaac/replay.npz", allow_pickle=True); pv = json.loads(str(P["variants"])).index("guided")
        R = np.load(f"{rd}/replay_isaac/isaac/replay.npz", allow_pickle=True); S = np.load(f"{rd}/replay_isaac/specs.npz")
        k = next(k for k in range(len(R["env_episode"])) if int(R["env_episode"][k]) == m and int(R["env_variant"][k]) == v)
        o0 = S["obj_pts0"][m]; c0 = o0.mean(0); f32 = lambda x: np.asarray(x, np.float32)
        d = os.path.join(OUT, ep, meth); os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "states.npz"),
            pred_points=f32(z["predicted"][m] + SHIFT), pred_roles=z["roles"][m].astype(np.int8), pred_object_mask=z["object_mask"][m], pred_particle_body=z["particle_body"][m].astype(np.int16),
            target=f32(z["target"][m] + SHIFT), sim_points=f32(P["sim"][pv, m] + SHIFT),
            sim_object_points=f32(np.einsum("tij,pj->tpi", qmat(R["obj_quat"][k]), o0 - c0) + R["obj_pos"][k][:, None] + SHIFT),
            sim_object_pos=f32(R["obj_pos"][k] + SHIFT), sim_object_quat_wxyz=R["obj_quat"][k], sim_hand_pos=f32(R["hand_pos"][k] + SHIFT), sim_hand_quat_wxyz=R["hand_quat"][k],
            sim_finger_q=R["finger_q"][k], sim_gripper_body_pos=f32(R["body_pos"][k] + SHIFT), sim_gripper_body_quat_wxyz=R["body_quat"][k], sim_gripper_closed=R["cmd_closed"][k],
            fps=np.float32(R["env_rate"][k]))
        meas = dict(start_to_target=None, pred_final_to_target=round(row["gen_to_target_mm"], 1), sim_final_to_target=round(row["sim_to_target_mm"], 1),
                    pred_peak_lift=round(row["gen_lift_mm"], 1), sim_peak_lift=round(row["sim_lift_mm"], 1), pred_vs_sim_max=round(row["sim_vs_gen_max_mm"], 1), pred_vs_sim_final=round(row["sim_vs_gen_final_mm"], 1))
        oc = z["gt"][m][0][z["object_mask"][m]].mean(0) if "gt" in z.files else o0.mean(0)
        meas["start_to_target"] = round(float(np.linalg.norm(R["obj_pos"][k][0] - z["target"][m])) * 1e3, 1)
        # renders of this run/member
        rend = os.path.join(d, "renders"); os.makedirs(rend, exist_ok=True); copied = []
        pats = [f"{FIG}/pick_panel_v2/{cat}__{run}__m{m}__*.png", f"{FIG}/pick_can_candidates/{cat}__{run}__m{m}__*.png", f"{FIG}/pick_can_choice/{cat}__{run}__m{m}__*.png"]
        if meth == "pointworld":
            pats += [f"{FIG}/pick_pw_scene_v2/{cat}__pw_benchmark__pwcloud__m{m}__*.png", f"{FIG}/pick_pw_evolution_v3/pointworld_{cat}_t{t}_f*.png", f"{FIG}/pick_pw_pred_start_end/pointworld_{cat}_t{t}_f*.png"]
        for p in pats:
            for f in sorted(glob.glob(p)):
                src_dir = os.path.basename(os.path.dirname(f)); dst = f"{src_dir}__{os.path.basename(f)}"; shutil.copy2(f, os.path.join(rend, dst)); copied.append(dst)
        meta = dict(episode=ep, object=oname, benchmark_target=f"t{t}", method=meth, method_description=METHOD[meth], source_run=rd, member=m,
                    source_episode=str(z["episodes"][m]), checkpoint=rj.get("checkpoint") or rj.get("ckpt"), measured_mm=meas, renders=copied)
        json.dump(meta, open(os.path.join(d, "case.json"), "w"), indent=1)
        index.append(dict(episode=ep, method=meth, member=m, **meas, renders=len(copied))); print(ep, meth, meas, len(copied), "renders", flush=True)
fd = os.path.join(OUT, "figures"); os.makedirs(fd, exist_ok=True)
for b in ("pick_baseline_pointworld_ours", "pick_pointworld_prediction", "pick_pointworld_prediction_object_centred"):
    for ext in (".pdf", ".png"): shutil.copy2(f"{FIG}/paper_pdf/{b}{ext}", fd)
json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1); print("wrote", OUT)
