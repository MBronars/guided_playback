"""Isaac Lab floating Franka hand replay of a pickup / rl_push rollout (packages/pickup_replay_isaac, isaaclab env).

rollout.npz -> specs.npz (pool link-local points Kabsch-fit to the generated gripper rows per frame: hand pose, finger
q, closed flag; object = frame-0 generated particles, density/friction from a matching pool row else pool medians)
-> replay_isaac.py (subprocess, ABSOLUTE paths) -> replay.npz contract (gripper = link-local points under the simulated
link poses, object = frame-0 particles under the simulated object pose)."""
import json, os, subprocess, sys
import numpy as np
from . import common as C

SCRIPTS = "/data/user_data/mbronars/packages/pickup_replay_isaac/scripts"
PY = "/data/user_data/mbronars/.miniconda3/envs/isaaclab/bin/python"
POOL = "/data/user_data/mbronars/packages/pickup_replay_isaac/out/pool_pick_val/specs.npz"
SHIFTS = [np.array([0.45, 0.0, 0.10]), np.array([0.45, 0.0, 0.11]), np.zeros(3)]


RLP_INDEX = "/data/user_data/mbronars/packages/guided_playback/sweeps/rl_push/test_index.json"
RLP_ROOT = "/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/rl_push/test/"


def rl_push_hulls(z):
    """Per member: the release object's CoACD hull vertices posed at the episode's frame 0 in the model frame (world), packed
    [B, Vmax, 3] with start offsets [B, Hmax + 1] (-1 padded). Checked against the frame-0 particles' bounding box."""
    idx = {r["loader"]: r for r in json.load(open(RLP_INDEX))}
    B = z["predicted"].shape[0]; V, O = [], []
    for m in range(B):
        r = idx[int(z["episodes"][m])]; o = np.load(RLP_ROOT + r["file"], allow_pickle=True)
        v = o["coacd_hull_vertices"].astype(np.float64); off = list(o["coacd_hull_offsets"].astype(int)) + [len(v)]
        w = np.einsum("ij,pj->pi", C.qmat(np.array(r["obj_quat0"], np.float64)), v) + np.array(r["obj_pos0"]) + np.array([-0.45, 0.0, 0.0])
        p0 = z["predicted"][m, 0, z["object_mask"][m]]
        err = max(np.abs(w.min(0) - p0.min(0)).max(), np.abs(w.max(0) - p0.max(0)).max())
        assert err < 0.02, f"member {m}: CoACD hulls off the frame-0 particles by {err * 1e3:.1f} mm"   # particles = 512-point subsample of the surface cloud
        V.append(w); O.append(np.array(off))
    Vmax, Hmax = max(len(v) for v in V), max(len(o) for o in O)
    hv = np.zeros((B, Vmax, 3)); ho = np.full((B, Hmax), -1, np.int64)
    for m in range(B): hv[m, :len(V[m])] = V[m]; ho[m, :len(O[m])] = O[m]
    return hv, ho


ARM_SHIFT = np.array([0.45, 0.0, 0.0])          # rl_push model frame -> robot base frame (data/rl_push.py CANONICAL_TRANSLATION undone)


def rl_push_arm_q0(z):
    """Per member: the recorded 7 arm joints at the episode's first frame (release joint_pos), the arm replay's IK seed."""
    idx = {r["loader"]: r for r in json.load(open(RLP_INDEX))}; q = []
    for m in range(z["predicted"].shape[0]):
        r = idx[int(z["episodes"][m])]; o = np.load(RLP_ROOT + r["file"], allow_pickle=True)
        q.append(o["joint_pos"][int(o["episode_start"][r["e"]]) + r["start"], :7].astype(np.float64))
    return np.stack(q)


def run(run_dir, out, variants=("guided", "unguided"), rate=20, close_threshold=0.035, gpu="0", arm=False):
    """arm=True: pickup_replay_isaac/scripts/replay_isaac_arm.py, the gripper on a full fixed-base Franka (rl_push only: the
    specs are shifted into the base frame, arm.npz holds the recorded frame-0 joints, results are shifted back)."""
    sys.path.insert(0, SCRIPTS)
    from build_specs import gripper_track
    z, rj = C.load_run(run_dir); os.makedirs(out, exist_ok=True)
    pool = np.load(POOL); counts = [int(c) for c in pool["link_counts"][0]]
    local = [pool["link_local"][0, li, :n].astype(np.float64) for li, n in enumerate(counts)]; n_grip = sum(counts)
    B, V = z["predicted"].shape[0], len(variants); T = z["predicted"].shape[1]
    roles = z["roles"]; obj_mask = z["object_mask"]
    arrays = None; checks = []
    for m in range(B):
        pb = z["particle_body"][m]
        assert [int((pb[:n_grip] == b).sum()) for b in (1, 2, 3)] == counts, "gripper row layout differs from the pool"
        obj_rows = obj_mask[m]; obj0 = z["predicted"][m, 0, obj_rows].astype(np.float64)
        best = (None, np.inf, None)
        if obj0.shape[0] == pool["obj_pts0"].shape[1]:
            for sh in SHIFTS:
                d = np.abs(pool["obj_pts0"] - (obj0 + sh)[None]).max(axis=(1, 2)); j = int(np.argmin(d))
                if d[j] < best[1]: best = (j, float(d[j]), sh.tolist())
        matched = best[0] is not None and best[1] < 0.002
        rho = float(pool["rho"][best[0]]) if matched else float(np.median(pool["rho"])); mu = float(pool["mu"][best[0]]) if matched else float(np.median(pool["mu"]))
        if arrays is None:
            arrays = dict(hand_pos=np.zeros((B, V, T, 3)), hand_quat=np.zeros((B, V, T, 4)), finger_q=np.zeros((B, V, T, 2)), fit_rms=np.zeros((B, V, T)),
                          obj_centroid=np.zeros((B, V, T, 3)), target=np.full((B, V, 3), np.nan), obj_pts0=np.zeros((B, obj0.shape[0], 3)), rho=np.zeros(B), mu=np.zeros(B),
                          asset_id=np.full(B, -1), link_local=np.repeat(pool["link_local"][:1], B, 0), link_counts=np.repeat(pool["link_counts"][:1], B, 0))
        for v, var in enumerate(variants):
            P = z[C.VARIANT_KEY[var]][m].astype(np.float64)
            hp, hq, fq, rms = gripper_track(P[:, :n_grip], local, counts)
            arrays["hand_pos"][m, v], arrays["hand_quat"][m, v], arrays["finger_q"][m, v], arrays["fit_rms"][m, v] = hp, hq, fq, rms
            arrays["obj_centroid"][m, v] = P[:, obj_rows].mean(1); arrays["target"][m, v] = z["target"][m]
        arrays["obj_pts0"][m] = obj0; arrays["rho"][m] = rho; arrays["mu"][m] = mu
        checks.append(dict(member=m, episode=str(z["episodes"][m]), pool_match=matched, pool_match_mm=round(best[1] * 1e3, 2) if best[0] is not None else None, rho=rho, mu=mu,
                           fit_rms_mm_max={var: round(float(arrays["fit_rms"][m, v].max()) * 1e3, 2) for v, var in enumerate(variants)}))
    if rj.get("domain") == "rl_push" and not os.environ.get("GP_ISAAC_POOL_PHYSICS"):
        # rl_push: the collection's object physics (RLSkills_IsaacLab franka_gripper base_env_cfg: density 500, friction 0.6-1.2 mean 1.0)
        # and the object's own CoACD convex decomposition (release file), posed at frame 0 like the particles, instead of one solid hull.
        arrays["rho"][:] = 500.0; arrays["mu"][:] = 1.0
        hv, ho = rl_push_hulls(z); arrays["hull_vertices"], arrays["hull_offsets"] = hv, ho
        print(f"[isaac] rl_push physics: density 500, friction 1.0, CoACD hulls per member {[int((o >= 0).sum() - 1) for o in ho]}", flush=True)
    if rj.get("task") == "place" and not os.environ.get("GP_ISAAC_POOL_PHYSICS"):
        # place runs (held start): the asset's CoACD decomposition posed at frame 0 + the recorded density / friction (gp/replay/place_assets.py).
        # Pick / lift runs never enter here: they keep the single convex hull + pool physics above.
        from .place_assets import place_hulls
        hv, ho, prho, pmu, prow = place_hulls(z); arrays["hull_vertices"], arrays["hull_offsets"] = hv, ho; arrays["rho"][:] = prho; arrays["mu"][:] = pmu
        for c_, pr in zip(checks, prow): c_.update(place_asset=pr)
        print(f"[isaac] place physics: CoACD pieces per member {[int((o >= 0).sum() - 1) for o in ho]}, rho {prho.round(0).tolist()}, mu {pmu.tolist()}", flush=True)
    if os.environ.get("GP_ISAAC_MU"): arrays["mu"][:] = float(os.environ["GP_ISAAC_MU"]); print("[isaac] object friction override", arrays["mu"][0], flush=True)
    if os.environ.get("GP_ISAAC_RHO"): arrays["rho"][:] = float(os.environ["GP_ISAAC_RHO"]); print("[isaac] object density override", arrays["rho"][0], flush=True)
    closed = arrays["finger_q"].mean(-1) < close_threshold
    specs = os.path.abspath(os.path.join(out, "specs.npz"))
    if arm:
        assert rj.get("domain") == "rl_push", "arm replay: rl_push runs only (base frame known)"
        for key in ("hand_pos", "obj_centroid", "target", "obj_pts0"): arrays[key] = arrays[key] + ARM_SHIFT
        if "hull_vertices" in arrays: arrays["hull_vertices"] = arrays["hull_vertices"] + ARM_SHIFT
        np.savez(os.path.join(out, "arm.npz"), arm_q0=rl_push_arm_q0(z))
    np.savez(specs, **arrays, closed=closed, close_threshold=close_threshold)
    json.dump(dict(variants=list(variants), episodes=[[os.path.basename(run_dir.rstrip("/")), c["member"]] for c in checks], checks=checks, shift=[0, 0, 0], close_threshold=close_threshold),
              open(specs[:-4] + ".json", "w"), indent=1)
    sim_dir = os.path.abspath(os.path.join(out, "isaac"))
    log = open(os.path.join(out, "replay.log"), "w")
    print(f"[isaac] replaying {B} members x {V} variants at {rate} fps (log {log.name})", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMNI_KIT_ACCEPT_EULA="YES")
    if os.path.exists(os.path.join(sim_dir, "replay.npz")) and os.environ.get("GP_REUSE_SIM"):
        print("[isaac] reusing existing replay.npz", flush=True)
    else:
        extra = os.environ.get("GP_ISAAC_EXTRA", "").split()                       # e.g. "--kp_pos 100": controller / physics overrides passed to replay_isaac.py
        if extra: print("[isaac] extra args", extra, flush=True)
        script = "replay_isaac_arm.py" if arm else "replay_isaac.py"
        subprocess.run([PY, "-u", os.path.join(SCRIPTS, script), "--specs", specs, "--out", sim_dir, "--rates", str(rate)] + extra, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    R = dict(np.load(os.path.join(sim_dir, "replay.npz")))
    if arm:                                                                          # back to the model frame
        R["body_pos"] = R["body_pos"] - ARM_SHIFT; R["obj_pos"] = R["obj_pos"] - ARM_SHIFT
        for key in ("hand_pos", "obj_centroid", "target", "obj_pts0"): arrays[key] = arrays[key] - ARM_SHIFT
    N = z["predicted"].shape[2]; sim = np.zeros((V, B, T, N, 3), np.float32); sim_roles = np.zeros((B, N), np.int8); rows = []
    for k in range(len(R["env_episode"])):
        m, v = int(R["env_episode"][k]), int(R["env_variant"][k]); var = variants[v]
        loc = [arrays["link_local"][m, i, :counts[i]] for i in range(3)]
        grip = np.concatenate([np.einsum("tij,pj->tpi", C.qmat(R["body_quat"][k, :, i]), loc[i]) + R["body_pos"][k, :, i][:, None] for i in range(3)], 1)
        o0 = arrays["obj_pts0"][m]; c0 = o0.mean(0)
        objp = np.einsum("tij,pj->tpi", C.qmat(R["obj_quat"][k]), o0 - c0) + R["obj_pos"][k][:, None]
        obj_rows = obj_mask[m]; Tk = min(T, grip.shape[0])
        dst = sim[v, m]; dst[:Tk, :n_grip] = grip[:Tk]; dst[:Tk, obj_rows] = objp[:Tk]
        if Tk < T: sim[v, m, Tk:] = sim[v, m, Tk - 1]
        sim_roles[m] = roles[m]
        row = C.analysis_row(z[C.VARIANT_KEY[var]][m], sim[v, m], roles[m], obj_rows, z["target"][m], m, var)
        row.update(fit_rms_mm_max=checks[m]["fit_rms_mm_max"][var], closed_frames=int(R["cmd_closed"][k].sum()), rho=checks[m]["rho"], mu=checks[m]["mu"], pool_match=checks[m]["pool_match"])
        rows.append(row)
    if arm:
        for row in rows:
            k = [i for i in range(len(R["env_episode"])) if int(R["env_episode"][i]) == row["member"] and variants[int(R["env_variant"][i])] == row["variant"]][0]
            pe = np.linalg.norm(R["hand_pos"][k] - R["target_pos"][k], axis=-1) * 1e3
            lim = R["joint_limits"]; marg = np.minimum(R["arm_q"][k] - lim[:, 0], lim[:, 1] - R["arm_q"][k])
            row.update(arm_hand_err_mm_mean=float(pe.mean()), arm_hand_err_mm_max=float(pe.max()), arm_min_joint_margin_rad=float(marg.min()))
    C.write(out, sim, sim_roles, list(variants), rows, extra=dict(sim="isaac full arm (fixed base)" if arm else "isaac floating hand", rate=rate, close_threshold=close_threshold, isaac_extra=os.environ.get("GP_ISAAC_EXTRA", ""), mu_override=os.environ.get("GP_ISAAC_MU"), rho_override=os.environ.get("GP_ISAAC_RHO")))
