"""Shared pieces of the replay backends: the replay.npz contract, rigid transforms, the common analysis rows.

replay.npz (next to rollout.npz, in <run>/replay_<sim>/): sim [V, B, T, Ns, 3] simulated point clouds at the MODEL's
particle resolution (rigid bodies = the frame-0 generated particles under the simulated body pose; MPM = the simulated
particles), sim_roles [B, Ns] (0 actuator, 1 finger, 2 object), variants (json list), plus analysis.json rows:
member, variant, gen_to_target_mm, sim_to_target_mm, sim_vs_gen_final_mm, gen_moved_mm, sim_moved_mm, gen_lift_mm,
sim_lift_mm, track_mm_mean, track_mm_max (simulated vs generated actuator centre) and backend extras."""
import json, os
import numpy as np

VARIANT_KEY = {"guided": "predicted", "unguided": "unguided"}


def qmat(q):
    """wxyz quaternion(s) -> rotation matrices."""
    q = np.asarray(q, np.float64); w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_quat(R):
    """rotation matrix -> wxyz."""
    m = np.asarray(R, np.float64); t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1) * 2; return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def kabsch(src, dst):
    """R, t with dst ~ src @ R.T + t."""
    cs, cd = src.mean(0), dst.mean(0); H = (src - cs).T @ (dst - cd); U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); D = np.diag([1, 1, d]); R = Vt.T @ D @ U.T
    return R, cd - cs @ R.T


def load_run(run):
    z = np.load(os.path.join(run, "rollout.npz")); rj = json.load(open(os.path.join(run, "result.json")))
    return z, rj


def analysis_row(gen, sim, roles, obj_mask, target, m, variant):
    """gen/sim [T, N, 3] (sim at the model resolution, same roles), frame 0 = the shared start."""
    go = gen[:, obj_mask].mean(1); so = sim[:, obj_mask].mean(1)
    act = roles == 0
    ga = gen[:, act].mean(1); sa = sim[:, act].mean(1); tr = np.linalg.norm(ga - sa, axis=-1) * 1e3
    row = dict(member=int(m), variant=variant, sim_vs_gen_final_mm=float(np.linalg.norm(go[-1] - so[-1]) * 1e3),
               sim_vs_gen_max_mm=float(np.linalg.norm(go - so, axis=-1).max() * 1e3),
               gen_moved_mm=float(np.linalg.norm(go[-1] - go[0]) * 1e3), sim_moved_mm=float(np.linalg.norm(so[-1] - so[0]) * 1e3),
               gen_lift_mm=float((go[:, 2].max() - go[0, 2]) * 1e3), sim_lift_mm=float((so[:, 2].max() - so[0, 2]) * 1e3),
               track_mm_mean=float(tr.mean()), track_mm_max=float(tr.max()))
    if target is not None and not np.isnan(target).any():
        row["gen_to_target_mm"] = float(np.linalg.norm(go[-1] - target) * 1e3); row["sim_to_target_mm"] = float(np.linalg.norm(so[-1] - target) * 1e3)
    return row


def write(out, sim, sim_roles, variants, rows, extra=None):
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "replay.npz"), sim=sim.astype(np.float32), sim_roles=sim_roles.astype(np.int8), variants=np.array(json.dumps(variants)))
    json.dump(dict(rows=rows, variants=variants, **(extra or {})), open(os.path.join(out, "analysis.json"), "w"), indent=1)
    print("wrote", os.path.join(out, "replay.npz"), "rows", len(rows), flush=True)
    for r in rows:
        print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
