"""Executed clouds -> the generic closed-loop history contract of gp/adapters/paper_prefix.py (domain-agnostic).

history_from_clouds(Q, body_poses0, particle_body, body_type, valid): Q [c+1, N, 3] executed positions (model frame,
the adapter's particle order, frame 0 = episode start); body_poses0 [K, 4, 4] the batch's frame-0 poses (rollout.npz
'body_poses0'). Each rigid body's pose at frame t = M_t @ pose0 with M_t the Kabsch fit of its frame-0 rows onto its
frame-t rows; non-rigid bodies keep pose0. Returns dict(positions, body_poses, fit_rms_mm)."""
import numpy as np


def kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0); H = (src - cs).T @ (dst - cd); U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)); R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cd - R @ cs


def history_from_clouds(Q, body_poses0, particle_body, body_type, valid=None):
    Q = np.asarray(Q, np.float64); T, N = Q.shape[:2]; K = body_poses0.shape[0]
    valid = np.ones(N, bool) if valid is None else np.asarray(valid, bool)
    bp = np.repeat(np.asarray(body_poses0, np.float64)[None], T, 0); rms = 0.0
    for k in range(K):
        if int(body_type[k]) != 1:
            continue
        rows = (particle_body == k) & valid
        if rows.sum() < 3:
            continue
        for t in range(1, T):
            R, tr = kabsch(Q[0, rows], Q[t, rows]); M = np.eye(4); M[:3, :3], M[:3, 3] = R, tr
            bp[t, k] = M @ body_poses0[k]
            rms = max(rms, float(np.sqrt((((Q[0, rows] @ R.T + tr) - Q[t, rows]) ** 2).sum(-1).mean())) * 1e3)
    return dict(positions=Q.astype(np.float32), body_poses=bp.astype(np.float32), fit_rms_mm=rms)
