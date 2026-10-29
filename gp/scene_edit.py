"""Frame-0 edits of a pwm_final_paper pickup batch before guidance: close the fingers, move the hand (mid / natural).
Lifted from reports/pickplace_server/run_guide.py (2026-09-22/23) unchanged in behaviour."""
import dataclasses, math
import numpy as np
import torch
from .objectives import centroid


def _replace_frame0(batch, pos0, bp0):
    positions = batch.positions.clone(); positions[:, 0] = pos0
    body_poses = batch.body_poses.clone(); body_poses[:, 0] = bp0
    out = dataclasses.replace(batch, positions=positions, body_poses=body_poses)
    out.gt_frames, out.episodes = batch.gt_frames, batch.episodes
    return out


def close_fingers(batch, closed_sep_mm=27.0, log=print):
    """Slide each prismatic finger along its joint axis so the finger-centroid separation is closed_sep_mm."""
    B = batch.positions.shape[0]
    pos0 = batch.positions[:, 0].clone(); bp0 = batch.body_poses[:, 0].clone()
    for b in range(B):
        act = batch.pose_action_mask[b]; jt = batch.joint_type[b]; par = batch.body_parent[b]
        for k in (act & (jt == 2)).nonzero()[:, 0].tolist():
            h = int(par[k]); Rh, th = bp0[b, h, :3, :3], bp0[b, h, :3, 3]
            axis = batch.joint_axis[b, k] / batch.joint_axis[b, k].norm()
            local = Rh.T @ (bp0[b, k, :3, 3] - th); q = float(local @ axis)
            q_closed = np.sign(q) * closed_sep_mm / 2000.0
            t_closed = th + Rh @ (local + (q_closed - q) * axis)
            bp0[b, k, :3, 3] = t_closed
            rows = (batch.particle_body[b] == k) & batch.get_valid_mask()[b]
            n = int(batch.body_template_mask[b, k].sum())
            pos0[b, rows] = (bp0[b, k, :3, :3] @ batch.body_templates[b, k, :n].T).T + t_closed
            log(f"member {b} finger body {k}: q {q * 1e3:+.1f} mm -> {q_closed * 1e3:+.1f}")
    return _replace_frame0(batch, pos0, bp0)


def _rot_a_to_b(a, b):
    a = a / a.norm(); b = b / b.norm(); v = torch.cross(a, b, dim=0); c = float(a @ b)
    if v.norm() < 1e-8:
        return torch.eye(3, device=a.device) if c > 0 else -torch.eye(3, device=a.device)
    vx = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], device=a.device)
    return torch.eye(3, device=a.device) + vx + vx @ vx * (1 / (1 + c))


def hand_start_natural(batch, target, obj_mask, seed, bearing_deg=60.0, bearing_jitter_deg=15.0, dist_mm=250.0, dist_jitter_mm=30.0,
                       dz_mm=85.0, tilt_deg=0.0, log=print):
    """Re-pose the whole gripper at frame 0 like an rl_push start: behind the object relative to the push direction (bearing),
    at a horizontal distance, dz above the object centre, approach axis tilted from straight down, opening axis toward the object."""
    dev = batch.positions.device; B = batch.positions.shape[0]
    pos0 = batch.positions[:, 0].clone(); bp0 = batch.body_poses[:, 0].clone()
    gj = torch.Generator().manual_seed(seed + 13)
    c0 = centroid(batch.positions[:, 0], obj_mask)
    for b in range(B):
        act = batch.pose_action_mask[b]; jt = batch.joint_type[b]; valid = batch.get_valid_mask()[b]
        hand_k = int((act & (jt == 0)).nonzero()[0]); fk = (act & (jt == 2)).nonzero()[:, 0].tolist()
        rows = {k: (batch.particle_body[b] == k) & valid for k in act.nonzero()[:, 0].tolist()}
        hc = pos0[b, rows[hand_k]].mean(0); fc = [pos0[b, rows[k]].mean(0) for k in fk]
        appr = (fc[0] + fc[1]) / 2 - hc; appr = appr / appr.norm()
        push = target[b] - c0[b]; push[2] = 0; push = push / push.norm()
        ang = math.radians(bearing_deg + float(torch.randn(1, generator=gj)) * bearing_jitter_deg)
        side = 1.0 if float(torch.rand(1, generator=gj)) < 0.5 else -1.0
        perp = torch.stack([-push[1], push[0], torch.zeros((), device=dev)])
        hand_dir = -push * math.cos(ang) + side * perp * math.sin(ang)
        dist = (dist_mm + float(torch.randn(1, generator=gj)) * dist_jitter_mm) / 1000.0
        want = c0[b] + hand_dir * dist; want[2] = c0[b, 2] + dz_mm / 1000.0
        down = torch.tensor([0.0, 0.0, -1.0], device=dev); bearing = -hand_dir
        tilt_dir = down * math.cos(math.radians(tilt_deg)) + bearing * math.sin(math.radians(tilt_deg))
        R1 = _rot_a_to_b(appr, tilt_dir)
        opening = R1 @ (fc[0] - fc[1]); opening[2] = 0; opening = opening / opening.norm()
        b_h = bearing.clone(); b_h[2] = 0; b_h = b_h / b_h.norm()
        cs = float(opening @ b_h); sn = float(opening[0] * b_h[1] - opening[1] * b_h[0])
        if cs < 0:
            cs, sn = -cs, -sn
        R2 = torch.tensor([[cs, -sn, 0], [sn, cs, 0], [0, 0, 1]], device=dev); Rfix = R2 @ R1
        for k, r in rows.items():
            bp0[b, k, :3, :3] = Rfix @ bp0[b, k, :3, :3]; bp0[b, k, :3, 3] = Rfix @ (bp0[b, k, :3, 3] - hc) + want
            pos0[b, r] = (Rfix @ (pos0[b, r] - hc).T).T + want
        log(f"member {b}: hand centroid -> {[round(float(v) * 1e3) for v in want]} mm, bearing {math.degrees(ang):.0f} deg {'L' if side > 0 else 'R'}, dist {dist * 1e3:.0f} mm")
    return _replace_frame0(batch, pos0, bp0)


def hand_start_mid(batch, target, obj_mask, seed, jitter_mm=30.0, z_mm=105.0, log=print):
    """Translate the gripper so the hand-body centroid sits at the midpoint between the object start and the goal (+ xy jitter), at z_mm."""
    dev = batch.positions.device; B = batch.positions.shape[0]
    pos0 = batch.positions[:, 0].clone(); bp0 = batch.body_poses[:, 0].clone()
    gj = torch.Generator().manual_seed(seed + 11)
    c0 = centroid(batch.positions[:, 0], obj_mask)
    for b in range(B):
        act = batch.pose_action_mask[b]; jt = batch.joint_type[b]; valid = batch.get_valid_mask()[b]
        hand_k = int((act & (jt == 0)).nonzero()[0]); hand_rows = (batch.particle_body[b] == hand_k) & valid
        cur = pos0[b, hand_rows].mean(0)
        want = 0.5 * (c0[b] + target[b]); want = want.clone(); want[:2] += torch.randn(2, generator=gj).to(dev) * jitter_mm / 1000.0
        want[2] = z_mm / 1000.0
        delta = want - cur
        for k in act.nonzero()[:, 0].tolist():
            bp0[b, k, :3, 3] += delta
            rows = (batch.particle_body[b] == k) & valid; pos0[b, rows] += delta
        log(f"member {b}: hand start moved by {[round(float(v) * 1e3) for v in delta]} mm")
    return _replace_frame0(batch, pos0, bp0)


def hand_start_above(batch, obj_mask, above_mm=300.0, log=print):
    """Translate the gripper (all actuated bodies together, orientation unchanged) so the hand-body centroid sits straight above the
    object centroid, above_mm higher than it."""
    B = batch.positions.shape[0]
    pos0 = batch.positions[:, 0].clone(); bp0 = batch.body_poses[:, 0].clone()
    c0 = centroid(batch.positions[:, 0], obj_mask)
    for b in range(B):
        act = batch.pose_action_mask[b]; jt = batch.joint_type[b]; valid = batch.get_valid_mask()[b]
        hand_k = int((act & (jt == 0)).nonzero()[0]); hand_rows = (batch.particle_body[b] == hand_k) & valid
        cur = pos0[b, hand_rows].mean(0)
        want = c0[b].clone(); want[2] = c0[b, 2] + above_mm / 1000.0
        delta = want - cur
        for k in act.nonzero()[:, 0].tolist():
            bp0[b, k, :3, 3] += delta
            rows = (batch.particle_body[b] == k) & valid; pos0[b, rows] += delta
        log(f"member {b}: hand start moved straight above the object by {[round(float(v) * 1e3) for v in delta]} mm (was {[round(float(v) * 1e3) for v in (cur - c0[b])]} mm off the centroid)")
    return _replace_frame0(batch, pos0, bp0)


def load_ply_vertices(path):
    """Vertices of a binary little-endian PLY (x,y,z + any scalar props), no trimesh needed."""
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"end_header\n"):
            hdr += f.readline()
        h = hdr.decode(); n = int([l for l in h.split("\n") if l.startswith("element vertex")][0].split()[-1])
        props = [l.split()[1:] for l in h.split("\n") if l.startswith("property ") and "list" not in l]
        fmt = {"float": "<f4", "uchar": "u1", "double": "<f8", "int": "<i4", "float32": "<f4", "uint8": "u1"}
        dt = np.dtype([(nm, fmt[t]) for t, nm in props]); arr = np.frombuffer(f.read(n * dt.itemsize), dtype=dt)
    return np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)


def replace_object(batch, obj_mask, mesh_path, seed=0, yaw_deg=0.0, log=print):
    """Swap the (single) object's particles for points sampled on `mesh_path` (metres, centred): same xy centroid, bottom at the
    original object's lowest particle, same particle count. Updates positions (all frames), the object's body template and its
    frame-0 body pose (rotation kept, translation = new centroid), so pose-native sampling rebuilds the new object from the template."""
    V = load_ply_vertices(mesh_path)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)); V = V @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    rng = np.random.RandomState(seed)
    positions = batch.positions.clone(); body_poses = batch.body_poses.clone(); templates = batch.body_templates.clone()
    B = positions.shape[0]
    for b in range(B):
        rows = obj_mask[b].nonzero()[:, 0]; n = int(rows.numel())
        k = int(batch.particle_body[b, rows[0]])
        old = positions[b, 0, rows]; c_old = old.mean(0); z_min_old = float(old[:, 2].min())
        idx = rng.choice(len(V), n, replace=len(V) < n); P = V[idx].copy()
        P[:, :2] -= P[:, :2].mean(0); P[:, 2] -= P[:, 2].min()                   # centre xy, bottom at 0
        P = torch.tensor(P, dtype=positions.dtype, device=positions.device)
        P[:, 0] += c_old[0]; P[:, 1] += c_old[1]; P[:, 2] += z_min_old
        c_new = P.mean(0)
        positions[b, :, rows] = P[None]                                          # every frame: no trace of the old object anywhere
        nt = int(batch.body_template_mask[b, k].sum()); assert nt == n, f"template rows {nt} != object rows {n}"
        R0 = body_poses[b, 0, k, :3, :3]
        templates[b, k, :n] = (R0.T @ (P - c_new).T).T
        body_poses[b, :, k, :3, 3] = c_new[None]
        log(f"member {b}: object body {k} -> {mesh_path.split('/')[-2]}/{mesh_path.split('/')[-1]}: {n} pts, extents {((P.max(0).values - P.min(0).values) * 1e3).round().tolist()} mm, centroid {(c_new * 1e3).round().tolist()} mm (was {(c_old * 1e3).round().tolist()})")
    out = dataclasses.replace(batch, positions=positions, body_poses=body_poses, body_templates=templates)
    out.gt_frames, out.episodes = batch.gt_frames, batch.episodes
    return out


def set_frame0_points(batch, points, log=print):
    """Closed loop: make an EXECUTED particle cloud (model frame, same row order as the batch) the new frame 0. Every body (hand,
    fingers, object) is moved rigidly: its pose is the Kabsch fit of its frame-0 rows onto the executed rows, applied to its
    frame-0 body pose, and its rows are the fitted rigid copy (so particles and poses stay consistent)."""
    B = batch.positions.shape[0]; dev = batch.positions.device
    P = torch.as_tensor(points, dtype=batch.positions.dtype, device=dev)
    pos0 = batch.positions[:, 0].clone(); bp0 = batch.body_poses[:, 0].clone(); err = 0.0
    for b in range(B):
        valid = batch.get_valid_mask()[b]
        for k in torch.unique(batch.particle_body[b][valid]).tolist():
            rows = (batch.particle_body[b] == k) & valid
            X = pos0[b, rows].double(); Y = P[rows].double(); cx, cy = X.mean(0), Y.mean(0)
            U, _, Vt = torch.linalg.svd((X - cx).T @ (Y - cy)); d = torch.sign(torch.linalg.det(Vt.T @ U.T))
            R = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=torch.float64, device=dev)) @ U.T; t = cy - R @ cx
            new = (X @ R.T + t); err = max(err, float((new - Y).norm(dim=-1).max()))
            pos0[b, rows] = new.to(pos0.dtype)
            M = torch.eye(4, dtype=torch.float64, device=dev); M[:3, :3] = R; M[:3, 3] = t
            bp0[b, k] = (M @ bp0[b, k].double()).to(bp0.dtype)
    log(f"frame 0 set from the executed cloud (rigid fit, max residual {err * 1e3:.2f} mm)")
    return _replace_frame0(batch, pos0, bp0)
