"""Tier-2 v2 `pose_native` input side (docs/tier2_v2_design.md), shared by the
TrajectoryTransformer and unet_std.

`pose_native_inputs` integrates the NOISY 6-D pose deltas from the C-1 anchor
pose, reconstructs each rigid body's particles as template@pose per frame, and
rewrites those rows of `noisy_future` as normalized per-frame Cartesian deltas
— the same (audit-proven exact) chain as v1's reconstruction, applied to the
noisy INPUT instead of the predicted output. The particle rows of rigid bodies
are therefore a deterministic function of the pose channel at every noise
level (v9's self-consistency, generalized).

Also here: `body_object_index` (which contiguous object each rigid body lives
in — the unet's within-object L0 attention needs it) and
`context_pose_deltas` (the clean GT context Δ6 for the context pose tokens).

Articulation (optional): a child body moves with its parent through one 1-DOF
joint (`compose_joints`, forward kinematics), so jointed bodies cannot leave their
joint at any noise level, in the input rewrite, the materialization, or rollouts.

The three batch geometry entrypoints disable autocast and use FP32 pose math.
Their callers keep control of neural-network precision and particle-buffer dtype.
"""
from __future__ import annotations

from functools import wraps

import torch

# Articulation (TrajectoryBatch.body_parent / joint_type / joint_axis / joint_pivot). A child body
# moves with its parent through one 1-DOF joint. Its pose channels carry only the joint coordinate
# change, prismatic in channel 0 (metres, normalized with the translation scale) and revolute in
# channel 3 (radians, the rotation scale); its other five channels are identically zero. Axis and
# pivot are expressed in the parent's body (template) frame.
JOINT_FREE, JOINT_FIXED, JOINT_PRISMATIC, JOINT_REVOLUTE = 0, 1, 2, 3
JOINT_CHANNEL = {JOINT_PRISMATIC: 0, JOINT_REVOLUTE: 3}


def _geometry_without_autocast(fn):
    @wraps(fn)
    def wrapped(batch, *args, **kwargs):
        # .float() alone does not protect matmul/einsum inside an AMP forward.
        # Scope this to geometry; attention and learned layers retain caller AMP.
        # Read the device from whichever tensor the caller has: a purely deformable batch carries no
        # body_poses and must reach the wrapped function's own guard rather than crash here, while the
        # pose-only fixtures in tests/test_pose_ops.py carry no positions.
        ref = getattr(batch, "positions", None)
        with torch.autocast(device_type=(batch.body_poses if ref is None else ref).device.type,
                            enabled=False):
            return fn(batch, *args, **kwargs)
    return wrapped


def layout_from(parent, jtype, axis, pivot, body_idx):
    """Articulation over the bodies `body_idx` (ascending). parent / jtype: one layout row [K]
    (domain-pure batches share it); axis / pivot [B, K, 3]. Returns (parent [n] as a local index
    or -1, jtype [n], axis [B, n, 3], pivot [B, n, 3]); a jointed body's parent must be listed."""
    idx = [int(k) for k in body_idx.tolist()]
    local = {k: j for j, k in enumerate(idx)}
    jt = jtype[body_idx].long().cpu()
    par = []
    for j, k in enumerate(idx):
        p = int(parent[k])
        if int(jt[j]) == JOINT_FREE:
            if p != -1:
                raise ValueError(f"free body {k} names parent {p}")
            par.append(-1)
        elif p not in local:
            raise ValueError(f"jointed body {k}: parent {p} is not among the bodies {idx}")
        else:
            par.append(local[p])
    return torch.tensor(par, dtype=torch.long), jt, axis[:, body_idx], pivot[:, body_idx]


def joint_layout(batch, body_idx):
    """`layout_from` for a batch, or None when it carries no articulation."""
    if getattr(batch, "joint_type", None) is None:
        return None
    return layout_from(batch.body_parent[0], batch.joint_type[0], batch.joint_axis,
                       batch.joint_pivot, body_idx)


def pose_channel_mask(batch, rigid_idx):
    """[Kr, 6] mask of the pose channels each rigid body uses, or None when every body uses all six."""
    layout = joint_layout(batch, rigid_idx)
    if layout is None:
        return None
    jt = layout[1]
    mask = torch.zeros(len(jt), 6)
    mask[jt == JOINT_FREE] = 1.0
    for kind, channel in JOINT_CHANNEL.items():
        mask[jt == kind, channel] = 1.0
    return mask


def joint_features(batch, rigid_idx):
    """(joint type [Kr] long, [axis, pivot] [B, Kr, 6]) of the rigid bodies: the model's view of the
    articulation. A batch without one reads as all-free bodies."""
    if getattr(batch, "joint_type", None) is None:
        return (torch.full((len(rigid_idx),), JOINT_FREE, dtype=torch.long, device=rigid_idx.device),
                torch.zeros(batch.batch_size, len(rigid_idx), 6, device=batch.positions.device))
    return (batch.joint_type[0, rigid_idx],
            torch.cat([batch.joint_axis[:, rigid_idx], batch.joint_pivot[:, rigid_idx]], dim=-1).float())


def _child_order(parent):
    """Jointed bodies (local indices) with every parent before its children."""
    done = {j for j, p in enumerate(parent.tolist()) if p < 0}
    pending = [j for j, p in enumerate(parent.tolist()) if p >= 0]
    order = []
    while pending:
        ready = [j for j in pending if int(parent[j]) in done]
        if not ready:
            raise ValueError("articulation has a cycle")
        order += ready
        done.update(ready)
        pending = [j for j in pending if j not in done]
    return order


def compose_joints(R, t, R_anchor, t_anchor, raw, layout):
    """Forward kinematics: rebuild jointed bodies from their parents.

    R [B, F, n, 3, 3], t [B, F, n, 3]: per-frame poses with the FREE bodies integrated (entries of
    jointed bodies are replaced); R_anchor / t_anchor [B, n, 3, 3] / [B, n, 3]: the poses the window
    starts from; raw [B, F, n, 6]: raw pose deltas. A child keeps its anchor pose relative to its
    parent, moved by its joint coordinate accumulated since the anchor: along the axis (prismatic),
    about the axis through the pivot (revolute), not at all (fixed)."""
    from utils.pose_ops import so3_exp
    parent, jt, axis, pivot = layout
    F = raw.shape[1]
    Rs, ts = list(R.unbind(dim=2)), list(t.unbind(dim=2))
    for c in _child_order(parent):
        p = int(parent[c])
        RpT = R_anchor[:, p].transpose(-1, -2)
        R_rel = (RpT @ R_anchor[:, c])[:, None].expand(-1, F, -1, -1)
        t_rel = (RpT @ (t_anchor[:, c] - t_anchor[:, p])[..., None])[:, None, :, 0].expand(-1, F, -1)
        kind = int(jt[c])
        if kind in JOINT_CHANNEL:
            q = raw[:, :, c, JOINT_CHANNEL[kind]].cumsum(dim=1)[..., None]      # [B, F, 1]
            a = axis[:, c, None].to(raw.dtype)                                   # [B, 1, 3]
            if kind == JOINT_PRISMATIC:
                t_rel = t_rel + q * a
            else:
                Rq = so3_exp(q * a)                                              # [B, F, 3, 3]
                o = pivot[:, c, None].to(raw.dtype)
                R_rel = Rq @ R_rel
                t_rel = (Rq @ (t_rel - o)[..., None])[..., 0] + o
        Rs[c] = Rs[p] @ R_rel
        ts[c] = ts[p] + (Rs[p] @ t_rel[..., None])[..., 0]
    return torch.stack(Rs, dim=2), torch.stack(ts, dim=2)


def articulated_deltas(poses, d6, layout):
    """Pose channels of an articulated set: d6 [B, T-1, n, 6] (world deltas of the transitions of
    poses [B, T, n, 4, 4]) with each jointed body's row replaced by its joint coordinate change in
    its one channel, zeros elsewhere (a fixed body's row is all zero)."""
    from utils.pose_ops import so3_log
    parent, jt, axis, _ = layout
    d6 = d6.clone()
    for c in range(len(jt)):
        kind = int(jt[c])
        if kind == JOINT_FREE:
            continue
        p = int(parent[c])
        RpT = poses[:, :, p, :3, :3].transpose(-1, -2)
        a = axis[:, c, None].to(d6.dtype)                                        # [B, 1, 3]
        d6[:, :, c] = 0.0
        if kind == JOINT_PRISMATIC:
            t_rel = (RpT @ (poses[:, :, c, :3, 3] - poses[:, :, p, :3, 3])[..., None])[..., 0]
            d6[:, :, c, 0] = ((t_rel[:, 1:] - t_rel[:, :-1]) * a).sum(-1)
        elif kind == JOINT_REVOLUTE:
            R_rel = RpT @ poses[:, :, c, :3, :3]
            dR = R_rel[:, 1:] @ R_rel[:, :-1].transpose(-1, -2)
            B, Tm1 = dR.shape[:2]
            d6[:, :, c, 3] = (so3_log(dR.reshape(-1, 3, 3)).reshape(B, Tm1, 3) * a).sum(-1)
    return d6


@_geometry_without_autocast
def _integrate_and_write(batch, rigid_idx, raw, R0, t0, target, valid):
    """Integrate raw pose deltas [B, S, Kr, 6] from the anchor (R0 [B, Kr, 3, 3], t0
    [B, Kr, 3]) — jointed bodies through forward kinematics from their parents — and
    rewrite the rigid particle rows of `target` [B, S, N, 3] (normalized deltas) as
    template @ pose per frame. Returns (target, R_all, t_all) with R_all / t_all
    [B, S, Kr, ...] the pose reached AFTER each transition."""
    from utils.pose_ops import so3_exp
    B, S = raw.shape[0], raw.shape[1]
    Kr = int(rigid_idx.numel())
    R_prev, t_prev = R0, t0
    # Sequential pose compose (inherently ordered, tiny [B,Kr] matmuls) …
    R_list, t_list = [], []
    for f in range(S):
        dR = so3_exp(raw[:, f, :, 3:].reshape(-1, 3)).reshape(B, Kr, 3, 3)
        R_prev = dR @ R_prev
        t_prev = t_prev + raw[:, f, :, :3]
        R_list.append(R_prev)
        t_list.append(t_prev)
    R_all = torch.stack(R_list, dim=1)                 # [B, S, Kr, 3, 3]
    t_all = torch.stack(t_list, dim=1)                 # [B, S, Kr, 3]
    layout = joint_layout(batch, rigid_idx)
    if layout is not None:                             # jointed bodies follow their parents
        R_all, t_all = compose_joints(R_all, t_all, R0, t0, raw, layout)
    # … then ONE batched reconstruction + normalize + write per body.
    b_idx = torch.arange(B, device=target.device)[:, None]
    for j, kk in enumerate(rigid_idx.tolist()):
        n_t_all = batch.body_template_mask[:, kk].sum(dim=1)
        n_t = int(n_t_all[0])
        assert bool((n_t_all == n_t).all()), \
            f"body {kk}: template counts differ across batch ({n_t_all.tolist()})"
        tmpl = batch.body_templates[:, kk, :n_t].float()               # [B, P, 3]
        pts = (torch.einsum("bfij,bpj->bfpi", R_all[:, :, j], tmpl)
               + t_all[:, :, j, None])                                  # [B, S, P, 3]
        prev0 = (torch.einsum("bij,bpj->bpi", R0[:, j], tmpl)
                 + t0[:, j, None])
        prev = torch.cat([prev0[:, None], pts[:, :-1]], dim=1)
        vals = batch.delta_stats.normalize(pts - prev, clip=False).to(target.dtype)
        # Row layout varies per element (collated cloth batches mix garment
        # sizes, shifting the EE segments) — select rows per element, never
        # from element 0. nonzero() is ascending, matching template order the
        # same way the old boolean-mask write did.
        rows = []
        for e in range(B):
            r = ((batch.particle_body[e] == kk) & valid[e]).nonzero(as_tuple=True)[0]
            assert r.numel() == n_t, \
                f"body {kk} elem {e}: {r.numel()} rows != {n_t} template points"
            rows.append(r)
        idx = torch.stack(rows)                                        # [B, P]
        target[b_idx, :, idx] = vals.transpose(1, 2)                   # [B, P, S, 3]
    return target, R_all, t_all


def pose_native_inputs(batch, noisy_future, noisy_pose, noisy_ctx_pose=None, noisy_context=None):
    """Returns (patched_noisy_future, pack).

    noisy_pose [B, F, K_rigid, 6] normalized; rigid order = ascending body
    index with body_type == 1 (domain-pure batches make this well-defined).
    pack: rigid_idx [Kr] body indices; centers [B, F, Kr, 3] integrated
    translation per FUTURE frame (inclusive); anchor_centers [B, Kr, 3] the
    C-1 anchor translation; noisy_pose (as given).

    noisy_ctx_pose [B, C-1, Kr, 6] (optional; train.pose_noise_context) makes the CONTEXT a
    noised channel too — the uniform-window recipe of a context-PREDICTING model. The C-1
    context transitions are then integrated from the clean FRAME-0 body pose (jointed bodies
    through the same forward kinematics), their particle rows are rewritten in `noisy_context`
    the same self-consistent way (so a rigid body stays rigid at every noise level, which the
    retired train.noise_context did not), and the future anchor becomes the pose reached by
    that NOISY integration instead of the clean frame C-1 pose, so no clean context pose is
    handed to the model. Returns (patched_noisy_future, patched_noisy_context, pack) in that
    case; pack additionally carries ctx_centers [B, C-1, Kr, 3] (start of each context
    transition) and noisy_ctx_pose.
    """
    B, F, N, _ = noisy_future.shape
    rigid_idx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    Kr = int(rigid_idx.numel())
    assert noisy_pose.shape[2] == Kr, (noisy_pose.shape, Kr)
    C = batch.context_len
    valid = batch.get_valid_mask()
    raw = batch.pose_delta_stats.denormalize(noisy_pose.float()).float()  # [B, F, Kr, 6]

    ctx_patched = ctx_centers = None
    if noisy_ctx_pose is None:
        anchor = batch.body_poses[:, C - 1]                           # [B, K, 4, 4]
        anchor_R = anchor[:, rigid_idx, :3, :3].float()
        anchor_T = anchor[:, rigid_idx, :3, 3].float()
    else:
        assert noisy_context is not None, "noisy_ctx_pose needs noisy_context"
        assert noisy_ctx_pose.shape[1] == C - 1, (noisy_ctx_pose.shape, C)
        craw = batch.pose_delta_stats.denormalize(noisy_ctx_pose.float()).float()
        p0 = batch.body_poses[:, 0]                                   # [B, K, 4, 4]
        R0 = p0[:, rigid_idx, :3, :3].float()
        t0 = p0[:, rigid_idx, :3, 3].float()
        ctx_patched, cR, cT = _integrate_and_write(
            batch, rigid_idx, craw, R0, t0, noisy_context.clone(), valid)
        # RoPE centres are the START of each context transition (SHIFT convention).
        ctx_centers = torch.cat([t0[:, None], cT[:, :-1]], dim=1)     # [B, C-1, Kr, 3]
        anchor_R, anchor_T = cR[:, -1], cT[:, -1]

    patched, R_all, t_all = _integrate_and_write(
        batch, rigid_idx, raw, anchor_R, anchor_T, noisy_future.clone(), valid)
    pack = {"rigid_idx": rigid_idx, "centers": t_all,
            "anchor_centers": anchor_T, "noisy_pose": noisy_pose,
            "ctx_centers": ctx_centers, "noisy_ctx_pose": noisy_ctx_pose}
    if noisy_ctx_pose is None:
        return patched, pack
    return patched, ctx_patched, pack


@_geometry_without_autocast
def context_pose_deltas(batch, rigid_idx):
    """Clean GT context pose deltas [B, C-1, Kr, 6] (world [dt, so3_log(dR)]; a
    jointed body: its joint coordinate change), the pose-side analog of the particle
    context stream. Un-normalized."""
    return pose_sequence_deltas(batch.body_poses[:, :batch.context_len, rigid_idx],
                                joint_layout(batch, rigid_idx))


def pose_sequence_deltas(poses, layout=None):
    """Raw transitions of [B, T, K, 4, 4] poses, shared by loading and augmentation.

    Free bodies use world translation and left-relative rotation logarithms; children
    use parent-frame joint coordinates. Geometry stays FP32, including under AMP.
    """
    from utils.pose_ops import so3_log
    with torch.autocast(device_type=poses.device.type, enabled=False):
        poses = poses.float()
        dt = poses[:, 1:, :, :3, 3] - poses[:, :-1, :, :3, 3]
        dR = poses[:, 1:, :, :3, :3] @ poses[:, :-1, :, :3, :3].transpose(-1, -2)
        B, F, K = dt.shape[:3]
        d6 = torch.cat([dt, so3_log(dR.reshape(-1, 3, 3)).reshape(B, F, K, 3)], dim=-1)
        return d6 if layout is None else articulated_deltas(poses, d6, layout)


def future_pose_deltas(batch, poses=None):
    """The `pose_deltas_gt` target that `poses` (default batch.body_poses) implies, [B, F, K, 6].

    The construction of data/unified_dataset._augment_pose_channels: raw world-frame
    [dt, so3_log(dR)] over the transitions from the C-1 anchor to the end of the window, with a
    jointed body's row replaced by its joint coordinate change. Recomputing it is how an anchor
    perturbation reaches the target (train.light_pose_noise_anchor): move the anchor, recompute, and
    the first transition is the one that still lands on the recorded frame C.
    """
    p = batch.body_poses if poses is None else poses
    layout = joint_layout(batch, torch.arange(p.shape[2], device=p.device))
    return pose_sequence_deltas(p[:, batch.context_len - 1:], layout)


def pose_noise_scope(anchor, scope="auto"):
    """Resolve historical body scope without changing existing noise recipes."""
    if scope not in ("auto", "all", "passive"):
        raise ValueError("train.light_pose_noise_scope must be auto, all, or passive")
    scope = ("passive" if anchor else "all") if scope == "auto" else scope
    if anchor and scope != "passive":
        raise ValueError("train.light_pose_noise_anchor currently requires passive scope")
    return scope


@_geometry_without_autocast
def jitter_rigid_history(batch, std, anchor=False, *, target="restore", scope="auto",
                         noise_type="iid", generator=None):
    """train.light_pose_noise_std: light noise on the rigid bodies' history, as whole-body pose error.

    The rigid-body analog of train.light_context_noise_std, which spares rigid rows. A body's history
    transitions get Gaussian noise, `std` in normalized pose-delta units (per channel the half-range of
    batch.pose_delta_stats), integrated back from the anchor frame C-1. A jointed body takes its
    parent's correction and keeps its joint coordinate. Each moved body's particle rows move with it,
    so pose and particle history stay consistent (ISSUES.md ISSUE-001). Draws from the ambient
    generator unless one is supplied. Returns a new batch; `batch` is not modified.

    anchor=False: the anchor stays exact. Scope defaults to every rigid body; explicit passive
    scope allows history-only and moved-anchor experiments to corrupt the same bodies.

    anchor=True (train.light_pose_noise_anchor): the PASSIVE bodies' anchor pose is perturbed too, the
    history is integrated back from the perturbed anchor. With target="restore", the target is patched so the
    first future transition starts at the perturbed anchor and still lands on the recorded frame C.
    The particle target follows with no work, because Batch.future_deltas differences positions from
    the anchor. The model then has to return to the recorded trajectory instead of carrying the error
    forward. With target="carry", recorded world pose increments and joint-coordinate increments
    are retained, and the future poses/particles move consistently with the noisy anchor. A free
    body's rotation then has a constant RIGHT/local offset, not a constant left/world offset.
    Carry and restore consume the same random draws and use the same context and body scope.

    Passive-only is a restricted augmentation policy: default rollouts can also have actuator
    anchor error. Given actions are left recorded. Actuated descendants of a passive parent are
    conservatively rejected until a policy for all action representations is defined.

    noise_type="iid" (default) draws each history transition independently. "random_walk" draws
    increments at std/sqrt(C-1) and accumulates them, so the transition noise is correlated across
    the window and `std` still names its magnitude at the MOST RECENT transition; the anchor offset
    is reached by integrating those increments FORWARD from the exact oldest frame, as GNS pins its
    oldest input frame and accumulates position noise forward. That is the noise GNS
    reports as best (noise_utils.get_random_walk_noise_for_position_sequence: per-step draws at
    noise_std_last_step/sqrt(num_velocities), cumulative-summed, positions integrated from them).
    Within the walk, the rotation-vector increments are summed, a first-order stand-in for composing
    them on SO(3) that suits these light magnitudes; each resulting perturbation is then applied to
    the recorded rotations by composition, never by adding axis-angles.

    With anchor=True, noise_type="random_walk", scope="passive" and target="carry", this adapts the
    GNS recipe to whole-body rigid poses: per-particle noise cannot be used on a rigid body
    (ISSUES.md ISSUE-001), and the prediction target is a multi-frame delta rather than one
    acceleration.
    """
    import dataclasses
    from utils.pose_ops import so3_exp
    scope = pose_noise_scope(anchor, scope)
    if target not in ("restore", "carry") or (target == "carry" and not anchor):
        raise ValueError("train.light_pose_noise_target must be restore, or carry with anchor=true")
    if noise_type not in ("iid", "random_walk"):
        raise ValueError("train.light_pose_noise_type must be iid or random_walk")
    C = batch.context_len
    if std <= 0 or C < 2 or batch.body_type is None:
        return batch
    rigid_idx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    if rigid_idx.numel() == 0:
        return batch
    poses = batch.body_poses[:, :C, rigid_idx].float()                     # [B, C, Kr, 4, 4]
    R, t = poses[..., :3, :3], poses[..., :3, 3]
    B, Kr = t.shape[0], t.shape[2]
    layout = joint_layout(batch, rigid_idx)
    half = (batch.pose_delta_stats.scale.float() / 2).to(t.device)          # raw units per normalized unit

    # Which bodies may move. With anchor=True only passive ones do, and a jointed body moves exactly
    # when its parent does, since it takes its parent's correction rather than noise of its own.
    move = torch.ones(B, Kr, dtype=torch.bool, device=t.device)
    if scope == "passive":
        if batch.pose_action_mask is None:
            raise ValueError("passive pose noise needs pose_action_mask to tell passive "
                             "bodies from actuated ones")
        act = batch.pose_action_mask[:, rigid_idx].to(t.device)
        move = ~act
        if layout is not None:
            for c in _child_order(layout[0]):
                move[:, c] = move[:, int(layout[0][c])]
        if bool((move & act).any()):
            raise ValueError("passive pose noise: an actuated body is a jointed child of a passive "
                             "one; this action layout is not supported by the augmentation policy")

    if noise_type == "random_walk":                                         # correlated, GNS's choice
        steps = torch.randn(B, C - 1, Kr, 6, device=t.device, generator=generator)
        eta = (steps * (std * half / (C - 1) ** 0.5)).cumsum(dim=1) * move[:, None, :, None]
    else:
        eta = torch.randn(B, C - 1, Kr, 6, device=t.device, generator=generator) * std * half * move[:, None, :, None]
    Rn, tn = R.clone(), t.clone()
    if anchor and noise_type == "random_walk":
        # GNS pins the OLDEST input frame and accumulates forward, so integrate that way and let the
        # anchor fall out of the noisy increments. Deriving the anchor by SUMMING the angular
        # perturbations instead is wrong to FIRST order whenever the body rotates: each backward step
        # conjugates the accumulated error by its own rotation increment (D^T Exp(v) D = Exp(D^T v)),
        # so the sum is taken in the wrong frame and the oldest frame does not come back clean.
        for f in range(C - 1):
            dR = R[:, f + 1] @ R[:, f].transpose(-1, -2)
            dRn = so3_exp(eta[:, f, :, 3:].reshape(-1, 3)).reshape(B, Kr, 3, 3) @ dR
            Rn[:, f + 1] = dRn @ Rn[:, f]
            tn[:, f + 1] = tn[:, f] + (t[:, f + 1] - t[:, f]) + eta[:, f, :, :3]
    else:
        if anchor:                                                          # the anchor moves first
            eta0 = torch.randn(B, Kr, 6, device=t.device, generator=generator) * std * half * move[..., None]
            Rn[:, C - 1] = so3_exp(eta0[..., 3:].reshape(-1, 3)).reshape(B, Kr, 3, 3) @ R[:, C - 1]
            tn[:, C - 1] = t[:, C - 1] + eta0[..., :3]
        for f in range(C - 2, -1, -1):                                      # back from the anchor
            dR = R[:, f + 1] @ R[:, f].transpose(-1, -2)
            dRn = so3_exp(eta[:, f, :, 3:].reshape(-1, 3)).reshape(B, Kr, 3, 3) @ dR
            Rn[:, f] = dRn.transpose(-1, -2) @ Rn[:, f + 1]
            tn[:, f] = tn[:, f + 1] - (t[:, f + 1] - t[:, f] + eta[:, f, :, :3])
    Q = Rn @ R.transpose(-1, -2)                                            # world-frame correction per frame
    if layout is not None:                                                  # jointed bodies follow their parents
        parent = layout[0]
        for c in _child_order(parent):
            p = int(parent[c])
            Q[:, :, c] = Q[:, :, p]
            Rn[:, :, c] = Q[:, :, p] @ R[:, :, c]
            tn[:, :, c] = tn[:, :, p] + (Q[:, :, p] @ (t[:, :, c] - t[:, :, p])[..., None])[..., 0]
    # A body that may not move keeps its recorded pose exactly, not to within the integration's error.
    keep = ~move[:, None, :]
    Rn = torch.where(keep[..., None, None], R, Rn)
    tn = torch.where(keep[..., None], t, tn)

    H = C if anchor else C - 1                                              # history frames that move
    body_poses = batch.body_poses.clone()
    body_poses[:, :H, rigid_idx, :3, :3] = Rn[:, :H].to(body_poses.dtype)
    body_poses[:, :H, rigid_idx, :3, 3] = tn[:, :H].to(body_poses.dtype)
    history = batch.positions[:, :H].float()                               # [B, H, N, 3]
    moved, valid = history, batch.get_valid_mask()
    for j, kk in enumerate(rigid_idx.tolist()):
        rows = ((batch.particle_body == kk) & valid)[:, None, :, None] & move[:, j, None, None, None]
        placed = (torch.einsum("bfij,bfnj->bfni", Q[:, :H, j], history - t[:, :H, j, None])
                  + tn[:, :H, j, None])
        moved = torch.where(rows, placed, moved)
    positions = batch.positions.clone()
    positions[:, :H] = moved.to(positions.dtype)
    out = dataclasses.replace(batch, positions=positions, body_poses=body_poses)
    if anchor and target == "carry":
        # Preserve WORLD increments: R'_f = R_f (R_a^T R'_a), t'_f = t_f + t'_a - t_a.
        # A constant left/world rotation would conjugate every angular target instead.
        future = batch.body_poses[:, C:, rigid_idx].float()
        Rf, tf = future[..., :3, :3], future[..., :3, 3]
        local = R[:, -1].transpose(-1, -2) @ Rn[:, -1]
        Rfn = Rf @ local[:, None]
        tfn = tf + (tn[:, -1] - t[:, -1])[:, None]
        Qf = Rfn @ Rf.transpose(-1, -2)
        if layout is not None:
            for c in _child_order(layout[0]):
                p = int(layout[0][c])
                Qf[:, :, c] = Qf[:, :, p]
                Rfn[:, :, c] = Qf[:, :, p] @ Rf[:, :, c]
                tfn[:, :, c] = tfn[:, :, p] + (Qf[:, :, p] @ (tf[:, :, c] - tf[:, :, p])[..., None])[..., 0]
        out.body_poses[:, C:, rigid_idx, :3, :3] = torch.where(keep[..., None, None], Rf, Rfn).to(body_poses.dtype)
        out.body_poses[:, C:, rigid_idx, :3, 3] = torch.where(keep[..., None], tf, tfn).to(body_poses.dtype)
        recorded = batch.positions[:, C:].float()
        moved = recorded
        for j, kk in enumerate(rigid_idx.tolist()):
            rows = ((batch.particle_body == kk) & valid)[:, None, :, None] & move[:, j, None, None, None]
            placed = torch.einsum("bfij,bfnj->bfni", Qf[:, :, j], recorded - tf[:, :, j, None]) + tfn[:, :, j, None]
            moved = torch.where(rows, placed, moved)
        out.positions[:, C:] = moved.to(positions.dtype)
    elif anchor and batch.pose_deltas_gt is not None:
        # The particle target moves with the anchor on its own (Batch.future_deltas differences
        # positions); the pose target is a stored tensor and has to be patched. Only the FIRST future
        # transition of a moved free body changes: later transitions run between recorded poses, and a
        # jointed body keeps its joint coordinate because it moved with its parent. Writing just those
        # entries leaves the loader's own target bitwise intact everywhere else, where recomputing it
        # would differ by float epsilon.
        free = move if layout is None else move & (layout[1].to(move.device) == JOINT_FREE)[None, :]
        if bool(free.any()):
            sel = torch.zeros(B, batch.pose_deltas_gt.shape[2], dtype=torch.bool, device=free.device)
            sel[:, rigid_idx] = free
            pdg = batch.pose_deltas_gt.clone()
            pdg[:, 0] = torch.where(sel[..., None], future_pose_deltas(out)[:, 0].to(pdg.dtype), pdg[:, 0])
            out = dataclasses.replace(out, pose_deltas_gt=pdg)
    return out


def body_object_index(batch, canonical_obj_ids, rigid_idx):
    """For each rigid body, the CANONICAL object index (contiguous block) its
    particles live in. canonical_obj_ids [B, N] (UNetWrapper._object_ids).
    Returns a python list of length Kr. Element 0 is representative (domain-
    pure batches: the body/object layout is identical across the batch)."""
    pb = batch.particle_body[0]
    oid = canonical_obj_ids[0]
    out = []
    for kk in rigid_idx.tolist():
        objs = oid[pb == kk].unique()
        assert objs.numel() == 1, f"rigid body {kk} spans objects {objs.tolist()}"
        out.append(int(objs[0]))
    return out


def clamp_actuated_pose(batch, x0pose):
    """Opt-in: replace the predicted (normalized) pose deltas of ACTUATED rigid bodies
    with their given ``pose_actions`` (same currency, same normalization as the
    action skip input), so given bodies (grippers, zero-motion fixtures) are
    materialized exactly from their input instead of the network's copy of it.
    Passive bodies are untouched. Returns a new tensor; ``x0pose`` is not modified."""
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    act = batch.pose_action_mask[:, ridx]                                   # [B, Kr]
    given = batch.pose_delta_stats.normalize(batch.pose_actions[:, :, ridx], clip=False).to(x0pose.dtype)
    return torch.where(act[:, None, :, None], given, x0pose)


@_geometry_without_autocast
def materialize_pose_native(batch, x0p, x0pose, *, stabilize_joints=False):
    """Tier-2 v2 output materialization: deformable rows from the particle x0,
    rigid rows = template @ integrate(denorm(x0_pose), anchor). Returns
    (pred_pos [B,F,N,3], R [B,F,Kr,3,3], t [B,F,Kr,3], rigid_idx).

    Autoregressive callers stabilize jointed rotations before carrying them into
    another window. Tiny non-orthogonality otherwise compounds when FK uses a
    parent's transpose as its inverse, especially along deep revolute chains.
    Training and ordinary single-window materialization keep their arithmetic.
    """
    from utils.pose_ops import so3_exp
    C = batch.context_len
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    raw = batch.pose_delta_stats.denormalize(x0pose.float()).float()
    B, F, Kr, _ = raw.shape
    R = batch.body_poses[:, C - 1, ridx, :3, :3].float().clone()
    t = batch.body_poses[:, C - 1, ridx, :3, 3].float().clone()
    R0, t0 = R, t
    layout = joint_layout(batch, ridx)
    if x0p is not None:
        pred_pos = batch.predictions_to_positions(
            batch.prediction_stats.denormalize(x0p))
    else:
        pred_pos = torch.zeros_like(batch.future)
    Rs = torch.zeros(B, F, Kr, 3, 3, device=raw.device)
    Ts = torch.zeros(B, F, Kr, 3, device=raw.device)
    valid = batch.get_valid_mask()
    # Row layout is resolved PER ELEMENT, not from element 0: collated cloth batches can
    # mix garment sizes, which shifts every EE segment (the same reason
    # utils/pose_native.py selects per element on the input side). Hoisted out of the
    # frame loop — it does not depend on f.
    b_idx = torch.arange(B, device=pred_pos.device)[:, None]
    rows, tmpls = [], []
    for kk in ridx.tolist():
        n_all = batch.body_template_mask[:, kk].sum(dim=1)
        n_t = int(n_all[0])
        assert bool((n_all == n_t).all()), \
            f"body {kk}: template counts differ across batch ({n_all.tolist()})"
        r = [((batch.particle_body[e] == kk) & valid[e]).nonzero(as_tuple=True)[0]
             for e in range(B)]
        for e, rr in enumerate(r):
            assert rr.numel() == n_t, \
                f"body {kk} elem {e}: {rr.numel()} rows != {n_t} template points"
        rows.append(torch.stack(r))                          # [B, P]
        tmpls.append(batch.body_templates[:, kk, :n_t].float())
    for f in range(F):
        dR = so3_exp(raw[:, f, :, 3:].reshape(-1, 3)).reshape(B, Kr, 3, 3)
        R = dR @ R
        t = t + raw[:, f, :, :3]
        Rs[:, f] = R
        Ts[:, f] = t
        if layout is None:
            for j in range(len(ridx)):
                pred_pos[b_idx, f, rows[j]] = (
                    torch.einsum("bij,bpj->bpi", R[:, j], tmpls[j]) + t[:, j, None])
    if layout is not None:                             # jointed bodies follow their parents
        Rs, Ts = compose_joints(Rs, Ts, R0, t0, raw, layout)
        if stabilize_joints:
            from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
            # quaternion_to_matrix normalizes the quaternion. Rebuild particles
            # below from these same rotations, keeping the carried pose track and
            # the next window's particle context consistent.
            Rs = quaternion_to_matrix(matrix_to_quaternion(Rs))
        for f in range(F):
            for j in range(len(ridx)):
                pred_pos[b_idx, f, rows[j]] = (
                    torch.einsum("bij,bpj->bpi", Rs[:, f, j], tmpls[j]) + Ts[:, f, j, None])
    return pred_pos, Rs, Ts, ridx
