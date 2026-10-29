"""PGND's simulation step in plain torch (the training code runs it as warp kernels).

One step, matching pgnd/sim/model.py grid_op_batch + g2p_batch:

  1. the network's predicted grid velocity field is the grid state;
  2. grid_op: the outer `bound` cells reflect inward-facing velocity with Coulomb friction, and
     nodes within a closed gripper sphere take the gripper's velocity (linear + w x r);
  3. g2p: every point reads its velocity from the grid with quadratic B-spline weights, then
     x <- clamp(x + dt * v) inside the box.

Everything is in the simulator's frame (the unit box scaled by num_grids * dx).
"""
import torch


def grid_op(grid_v, dx, bound, friction, gripper=None):
    """grid_v [B, X, Y, Z, 3] -> the same field after walls and the gripper sphere.

    gripper: None, or (centre [B, 3], velocity [B, 3], angular velocity [B, 3], radius [B], open [B]).
    """
    B, X, Y, Z, _ = grid_v.shape
    v = grid_v.clone()
    sizes = (X, Y, Z)
    # same order as the warp kernel: the three low walls, then the three high walls, each seeing
    # the velocity the previous wall left behind
    walls = [(axis, -1.0) for axis in range(3)] + [(axis, 1.0) for axis in range(3)]
    for axis, side in walls:
        n = sizes[axis]
        idx = torch.arange(n, device=v.device)
        sel = idx < bound if side < 0 else idx > n - bound
        shape = [1, 1, 1, 1]
        shape[axis + 1] = n
        comp = v[..., axis]
        wall = sel.reshape(shape[1:])[None].expand(B, X, Y, Z)   # broadcast along the other two axes
        hit = wall & ((comp < 0) if side < 0 else (comp > 0))
        if not hit.any():
            continue
        impulse = torch.abs(comp)                                # |v_axis| against the wall
        stopped = v.clone()
        stopped[..., axis] = torch.where(hit, torch.zeros_like(comp), comp)
        speed = stopped.norm(dim=-1, keepdim=True)
        direction = stopped / (speed + 1e-10)
        fr = -friction * impulse.unsqueeze(-1) * direction
        scale = torch.clamp(speed / (fr.norm(dim=-1, keepdim=True) + 1e-10), max=1.0)
        v = torch.where(hit.unsqueeze(-1), stopped + fr * scale, v)
    if gripper is not None:
        centre, g_v, g_w, radius, open_flag = gripper
        grid = torch.stack(torch.meshgrid(
            *[torch.arange(n, device=v.device, dtype=v.dtype) * dx for n in sizes], indexing='ij'), -1)
        rel = grid[None] - centre[:, None, None, None]               # [B, X, Y, Z, 3]
        inside = rel.norm(dim=-1) < radius[:, None, None, None]
        inside = inside & (open_flag < .5)[:, None, None, None]
        speed = g_w.norm(dim=-1, keepdim=True)
        axis = g_w / (speed + 1e-10)
        perp = rel - (axis[:, None, None, None] * rel).sum(-1, keepdim=True) * axis[:, None, None, None]
        moving = torch.cross(g_w[:, None, None, None].expand_as(perp), perp, dim=-1) + g_v[:, None, None, None]
        v = torch.where(inside.unsqueeze(-1), moving, v)
    return v


def g2p(x, grid_v, dx, dt, clip_bound, num_grids):
    """Points [B, N, 3] read the grid velocity and advance one step -> (x_next, v_next)."""
    B, N, _ = x.shape
    p = x / dx
    base = (p - 0.5).to(torch.int32)                                  # same truncation as the warp kernel
    fx = p - base.to(p.dtype)
    w = torch.stack([0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2], dim=0)  # [3, B, N, 3]
    v = torch.zeros_like(x)
    b_idx = torch.arange(B, device=x.device)[:, None].expand(B, N)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                weight = w[i, ..., 0] * w[j, ..., 1] * w[k, ..., 2]   # [B, N]
                gx, gy, gz = base[..., 0] + i, base[..., 1] + j, base[..., 2] + k
                v = v + weight[..., None] * grid_v[b_idx, gx.long(), gy.long(), gz.long()]
    bound = clip_bound * dx + 1e-7
    x_next = x + dt * v
    hi = torch.tensor([n * dx for n in num_grids], device=x.device, dtype=x.dtype)
    x_next = torch.max(torch.min(x_next, hi - bound), bound.expand_as(x_next) if bound.dim() else bound + 0 * x_next)
    return x_next, v


def step(model, x, v, x_his, v_his, enabled, cfg, gripper=None, friction=None):
    """One PGND step: predict the grid field, apply walls and gripper, move the points."""
    num_grids, dx = list(cfg.sim.num_grids[:3]), float(cfg.sim.num_grids[3])
    dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else float(cfg.sim.dt)
    fr = float(cfg.model.friction.value if friction is None else friction)
    pred = model(x, v, x_his, v_his, enabled)                         # [B, X, Y, Z, 3]
    grid_v = grid_op(pred, dx, int(cfg.sim.bound), fr, gripper)
    clip = torch.as_tensor(float(cfg.model.clip_bound), device=x.device, dtype=x.dtype)
    return g2p(x, grid_v, dx, dt, clip, num_grids)
