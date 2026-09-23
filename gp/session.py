"""One guided run from a loaded adapter and a namespace of knobs (the CLI's argparse namespace or a server request)."""
import json, os, time
import numpy as np, torch
from .objectives import Objective, centroid, wall_between
from . import guide as G, scene_edit

DEFAULTS = dict(domain=None, task="push", episodes="0,1,2,3", transitions=60, steps=10, keep=3, offset="0.15,0,0", random_offset=None, own_goal=None, goal_dist_m=None,
                iterations=10, lr=0.2, lr_final=None, optimizer="adam", shell=False, seed=0, fix_fps=False, churn=False, t_freeze=0.0, joint_limits=None,
                close_fingers=False, closed_sep_mm=27.0, hand_start=None, hand_jitter_mm=30.0, hand_z_mm=105.0, hand_bearing_deg=60.0, hand_bearing_jitter_deg=15.0,
                hand_dist_mm=250.0, hand_dist_jitter_mm=30.0, hand_dz_mm=85.0, hand_tilt_deg=0.0, wall_height=0.10, wall_thickness=0.02, wall_width=0.30,
                clearance_mm=5.0, box_weight=5.0, floor_weight=0.0, eager=False, name=None, runs=None, desc="", adapter="paper", ckpt="", sampler="stream")


class Args(dict):
    __getattr__ = dict.__getitem__


def load_adapter(A):
    from .adapters import get_adapter
    ad = get_adapter(A.adapter).load(A.ckpt, torch.device("cuda"), compile=not A.eager)
    if A.joint_limits:
        ad.joint_limits = tuple(float(v) / 1000.0 for v in A.joint_limits.split(","))
    return ad


def guided_run(ad, A):
    DEV = torch.device("cuda")
    eps = [int(v) for v in A.episodes.split(",")]; B = len(eps)
    batch = ad.episodes(eps, A.transitions, domain=A.domain)
    print(f"{A.adapter} step {ad.step} domain {ad.domain}: {B} members, {A.transitions} transitions, {batch.positions.shape[2]} points", flush=True)
    if A.close_fingers:
        batch = scene_edit.close_fingers(batch, A.closed_sep_mm)
    mask = ad.object_mask(batch)
    c0 = centroid(batch.positions[:, 0], mask)
    if A.own_goal:
        tags = {t["loader"]: t for t in json.load(open(A.own_goal))}
        R = "/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/rl_push/test"
        goals = []
        for i in eps:
            t = tags[i]; o = np.load(f"{R}/{t['file']}", allow_pickle=True)
            goals.append(o["goal_pos"][t["e"]] + np.array([-0.45, 0.0, 0.0]))
            print(f"member ep{i}: {t['cell']} push_dir {t['push_dir']:.0f} reposition {t['repo']:.0f} goal_dist {t['goal_dist']:.2f} m", flush=True)
        target = torch.tensor(np.stack(goals), device=DEV, dtype=torch.float32); target[:, 2] = c0[:, 2]
        if A.goal_dist_m is not None:
            v = target - c0; target = c0 + v / v.norm(dim=-1, keepdim=True) * A.goal_dist_m
        off = target - c0
    elif A.random_offset:
        rmin, rmax = [float(v) for v in A.random_offset.split(",")]
        gr = torch.Generator().manual_seed(A.seed + 7)
        ang = torch.rand(B, generator=gr) * 2 * np.pi; rad = rmin + (rmax - rmin) * torch.rand(B, generator=gr)
        off = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang), torch.zeros(B)], 1).to(DEV); target = c0 + off
    else:
        off = torch.tensor([float(v) for v in A.offset.split(",")], device=DEV)[None].expand(B, -1).clone()
        if A.task in ("push", "wall", "move"): off[:, 2] = 0
        target = c0 + off
    print("targets (mm):", [[round(float(v) * 1e3) for v in t] for t in target], flush=True)
    if A.hand_start == "natural":
        batch = scene_edit.hand_start_natural(batch, target, mask, A.seed, A.hand_bearing_deg, A.hand_bearing_jitter_deg, A.hand_dist_mm, A.hand_dist_jitter_mm, A.hand_dz_mm, A.hand_tilt_deg)
    elif A.hand_start == "mid":
        batch = scene_edit.hand_start_mid(batch, target, mask, A.seed, A.hand_jitter_mm, A.hand_z_mm)
    boxes = []
    if A.task == "wall":
        if A.random_offset: raise SystemExit("--task wall needs one shared --offset")
        boxes = [wall_between(c0[0], off[0], A.wall_height, A.wall_thickness, A.wall_width)]
        print(f"wall centre {boxes[0][0].tolist()} half {boxes[0][1].tolist()} m", flush=True)
    obj = Objective(target, mask, batch.get_valid_mask(), boxes, A.clearance_mm / 1000, A.box_weight, A.floor_weight)
    ad.sampler_mode = getattr(A, 'sampler', None) or (A.get('sampler') if isinstance(A, dict) else None) or 'stream'
    state = ad.prepare(batch, A.transitions, A.steps, A.keep, A.seed, A.churn, A.t_freeze, A.fix_fps)
    print("state:", state.info, flush=True)
    res = G.run(ad, batch, state, obj, A.iterations, A.lr, A.lr_final, A.optimizer, A.shell)
    out = os.path.join(A.runs, ad.domain, A.name)
    meta = dict((dict(A) if isinstance(A, dict) else vars(A)), offsets=off.cpu().numpy(), model_calls=ad.model_calls(state), sampler=state.info.get("sampler"), state_info={k: v for k, v in state.info.items() if isinstance(v, (int, float, str))},
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    G.save(out, ad, batch, res, obj, meta)
    print("best mm", res["best_mm"], "wrote", out, flush=True)
    return out, res
