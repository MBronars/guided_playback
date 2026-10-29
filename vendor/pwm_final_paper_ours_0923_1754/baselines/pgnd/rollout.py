#!/usr/bin/env python3
"""Standalone PGND inference on mpm_drag episodes -- torch only, no warp, no PGND checkout.

Rolls a trained PGND checkpoint forward over a whole episode of this repo's mpm_drag release:
the model predicts every frame from the first few context frames, while PGND's gripper sphere
follows the driven patch's recorded motion (what the baseline was trained with).

    python baselines/pgnd/rollout.py --out /tmp/pgnd_rollouts --n 6

which loads the mpm_drag run's latest checkpoint and rolls it out on that release's test split.
Other checkpoints: --step best | final | 40000, --task three | dex, or --ckpt with a path.

Outputs one npz per episode with `pred` and `gt` points in world metres, the gripper track, and
the per-frame mean position error in mm. `rollout_points()` is the entry point for other data:
give it points and a gripper track, and it returns the predicted points.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.pgnd.network import PGNDModel  # noqa: E402
from baselines.pgnd.sim_torch import step  # noqa: E402

# PGND was trained with full float32 matmuls; newer torch defaults to TF32 on Ampere and later,
# which moves a rollout by millimetres within a few steps.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

# PGND runs live beside the GNS baselines; each run keeps latest.pt / best.pt / final.pt and a
# model-<step>.pt every 10,000 steps in <run>/train/ckpt, with its config at <run>/train/hydra.yaml.
RUN_ROOT = Path('/data/group_data/katefgroup/runs_gns_mbronars')
RUNS = {'mpm_drag': RUN_ROOT / 'pgnd_mpm_drag', 'three': RUN_ROOT / 'pgnd_three', 'dex': RUN_ROOT / 'pgnd_dex'}
DEFAULT_SPLIT = ('/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/'
                 'pwm-data-v1/mpm_drag/test')

FLIP = np.array([1., -1., -1.], np.float32)          # world (z up) -> PGND source axes
AXIS = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # PGND's preprocess axis map


def place_in_box(points_world, cfg, floor_points=None, floor_margin=0.0):
    """World metres -> the simulator's box, as PGND's preprocess does.

    points_world [T, N, 3]; returns (points_sim [T, N, 3], scale, translation) so that
    world -> sim is  ((p * FLIP) @ AXIS.T) * scale + translation.
    """
    scale = float(cfg.sim.preprocess_scale)
    dx = float(cfg.sim.num_grids[3])
    box = np.array([cfg.sim.num_grids[i] * dx for i in range(3)])
    lo = dx * (float(cfg.model.clip_bound) + 0.5)
    p = np.einsum('tnj,jk->tnk', points_world * FLIP, AXIS.T) * scale
    floor = None
    if floor_points is not None:
        f = np.einsum('tnj,jk->tnk', floor_points * FLIP, AXIS.T) * scale
        floor = min(float(f[:, :, 1].min()), float(p[:, :, 1].min())) - floor_margin * scale
    if bool(cfg.sim.preprocess_with_table):
        t = np.array([box[0] / 2 - (p[:, :, 0].max() + p[:, :, 0].min()) / 2,
                      lo + 1e-5 - (p[:, :, 1].min() if floor is None else floor),
                      box[2] / 2 - (p[:, :, 2].max() + p[:, :, 2].min()) / 2])
    else:
        t = np.array([box[i] / 2 - (p[:, :, i].max() + p[:, :, i].min()) / 2 for i in range(3)])
    return (p + t).astype(np.float32), scale, t


def sphere_track(centres_world, closed, cfg, translation, dt_frame):
    """The 15-number gripper state per frame PGND's sim reads: centre, velocity, identity
    rotation, zero angular velocity, radius, open flag. `centres_world` [T, 3] is the sphere
    centre in world metres, `closed` [T] whether it grips that frame."""
    scale = float(cfg.sim.preprocess_scale)
    c = np.einsum('tj,jk->tk', centres_world * FLIP, AXIS.T) * scale + translation
    v = np.zeros_like(c)
    v[:-1] = (c[1:] - c[:-1]) / dt_frame      # forward difference, as PGND's preprocess
    v[-1] = v[-2]
    g = np.zeros((len(c), 15), np.float32)
    g[:, :3], g[:, 3:6], g[:, 6] = c, v, 1.0
    g[:, 13] = float(cfg.model.gripper_radius)
    g[:, 14] = np.where(closed, 0.0, 1.0)     # PGND: < 0.5 is closed
    return g


def rollout_points(model, cfg, points_sim, grippers, device, n_history=None):
    """points_sim [T, N, 3] (only the first n_history + 1 frames are read), grippers [T, 15].
    Returns the predicted points [T - n_history - 1, N, 3] in the simulator's frame."""
    n_history = cfg.sim.n_history if n_history is None else n_history
    dt = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else float(cfg.sim.dt)
    xs = torch.as_tensor(points_sim, dtype=torch.float32, device=device)
    g = torch.as_tensor(grippers, dtype=torch.float32, device=device)
    T, N, _ = xs.shape
    vs = torch.zeros_like(xs)
    vs[1:] = (xs[1:] - xs[:-1]) / dt          # backward difference, as PGND's dataset
    vs[0] = vs[1]
    f0 = n_history
    x, v = xs[f0][None], vs[f0][None]
    x_his = torch.cat([xs[f0 - n_history + i] for i in range(n_history)], -1)[None]
    v_his = torch.cat([vs[f0 - n_history + i] for i in range(n_history)], -1)[None]
    enabled = torch.ones(1, N, dtype=torch.bool, device=device)
    radius = float(cfg.model.gripper_radius)
    out = []
    with torch.no_grad():
        for s in range(f0 + 1, T):
            gi = g[s][None]
            x_in = x.clone()
            x, v = step(model, x, v, x_his, v_his, enabled, cfg,
                        gripper=(gi[:, :3], gi[:, 3:6], gi[:, 10:13], gi[:, 13], gi[:, 14]))
            # points inside the closed sphere move with it (PGND's gripper forcing)
            rel = x_in - gi[:, None, :3]
            grabbed = (rel.norm(dim=-1) < radius) & (gi[:, 14:15] < .5)
            x = torch.where(grabbed[..., None], x_in + dt * gi[:, None, 3:6], x)
            v = torch.where(grabbed[..., None], gi[:, None, 3:6].expand_as(v), v)
            out.append(x[0].cpu().clone())
            if n_history > 0:
                x_his = torch.cat([x_his.reshape(1, N, -1, 3)[:, :, 1:], x[:, :, None]], 2).reshape(1, N, -1)
                v_his = torch.cat([v_his.reshape(1, N, -1, 3)[:, :, 1:], v[:, :, None]], 2).reshape(1, N, -1)
    return torch.stack(out).numpy()


def drag_episodes(root, n, seed):
    """mpm_drag episodes from this repo's own loader: world points, patch mask, active flags."""
    from data.indexed_episodes import IndexedEpisodeDataset
    ds = IndexedEpisodeDataset(root, domain='mpm_drag', context_len=1, future_len=-1,
                               training=False, shuffle=False, raw=True)
    picks = sorted(random.Random(seed).sample(range(len(ds.records)), min(n, len(ds.records))))
    for i in picks:
        rec = ds.records[i]
        b = ds.window(rec, 0)
        sel = b.action_mask[0].numpy().astype(bool)
        yield rec['episode_id'], b.positions[0].numpy(), sel, b.action_active[0].numpy()[:, sel].any(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', default='mpm_drag', choices=sorted(RUNS), help='which PGND run to load')
    ap.add_argument('--step', default='latest', help="checkpoint in the run's ckpt dir: latest, best, final, or a step number")
    ap.add_argument('--ckpt', default=None, help='overrides --task/--step')
    ap.add_argument('--cfg', default=None, help="the run's hydra.yaml (default: next to the checkpoint)")
    ap.add_argument('--root', default=None, help='mpm_drag split directory (its INDEX.json)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    run = RUNS[a.task]
    ckpt_path = Path(a.ckpt) if a.ckpt else run / 'train/ckpt' / (
        a.step if a.step.endswith('.pt') else
        f'{a.step}.pt' if a.step in ('latest', 'best', 'final') else f'model-{int(a.step)}.pt')
    cfg_path = Path(a.cfg) if a.cfg else ckpt_path.parent.parent / 'hydra.yaml'
    root = a.root or DEFAULT_SPLIT
    for label, p in (('checkpoint', ckpt_path), ('config', cfg_path), ('episodes', Path(root))):
        if not p.exists():
            raise SystemExit(f'{label} not found: {p}')

    cfg = OmegaConf.load(cfg_path)
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    model = PGNDModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['material'])
    model.eval()
    print(f'{ckpt_path}: iteration {ckpt.get("iteration")}, {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters')

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dt_frame = eval(cfg.sim.dt) if isinstance(cfg.sim.dt, str) else float(cfg.sim.dt)
    for k, (episode_id, pos, sel, active) in enumerate(drag_episodes(root, a.n, a.seed)):
        centres = pos[:, sel].mean(1)
        centres = np.concatenate([centres[:1], centres[:-1]])          # the sphere sits at the step's start
        closed = np.concatenate([[False], active])
        sim, scale, translation = place_in_box(pos, cfg)
        grippers = sphere_track(centres, closed, cfg, translation, dt_frame)
        pred_sim = rollout_points(model, cfg, sim, grippers, device)
        n_hist = cfg.sim.n_history
        gt_sim = sim[n_hist + 1:]
        err_mm = np.linalg.norm(pred_sim - gt_sim, axis=-1).mean(-1) / scale * 1000
        inv = lambda p: np.einsum('tnj,jk->tnk', (p - translation) / scale, np.linalg.inv(AXIS).T) * FLIP
        np.savez(out / f'ep{k:02d}.npz', pred=inv(pred_sim).astype(np.float32), gt=inv(gt_sim).astype(np.float32),
                 pred_sim=pred_sim, gt_sim=gt_sim, gripper=grippers[n_hist + 1:], err_mm=err_mm,
                 episode=episode_id, scale=scale, translation=translation)
        print(f'  ep{k:02d} {episode_id[:24]}...: {len(gt_sim)} steps, {pos.shape[1]} points, '
              f'mean error {err_mm.mean():.1f} mm, final {err_mm[-1]:.1f} mm', flush=True)
    print(f'wrote rollouts -> {out}')


if __name__ == '__main__':
    main()
