"""Guided planning on unified checkpoints: choose the actuator trajectory so that a
passive rigid body reaches a target, one JSON receipt per run.

    PYTHONPATH=. python guidance/run_guidance.py --checkpoint runs/x/checkpoints/final.pt \
        --domain pushing --planner cem --objective offset:0.05,0,0 --max_batches 8 \
        --horizon 16 --stride 16 --out reports/guidance/pushing_cem.json

Objectives: `gt_final` (the passive body's ground-truth pose at the horizon; the
in-distribution check, the batch's own plan already reaches it) or
`offset:dx,dy,dz` (metres from the anchor-frame position; the planning case).
Phantom-motion and floor penalties are added with --phantom_weight/--floor_weight.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from guidance import objectives as objlib
from guidance import planners
from guidance.horizon import receding_horizon
from guidance.plan import actuated_bodies, plan_from_batch
from guidance.planners import window_nfe


def passive_body(batch, requested: int | None) -> int:
    rigid = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0].tolist()
    passive = [b for b in rigid if not bool(batch.pose_action_mask[0, b])]
    if requested is not None:
        if requested not in passive:
            raise ValueError(f"--body {requested} is not a passive rigid body (passive: {passive})")
        return requested
    if len(passive) != 1:
        raise ValueError(f"{len(passive)} passive rigid bodies {passive}; pass --body")
    return passive[0]


def build_objective(args, batch, body, horizon):
    C = batch.context_len
    if args.objective == "gt_final":
        target_t = batch.body_poses[:, C + horizon - 1, body, :3, 3].float()
        target_R = batch.body_poses[:, C + horizon - 1, body, :3, :3].float()
    elif args.objective.startswith("offset:"):
        dx, dy, dz = (float(v) for v in args.objective[len("offset:"):].split(","))
        target_t = batch.body_poses[:, C - 1, body, :3, 3].float() + torch.tensor([dx, dy, dz], device=batch.device)
        target_R = None
    else:
        raise ValueError(f"unknown objective {args.objective!r}")
    terms = [(1.0, objlib.body_target(body, target_t, target_R, rotation_weight=args.rotation_weight))]
    if args.phantom_weight > 0:
        terms.append((args.phantom_weight, objlib.phantom_motion(body, args.contact_radius)))
    if args.floor_weight > 0:
        terms.append((args.floor_weight, objlib.floor_penetration(args.floor_z)))
    return objlib.weighted(*terms), target_t, target_R


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--domain", required=True, choices=["pushing", "pickup", "cloth"])
    p.add_argument("--max_batches", type=int, default=8)
    p.add_argument("--planner", default="best_of_k", choices=sorted(planners.PLANNERS))
    p.add_argument("--planner_kwargs", default="{}", help="JSON dict passed to the planner")
    p.add_argument("--objective", default="gt_final")
    p.add_argument("--body", type=int, default=None, help="passive rigid body index (default: the only one)")
    p.add_argument("--rotation_weight", type=float, default=0.0)
    p.add_argument("--phantom_weight", type=float, default=100.0)
    p.add_argument("--contact_radius", type=float, default=0.02)
    p.add_argument("--floor_weight", type=float, default=0.0)
    p.add_argument("--floor_z", type=float, default=0.0)
    p.add_argument("--horizon", type=int, default=None, help="frames to plan (default: one training window)")
    p.add_argument("--stride", type=int, default=None, help="frames committed per window (default: horizon)")
    p.add_argument("--steps", type=int, default=None, help="sampler steps (default: checkpoint recipe)")
    p.add_argument("--context_len", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--dump_dir", default=None, help="write per-episode NPZ trajectories (GT, planned, target) here")
    args = p.parse_args()

    from data.unified_dataset import DOMAIN_FUTURE_LEN
    from inference import load_model_and_diffusion
    from scripts.unified.eval_unified import evaluation_dataset
    from utils.reproducibility import code_revision, file_sha256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, cfg, ckpt_steps = load_model_and_diffusion(args.checkpoint, device)
    if float(cfg.train.get("action_dropout", 0)) == 1.0:
        raise ValueError("This CLI plans supplied actuator actions; use guidance.run_unconditional for action_dropout=1 checkpoints")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trained_f = int(cfg.data.get("future_len", -1) or -1)
    F_train = trained_f if trained_f > 0 else DOMAIN_FUTURE_LEN[args.domain]
    C = args.context_len or int(cfg.data.context_len)
    horizon = args.horizon or F_train
    stride = args.stride or horizon
    steps = args.steps or int(ckpt_steps)
    schedule, uscale = cfg.get("df_schedule", "pyramid"), float(cfg.get("df_uncertainty_scale", 1.0))
    ds = evaluation_dataset(cfg, args.domain, C, F_train)
    ds.emit_sidecar = True
    ds.articulated = bool(getattr(model.cfg, "articulated_bodies", False))
    ds._pose_stats()
    ds.future_len = 10_000            # whole episodes; windows are cut by the horizon driver
    planner = planners.PLANNERS[args.planner]
    kwargs = json.loads(args.planner_kwargs)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    episodes = []
    for i, batch in enumerate(ds):
        if i >= args.max_batches:
            break
        batch = batch.to(device)
        if batch.pose_deltas_gt is None:
            raise ValueError("guidance needs pose-native batches (rigid bodies with pose channels)")
        H = min(horizon, batch.positions.shape[1] - C)
        body = passive_body(batch, args.body)
        objective, target_t, target_R = build_objective(args, batch, body, H)
        t0 = time.time()
        out = receding_horizon(model, diffusion, batch, planner, objective, horizon=H, stride=stride,
                               future_len=F_train, steps=steps, schedule=schedule, uncertainty_scale=uscale,
                               generator=gen, planner_kwargs=kwargs)
        wall = time.time() - t0
        col = int((out["rigid_idx"] == body).nonzero()[0])
        final_t = out["T"][:, -1, col]
        dist_mm = (final_t - target_t).norm(dim=-1) * 1000.0
        gt_plan = plan_from_batch(batch)[:, :H]
        plan_change = (out["deltas"] - gt_plan)[..., :3].norm(dim=-1)      # [B, H, Ka] metres
        rec = {
            "episode": i, "batch_size": int(batch.batch_size), "horizon": H, "windows": len(out["windows"]),
            "passive_body": body, "actuated_bodies": actuated_bodies(batch).tolist(),
            "target_distance_mm": [float(v) for v in dist_mm],
            "planned_vs_gt_translation_mm": {"mean": float(plan_change.mean()) * 1000, "max": float(plan_change.max()) * 1000},
            "plan_max_step_mm": float(out["deltas"][..., :3].norm(dim=-1).max()) * 1000,
            "window_losses": [w.loss for w in out["windows"]],
            "window_diagnostics": [w.diagnostics for w in out["windows"]],
            "nfe": out["nfe"], "wall_s": wall,
        }
        if target_R is not None:
            from utils.pose_ops import rotation_geodesic_error
            rec["target_rotation_deg"] = [float(v) * 180 / torch.pi for v in
                                          rotation_geodesic_error(out["R"][:, -1, col], target_R)]
        episodes.append(rec)
        if args.dump_dir:
            import numpy as np
            os.makedirs(args.dump_dir, exist_ok=True)
            np.savez(os.path.join(args.dump_dir, f"{args.domain}_ep{i:03d}_{args.planner}.npz"),
                     planned=out["positions"].cpu().numpy(), gt=batch.positions[:, :C + H].cpu().numpy(),
                     target_t=target_t.cpu().numpy(), context_len=C, passive_rows=(batch.particle_body == body).cpu().numpy(),
                     action_mask=batch.action_mask.cpu().numpy(), object_ids=batch.object_ids.cpu().numpy(),
                     valid=batch.get_valid_mask().cpu().numpy(), target_distance_mm=dist_mm.cpu().numpy(),
                     planner=args.planner, objective=args.objective)
        print(f"[{args.domain} ep {i}] {args.planner}: target {[round(v, 2) for v in rec['target_distance_mm']]} mm, "
              f"nfe {out['nfe']}, {wall:.1f}s")

    # Nominal cost of a full model window; totals below use actual planner calls.
    per_window = window_nfe(diffusion, batch, steps, schedule, uscale, future_len=F_train)
    all_d = [v for e in episodes for v in e["target_distance_mm"]]
    results = {
        "checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint), **code_revision(),
        "domain": args.domain, "planner": args.planner, "planner_kwargs": kwargs, "objective": args.objective,
        "phantom_weight": args.phantom_weight, "contact_radius": args.contact_radius, "floor_weight": args.floor_weight,
        "rotation_weight": args.rotation_weight, "context_len": C, "future_len": F_train, "horizon": horizon,
        "stride": stride, "steps": steps, "schedule": schedule, "uncertainty_scale": uscale,
        "nfe_per_window": per_window, "seed": args.seed, "torch_version": str(torch.__version__),
        "summary": {"episodes": len(episodes), "samples": len(all_d),
                    "target_distance_mm_mean": sum(all_d) / max(len(all_d), 1),
                    "nfe_total": sum(e["nfe"] for e in episodes), "wall_s_total": sum(e["wall_s"] for e in episodes)},
        "episodes": episodes,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[done] {args.out}: mean target distance {results['summary']['target_distance_mm_mean']:.2f} mm over {len(all_d)} samples")


if __name__ == "__main__":
    main()
