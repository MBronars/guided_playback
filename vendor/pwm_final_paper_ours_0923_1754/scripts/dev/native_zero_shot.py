"""Zero-shot check of a unified checkpoint on native production episodes.

Single-window sampling (the production DF sampler) and a short overlapping
autoregression (the evaluator's window contract) on motion-rich windows of each
episode, with per-group particle errors (manipulated object, distractors, static
parts, gripper), rigid pose errors, GIFs, and a pickup-corpus control through the
same code path.

    PYTHONPATH=. python scripts/dev/native_zero_shot.py --checkpoint <ckpt> \
        --episodes /data/user_data/robertre/yikai_native_0910/*/trajectory.npz \
        --out /data/user_data/robertre/scratchpad/yikai_native_0910/zero_shot
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from data.native_episodes import (GRIPPER_PARTS, load_native_episode, native_window_batch,
                                  select_windows)
from data.unified_dataset import DOMAIN_FUTURE_LEN
from inference import load_model_and_diffusion, run_diffusion_inference_df, windowed_rollout
from scripts.unified.eval_unified import evaluation_dataset, hides_given_actions
from utils.conditioning import hide_actions
from utils.pose_native import materialize_pose_native
from utils.evaluation import TrajectoryScores, rigid_pose_errors

STATIC_NAMES = ("table", "shelf")


def groups(ep):
    """Particle masks: src (manipulated), distractor (other objects), static (table/shelf), gripper."""
    names = ep.part_names
    static = torch.zeros(ep.num_particles, dtype=torch.bool)
    for k, n in enumerate(names):
        if any(s in n for s in STATIC_NAMES):
            static |= ep.part_mask(k)
    src = ep.part_mask(ep.src_part)
    return {"src": src, "distractor": ~(src | static | ep.gripper), "static": static, "gripper": ep.gripper}


def sample_window(model, diffusion, batch, steps, schedule, uscale, hide_given_actions=False):
    """One production DF window. hide_given_actions samples it without the actuator's future (a
    train.action_dropout=1 checkpoint); positions are still materialized from the unhidden batch."""
    out = run_diffusion_inference_df(model, hide_actions(batch, 1.0) if hide_given_actions else batch,
                                     diffusion, num_inference_steps=steps,
                                     schedule=schedule, uncertainty_scale=uscale)
    pred_pos, R, t, ridx = materialize_pose_native(batch, *out)
    return pred_pos, R, t, ridx


def particle_errors(batch, pred, gt, masks):
    """Per particle group, the evaluation vocabulary's particle errors (utils.evaluation)."""
    out = {}
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        scores = TrajectoryScores(marks=())
        scores.add(batch, pred, gt, rows=mask.reshape(1, -1).expand(pred.shape[0], -1), poses=False)
        result = scores.result()
        if result["n_episodes"]:
            out[name] = {key: result[key] for key in ("particle_mean_mm", "particle_final_mm")}
    return out


def pose_errors(batch, pred, ep):
    """Final-frame pose error per rigid part (utils.evaluation.rigid_pose_errors: Kabsch of each
    part's template onto its predicted particles, the evaluator's pose metric)."""
    C, H = batch.context_len, pred.shape[1]
    return {ep.part_names[k]: {"pose_final_mm": round(float(mm[-1]), 3), "pose_final_deg": round(float(deg[-1]), 3)}
            for _, k, _, deg, mm in rigid_pose_errors(batch, pred, batch.body_poses[:, C:C + H])}


def rollout(model, diffusion, full, F_train, steps, schedule, uscale, hide_given_actions=False):
    """The evaluator's rollout (inference.windowed_rollout) over the whole future of `full`."""
    C = full.context_len
    out = windowed_rollout(model, diffusion, full, horizon=full.positions.shape[1] - C, future_len=F_train,
                           num_inference_steps=steps, schedule=schedule, uncertainty_scale=uscale,
                           hide_given_actions=hide_given_actions)
    return out["positions"][:, C:]


def render(gt, pred, batch, path, C, F):
    from utils.visualization import create_trajectory_comparison_gif
    ctx = batch.positions[:1, :C].cpu()
    g = torch.cat([ctx, gt[:1].cpu()], 1); p = torch.cat([ctx, pred[:1].cpu()], 1)
    lo = g.amin(dim=(0, 1, 2)); hi = g.amax(dim=(0, 1, 2)); pad = 0.05 * (hi - lo).max()
    create_trajectory_comparison_gif(g, p, batch.action_mask[:1].cpu(), path, fps=8,
                                     object_ids=batch.object_ids[:1].cpu(), context_len=C, future_len=F,
                                     ground_plane_z=None,
                                     axis_limits=tuple((float(lo[i] - pad), float(hi[i] + pad)) for i in range(3)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--rollout_frames", type=int, default=48)
    ap.add_argument("--pickup_control_batches", type=int, default=2)
    ap.add_argument("--layouts", default="all_parts,moving_only,pickup",
                    help="comma list of: all_parts (every part), moving_only (drop static parts), "
                         "pickup (exactly the pickup corpus layout: object + hand + two fingers)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, cfg, steps = load_model_and_diffusion(args.checkpoint, device)
    trained_f = int(cfg.data.get("future_len", -1) or -1)
    C = int(cfg.data.context_len)
    F = trained_f if trained_f > 0 else DOMAIN_FUTURE_LEN["pickup"]
    schedule, uscale = cfg.get("df_schedule", "pyramid"), float(cfg.get("df_uncertainty_scale", 1.0))
    hide = hides_given_actions(cfg)
    results = {"checkpoint": args.checkpoint, "context_len": C, "future_len": F, "steps": steps,
               "schedule": schedule, "actions": "hidden" if hide else "given", "episodes": {}}

    for path in sorted(sum((glob.glob(p) for p in args.episodes), [])):
        name = os.path.basename(os.path.dirname(path))
        for variant in args.layouts.split(","):
            ep = load_native_episode(path, drop_static=(variant == "moving_only"), pickup_layout=(variant == "pickup"))
            masks = groups(ep)
            starts = select_windows(ep, args.windows, C, F)
            rec = {"N": ep.num_particles, "parts": ep.part_names, "window_starts": starts, "windows": []}
            for s in starts:                                   # one window at a time: N is large
                b = native_window_batch(ep, [s], context_len=C, future_len=F, normalization=cfg.normalization).to(device)
                pred = sample_window(model, diffusion, b, steps, schedule, uscale, hide)[0]
                w = {"start": s, "particles": particle_errors(b, pred, b.future, masks), "poses": pose_errors(b, pred, ep)}
                rec["windows"].append(w)
                if s == starts[0]:
                    render(b.future, pred, b, os.path.join(args.out, f"{name}_{variant}_window{s}.gif"), C, F)
                print(f"[{name} {variant}] window {s}: " + ", ".join(f"{k} {v['particle_mean_mm']:.1f}/{v['particle_final_mm']:.1f}" for k, v in w["particles"].items()))
            # rollout on the most active window start, if the episode has room
            H = args.rollout_frames
            s = starts[0] if starts[0] + C + H <= ep.num_frames else max(0, ep.num_frames - C - H)
            full = native_window_batch(ep, [s], context_len=C, future_len=H, normalization=cfg.normalization).to(device)
            pred = rollout(model, diffusion, full, F, steps, schedule, uscale, hide)
            rec["rollout"] = {"start": s, "frames": H, "particles": particle_errors(full, pred, full.future, masks),
                              "poses": pose_errors(full, pred, ep)}
            render(full.future, pred, full, os.path.join(args.out, f"{name}_{variant}_rollout{H}_from{s}.gif"), C, H)
            print(f"[{name} {variant}] rollout {H} from {s}: " + ", ".join(f"{k} {v['particle_mean_mm']:.1f}/{v['particle_final_mm']:.1f}" for k, v in rec["rollout"]["particles"].items()))
            results["episodes"][f"{name}/{variant}"] = rec

    if args.pickup_control_batches > 0:
        ds = evaluation_dataset(cfg, "pickup", C, F)
        ds.emit_sidecar = True
        ctrl = []
        for i, b in enumerate(ds):
            if i >= args.pickup_control_batches:
                break
            b = b.to(device)
            pred = sample_window(model, diffusion, b, steps, schedule, uscale, hide)[0]
            masks = {"object": ~b.action_mask[0], "gripper": b.action_mask[0]}
            ctrl.append(particle_errors(b, pred, b.future, masks))
            if i == 0:
                render(b.future, pred, b, os.path.join(args.out, "pickup_control_window.gif"), C, F)
        results["pickup_control"] = ctrl
        print("[pickup control] " + "; ".join(", ".join(f"{k} {v['particle_mean_mm']:.1f}/{v['particle_final_mm']:.1f}" for k, v in c.items()) for c in ctrl))

    with open(os.path.join(args.out, "zero_shot.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("[done]", os.path.join(args.out, "zero_shot.json"))


if __name__ == "__main__":
    main()
