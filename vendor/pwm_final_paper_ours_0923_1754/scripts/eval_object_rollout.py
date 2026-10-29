"""Unified-model evaluation on passive-MPM populations, in millimetres.

The checkpoint owns normalization, context/window lengths and sampler settings.
An optional --config checks compatibility; it does not replace those settings.
Raw sources receive the same fixed canonical shift and shared position scale as
unified training. Historical specialist checkpoints require the frozen revision.

--rollout window scores one model window. full is inference.windowed_rollout to
the end of each clip, the same rollout as eval_unified. stream uses a continuously
rolling pyramid buffer (--schedule is ignored in stream mode), a different protocol.
Scores are utils.evaluation.TrajectoryScores, so keys match eval_unified;
particle_per_frame_mm averages, at each frame, the clips that reach it (particle_per_frame_n counts them).
--clips all is one finite pass.

Evaluate each {N}obj directory separately and state how counts are weighted.
For a v2 specialist selected on robert_mpm_v2_val_unseen_fulltar, the untouched
unseen test root is robert_mpm_v2_test_unseen_00001plus: the full test_unseen root
includes the validation tars. See docs/CODE_WALKTHROUGH.md for split roles.
Use max_objects=7 for v2 OOD counts 5–7. Corpus sizes are measured in the output,
not inferred from old raw-scene counts.

Usage (from repo root, PYTHONPATH=.):
  python scripts/eval_object_rollout.py --checkpoint runs/<run>/checkpoints/best.pt \
      --rollout full --clips all --schedule pyramid --max_objects 4 \
      --wds .../robert_mpm_v2_test_unseen_00001plus/3obj --out out.json
"""
from __future__ import annotations
import argparse, json, os
import random
from pathlib import Path

import numpy as np
import torch
from collections import Counter

from data.unified_dataset import UnifiedMPMDataset, _shift, GROUP
from data.normalization import resolve_normalization
from data.test_batch import LongTrajectoryBatch
from inference import (
    AutoregressiveConfig, autoregressive_rollout, load_model_and_diffusion,
    run_diffusion_inference_df, windowed_rollout,
)
from utils.evaluation import (TrajectoryScores, finish_metrics, iter_clips, parse_clips, read_provenance,
                              validate_checkpoint_fields, validate_request)
from utils.reproducibility import code_revision, file_sha256


# These are distinct populations/protocol inputs, not interchangeable names for
# a single 'test' set. Active drag is deliberately absent (unsupported adapter).
MPM_POPULATIONS = {
    "v2_seen": ("robert_mpm_v2_test_seen", "unified checkpoint selection; not untouched final test"),
    "v2_unseen_full": ("robert_mpm_v2_test_unseen", "includes specialist validation tars"),
    "v2_unseen_validation": ("robert_mpm_v2_val_unseen_fulltar", "historical specialist validation"),
    "v2_unseen_00001plus": ("robert_mpm_v2_test_unseen_00001plus", "excludes specialist validation tars"),
    "v2_unseen_clean": ("robert_mpm_v2_test_unseen_clean", "alternate 00001+ view, not independent population"),
    "v2_ood": ("robert_mpm_v2_ood", "full held-out counts 5-7"),
    "v2_ood_first_tar": ("robert_mpm_v2_ood_first_tar", "diagnostic subset, not full OOD"),
    "v2_viz_pinned": ("robert_mpm_v2_viz_pinned", "visualization subset, not representative benchmark"),
    "paper_v1_id": ("robert_mpm_test/webdataset_full49", "paper-era ID inputs; this CLI does not impose paper scoring crop"),
    "legacy_v1_test": ("robert_mpm/webdataset_test", "legacy test; not a fresh training-validation split"),
}


def create_eval_loader(wds, cfg, *, rollout, num_workers):
    """Finite deterministic source traversal, with canonical unified preprocessing."""
    ds = UnifiedMPMDataset(
        wds_location=wds, normalization=cfg.normalization,
        context_len=int(cfg.data.context_len),
        future_len=(-1 if rollout in ("full", "stream") else int(cfg.data.future_len)),
        num_collate=int(cfg.data.get("num_collate", 1)), training=False, shuffle=False,
        prediction_target="delta")
    # No exclusions are newly applied to historical evaluation populations.
    ds.strict_read = True
    return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers)


def _to_long_batch(batch, prediction_target):
    """TrajectoryBatch holding a FULL clip -> LongTrajectoryBatch for autoregressive rollout.
    Passive MPM data has no actions, so actions are zeros over all T-1 transitions."""
    B, T, N, _ = batch.positions.shape
    if batch.action_active is not None:
        raise ValueError('Passive object rollout cannot discard scheduled drag commands; use a drag-aware long batch')
    return LongTrajectoryBatch(
        positions=batch.positions,
        actions=torch.zeros(B, T - 1, N, 3, device=batch.positions.device,
                            dtype=batch.positions.dtype),
        action_mask=batch.action_mask,
        object_ids=batch.object_ids,
        valid_lengths=batch.valid_lengths,
        E=batch.E, nu=batch.nu, rho=batch.rho,
        mat_type=getattr(batch, "mat_type", None),
        delta_stats=batch.delta_stats,
        position_stats=batch.position_stats,
        prediction_target=prediction_target,
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", help="Optional compatibility check against checkpoint representation")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=2)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--wds", help="explicit passive-MPM split or object-count directory")
    source.add_argument("--population", choices=sorted(MPM_POPULATIONS),
                        help="named population; distinct split roles are recorded in output")
    ap.add_argument("--max_objects", type=int, default=4, help="4 for seen/unseen, 7 for ood")
    ap.add_argument("--schedule", default=None, choices=["pyramid", "full_sequence"],
                    help="explicit sampler ablation; default is the checkpoint schedule")
    ap.add_argument("--clips", required=True,
                    help="'all' = ONE FULL PASS over the population (exact coverage); N = the first "
                         "N clips of its stratified order")
    ap.add_argument("--max_batches", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rollout", default="window", choices=["window", "full", "stream"],
                    help="window = ONE future_len chunk (per-step accuracy); "
                         "full = windowed_rollout to the end of the clip (long-horizon); "
                         "stream = rolling-buffer streaming rollout (1 commit/forward, sliding "
                         "context; --schedule ignored — the buffer is a continuously rolling pyramid)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    population_role = "explicit custom source; disjointness not established by this evaluator"
    if args.population:
        relative, population_role = MPM_POPULATIONS[args.population]
        args.wds = str(Path(GROUP) / relative)
    if "drag" in str(Path(args.wds).resolve()).lower():
        raise ValueError("Active MPM drag is not supported by the passive unified adapter")

    if args.max_batches is not None:
        raise ValueError("--max_batches is retired: it counted shard records, not clips. "
                         "Use --clips N or --clips all")
    max_clips = parse_clips(args.clips)
    if args.max_objects <= 0 or args.num_workers < 0:
        raise ValueError("max_objects must be positive and num_workers nonnegative")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, diffusion, cfg, inference_steps = load_model_and_diffusion(args.checkpoint, device=device)
    d = cfg.data
    if d.format != "unified" or cfg.get("prediction_target", "delta") != "delta":
        raise ValueError("MPM evaluator requires a unified delta checkpoint; use "
                         "robert/pre-unified-only-2026-09-06 for historical specialists")
    if not cfg.get("diffusion_forcing", False):
        raise ValueError("Native evaluator requires a diffusion-forcing checkpoint")
    validate_request(1, int(d.context_len), int(d.future_len))
    normalization = resolve_normalization(cfg.normalization)
    args.schedule = args.schedule or cfg.get("df_schedule", "pyramid")
    if args.config:
        from utils.config import load_config
        external = load_config(args.config)
        validate_checkpoint_fields(external.data, d,
                                   ("format", "context_len", "future_len"))
        if resolve_normalization(external.normalization)["sha256"] != normalization["sha256"]:
            raise ValueError("normalization disagrees with checkpoint")
        validate_checkpoint_fields(external, cfg,
                                   ("prediction_target", "denoise_method", "diffusion_forcing",
                                    "df_uncertainty_scale"), prefix="")

    # rollout=full needs the WHOLE clip (future_len=-1 disables random sub-window slicing);
    # windowed_rollout then rolls out over it one trained-length window at a time.
    loader = create_eval_loader(args.wds, cfg, rollout=args.rollout, num_workers=args.num_workers)
    ar_cfg = AutoregressiveConfig(
        context_len=d.context_len, future_len=d.future_len,
        num_inference_steps=inference_steps, diffusion_forcing=cfg.diffusion_forcing,
        df_schedule=args.schedule, df_uncertainty_scale=cfg.df_uncertainty_scale,
        stride=-1, streaming=(args.rollout == "stream"))

    torch.manual_seed(args.seed)
    C = int(d.context_len)
    scores, per_count, sources = TrajectoryScores(per_frame=True), {}, Counter()
    for i, (batch, source) in enumerate(iter_clips(loader, max_clips)):
        sources[source] += batch.batch_size
        batch = batch.to(device)
        if args.rollout == "window":
            pred_norm = run_diffusion_inference_df(
                model, batch, diffusion, num_inference_steps=inference_steps,
                schedule=args.schedule, uncertainty_scale=float(cfg.df_uncertainty_scale))
            pred_pos = batch.predictions_to_positions(batch.prediction_stats.denormalize(pred_norm))
            gt_pos = batch.future                                         # [B,F,N,3]
        elif args.rollout == "full":
            # The evaluator's rollout to the end of the clip: predicted frames feed the
            # next context, so this measures COMPOUNDING error over the full horizon.
            res = windowed_rollout(model, diffusion, batch, horizon=batch.positions.shape[1] - C,
                                   future_len=int(d.future_len), num_inference_steps=inference_steps,
                                   schedule=args.schedule, uncertainty_scale=float(cfg.df_uncertainty_scale))
            pred_pos, gt_pos = res["positions"][:, C:], batch.positions[:, C:]
        else:
            seq = _to_long_batch(batch, cfg.prediction_target)
            res = autoregressive_rollout(model, diffusion, seq, ar_cfg)       # dispatches to streaming
            pred_pos, gt_pos = res["predicted_positions"][:, C:], seq.positions[:, C:]
        valid = batch.get_valid_mask()
        counts = [int(batch.object_ids[b][valid[b]].unique().numel()) for b in range(batch.batch_size)]
        if max(counts) > args.max_objects:
            raise ValueError(f"Observed {max(counts)} objects exceeds --max_objects {args.max_objects}; "
                             "use 7 for the full OOD counts 5-7")
        scores.add(batch, pred_pos, gt_pos)
        for nobj in sorted(set(counts)):
            clips = torch.tensor([count == nobj for count in counts], device=valid.device)
            per_count.setdefault(nobj, TrajectoryScores()).add(
                batch, pred_pos, gt_pos, rows=clips[:, None].expand_as(valid))
        if (i + 1) % 5 == 0 or i == 0:
            running = scores.result()
            print(f"[progress] batch {i + 1} | {scores.n} clips | "
                  f"running mean {running['particle_mean_mm']:.2f}mm "
                  f"final {running['particle_final_mm']:.2f}mm", flush=True)

    per_count_keys = ("n_episodes", "particle_mean_mm", "particle_final_mm")
    res = {"checkpoint": args.checkpoint, "wds": args.wds, "schedule": args.schedule,
           "clips_requested": "all" if max_clips is None else max_clips,
           "rollout": args.rollout, "max_objects": args.max_objects, "num_workers": args.num_workers,
           "context_len": C, "window_len": int(d.future_len),
           "normalization": normalization, "shift_m": _shift("mpm").tolist(), "seed": args.seed,
           "population": args.population, "population_role": population_role,
           "checkpoint_sha256": file_sha256(args.checkpoint),
           "inference_steps": int(inference_steps), "uncertainty_scale": float(cfg.df_uncertainty_scale),
           "effective_schedule": "streaming_pyramid" if args.rollout == "stream" else args.schedule,
           "config_check": args.config, **code_revision(), **scores.result(),
           "per_count": {str(k): {key: v.result()[key] for key in per_count_keys}
                         for k, v in sorted(per_count.items())},
           **read_provenance(sources, getattr(getattr(loader, "dataset", None), "shards", ()))}
    finish_metrics(res)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(res, stream, indent=1, allow_nan=False)
    pc = " ".join(f"{k}obj:{v['particle_mean_mm']:.1f}" for k, v in res["per_count"].items())
    print(f"[{os.path.basename(args.checkpoint)} | {os.path.basename(os.path.dirname(args.wds))} | {args.schedule}"
          f" | {args.rollout}:{res['horizon_frames_max']}f | n={res['n_episodes']}] "
          f"mean {res['particle_mean_mm']:.2f}mm final {res['particle_final_mm']:.2f}mm  ({pc})")


if __name__ == "__main__":
    main()
