"""Unified model on the EXACT paper Table 1 MPM protocol.

Replicates archive eval_paper.py (the writer of inference_results/paper_eval/
v6_1_full49/summary.json, whose training_horizon_avg reproduces the published
Table 1 row: ID 6.48 = mean of 1-4obj group means, OOD-5/6/7 = 5.83/5.94/5.77):

  episodes   robert_mpm_test/webdataset_full49/{n}obj/*.tar, first 60/group
  rollout    autoregressive from context to frame 48 (49-frame episodes)
  metric     per-frame mean per-particle Euclidean distance (cm), per-file mean
             over the frame window, then mean across files, then across groups

Context and AR window lengths, normalization and coordinate convention default
to the checkpoint. To keep the predicted segment identical (frames 8-23, GT
context ending at frame 7), crop to [8-C:49]: C=2, C=4 and C=8 models see different
amounts of context but share the same prediction boundary and scoring frames.

Published constants (decoded from v6_1_full49/summary.json):
  train_h (frames 8-23):  1obj 4.99  2obj 7.81  3obj 6.93  4obj 6.20  -> ID 6.48
  long    (frames 8-48):  1obj 5.77  2obj 9.70  3obj 8.44  4obj 7.22  -> ID 7.78
  final   (frame 48):     1obj 6.41  2obj 11.04 3obj 9.52  4obj 8.00  -> ID 8.74

Usage:
  PYTHONPATH=. python scripts/unified/eval_paper_mpm_unified.py \
      --checkpoint runs/unified_v1_tier2v2/checkpoints/step_100000.pt \
      --out reports/unified/paper_mpm_table1_100k.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import tarfile
from pathlib import Path

import numpy as np
import torch

from data.normalization import domain_normalizers, resolve_normalization
from data.test_batch import LongTrajectoryBatch
from data.unified_dataset import _shift
from inference import AutoregressiveConfig, autoregressive_rollout, load_model_and_diffusion
from utils.evaluation import validate_request, validate_prediction, finish_metrics
from utils.reproducibility import file_sha256, code_revision

PUBLISHED = {  # dist_cm from the paper-era summary (Table 1 provenance)
    "train_h": {1: 4.99, 2: 7.81, 3: 6.93, 4: 6.20},
    "long":    {1: 5.77, 2: 9.70, 3: 8.44, 4: 7.22},
    "final":   {1: 6.41, 2: 11.04, 3: 9.52, 4: 8.00},
}
TH_NATIVE = 24      # Table 1 window end (exclusive), native frame index
SEG_START = 8       # first predicted frame of the published aggregate


def resolve_checkpoint_protocol(cfg, context_len=None):
    fmt = cfg.data.format
    if fmt != "unified" or cfg.get("prediction_target", "delta") != "delta":
        raise ValueError("Paper MPM protocol requires a unified delta checkpoint; historical specialists "
                         "are reproducible at robert/pre-unified-only-2026-09-06")
    trained_context = int(cfg.data.context_len)
    C = trained_context if context_len is None else context_len
    if C != trained_context:
        raise ValueError("context_len must match the checkpoint; crop follows its trained context")
    F = int(cfg.data.future_len)
    validate_request(1, C, F)
    if C > SEG_START:
        raise ValueError("Paper context_len must be at most 8 (context ends at frame 7)")
    return C, F, resolve_normalization(cfg.normalization)


def paper_group_metrics(rows, crop):
    if not rows:
        raise ValueError("No episodes evaluated in requested MPM object group")
    arr = np.stack(rows)
    if arr.ndim != 2 or arr.shape[1] != 49 - crop:
        raise ValueError("Paper scoring requires exactly native frames 0..48 before cropping")
    if not np.isfinite(arr).all():
        raise ValueError("Non-finite paper MPM metrics")
    s, th = SEG_START - crop, TH_NATIVE - crop
    return {"n_files": len(rows),
            "train_h": float(np.mean(arr[:, s:th].mean(axis=1))),
            "long": float(np.mean(arr[:, s:].mean(axis=1))),
            "final": float(np.mean(arr[:, -1])),
            "per_frame_dist_cm": [round(float(x), 3) for x in arr.mean(axis=0)]}


def iter_wds_test_shards(shard_dir, delta_stats, position_stats,
                         prediction_target="delta", max_trajectories=None):
    """Paper-era episode order/decoding, with explicit shape validation."""
    if max_trajectories is not None and max_trajectories <= 0:
        raise ValueError("max_trajectories must be positive")
    shards = sorted(Path(shard_dir).glob("*.tar"))
    count = 0
    for shard_path in shards:
        with tarfile.open(shard_path) as tf:
            batch_files: dict = {}
            for member_name in tf.getnames():
                if not member_name.endswith(".pyd"):
                    continue
                stem = member_name[:-4]
                bidx, key = stem.split(".", 1)
                batch_files.setdefault(bidx, {})[key] = member_name
            for bidx in sorted(batch_files):
                files = batch_files[bidx]

                def _load(key):
                    return pickle.load(tf.extractfile(files[key]))

                from collection.mpm.contracts import require_passive_consumer_input
                require_passive_consumer_input({key+'.pyd':_load(key) for key in
                    ('grab_velocity','is_grabbed','drag_active') if key in files})

                positions = _load("particle_pos")
                object_ids = _load("object_ids")
                material_props = _load("material_properties")
                if positions.ndim != 4:
                    raise ValueError(f"{shard_path}/{bidx}: expected [B,49,N,3] positions")
                B, T, N, _ = positions.shape
                if B <= 0 or T != 49 or N <= 0 or positions.shape[-1] != 3:
                    raise ValueError(f"{shard_path}/{bidx}: paper protocol requires nonempty [B,49,N,3]")
                valid = np.asarray(_load("valid_particles"), dtype=bool) if "valid_particles" in files else np.ones((B, T, N), bool)
                if valid.shape != (B, T, N) or not np.array_equal(valid, np.broadcast_to(valid[:, :1], valid.shape)):
                    raise ValueError("Paper MPM requires a temporally constant particle mask")
                lengths = valid[:, 0].sum(axis=1)
                if np.any(lengths == 0) or not np.array_equal(valid[:, 0], np.arange(N)[None] < lengths[:, None]):
                    raise ValueError("Paper MPM requires nonempty prefix-packed valid particles")
                for b in range(B):
                    pos = torch.from_numpy(np.ascontiguousarray(positions[b])).float()
                    oid = torch.from_numpy(np.ascontiguousarray(object_ids[b])).long()
                    mat = torch.from_numpy(np.ascontiguousarray(material_props[b])).float()
                    yield f"{shard_path.stem}_{bidx}_b{b}", LongTrajectoryBatch(
                        positions=pos.unsqueeze(0),
                        actions=torch.zeros(1, T - 1, N, 3, dtype=pos.dtype),
                        action_mask=torch.zeros(1, N, dtype=torch.bool),
                        object_ids=oid.unsqueeze(0),
                        valid_lengths=torch.tensor([int(lengths[b])]),
                        E=mat[:, 0].unsqueeze(0), nu=mat[:, 1].unsqueeze(0),
                        rho=mat[:, 2].unsqueeze(0),
                        mat_type=mat[:, 3].long().unsqueeze(0),
                        delta_stats=delta_stats, position_stats=position_stats,
                        prediction_target=prediction_target,
                    )
                    count += 1
                    if max_trajectories is not None and count >= max_trajectories:
                        return


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--id_wds_root",
                   default="/data/user_data/robertre/Genesis/3DPCWM_Data/"
                           "robert_mpm_test/webdataset_full49")
    p.add_argument("--max_trajectories", "--max_files", dest="max_files", type=int, default=60,
                   help="episodes per object-count group (legacy --max_files alias; not tar count)")
    p.add_argument("--obj_groups", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--context_len", type=int, default=None,
                   help="defaults to checkpoint context; crop is set so context always ends "
                        "at native frame 7 (the paper's information boundary)")
    p.add_argument("--streaming", action="store_true",
                   help="rolling-buffer streaming rollout (inference.streaming_rollout) "
                        "instead of strided chunked AR; same files/frames/metric")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    validate_request(args.max_files, 1, None)
    if len(set(args.obj_groups)) != len(args.obj_groups) or any(n < 1 or n > 7 for n in args.obj_groups):
        raise ValueError("obj_groups must be distinct counts in 1..7")
    if "drag" in str(Path(args.id_wds_root).resolve()).lower():
        raise ValueError("Active MPM drag is not supported by the passive unified adapter")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, diffusion, cfg, steps = load_model_and_diffusion(args.checkpoint, device)
    model.eval()
    C, F, normalization = resolve_checkpoint_protocol(cfg, args.context_len)
    crop = SEG_START - C
    delta_stats, position_stats, _ = domain_normalizers(normalization, "mpm")
    shift = _shift("mpm")
    # Strided chunks with a shorter last chunk: the published protocol's rollout, kept as
    # is. Every other evaluator uses inference.windowed_rollout (docs/EVALUATION.md).
    ar = AutoregressiveConfig(
        context_len=C, future_len=F, num_inference_steps=steps,
        diffusion_forcing=cfg.get("diffusion_forcing", False),
        df_schedule=cfg.get("df_schedule", "pyramid"),
        df_uncertainty_scale=cfg.get("df_uncertainty_scale", 1.0),
        streaming=bool(args.streaming))

    out = {"checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint),
           **code_revision(), "context_len": C, "future_len": F, "crop": crop,
           "normalization": normalization, "coordinate_convention": "unified",
           "shift_m": shift.tolist(), "seed": args.seed, "inference_steps": steps,
           "data_root": str(Path(args.id_wds_root).resolve()),
           "max_trajectories_per_group": args.max_files,
           "sampler": "streaming" if args.streaming else "strided_ar",
           "protocol": f"paper Table 1 (eval_paper.py): first-{args.max_files}/group, "
                       "dist_cm windowed means; segment frames 8-23 / 8-48 / 48",
           "per_group": {}}
    for n_obj in args.obj_groups:
        src = Path(args.id_wds_root) / f"{n_obj}obj"
        rows = []          # per-file per-frame dist_cm, native frames 8..48
        episode_ids = []
        for label, batch in iter_wds_test_shards(
                src, delta_stats, position_stats,
                max_trajectories=args.max_files):
            b = batch
            if b.positions.shape[1] != 49:
                raise ValueError(f"{label}: paper protocol requires 49 native frames")
            b.positions = b.positions[:, crop:] + shift.to(b.positions.dtype)
            b.actions = b.actions[:, crop:]
            b = b.to(device)
            with torch.no_grad():
                res = autoregressive_rollout(model=model, diffusion=diffusion,
                                             sequence_data=b, config=ar, verbose=False)
            pred = res["predicted_positions"][0]      # [T', N, 3], T' = 43
            gt = res["gt_positions"][0]
            v = b.get_valid_mask()[0]
            validate_prediction(pred[None], gt[None], v[None])
            d = (pred[:, v] - gt[:, v]).norm(dim=-1).mean(dim=1) * 100.0  # [T'] cm
            rows.append(d.cpu().numpy())
            episode_ids.append(label)
        g = paper_group_metrics(rows, crop)
        g["episode_ids"] = episode_ids
        g["published"] = {k: PUBLISHED[k].get(n_obj) for k in PUBLISHED}
        out["per_group"][n_obj] = g
        print(f"[{n_obj}obj] n={g['n_files']} train_h {g['train_h']:.2f} "
              f"(pub {PUBLISHED['train_h'].get(n_obj)}) long {g['long']:.2f} "
              f"(pub {PUBLISHED['long'].get(n_obj)}) final {g['final']:.2f} "
              f"(pub {PUBLISHED['final'].get(n_obj)})", flush=True)

    ids = [out["per_group"][n] for n in args.obj_groups]
    for k, pub_id in (("train_h", 6.48), ("long", 7.78), ("final", 8.74)):
        ours = float(np.mean([g[k] for g in ids]))
        out[f"selected_groups_{k}"] = round(ours, 3)
        if set(args.obj_groups) == {1, 2, 3, 4}:
            out[f"ID_{k}"] = round(ours, 3)
            out[f"ID_{k}_published"] = pub_id
            print(f"ID {k}: ours {ours:.2f} vs published {pub_id}", flush=True)
    out["n_episodes"] = sum(g["n_files"] for g in ids)
    finish_metrics(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(out, stream, indent=1, allow_nan=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
