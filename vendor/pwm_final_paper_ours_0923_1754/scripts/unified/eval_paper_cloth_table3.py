"""Unified model on the EXACT paper Table 3 cloth protocol (SIM1 Cloth Folding).

Provenance (reconstructed + reproduced to <=0.001 on all 25 published cells):
  episodes   sim1_cloth_diverse_v1/bimanual/existing_000000.npz (pants, 2800 native)
             sim1_cloth_diverse_v1/bimanual/existing_000004.npz (long-shirt, 2918)
  protocol   stride-2 rollout starting at post-stride frame 30 (native 60);
             published "Ours" pred[t] is compared to GT at native 60+2t
  metric     symmetric Chamfer, SUM of directional NN means, x100 (cm), FULL
             cloth particle set, unweighted mean over the 2 episodes
  published  Ours: t=10/50/100/500/1000 -> 0.179 / 0.578 / 0.919 / 2.459 / 3.476

These episodes are from a different data vintage than cloth_curated_med. For a
model trained on that corpus this is a cross-vintage comparison; episode/asset
disjointness must still be established for the checkpoint being evaluated.
The model uses its checkpoint's C context frames ending at post-stride frame 29,
so its first prediction lands exactly on the published clock
(post-stride 30 = native 60); eval_unified's chamfer_cm_at{m} indexes the m-th
predicted frame, so published t corresponds to m = t + 1.

Usage:
  PYTHONPATH=. python scripts/unified/eval_paper_cloth_table3.py \
      --checkpoint runs/unified_v1_tier2v2/checkpoints/step_100000.pt \
      --out reports/unified/paper_cloth_table3_100k.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from utils.evaluation import validate_request, finish_metrics
from utils.reproducibility import file_sha256, code_revision

EPISODES = [
    ("/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/"
     "bimanual/existing_000000.npz", "pants"),
    ("/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/sim1_cloth_diverse_v1/"
     "bimanual/existing_000004.npz", "long-shirt"),
]
PUBLISHED_OURS = {10: 0.179, 50: 0.578, 100: 0.919, 500: 2.459, 1000: 3.476}
FIRST_PRED_S2 = 30     # prediction boundary, independent of the checkpoint context
MARKS = [t + 1 for t in PUBLISHED_OURS]          # chamfer_cm_at{m}: m-th predicted frame


def materialize(root: str, context_len: int = 2) -> str:
    """Cache the two paper episodes; reject stale/partial output instead of overwriting it."""
    from scripts.unified.materialize_cloth_wds import (
        SHIFT, STRIDE, N_CLOTH_MAX, N_EE, ShardWriter)
    if not 1 <= context_len <= FIRST_PRED_S2:
        raise ValueError("Paper cloth context_len must be in 1..30")
    start = FIRST_PRED_S2 - context_len
    span = context_len + max(MARKS)
    val_dir = os.path.join(root, "val")
    sidecar_path = os.path.join(root, "cloth_sidecar_table3.npz")
    manifest_path = Path(root) / "manifest.json"
    inputs = {"version": 1, "context_len": context_len, "start_s2": start,
              "span": span, "stride": STRIDE, "shift_m": SHIFT.tolist(),
              "n_cloth_max": N_CLOTH_MAX, "n_ee": N_EE,
              "episodes_sha256": {path: file_sha256(path) for path, _ in EPISODES}}
    if manifest_path.exists():
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        if manifest.get("inputs") != inputs:
            raise ValueError("Paper cloth cache inputs changed; choose a fresh --root")
        outputs = manifest.get("outputs", {})
        if set(outputs) != {"val/shard-00000.tar", "cloth_sidecar_table3.npz"}:
            raise ValueError("Incomplete paper cloth cache; choose a fresh --root")
        for name, digest in outputs.items():
            path = Path(root) / name
            if not path.is_file() or file_sha256(path) != digest:
                raise ValueError(f"Paper cloth cache is incomplete/corrupt: {path}; choose a fresh --root")
        return sidecar_path
    if list(Path(val_dir).glob("*.tar")) or Path(sidecar_path).exists():
        raise ValueError("Unverified paper cloth cache has no manifest; choose a fresh --root")
    w = ShardWriter(val_dir)
    sidecar = {}
    n_max = N_CLOTH_MAX + 2 * N_EE
    try:
        for path, name in EPISODES:
            with np.load(path, allow_pickle=False) as z:
                cp, le, re_ = [z[k][::STRIDE][start:start + span].astype(np.float32)
                               for k in ("cloth_points", "left_ee_points", "right_ee_points")]
                nc = cp.shape[1]
                if cp.shape != (span, nc, 3) or not 0 < nc <= N_CLOTH_MAX:
                    raise ValueError(f"{path}: invalid cloth shape or insufficient paper horizon")
                if le.shape != (span, N_EE, 3) or re_.shape != le.shape:
                    raise ValueError(f"{path}: gripper shape does not match the paper materializer")
                pos = np.concatenate([cp, le, re_], axis=1) + SHIFT
                if not np.isfinite(pos).all():
                    raise ValueError(f"{path}: non-finite paper cloth positions")
                n_valid = pos.shape[1]
                if n_valid < n_max:
                    pad = np.repeat(pos[:, -1:, :], n_max - n_valid, axis=1)
                    pos = np.concatenate([pos, pad], axis=1)
                oid = np.zeros(n_max, dtype=np.int64)
                oid[nc:nc + N_EE] = 1
                oid[nc + N_EE:n_valid] = 2
                valid = np.arange(n_max) < n_valid
                key = os.path.basename(path)[:-4]
                w.add(key, {"particle_pos": pos, "obj_ids": oid, "valid_particles": valid,
                            "n_cloth": np.int64(nc),
                            "dt": np.float32(STRIDE / float(z["fps"])) if "fps" in z.files
                            else np.float32(1 / 30)})
                sidecar[key] = z["ee_link_ids"]
                print(f"materialized {key} ({name}): T_s2={pos.shape[0]} nc={nc}", flush=True)
    finally:
        w.close()
    np.savez(sidecar_path, **sidecar)
    outputs = {name: file_sha256(Path(root) / name)
               for name in ("val/shard-00000.tar", "cloth_sidecar_table3.npz")}
    with manifest_path.open("w") as stream:
        json.dump({"inputs": inputs, "outputs": outputs}, stream, indent=2, allow_nan=False)
    return sidecar_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--root", default="reports/unified/cloth_table3",
                   help="cache root; each checkpoint context gets a separate verified subdirectory")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import data.unified_dataset as ud
    from inference import load_model_and_diffusion
    from scripts.unified.eval_unified import rollout_domain
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, diffusion, cfg, _ = load_model_and_diffusion(args.checkpoint, device)
    model.eval()
    if cfg.data.format != "unified" or cfg.get("prediction_target", "delta") != "delta":
        raise ValueError("Paper cloth wrapper requires a unified delta checkpoint")
    C, F = int(cfg.data.context_len), int(cfg.data.future_len)
    validate_request(2, C, F)
    cache = Path(args.root) / f"context{C}"
    sidecar_path = materialize(str(cache), context_len=C)
    from data.normalization import resolve_normalization
    normalization = resolve_normalization(cfg.normalization)
    ds = ud.UnifiedClothDataset(
        [str(cache / "val")], split="all",
        context_len=C, future_len=F, training=False, shuffle=False, normalization=normalization)
    ds.SIDECAR_IDS = sidecar_path
    res = rollout_domain(model, diffusion, cfg, "cloth", max_clips=None, device=device,
                         context_len=C, future_len=F, horizon=1001, chamfer_marks=MARKS,
                         dataset=ds, seed=args.seed)
    if res["n_episodes"] != len(EPISODES) or any(f"chamfer_cm_at{m}" not in res for m in MARKS):
        raise ValueError("Paper cloth requires both episodes at every published scoring time")

    print("\nTable 3 protocol, unified vs published Ours (Chamfer cm):")
    for t in sorted(PUBLISHED_OURS):
        k = f"chamfer_cm_at{t + 1}"
        ours = res.get(k)
        print(f"  t={t:4d}: ours {ours if ours is None else round(ours, 3)}"
              f"  vs published {PUBLISHED_OURS[t]}")
    res["published_ours"] = PUBLISHED_OURS
    res["checkpoint"] = args.checkpoint
    res.update(context_len=C, future_len=F, start_s2=FIRST_PRED_S2-C,
               seed=args.seed, cache=str(cache), checkpoint_sha256=file_sha256(args.checkpoint),
               normalization=normalization, **code_revision())
    finish_metrics(res)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as stream:
        json.dump(res, stream, indent=1, allow_nan=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
