"""Save water-domain windows of ONE Water-3D trajectory as an overfit-harness fixture (.pt + receipt).

Reads a split materialized by scripts/unified/materialize_water3d.py; the windows are the
``--windows`` non-overlapping ones with the most motion (or explicit ``--starts``), one seeded
particle subset, mapped to the model frame exactly as UnifiedWaterDataset does. Drive the harness with
``--domain water --batch_name <name> --batch_dir <dir> --require_saved_batches`` and the same
normalization artifact and ``model.num_mat_types=7``.

    PYTHONPATH=. python scripts/dev/water3d_overfit_fixture.py --trajectory 0 --name water_valid0_c4f16b4 \
        --batch_dir <dir> --normalization_artifact stats/normalization_water3d.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from data.normalization import domain_normalizers, resolve_normalization
from data.unified_dataset import WATER_ROOT
from data.water3d import (WATER3D_DT_FILE, WATER3D_PARTICLES, WATER3D_SHIFT, WATER3D_STRIDE, WATER3D_SUBSET_SEED,
                          particle_subset, select_windows, to_model_frame, water_batch)
from utils.reproducibility import code_revision, file_sha256


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_dir", default=f"{WATER_ROOT}/valid", help="one materialized split directory")
    ap.add_argument("--trajectory", type=int, default=0, help="index into the split manifest")
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch_dir", required=True)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--starts", default=None, help="comma-separated file-frame window starts (default: most motion)")
    ap.add_argument("--context_len", type=int, default=4)
    ap.add_argument("--future_len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=WATER3D_STRIDE)
    ap.add_argument("--particles", type=int, default=WATER3D_PARTICLES)
    ap.add_argument("--subset_seed", type=int, default=WATER3D_SUBSET_SEED)
    ap.add_argument("--profile", default="xy_p99_per_domain")
    ap.add_argument("--normalization_artifact", required=True, help="an artifact with a water entry")
    args = ap.parse_args()
    for ext in ("pt", "json"):
        if os.path.exists(os.path.join(args.batch_dir, f"{args.name}.{ext}")):
            raise FileExistsError(f"{args.name}.{ext} exists in {args.batch_dir}; fixtures are never overwritten")
    os.makedirs(args.batch_dir, exist_ok=True)
    snapshot = resolve_normalization({"profile": args.profile, "artifact": args.normalization_artifact})
    delta_stats, position_stats, _ = domain_normalizers(snapshot, "water")
    manifest = json.load(open(os.path.join(args.split_dir, "manifest.json")))
    item = manifest["trajectories"][args.trajectory]
    positions = np.load(os.path.join(args.split_dir, item["file"]), mmap_mode="r")
    idx = particle_subset(int(item["particles"]), args.particles, args.subset_seed)
    starts = ([int(s) for s in args.starts.split(",")] if args.starts else
              select_windows(positions, args.windows, args.context_len, args.future_len, args.stride, idx))
    L = args.context_len + args.future_len
    windows = [to_model_frame(np.asarray(positions[s + args.stride * np.arange(L)][:, idx], dtype=np.float32))
               for s in starts]
    batch = water_batch(windows, context_len=args.context_len, delta_stats=delta_stats,
                        position_stats=position_stats, particles=args.particles)
    d = (batch.positions[:, 1:] - batch.positions[:, :-1]) * 1000.0
    xy, z = d[..., :2].abs().flatten(), d[..., 2].abs().flatten()
    sx, sz = (float(v) * 1000 for v in delta_stats.max_val[1:])
    motion = {"step_mean_mm": float(d.norm(dim=-1).mean()), "step_p99_mm": float(d.norm(dim=-1).flatten().quantile(0.99)),
              "abs_dxy_p99_mm": float(xy.quantile(0.99)), "abs_dz_p99_mm": float(z.quantile(0.99)),
              "past_scale_xy_frac": float((xy > sx).float().mean()), "past_scale_z_frac": float((z > sz).float().mean()),
              "scale_xy_mm": sx, "scale_z_mm": sz}
    path, mpath = os.path.join(args.batch_dir, f"{args.name}.pt"), os.path.join(args.batch_dir, f"{args.name}.json")
    torch.save(batch, path)
    receipt = {"domain": "water", "stats_set": snapshot["profile"], "normalization": snapshot,
               "normalization_sha256": snapshot["sha256"], "drawn": time.ctime(), "sha256": file_sha256(path),
               "B": batch.batch_size, "N": int(batch.positions.shape[2]), "C": int(batch.context_len),
               "F": int(batch.num_future_frames), "shape": list(batch.positions.shape),
               "source": {"split_dir": os.path.realpath(args.split_dir), "manifest_index": args.trajectory,
                          "key": item["key"], "frames": item["frames"], "particles_total": item["particles"],
                          "particles_kept": int(len(idx)), "subset_seed": args.subset_seed,
                          "window_starts_file_frames": starts, "stride": args.stride,
                          "fps": 1.0 / (WATER3D_DT_FILE * args.stride), "frame_shift_m": list(WATER3D_SHIFT),
                          "motion": motion, "adapter": "data/water3d.py", **code_revision()}}
    json.dump(receipt, open(mpath, "w"), indent=1)
    print(f"[fixture] {path}: B={batch.batch_size} N={receipt['N']} C={receipt['C']} F={receipt['F']} starts={starts} "
          f"key={item['key']} step mean {motion['step_mean_mm']:.1f} mm p99 {motion['step_p99_mm']:.1f}")


if __name__ == "__main__":
    main()
