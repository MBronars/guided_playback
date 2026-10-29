"""Save a native-episode window batch as an overfit-harness fixture (.pt + receipt).

The fixture follows the pickup domain contract so `scripts/unified/overfit_prod.py`
can be driven with `--domain pickup --batch_name <name> --batch_dir <dir>
--require_saved_batches`, using the pickup per-domain scales; nothing is refit.

    PYTHONPATH=. python scripts/dev/native_overfit_fixture.py --episode <trajectory.npz> \
        --name native_shelf --batch_dir <dir> --windows 2 --drop_static
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from data.native_episodes import (load_native_episode, native_window_batch, reconstruction_error,
                                  select_windows)
from data.normalization import resolve_normalization
from utils.reproducibility import code_revision, file_sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch_dir", required=True)
    ap.add_argument("--windows", type=int, default=2)
    ap.add_argument("--context_len", type=int, default=4)
    ap.add_argument("--future_len", type=int, default=38)
    ap.add_argument("--drop_static", action="store_true")
    ap.add_argument("--pickup_layout", action="store_true", help="exactly the pickup corpus layout (object + 3 gripper links)")
    ap.add_argument("--scene_layout", action="store_true", help="pickup gripper + every object predicted + table/shelf as zero-motion input")
    ap.add_argument("--profile", default="xy_p99_per_domain")
    ap.add_argument("--no_fixtures", action="store_true",
                    help="with --scene_layout: drop the table and shelf, as the pickup corpus has no table")
    ap.add_argument("--articulated", action="store_true",
                    help="fingers as prismatic children of the hand (for model.articulated_bodies)")
    args = ap.parse_args()
    os.makedirs(args.batch_dir, exist_ok=True)
    spec = {"profile": args.profile}
    snapshot = resolve_normalization(spec)
    ep = load_native_episode(args.episode, drop_static=args.drop_static, pickup_layout=args.pickup_layout,
                             scene_layout=args.scene_layout, keep_fixtures=not args.no_fixtures)
    starts = select_windows(ep, args.windows, args.context_len, args.future_len)
    batch = native_window_batch(ep, starts, context_len=args.context_len, future_len=args.future_len,
                                normalization=spec, normalization_domain="pickup", articulated=args.articulated)
    batch.domain = "pickup"
    path, mpath = os.path.join(args.batch_dir, f"{args.name}.pt"), os.path.join(args.batch_dir, f"{args.name}.json")
    torch.save(batch, path)
    meta = {"domain": "pickup", "stats_set": snapshot["profile"], "normalization": snapshot,
            "normalization_sha256": snapshot["sha256"], "drawn": time.ctime(), "sha256": file_sha256(path),
            "B": batch.batch_size, "N": int(batch.positions.shape[2]), "C": int(batch.context_len),
            "F": int(batch.num_future_frames), "shape": list(batch.positions.shape),
            "source": {"episode": os.path.abspath(args.episode), "window_starts": starts,
                       "drop_static": bool(args.drop_static), "pickup_layout": bool(args.pickup_layout),
                       "scene_layout": bool(args.scene_layout), "fixtures_dropped": bool(args.no_fixtures),
                       "articulated": bool(args.articulated),
                       "parts": ep.part_names,
                       "kinematic_parts": [ep.part_names[k] for k in sorted(set(ep.part_index[ep.kinematic].tolist()))],
                       "manipulated_part": ep.part_names[ep.src_part], "shift": list(ep.shift),
                       "reconstruction_error_m": reconstruction_error(batch),
                       "adapter": "data/native_episodes.py", **code_revision()},
            "source_note": "native production episode windows through data/native_episodes.py under the pickup contract"}
    json.dump(meta, open(mpath, "w"), indent=1)
    print(f"[fixture] {path}: B={batch.batch_size} N={meta['N']} C={meta['C']} F={meta['F']} starts={starts} "
          f"parts={len(ep.part_names)} recon={meta['source']['reconstruction_error_m'] * 1000:.4f} mm")


if __name__ == "__main__":
    main()
