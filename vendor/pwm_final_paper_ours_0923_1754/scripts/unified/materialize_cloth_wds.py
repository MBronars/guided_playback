"""Materialize cloth_curated_med (npz episodes) into unified-v1 WDS shards.

Why materialize: multi-worker npz reads over NFS thrash (measured on this cluster —
the eager_load lesson); WDS streams sequentially. Done ONCE, then UnifiedClothDataset
reads shards like every other corpus.

Per episode:
  - temporal stride 2 (the training view; fps 60 -> 30 Hz, dt=1/30)
  - concat [cloth (Nc), left_ee (1000), right_ee (1000), PAD to N_MAX] — padding at the
    TAIL so valid_lengths count semantics hold; pad rows repeat the last EE point (inert,
    marked invalid) rather than sitting at the canonical origin
  - canonical shift applied HERE (cloth: -0.65, 0, -0.80) — the adapter stays thin
  - obj_ids: 0=cloth, 1=left EE, 2=right EE (opaque labels; pad rows get 0 + invalid)
  - route to train/ or val/ by the frozen 38-garment manifest in data/manifests/

Layout: <out>/{train,val}/shard-%05d.tar with keys particle_pos/obj_ids/valid_particles/
n_cloth/dt (.pyd pickles, matching every other corpus).

Usage: PYTHONPATH=. python scripts/unified/materialize_cloth_wds.py --out NEW_DIRECTORY [--limit N]
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np

SRC = "/data/group_data/katefgroup/VLA/3DPCWM/robert_sim1/cloth_curated_med"
HOLDOUT_MANIFEST = Path(__file__).resolve().parents[2] / "data/manifests/cloth_holdout.json"
SHIFT = np.array([-0.65, 0.00, -0.80], dtype=np.float32)
STRIDE = 2
N_CLOTH_MAX = 12000      # med band is 7-12k by construction; larger episodes are skipped+logged
N_EE = 1000              # per gripper, verified constant in Phase 0
EPS_PER_SHARD = 8


def load_holdout_files(path=HOLDOUT_MANIFEST):
    """Exact episode membership, independent of model configuration and cwd."""
    with open(path) as stream:
        manifest = json.load(stream)
    files = manifest["val_files"]
    if manifest.get("schema_version") != 1 or not files or len(set(files)) != len(files):
        raise ValueError(f"Invalid cloth holdout manifest: {path}")
    if any(Path(name).name != name or not name.endswith(".npz") for name in files):
        raise ValueError("Cloth holdout entries must be .npz basenames")
    return set(files)


class ShardWriter:
    def __init__(self, out_dir: str):
        if list(Path(out_dir).glob("*.tar")):
            raise FileExistsError(f"Refusing to replace existing materialized shards under {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.idx = 0
        self.count = 0
        self.tf = None

    def _roll(self):
        if self.tf is not None:
            self.tf.close()
        path = os.path.join(self.out_dir, f"shard-{self.idx:05d}.tar")
        self.tf = tarfile.open(path, "x")
        self.idx += 1
        self.count = 0

    def add(self, key: str, arrays: dict):
        if self.tf is None or self.count >= EPS_PER_SHARD:
            self._roll()
        for name, arr in arrays.items():
            blob = pickle.dumps(arr, protocol=4)
            info = tarfile.TarInfo(f"{key}.{name}.pyd")
            info.size = len(blob)
            self.tf.addfile(info, io.BytesIO(blob))
        self.count += 1

    def close(self):
        if self.tf is not None:
            self.tf.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=-1)
    p.add_argument("--exclude_json", default=None,
                   help="reports/unified/cloth_exclusion.json — episodes whose |coord| exceeds S "
                        "(user decision 2026-08-06: delete rather than clip or widen S).")
    p.add_argument("--out", required=True, help="New output root; existing shards are never overwritten.")
    args = p.parse_args()

    excluded = set()
    if args.exclude_json:
        ex = json.load(open(args.exclude_json))
        excluded = {r["episode"] for r in ex["excluded"]}
        print(f"excluding {len(excluded)} episodes (bound {ex['S']} m) per {args.exclude_json}",
              flush=True)

    val_files = load_holdout_files()
    files = sorted(glob.glob(f"{SRC}/*.npz"))
    if args.limit > 0:
        files = files[:args.limit]
    files = [f for f in files if os.path.basename(f)[:-4] not in excluded]
    print(f"{len(files)} episodes; {len(val_files)} held-out filenames from {HOLDOUT_MANIFEST}", flush=True)

    writers = {"train": ShardWriter(f"{args.out}/train"), "val": ShardWriter(f"{args.out}/val")}
    stats = {"train": 0, "val": 0, "skipped_band": 0, "skipped_err": 0}
    N_MAX = N_CLOTH_MAX + 2 * N_EE

    for i, f in enumerate(files):
        base = os.path.basename(f)
        split = "val" if base in val_files else "train"
        try:
            z = np.load(f, allow_pickle=True)
            cp = z["cloth_points"][::STRIDE].astype(np.float32)
            le = z["left_ee_points"][::STRIDE].astype(np.float32)
            re_ = z["right_ee_points"][::STRIDE].astype(np.float32)
        except Exception as e:
            print(f"  ERR {base}: {type(e).__name__}: {e}", flush=True)
            stats["skipped_err"] += 1
            continue
        nc = cp.shape[1]
        if nc > N_CLOTH_MAX or le.shape[1] != N_EE or re_.shape[1] != N_EE:
            print(f"  SKIP {base}: nc={nc} le={le.shape[1]} re={re_.shape[1]}", flush=True)
            stats["skipped_band"] += 1
            continue

        pos = np.concatenate([cp, le, re_], axis=1) + SHIFT          # [T, nc+2000, 3]
        n_valid = pos.shape[1]
        if n_valid < N_MAX:                                          # tail padding, inert
            pad = np.repeat(pos[:, -1:, :], N_MAX - n_valid, axis=1)
            pos = np.concatenate([pos, pad], axis=1)
        oid = np.zeros(N_MAX, dtype=np.int64)
        oid[nc:nc + N_EE] = 1
        oid[nc + N_EE:n_valid] = 2
        valid = np.zeros(N_MAX, dtype=bool)
        valid[:n_valid] = True

        writers[split].add(base[:-4], {
            "particle_pos": pos,
            "obj_ids": oid,
            "valid_particles": valid,
            "n_cloth": np.int64(nc),
            "dt": np.float32(STRIDE / float(z["fps"])) if "fps" in z.files else np.float32(1 / 30),
        })
        stats[split] += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}  {stats}", flush=True)

    for w in writers.values():
        w.close()
    print(f"DONE {stats}", flush=True)
    if stats["train"] == 0 or stats["val"] == 0:
        print("WARNING: a split is empty — check val_files routing", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
