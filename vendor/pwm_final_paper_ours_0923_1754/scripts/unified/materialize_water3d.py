#!/usr/bin/env python3
"""Materialize DeepMind Water-3D TFRecords into one .npy per trajectory (the water domain's storage).

    python scripts/unified/materialize_water3d.py \
        --src /data/group_data/katefgroup/datasets/deepmind_lts/Water-3D \
        --out /data/group_data/katefgroup/datasets/deepmind_lts/Water-3D/materialized_v1 --splits train valid test

A TFRecord is one sequential stream of whole trajectories (~75 MB each), unusable for random window
access. Every trajectory becomes <out>/<split>/<key:05d>.npy: float32 positions [T, N, 3] in the file's own
frame (y up, box units, every file frame); the adapter owns the frame mapping and the frame stride, and
reads only the frames a window needs (memory-mapped). <out>/<split>/manifest.json lists, per trajectory,
key, file, frames, particles and the fraction of rows outside the metadata bounds, with the source file and
its size; metadata.json is copied to <out>. Every particle must be water (type 5). A split directory that
already exists is refused, never overwritten; a split is written under <split>.partial and renamed last.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.water3d import WATER_PARTICLE_TYPE, iter_water3d, load_metadata


def materialize_split(src: str, out: str, split: str, metadata: dict) -> dict:
    tfrecord = os.path.join(src, f"{split}.tfrecord")
    final, partial = os.path.join(out, split), os.path.join(out, f"{split}.partial")
    for path in (final, partial):
        if os.path.exists(path):
            raise FileExistsError(f"{path} exists; materialized splits are never overwritten")
    os.makedirs(partial)
    rows, keys, t0 = [], set(), time.time()
    for tr in iter_water3d(tfrecord, metadata):
        if tr.key in keys:
            raise ValueError(f"{split}: duplicate trajectory key {tr.key}")
        if not bool((tr.particle_type == WATER_PARTICLE_TYPE).all()):
            raise ValueError(f"{split} key {tr.key}: particle types {sorted(set(tr.particle_type.tolist()))}, expected water only")
        keys.add(tr.key)
        name = f"{tr.key:05d}.npy"
        np.save(os.path.join(partial, name), np.ascontiguousarray(tr.positions, dtype=np.float32))
        rows.append({"key": tr.key, "file": name, "frames": tr.num_frames, "particles": tr.num_particles,
                     "outside_bounds": tr.outside_bounds})
        if len(rows) % 100 == 0:
            print(f"[{split}] {len(rows)} trajectories, {time.time() - t0:.0f} s", flush=True)
    manifest = {"split": split, "source": os.path.realpath(tfrecord), "source_bytes": os.path.getsize(tfrecord),
                "frame": "file frame: y up, box units (unit box = 1 m in the water domain)",
                "dt_file_s": float(metadata["dt"]), "bounds": metadata["bounds"],
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "trajectories": rows}
    with open(os.path.join(partial, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    os.rename(partial, final)
    return {"split": split, "trajectories": len(rows), "seconds": round(time.time() - t0, 1),
            "particles_min": min(r["particles"] for r in rows), "particles_max": max(r["particles"] for r in rows)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="directory with metadata.json and {train,valid,test}.tfrecord")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", choices=("train", "valid", "test"), default=["train", "valid", "test"])
    a = ap.parse_args(argv)
    metadata = load_metadata(a.src)
    os.makedirs(a.out, exist_ok=True)
    meta_out = os.path.join(a.out, "metadata.json")
    if not os.path.exists(meta_out):
        shutil.copy2(os.path.join(a.src, "metadata.json"), meta_out)
    elif json.load(open(meta_out)) != metadata:
        raise ValueError(f"{meta_out} differs from the source metadata")
    for split in a.splits:
        print(json.dumps(materialize_split(a.src, a.out, split, metadata)), flush=True)


if __name__ == "__main__":
    main()
