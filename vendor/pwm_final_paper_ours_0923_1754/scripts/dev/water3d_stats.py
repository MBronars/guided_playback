#!/usr/bin/env python3
"""Fit Water-3D per-frame delta scales and write a normalization artifact with a ``water`` entry.

Estimator of data/stats_fitting.py: per-component absolute p99 (linear) of the future
deltas of C/F windows, xy = max(p99|dx|, p99|dy|), z = p99|dz|, componentwise maximum
over the chosen window policies ('prefix' = the first window of every trajectory,
'uniform' = a seeded sample of every window start, the stand-in for training windows,
which are uniform over start frames). Deltas are measured in the water domain's model
frame (z up, unit box = 1 m) at its frame stride. The artifact is the base artifact
(default: the frozen stats/normalization.json) with ``particle.water`` added to the
per-domain profile; every reference entry and the global profile are copied unchanged
(water is never part of the global aggregation). The base file is never written; select
the result with ``normalization.artifact=<out>``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.normalization import DEFAULT_ARTIFACT, load_normalization_artifact, resolve_normalization
from data.water3d import WATER3D_SHIFT, WATER3D_STRIDE, iter_water3d, load_metadata, to_model_frame


def _f32(x):
    return float(np.float32(x))


def fit(tfrecord, metadata, stride, context_len, future_len, uniform_rows, seed, use=("prefix", "uniform")):
    rng = np.random.default_rng(seed)
    rows = {"prefix": [], "uniform": []}
    n = 0
    for tr in iter_water3d(tfrecord, metadata):
        d = np.diff(to_model_frame(tr.positions[::stride]), axis=0)       # delta into frame t + 1
        rows["prefix"].append(d[context_len - 1:context_len - 1 + future_len].reshape(-1, 3))
        u = d[context_len - 1:].reshape(-1, 3)                            # every future delta of every window
        rows["uniform"].append(u[rng.choice(len(u), min(len(u), uniform_rows), replace=False)])
        n += 1
    policies = {}
    for name, parts in rows.items():
        r = np.abs(np.concatenate(parts))
        q = np.percentile(r, 99, axis=0, method="linear")
        policies[name] = {"rows": int(len(r)), "p99_abs_xyz_m": q.tolist(),
                          "scale_xy_z_m": [float(max(q[0], q[1])), float(q[2])]}
    scale = [max(policies[p]["scale_xy_z_m"][i] for p in use) for i in (0, 1)]
    return n, policies, [_f32(s) for s in scale]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/data/group_data/katefgroup/datasets/deepmind_lts/Water-3D")
    ap.add_argument("--split", choices=("train", "valid", "test"), default="train")
    ap.add_argument("--stride", type=int, default=WATER3D_STRIDE)
    ap.add_argument("--context_len", type=int, default=4)
    ap.add_argument("--future_len", type=int, default=16)
    ap.add_argument("--uniform_rows", type=int, default=4000, help="uniform-policy rows sampled per trajectory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policies", nargs="+", choices=("prefix", "uniform"), default=["prefix", "uniform"],
                    help="window policies whose componentwise maximum sets the scale")
    ap.add_argument("--base", default=str(DEFAULT_ARTIFACT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifact_version", required=True)
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise FileExistsError(a.out)

    metadata = load_metadata(a.root)
    tfrecord = f"{a.root}/{a.split}.tfrecord"
    n, policies, scale = fit(tfrecord, metadata, a.stride, a.context_len, a.future_len, a.uniform_rows, a.seed, tuple(a.policies))

    base = json.load(open(a.base))
    if "water" in base["profiles"]["xy_p99_per_domain"]["particle"]:
        raise ValueError(f"{a.base} already has a water entry")
    art = copy.deepcopy(base)
    art["profiles"]["xy_p99_per_domain"]["particle"]["water"] = scale
    art["artifact_version"] = a.artifact_version
    art["provenance"] = {
        "kind": "water_entry_added",
        "water": "Water-3D (Learning to Simulate, arXiv 2002.09405) particle scales for the opt-in water domain; "
                 "every other entry and the global profile are copied from the base artifact.",
        "base_artifact_version": base["artifact_version"],
        "base_file_sha256": hashlib.sha256(open(a.base, "rb").read()).hexdigest(),
        "fit": {"script": "scripts/dev/water3d_stats.py", "tfrecord": os.path.realpath(tfrecord),
                "tfrecord_bytes": os.path.getsize(tfrecord), "split": a.split, "trajectories": n,
                "frame_stride": a.stride, "fps": 1.0 / (float(metadata["dt"]) * a.stride),
                "context_len": a.context_len, "future_len": a.future_len,
                "frame": {"axes": "(x, y, z)_file -> (x, -z, y)", "shift_m": list(WATER3D_SHIFT), "unit_box_m": 1.0},
                "particles": "all particles of every trajectory",
                "estimator": "p99(abs) linear per component; xy = max(p99|x|, p99|y|); "
                             "componentwise maximum over the policies used",
                "policies_used": list(a.policies),
                "uniform_rows_per_trajectory": a.uniform_rows, "seed": a.seed, "policies": policies},
        "base_provenance": base["provenance"],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(art, f, indent=1)
    load_normalization_artifact(a.out)                      # schema + global-aggregation check
    snap = resolve_normalization({"profile": "xy_p99_per_domain", "artifact": a.out})
    print(json.dumps({"out": a.out, "split": a.split, "trajectories": n,
                      "water_xy_z_mm": [round(s * 1e3, 2) for s in snap["scales"]["particle"]["water"]],
                      "policies_xy_z_mm": {k: [round(s * 1e3, 2) for s in v["scale_xy_z_m"]]
                                           for k, v in policies.items()},
                      "snapshot_sha256": snap["sha256"]}, indent=1))


if __name__ == "__main__":
    main()
