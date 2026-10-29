#!/usr/bin/env python3
"""Fit rl_push particle and pose scales and write a normalization artifact with an ``rl_push`` entry.

  PYTHONPATH=. python scripts/dev/rl_push_stats.py --out stats/normalization_rl_push.json \
      --artifact_version rl-push-xy-p99-<date>

Estimator of data/stats_fitting.py (_half_ranges): per-component absolute p99 (linear); xy = max(p99|x|, p99|y|),
and likewise for the rotation-vector channels; componentwise maximum over the window policies. Rows come from the
adapter's own raw training batches through data.stats_fitting.raw_motion_rows: particle steps between every adjacent
pair of frames of each C+F window, and the future per-link rigid-body pose deltas (not articulated). Windows:
'prefix' = every usable training episode at the first allowed frame; 'uniform' = one seeded random start per usable
episode, the stand-in for training windows, which are uniform over starts. Particle rows keep --particles random
particles per window (every frame pair), so each window weighs the same. The training split applies the registered
rl_push manifests (validation objects and outlier episodes). The artifact is the base artifact (default: the frozen
stats/normalization.json) with particle.rl_push and pose.rl_push added to the per-domain profile; every reference
entry and the global profile are copied unchanged. The base file is never written; select the result with
normalization.artifact=<out>.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.blocklists import manifest_path
from data.normalization import DEFAULT_ARTIFACT, load_normalization_artifact, resolve_normalization
from data.rl_push import GRIPPER_LINKS, UnifiedRLPushDataset, push_batch, window_points
from data.stats_fitting import _half_ranges, raw_motion_rows
from data.unified_dataset import RL_PUSH_ROOT, exclusion_files
from utils.reproducibility import code_revision


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def fit(ds, policies, particles, seed, chunk):
    C, F = ds.context_len, ds._future()
    L = C + F
    usable = [(i, e) for i, e in ds.episodes if int(ds.objects[i].lengths[e]) >= ds.first_start + L]
    rng = np.random.default_rng(seed)
    result = {}
    for policy in policies:
        particle_rows, pose_rows, windows = [], [], []

        def flush():
            batch = push_batch(windows, context_len=C, delta_stats=ds.delta_stats, position_stats=ds.position_stats,
                               pose_stats=ds._pose, link_counts=ds.link_counts, emit_sidecar=True)
            B, T, N, _ = batch.positions.shape
            steps, poses = raw_motion_rows(batch, {"valid_particle_mask": torch.ones(B, T, N, dtype=torch.bool)})
            keep = np.stack([rng.choice(N, particles, replace=False) for _ in range(B)])
            particle_rows.append(np.take_along_axis(steps.reshape(B, T - 1, N, 3), keep[:, None, :, None], axis=2).reshape(-1, 3))
            pose_rows.append(poses)
            windows.clear()

        for i, e in usable:
            obj = ds.objects[i]
            start = (ds.first_start if policy == "prefix" else
                     int(rng.integers(ds.first_start, int(obj.lengths[e]) - L + 1)))
            windows.append(window_points(obj, e, start, L, ds.links, ds._object_rows(obj, rng)))
            if len(windows) == chunk:
                flush()
        if windows:
            flush()
        p, r = np.concatenate(particle_rows), np.concatenate(pose_rows)
        result[policy] = {"windows": len(usable), "particle_rows": int(len(p)), "pose_rows": int(len(r)),
                          "particle_xy_z_m": _half_ranges(p).tolist(),
                          "pose_txy_tz_rxy_rz": _half_ranges(r, pose=True).tolist()}
    particle = np.max([v["particle_xy_z_m"] for v in result.values()], axis=0)
    pose = np.max([v["pose_txy_tz_rxy_rz"] for v in result.values()], axis=0)
    return result, [float(np.float32(x)) for x in particle], [float(np.float32(x)) for x in pose], len(usable)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=RL_PUSH_ROOT)
    ap.add_argument("--context_len", type=int, default=4)
    ap.add_argument("--future_len", type=int, default=16)
    ap.add_argument("--policies", nargs="+", choices=("prefix", "uniform"), default=["prefix", "uniform"])
    ap.add_argument("--particles", type=int, default=32, help="random particles kept per window for particle rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--base", default=str(DEFAULT_ARTIFACT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifact_version", required=True)
    ap.add_argument("--manifests", nargs="*", help="rl_push manifests to apply instead of the registered ones; "
                    "without a validation object list among them the whole root is the training split "
                    "(a release's train folder)")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise FileExistsError(a.out)

    manifests = exclusion_files("rl_push", "train") if a.manifests is None else list(a.manifests)
    from data.rl_push import read_manifests
    split = "train" if read_manifests(manifests)[0] is not None else "all"
    ds = UnifiedRLPushDataset(a.root, split=split, context_len=a.context_len, future_len=a.future_len,
                              training=False, manifests=manifests, raw=True)
    policies, particle, pose, usable = fit(ds, a.policies, a.particles, a.seed, a.chunk)

    base = json.load(open(a.base))
    per_domain = base["profiles"]["xy_p99_per_domain"]
    if "rl_push" in per_domain["particle"] or "rl_push" in per_domain["pose"]:
        raise ValueError(f"{a.base} already has an rl_push entry")
    art = copy.deepcopy(base)
    art["profiles"]["xy_p99_per_domain"]["particle"]["rl_push"] = particle
    art["profiles"]["xy_p99_per_domain"]["pose"]["rl_push"] = pose
    art["artifact_version"] = a.artifact_version
    mtimes = [os.path.getmtime(os.path.join(ds.root, rel)) for rel in ds.shards]
    art["provenance"] = {
        "kind": "rl_push_entry_added",
        "rl_push": "RL push release particle and pose scales for the opt-in rl_push domain; every other entry and "
                   "the global profile are copied from the base artifact.",
        "base_artifact_version": base["artifact_version"], "base_file_sha256": _sha256(a.base),
        "fit": {"script": "scripts/dev/rl_push_stats.py", "release": os.path.realpath(a.root), "split": split,
                "objects": len(ds.objects), "episodes": len(ds.episodes), "usable_episodes": usable,
                "excluded_episodes": ds.excluded_episodes, "object_files_without_episodes": ds.empty_files,
                "object_files_modified": [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) for t in (min(mtimes), max(mtimes))],
                "manifests_sha256": {str(manifest_path(p)): _sha256(manifest_path(p)) for p in manifests},
                "context_len": a.context_len, "future_len": a.future_len, "first_start": ds.first_start,
                "layout": {"gripper_points": list(ds.link_counts), "object_points": ds.object_points,
                           "gripper_links_sha256": _sha256(GRIPPER_LINKS)},
                "estimator": "p99(abs) linear per component (data.stats_fitting._half_ranges); xy = max over x and y, "
                             "rotation-vector xy likewise; componentwise maximum over the policies used",
                "particle_rows": f"{a.particles} random particles per window x every adjacent frame pair of the window",
                "pose_rows": "future transitions of every rigid body, per-link world-frame deltas",
                "policies_used": list(a.policies), "seed": a.seed, "policies": policies, **code_revision()},
        "base_provenance": base["provenance"],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "x") as f:
        json.dump(art, f, indent=1)
    load_normalization_artifact(a.out)                     # schema + global-aggregation check
    snap = resolve_normalization({"profile": "xy_p99_per_domain", "artifact": a.out})
    print(json.dumps({"out": a.out, "usable_episodes": usable,
                      "particle_xy_z_mm": [round(s * 1e3, 2) for s in snap["scales"]["particle"]["rl_push"]],
                      "pose_txy_tz_mm_rxy_rz_rad": [round(s * 1e3, 2) for s in snap["scales"]["pose"]["rl_push"][:2]]
                      + [round(s, 4) for s in snap["scales"]["pose"]["rl_push"][2:]],
                      "policies": {k: {"particle_xy_z_mm": [round(s * 1e3, 2) for s in v["particle_xy_z_m"]],
                                       "pose": [round(s, 5) for s in v["pose_txy_tz_rxy_rz"]]} for k, v in policies.items()},
                      "snapshot_sha256": snap["sha256"]}, indent=1))


if __name__ == "__main__":
    main()
