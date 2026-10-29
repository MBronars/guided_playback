#!/usr/bin/env python3
"""Extract the pick-and-place corpus's gripper link point sets, for adapters that build points from poses.

  PYTHONPATH=. python scripts/dev/pickup_gripper_points.py --out data/embodiments/panda_gripper_pickup.npz

Reads the first record of the first shard of each registered pickup root (data.unified_dataset.PICKUP_ROOTS)
and checks, before writing anything, that
- the gripper rows of particle_pos equal link pose @ link_local_pts for panda_hand, panda_leftfinger and
  panda_rightfinger (link rows 8, 9 and 10, in that particle order; link quaternions wxyz);
- the three link point sets are identical in every root;
- each finger link has no rotation relative to the hand and sits at (0, +-q, 58.4 mm) in the hand frame, q being
  that finger's joint position.
Writes the three point sets (link frames, metres) and a JSON receipt next to them. Never overwrites.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.unified_dataset import PICKUP_ROOTS
from utils.reproducibility import code_revision

ROWS = {"panda_hand": 8, "panda_leftfinger": 9, "panda_rightfinger": 10}
FINGER_Z = 0.0584


def _rotation(q):
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def first_record(tar_path):
    fields = {}
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            key, name = member.name.split(".", 1)
            if "__key__" in fields and key != fields["__key__"]:
                break
            fields["__key__"] = key
            fields[name] = pickle.loads(tf.extractfile(member).read())
    return fields


def check_root(root):
    shard = sorted(glob.glob(f"{root}/**/*.tar", recursive=True))[0]
    r = first_record(shard)
    P = np.asarray(r["particle_pos.pyd"], np.float64)               # [B, T, N, 3]
    lp, lq = np.asarray(r["link_pos.pyd"], np.float64), np.asarray(r["link_quat.pyd"], np.float64)
    pts, npts = np.asarray(r["link_local_pts.pyd"], np.float64), np.asarray(r["link_local_npts.pyd"]).astype(int)
    joints = np.asarray(r["joint_pos_obs.pyd"], np.float64)           # [B, T, 9]
    links, off, recon = {}, 0, 0.0
    for name, row in ROWS.items():
        n = int(npts[row])
        links[name] = pts[row, :n]
        rebuilt = np.einsum("btij,pj->btpi", _rotation(lq[:, :, row]), links[name]) + lp[:, :, row, None]
        recon = max(recon, float(np.abs(rebuilt - P[:, :, off:off + n]).max()))
        off += n
    if not (np.asarray(r["obj_ids.pyd"])[:, :off] == 1).all():
        raise ValueError(f"{shard}: gripper particles are not the first {off} rows")
    Rh = _rotation(lq[:, :, ROWS["panda_hand"]])
    fingers = {}
    for name, col, sign in (("panda_leftfinger", 7, 1.0), ("panda_rightfinger", 8, -1.0)):
        row = ROWS[name]
        R_rel = np.einsum("btji,btjk->btik", Rh, _rotation(lq[:, :, row]))
        angle = np.degrees(np.arccos(np.clip((np.trace(R_rel, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
        t_rel = np.einsum("btji,btj->bti", Rh, lp[:, :, row] - lp[:, :, ROWS["panda_hand"]])
        expected = np.stack([np.zeros_like(joints[..., col]), sign * joints[..., col], np.full_like(joints[..., col], FINGER_Z)], -1)
        fingers[name] = {"rotation_deg_max": float(angle.max()), "translation_err_m_max": float(np.abs(t_rel - expected).max())}
    return {"shard": shard, "key": r["__key__"], "particle_reconstruction_err_m_max": recon, "fingers": fingers}, links


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    receipt_path = os.path.splitext(a.out)[0] + ".json"
    for path in (a.out, receipt_path):
        if os.path.exists(path):
            raise FileExistsError(path)
    checks, sets = [], []
    for root in PICKUP_ROOTS:
        check, links = check_root(root)
        checks.append(check)
        sets.append(links)
    for check in checks:
        if check["particle_reconstruction_err_m_max"] > 1e-5:
            raise ValueError(f"{check['shard']}: gripper particles are not link pose @ link points")
        for name, f in check["fingers"].items():
            if f["rotation_deg_max"] > 1e-3 or f["translation_err_m_max"] > 1e-5:
                raise ValueError(f"{check['shard']}: {name} is not at (0, +-q, {FINGER_Z}) m in the hand frame")
    for links in sets[1:]:
        if any(not np.array_equal(links[k], sets[0][k]) for k in ROWS):
            raise ValueError("pickup roots carry different gripper link point sets")
    np.savez(a.out, **{k: v.astype(np.float32) for k, v in sets[0].items()})
    receipt = {"points": {k: int(len(v)) for k, v in sets[0].items()}, "frames": "link frames, metres",
               "finger_convention": f"finger link = hand pose with translation (0, +q_left, {FINGER_Z}) or "
                                    f"(0, -q_right, {FINGER_Z}) m and no rotation of its own",
               "identical_across_roots": True, "roots": checks,
               "script": "scripts/dev/pickup_gripper_points.py", "made": time.strftime("%Y-%m-%d %H:%M"), **code_revision()}
    with open(receipt_path, "x") as f:
        json.dump(receipt, f, indent=1)
    print(json.dumps({"out": a.out, "points": receipt["points"]}))


if __name__ == "__main__":
    main()
