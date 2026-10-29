"""Reference SIM1 gripper for the cloth adapter's articulation (data/embodiments/sim1_gripper.npz).

The cloth corpus records particles only. Both grippers are one parallel-jaw geometry whose points
come in the same order in every episode, so a finger's opening direction in any episode is the
reference direction rotated by the registration of that episode's hand points to the reference hand.
This script takes the first validation episode whose left fingers slide by more than 30 mm, stores
its left hand points (frame-0 centred) and each finger's slide direction in that hand's frame (the
principal direction of the finger's hand-frame offset, oriented away from the other finger), and
checks the transfer on the first --check validation episodes (both grippers).

    PYTHONPATH=. python scripts/dev/make_sim1_gripper_reference.py --out data/embodiments/sim1_gripper.npz
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from data.unified_dataset import build_domain_dataset
from utils.rigid_ops import kabsch_batched


def finger_offsets(poses, hand, finger):
    """The finger's origin in the hand's body frame per frame [T, 3]; poses [T, K, 4, 4]."""
    Rh, th = poses[:, hand, :3, :3], poses[:, hand, :3, 3]
    return (Rh.transpose(-1, -2) @ (poses[:, finger, :3, 3] - th)[..., None])[..., 0]


def slide(offsets):
    """(principal direction, range m, max distance from that line m) of a finger's offsets."""
    d = offsets - offsets.mean(0)
    u = torch.linalg.svd(d, full_matrices=False).Vh[0]
    along = d @ u
    return u, float(along.max() - along.min()), float((d - along[:, None] * u).norm(dim=-1).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", type=int, default=40)
    a = ap.parse_args()
    ds = build_domain_dataset("cloth", split="val", training=False, context_len=4,
                              normalization="xy_p99_per_domain")
    ds.emit_sidecar = True
    ds.future_len = 10_000                                   # full episodes
    keys, process = [], ds._process_sample

    def record(sample):
        keys.append({k: str(sample.get(k, "")) for k in ("__url__", "__key__")})
        return process(sample)

    ds._process_sample = record
    episodes = []
    for i, b in enumerate(ds):
        if i >= a.check:
            break
        episodes.append((keys[-1], b.body_poses[0].double(), b.body_templates[0].double(),
                         b.body_template_mask[0]))
    ref = None
    for key, poses, tmpl, mask in episodes:
        left, right = finger_offsets(poses, 1, 2), finger_offsets(poses, 1, 3)
        (ul, range_l, line_l), (ur, range_r, line_r) = slide(left), slide(right)
        if range_l > 0.03:
            sep = left[0] - right[0]                          # right finger -> left finger
            ul = ul if float(ul @ sep) > 0 else -ul
            ur = ur if float(ur @ -sep) > 0 else -ur
            ref = dict(key=key, hand=tmpl[1][mask[1]], axes=torch.stack([ul, ur]),
                       range_mm=[range_l * 1e3, range_r * 1e3], line_mm=[line_l * 1e3, line_r * 1e3])
            break
    assert ref is not None, "no validation episode with a >30 mm finger slide among the checked ones"
    worst = {"registration_mm": 0.0, "axis_deg": 0.0, "moving_fingers": 0}
    for key, poses, tmpl, mask in episodes:
        for hand, fingers in ((1, (2, 3)), (4, (5, 6))):
            P = tmpl[hand][mask[hand]]
            R, _ = kabsch_batched(ref["hand"][None], P[None])
            worst["registration_mm"] = max(worst["registration_mm"],
                                           float((ref["hand"] @ R[0].T - P).norm(dim=-1).max()) * 1e3)
            for j, f in enumerate(fingers):
                u, rng, _ = slide(finger_offsets(poses, hand, f))
                if rng > 0.005:
                    ang = float(torch.rad2deg(torch.arccos((R[0] @ ref["axes"][j]).dot(u).abs().clamp(max=1.0))))
                    worst["axis_deg"] = max(worst["axis_deg"], ang)
                    worst["moving_fingers"] += 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, hand_points=ref["hand"].float().numpy(), finger_axes=ref["axes"].float().numpy())
    receipt = {"source": ref["key"], "hand_points": int(ref["hand"].shape[0]),
               "finger_axes": ref["axes"].tolist(), "reference_slide_range_mm": ref["range_mm"],
               "reference_line_residual_mm": ref["line_mm"],
               "left_right_axis_angle_deg": float(torch.rad2deg(torch.arccos(
                   (-ref["axes"][0]).dot(ref["axes"][1]).clamp(-1.0, 1.0)))),
               "transfer_check": {"episodes": len(episodes), **worst},
               "revision": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                          text=True).stdout.strip(),
               "made": time.strftime("%Y-%m-%d %H:%M")}
    json.dump(receipt, open(a.out.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
