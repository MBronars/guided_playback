#!/usr/bin/env python3
"""
infer.py — Run a trained PointWorld checkpoint on one window. Inference only.

Part of baselines/pointworld, a vendored inference-only copy of PointWorld (Apache-2.0; see
LICENSE and PROVENANCE.md). Nothing here imports PointWorld's trainer or evaluator, or any of
pwm_final_paper: the dependency set is the model, its preprocessing transforms, and the four
compiled packages README.md lists. PointWorld's own top-level `utils.py` is `pw_utils.py` here,
so it cannot shadow this repo's `utils` package.

WHAT ONE CALL DOES
    In:  the scene's points at the anchor frame          (Ns, 3) metres
         the actuator's points over the whole window     (1 + H, Nr, 3) metres, the anchor
                                                          frame first, then H given frames
    Out: the scene's predicted points                    (H, Ns, 3) metres, row for row with
                                                          the scene points that went in

    H is the checkpoint's own prediction horizon (`pred_horizon` in its model contract, 10 for
    the release, 16 for the runs trained with --pred_horizon 16). The actuator is GIVEN: the
    model reads its whole trajectory and never predicts it.

WHAT YOU NEED BESIDES THE CODE
    1. the checkpoint (.pt), and
    2. the normalization statistics folder it was trained with. The checkpoint's model contract
       records where that folder was on the training machine; copy the folder along and pass
       --norm_stats, since that path will not exist elsewhere.

ROW ORDER
    PointWorld voxel-downsamples the scene at `grid_size` (1 mm in these runs), which reorders
    the cloud and keeps one point per voxel. This script maps every input row back to the
    surviving point nearest its anchor-frame position, so the output is row-for-row with the
    input; rows that shared a voxel share that voxel's trajectory. `--report` prints how many
    rows that affected and the largest mapping distance.

USAGE (from anywhere; README.md has the environment)
    python baselines/pointworld/infer.py --checkpoint <shared>/matched_step_060000.pt \\
        --stats three_domain_h10 --selftest --report
    python baselines/pointworld/infer.py --checkpoint ... --stats dex_h16 \\
        --window window.npz --out pred.npz
    # window.npz: scene (Ns, 3) and robot (1 + H, Nr, 3), metres
    # --stats names a bundled ruler in norm_stats/; --norm_stats takes any folder instead.

AS A LIBRARY
    from infer import PointWorldInference
    model = PointWorldInference(checkpoint, norm_stats, device="cuda")
    pred = model.predict(scene0, robot)        # (H, Ns, 3)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from functools import partial

import pathlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


class _Chain:
    """The `.map(f, handler=...)` interface PointWorld's transform pipeline chains onto."""

    def __init__(self, source):
        self._source = source

    def __iter__(self):
        return iter(self._source() if callable(self._source) else self._source)

    def map(self, f, handler=None):
        parent = self

        def mapped():
            for sample in parent:
                try:
                    out = f(sample)
                except Exception as exc:                      # webdataset's handler contract
                    if handler is None or not handler(exc):
                        raise
                    continue
                if out is not None:
                    yield out

        return _Chain(mapped)


class PointWorldInference:
    """A checkpoint, its preprocessing, and one `predict` call."""

    def __init__(self, checkpoint: str, norm_stats: str, device: str = "cuda", domain: str | None = None):
        from arguments import parse_args
        from pointworld.base import BaseModel
        from pointworld.checkpoint_contract import read_checkpoint_contract, apply_model_contract_to_args

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_contract, data_contract = read_checkpoint_contract(state, context=checkpoint)
        self.domains = data_contract["domains"]
        # The pipeline branches on the domain name; any of the checkpoint's own names gives the
        # same path for these runs (all of them carry their points in the sample).
        self.domain = domain or self.domains[0]

        # norm_stats is NOT passed on this argv: the contract refuses a norm_stats_path that differs
        # from the training machine's, and that path does not exist here. It is set below instead,
        # after the contract has been applied — the statistics must still be the ones this
        # checkpoint trained with (README.md says which).
        argv = ["infer", "--domains", ",".join(self.domains), "--data_dirs", ",".join(["x"] * len(self.domains)),
                "--robot_features", "robot_flows,robot_velocity,robot_acceleration",
                "--scene_features", "scene_flows,dist2robot"]
        old, sys.argv = sys.argv, argv
        try:
            self.args = parse_args()
        finally:
            sys.argv = old
        apply_model_contract_to_args(self.args, model_contract, context=checkpoint)
        self.trained_norm_stats_path = model_contract["norm_stats_path"]   # where it was on our machine
        self.args.norm_stats_path = norm_stats
        self.args.device = device
        self.horizon = int(getattr(self.args, "pred_horizon", 10) or 10)
        self.T = self.horizon + 1

        self.model = BaseModel(self.args, {"scene_features_dim": 3 + self.T, "robot_features_dim": 9}, rank=0)
        self.model.load_state_dict(state["model"])
        self.model.to(device).eval()
        self.stats = {"rows": 0, "rows_sharing_a_voxel": 0, "max_match_mm": 0.0}

    # --- the transform chain, without the dataloader
    def _batch(self, scene0: np.ndarray, robot: np.ndarray):
        from dataset_components.cameras import sample_cameras
        from dataset_components.collate import custom_collate_fn
        from dataset_components.decoders import build_flow_sample, decode_data
        from dataset_components.pipeline import sample_transform_pipeline
        from dataset_components.robot import canonicalize_gripper_keys_and_flags

        zeros = np.zeros((self.T, 1), dtype=np.float32)
        shift = np.concatenate([scene0, robot[0]], axis=0).mean(axis=0).astype(np.float32)
        sample = {
            "view0_scene_flows.npy": _npy_bytes(np.repeat((scene0 - shift)[None], self.T, axis=0).astype(np.float32)),
            "view0_scene_visibility.npy": _npy_bytes(np.ones((self.T, len(scene0)), dtype=bool)),
            "robot_flows.npy": _npy_bytes((robot - shift).astype(np.float32)),
            "joint_positions.npy": _npy_bytes(zeros),
            "right_gripper_open.npy": _npy_bytes(zeros),
            "left_gripper_open.npy": _npy_bytes(zeros),
        }
        seed = int(self.args.seed)
        chain = (_Chain([sample])
                 .map(partial(decode_data, domain=self.domain))
                 .map(partial(build_flow_sample, domain=self.domain, robot_sampler=None,
                              max_robot_points=self.args.max_robot_points, deterministic=True, seed=seed))
                 .map(partial(sample_cameras, min_num_cameras=self.args.eval_min_num_cameras,
                              max_num_cameras=self.args.eval_max_num_cameras, deterministic=True, seed=seed))
                 .map(canonicalize_gripper_keys_and_flags))
        chain = sample_transform_pipeline(chain, "infer://", self.domain, "test", self.args, False, rank=0)
        items = list(chain)
        if len(items) != 1:
            raise RuntimeError(f"preprocessing dropped the window ({len(items)} samples out)")
        return custom_collate_fn(items, self.args), shift

    @torch.no_grad()
    def predict(self, scene0: np.ndarray, robot: np.ndarray) -> np.ndarray:
        """scene0 (Ns, 3), robot (1 + H, Nr, 3), metres -> (H, Ns, 3) metres, row for row."""
        from scipy.spatial import cKDTree
        scene0 = np.ascontiguousarray(scene0, dtype=np.float32)
        robot = np.ascontiguousarray(robot, dtype=np.float32)
        if robot.shape[0] != self.T:
            raise ValueError(f"robot needs {self.T} frames (1 anchor + {self.horizon} given), got {robot.shape[0]}")
        batch, shift = self._batch(scene0, robot)
        gpu = {k: (v.to(self.args.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        out = self.model(gpu, training=False)
        pred = out["scene_flows"][0].float().cpu().numpy()
        exists = batch["scene_exists"][0, 0].numpy().astype(bool)
        # __shift_amount__ holds MINUS the pipeline's accumulated centring shift.
        pipe_shift = (np.asarray(batch["__shift_amount__"][0], dtype=np.float32).reshape(-1)[:3]
                      if "__shift_amount__" in batch else np.zeros(3, np.float32))
        back = lambda a: a - pipe_shift + shift
        kept0 = back(batch["scene_flows"][0, 0].float().numpy()[exists])
        dist, idx = cKDTree(kept0).query(scene0)
        self.stats = {"rows": int(len(scene0)),
                      "rows_sharing_a_voxel": int(len(scene0) - len(np.unique(idx))),
                      "max_match_mm": float(dist.max()) * 1000.0}
        return back(pred[:, exists])[1:][:, idx]


def _selftest(model: PointWorldInference) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A synthetic window in the shape real ones have: a small cloud and a gripper sliding 2 cm."""
    rng = np.random.default_rng(0)
    scene0 = (rng.normal(scale=0.03, size=(512, 3))).astype(np.float32)
    base = rng.normal(scale=0.02, size=(671, 3)) + np.array([0.05, 0.0, 0.0])
    robot = (base[None] + np.linspace(0, 0.02, model.T)[:, None, None] * np.array([1.0, 0, 0])).astype(np.float32)
    return scene0, robot, model.predict(scene0, robot)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stats", choices=sorted(x.name for x in (pathlib.Path(HERE) / "norm_stats").iterdir() if x.is_dir()),
                   help="a ruler bundled with this folder; must be the one the checkpoint trained with")
    p.add_argument("--norm_stats", help="or any folder holding norm_stats.json")
    p.add_argument("--window", help="npz with `scene` (Ns,3) and `robot` (1+H,Nr,3), metres")
    p.add_argument("--out", help="npz to write `pred` (H,Ns,3) into")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--selftest", action="store_true", help="run a synthetic window instead of --window")
    p.add_argument("--report", action="store_true", help="print the row-mapping statistics")
    a = p.parse_args()
    if not a.window and not a.selftest:
        p.error("pass --window or --selftest")
    if bool(a.stats) == bool(a.norm_stats):
        p.error("pass exactly one of --stats (a bundled ruler) or --norm_stats (a folder)")
    stats = a.norm_stats or os.path.join(HERE, "norm_stats", a.stats)

    model = PointWorldInference(a.checkpoint, stats, device=a.device)
    print(f"checkpoint: horizon {model.horizon} frames, domains {model.domains}, device {a.device}")
    if a.selftest:
        scene0, robot, pred = _selftest(model)
    else:
        w = np.load(a.window)
        scene0, robot = w["scene"], w["robot"]
        pred = model.predict(scene0, robot)
    moved = np.linalg.norm(pred - scene0[None], axis=-1).mean(axis=1) * 1000
    print(f"prediction {pred.shape}: mean motion from the anchor, per frame (mm): "
          + ", ".join(f"{v:.1f}" for v in moved))
    if a.report:
        print(f"rows {model.stats['rows']}, sharing a voxel {model.stats['rows_sharing_a_voxel']}, "
              f"largest mapping distance {model.stats['max_match_mm']:.3f} mm")
    if a.out:
        np.savez(a.out, pred=pred.astype(np.float32), scene=scene0.astype(np.float32), robot=robot.astype(np.float32))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
