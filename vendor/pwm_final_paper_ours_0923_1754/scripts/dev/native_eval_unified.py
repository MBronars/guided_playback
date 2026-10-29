"""Native episodes through the production evaluator (same code path, receipts and
dumps as the pickup validation run).

    PYTHONPATH=. python scripts/dev/native_eval_unified.py --checkpoint <ckpt> \
        --episodes /data/user_data/robertre/yikai_native_0910/*/trajectory.npz --layout pickup \
        --rollout --rollout_frames 48 --seed 0 --dump_dir <dir> --dump_pick max_motion --dump_k 2 --out <json>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch

from data.native_episodes import NativeEpisodeDataset
from data.unified_dataset import DOMAIN_FUTURE_LEN
from inference import load_model_and_diffusion
from scripts.unified.eval_unified import eval_domain, hides_given_actions, rollout_domain
from utils.reproducibility import code_revision, file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--layout", default="pickup", choices=["pickup", "scene", "all_parts", "moving_only"])
    p.add_argument("--windows_per_episode", type=int, default=3)
    p.add_argument("--rollout", action="store_true")
    p.add_argument("--rollout_frames", type=int, default=48)
    p.add_argument("--dump_dir", default=None)
    p.add_argument("--dump_pick", default="first", choices=["first", "max_motion"])
    p.add_argument("--dump_k", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clamp_actuated", action="store_true", help="see scripts/unified/eval_unified.py")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, diffusion, cfg, _ = load_model_and_diffusion(args.checkpoint, device)
    model.eval()
    hide_given = hides_given_actions(cfg)
    trained_f = int(cfg.data.get("future_len", -1) or -1)
    C = int(cfg.data.context_len)
    F = trained_f if trained_f > 0 else DOMAIN_FUTURE_LEN["pickup"]
    paths = sorted(sum((glob.glob(e) for e in args.episodes), []))
    ds = NativeEpisodeDataset(paths, context_len=C, future_len=F, windows_per_episode=args.windows_per_episode,
                              layout=args.layout, normalization=cfg.normalization)
    results = {"checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint),
               "episodes": paths, "layout": args.layout,
               "windows": [(os.path.basename(os.path.dirname(paths[i])), s) for i, s in ds.items],
               "clips_requested": "all", "seed": args.seed, "rollout": args.rollout, "context_len": C, "future_len": F,
               "clamp_actuated": bool(args.clamp_actuated), "actions": "hidden" if hide_given else "given",
               "normalization": ds.normalization, "inference_steps": int(cfg.flow_matching.inference_steps),
               "df_schedule": cfg.get("df_schedule", "pyramid"), "torch_version": str(torch.__version__), **code_revision()}
    if args.rollout:
        results["rollout_frames_requested"] = args.rollout_frames
        results["pickup"] = rollout_domain(model, diffusion, cfg, "pickup", None, device, horizon=args.rollout_frames,
                                           dump_dir=args.dump_dir, dump_pick=args.dump_pick, dump_k=args.dump_k,
                                           future_len=F, context_len=C, dataset=ds, clamp_actuated=args.clamp_actuated,
                                           seed=args.seed, hide_given_actions=hide_given)
    else:
        results["pickup"] = eval_domain(model, diffusion, cfg, "pickup", None, device, future_len=F, context_len=C, dataset=ds,
                                        clamp_actuated=args.clamp_actuated, seed=args.seed,
                                        hide_given_actions=hide_given)
    print(results["pickup"])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
