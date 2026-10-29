"""One production optimizer step on the largest retained folding cloud.

This tests the actual model and loader at native resolution, with no point cap.
Run one architecture per process on an idle GPU. No checkpoint is written.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from accelerate import Accelerator

from data.batch import TrajectoryBatch
from data.indexed_episodes import IndexedEpisodeDataset
from diffusion import TrajectoryFlowMatching
from models.factory import build_particle_model
from scripts.unified.overfit_prod import build_optimizer
from scripts.unified.verify_collections import audit_batch
from train import train_step
from utils.config import load_config
from utils.reproducibility import code_revision, working_source_versions
from utils.trajectory_objective import trajectory_loss_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(916)
    cfg = load_config(args.config, ["train.compile=false"])
    ds = IndexedEpisodeDataset(args.root, domain="folding", normalization=cfg.normalization)
    ds.articulated = True
    record = max(ds.records, key=lambda r: r["particles"])
    start = (record["frames"] - 20) // 2
    batch = TrajectoryBatch.collate([ds.window(record, start), ds.window(record, start + 1)])
    audit = audit_batch(batch)
    print(f"Native maximum shape {list(batch.positions.shape)}; ground truth audit {audit}", flush=True)
    batch = batch.to("cuda")
    model = build_particle_model(cfg).to("cuda")
    optimizer = build_optimizer(model, cfg.train)
    accelerator = Accelerator(mixed_precision=cfg.train.mixed_precision)
    model, optimizer = accelerator.prepare(model, optimizer)
    fm = cfg.flow_matching
    diffusion = TrajectoryFlowMatching(num_train_timesteps=fm.num_train_timesteps,
        shift=fm.shift, prediction_type=fm.prediction_type,
        time_sampler=fm.get("time_sampler", "uniform"))
    torch.cuda.reset_peak_memory_stats()
    start_time = time.monotonic()
    loss, metrics = train_step(model, batch, diffusion, optimizer, accelerator,
        grad_clip=cfg.train.grad_clip, loss_spike_threshold=float("inf"),
        grad_norm_spike_threshold=float("inf"), **trajectory_loss_options(cfg))
    torch.cuda.synchronize()
    if not metrics["optimizer_step"] or metrics["skipped"]:
        raise RuntimeError(f"Native maximum optimizer step rejected: {metrics}")
    receipt = dict(code=code_revision(), source_versions=working_source_versions(), architecture=cfg.model.architecture,
        shape=list(batch.positions.shape), episode=record["episode_id"],
        index_sha256=ds.index["index_sha256"], audit=audit, loss=float(loss),
        grad_norm=float(metrics["grad_norm"]), seconds=time.monotonic() - start_time,
        peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
    with Path(args.out).open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
