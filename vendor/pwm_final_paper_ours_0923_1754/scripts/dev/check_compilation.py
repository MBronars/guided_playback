"""Exercise actual regional compilation on full-size models and pinned real batches.

Checks eager initialization, repeated compiled optimizer updates, and train/eval
transitions. Uses SGD at 1e-6 to bound weight changes; not a throughput benchmark,
training run, or claim of bitwise compiled/eager equivalence.
"""
import argparse
import json
from pathlib import Path
import time

from accelerate import Accelerator
from omegaconf import OmegaConf
import torch

from diffusion import TrajectoryFlowMatching
from inference import run_diffusion_inference_df
from models.factory import build_particle_model
from train import train_step, trajectory_loss_options
from utils.checkpoint import read_checkpoint, validate_resume_normalization
from utils.config import load_config
from utils.compilation import prepare_compilation
from utils.pose_native import materialize_pose_native
from utils.reproducibility import code_revision, file_sha256
from utils.state_dict import load_model_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", nargs="+", required=True, metavar="DOMAIN=PATH")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.train.get("compile") or not cfg.train.get("compile_regional") or args.rounds < 1:
        parser.error("Requires a regional-compile recipe and positive rounds")
    acc = Accelerator(mixed_precision=cfg.train.mixed_precision)
    assert acc.num_processes == 1, "Use the dedicated DDP gates for multi-rank checks"
    torch.set_num_threads(1)
    torch.manual_seed(314)
    model = build_particle_model(cfg)
    state = read_checkpoint(args.checkpoint)
    validate_resume_normalization(state, OmegaConf.to_container(cfg.normalization, resolve=True))
    load_model_state(model, state["ema"] if state.get("ema") else state["model"])
    del state
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    model, optimizer = acc.prepare(model, optimizer)
    raw = acc.unwrap_model(model)
    schema = tuple(raw.state_dict())
    diffusion = TrajectoryFlowMatching(
        num_train_timesteps=cfg.flow_matching.num_train_timesteps,
        shift=cfg.flow_matching.shift, prediction_type=cfg.flow_matching.prediction_type)
    batches, receipts = {}, {}
    for value in args.batches:
        domain, path = value.split("=", 1)
        path = Path(path)
        receipt = json.loads(path.with_suffix(".json").read_text())
        digest = file_sha256(path)
        assert digest == receipt.get("sha256", receipt.get("batch_sha256")), path
        batch = torch.load(path, map_location="cpu", weights_only=False)
        assert batch.context_len == cfg.data.context_len and batch.num_future_frames == cfg.data.future_len
        batches[domain] = batch.to(acc.device)
        receipts[domain] = dict(path=str(path.resolve()), sha256=digest, shape=list(batch.positions.shape))
    info = dict(passed=False, provenance=code_revision(), config=str(args.config),
                checkpoint=str(args.checkpoint.resolve()), batches=receipts,
                parameters=sum(p.numel() for p in raw.parameters()), mixed_precision=cfg.train.mixed_precision,
                note="Engineering gate: B=1 pinned fixtures, eager lr=0 then compiled SGD lr=1e-6; no throughput/generalization claim",
                records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(phase, domain, operation):
        print(f"START {phase}/{domain}", flush=True)
        start = time.monotonic()
        metrics = operation()
        torch.cuda.synchronize()
        row = dict(phase=phase, domain=domain, seconds=time.monotonic() - start, **metrics)
        info["records"].append(row)
        info["peak_allocated_gb"] = torch.cuda.max_memory_allocated(acc.device) / 1e9
        args.output.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    def update(batch):
        model.train()
        with acc.accumulate(model):
            loss, metrics = train_step(model, batch, diffusion, optimizer, acc, grad_clip=1,
                                       grad_norm_spike_threshold=float("inf"), **trajectory_loss_options(cfg))
        assert metrics["optimizer_step"], metrics
        return dict(loss=float(loss), grad_norm=metrics["grad_norm"], accepted=True)

    def sample(batch):
        raw.eval()
        torch.manual_seed(4242)
        with torch.no_grad():
            value = run_diffusion_inference_df(raw, batch, diffusion, num_inference_steps=4,
                                               schedule=cfg.get("df_schedule", "pyramid"))
            pred = (materialize_pose_native(batch, *value)[0] if isinstance(value, tuple)
                    else batch.predictions_to_positions(batch.prediction_stats.denormalize(value)))
        assert torch.isfinite(pred).all()
        return dict(finite=True, shape=list(pred.shape))

    for domain, batch in batches.items():
        record("eager_update_lr0", domain, lambda batch=batch: update(batch))
    model = prepare_compilation(model, cfg.train)
    optimizer.param_groups[0]["lr"] = 1e-6
    for round_index in range(args.rounds):
        for domain, batch in batches.items():
            record(f"compiled_update_{round_index}", domain, lambda batch=batch: update(batch))
        for domain, batch in batches.items():
            record(f"compiled_eval_{round_index}", domain, lambda batch=batch: sample(batch))
    assert tuple(raw.state_dict()) == schema, "Compilation changed checkpoint names"
    from torch._dynamo.utils import counters
    info.update(passed=True, state_dict_schema_unchanged=True,
                dynamo_counters={group: dict(values) for group, values in counters.items()})
    args.output.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")
    print(f"PASS: {args.output}", flush=True)


if __name__ == "__main__":
    main()
