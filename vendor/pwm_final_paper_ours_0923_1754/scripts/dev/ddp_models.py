"""Two-rank real-model update gate, including rank-dependent rigid/deformable heads.

Run with torchrun --standalone --nproc_per_node=2. Small model sizes isolate
distributed correctness; full-sized single-GPU checks are separate.
"""
import argparse
import json
from pathlib import Path

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import torch
import torch.distributed as dist

from diffusion import TrajectoryFlowMatching
from models.transformer import ModelConfig, TrajectoryTransformer
from models.unet_std import UNetWrapper
from tests.test_unet_std_pose_native import make_batch, _model_cfg
from train import train_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mixed-precision", default="no", choices=("no", "bf16", "fp16"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acc = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision,
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True,
                                                                    broadcast_buffers=False)])
    assert acc.num_processes == 2
    torch.set_num_threads(1)
    records = []
    for arch in ("transformer", "unet"):
        torch.manual_seed(1729)
        if arch == "transformer":
            model = TrajectoryTransformer(ModelConfig(
                dim=64, depth=2, heads=4, num_latents=24, k_neighbors=8, dropout=0,
                context_len=3, all_relative=True, pose_native=True, num_mat_types=6, causal_temporal=True,
                bypass_compression=False, use_stratified_encoding=True, use_stratified_decoding=True))
        elif arch == "unet":
            model = UNetWrapper(_model_cfg(dropout=0))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        model, optimizer = acc.prepare(model, optimizer)
        fm = TrajectoryFlowMatching()
        for step in range(3):
            if arch in ("transformer", "unet"):
                mode = (step + acc.process_index) % 3
                options = (dict(rigid=(), sidecar=False), {},
                           dict(deform_n=0, rigid=((1, 24), (2, 16))))[mode]
                batch = make_batch(seed=100 + acc.process_index, **options).to(acc.device)
                with acc.accumulate(model):
                    loss, metrics = train_step(model, batch, fm, optimizer, acc, grad_clip=1,
                                               diffusion_forcing=True, noise_context=True,
                                               grad_norm_spike_threshold=float("inf"))
            assert metrics["optimizer_step"], (arch, step, metrics)
            flat = torch.cat([p.detach().reshape(-1) for p in acc.unwrap_model(model).parameters()])
            other = flat.clone()
            dist.broadcast(other, src=0)
            error = (flat - other).abs().max()
            assert error == 0, (arch, step, float(error))
            records.append(dict(architecture=arch, step=step, loss=loss, rank_weight_error=float(error)))
        del model, optimizer, batch, flat, other
        acc.free_memory()
    if acc.is_main_process:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(passed=True, backend=dist.get_backend(),
                                               mixed_precision=args.mixed_precision, cases=records), indent=2) + "\n")
        print(f"All {len(records)} real-model distributed updates agree", flush=True)
    acc.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
