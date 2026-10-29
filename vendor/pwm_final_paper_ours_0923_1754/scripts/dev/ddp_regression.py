"""Two-rank optimizer safety gate; run with torchrun --nproc_per_node=2.

Uses the production train_step with a one-parameter model so the expected update
is analytic. Checks collective rejection and successful recovery, not throughput.
"""
import argparse
import copy
import json
import math
import random
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.distributed as dist
import numpy as np
from accelerate import Accelerator

from diffusion.flow_matching import TrajectoryFlowMatching
from tests.test_train import create_dummy_batch
from tests.test_training_safety import ConstantModel
from train import _reduce_validation, train_step
from utils.ema import EMA
from utils.lr_schedule import make_scheduler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "bf16", "fp16"), default="no")
    parser.add_argument("--accumulation", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision,
                              gradient_accumulation_steps=args.accumulation)
    assert accelerator.num_processes == 2, "launch with torchrun --nproc_per_node=2"
    model = ConstantModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = make_scheduler(optimizer, dict(lr=.1, steps=20, lr_schedule="constant"),
                               num_processes=accelerator.num_processes)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    raw = accelerator.unwrap_model(model)
    shadow = EMA(raw, decay=.9, warmup=True)
    batch = create_dummy_batch(B=1, T=6, N=8, context_len=2).to(accelerator.device)
    for t in range(6):
        batch.positions[:, t].fill_(max(0, t - 1))
    diffusion = TrajectoryFlowMatching(prediction_type="sample")
    records = []
    for case in ("healthy_initial", "rank_local_spike", "nonfinite_loss", "nonfinite_gradient", "healthy"):
        with torch.no_grad():
            raw.weight.fill_(0.25)
        optimizer.zero_grad()
        prior_optimizer = copy.deepcopy(optimizer.state_dict())
        prior_scheduler = copy.deepcopy(scheduler.state_dict())
        prior_shadow = copy.deepcopy(shadow.state_dict())
        prior_meta = shadow.metadata()
        momentum = optimizer.state[raw.weight].get("momentum_buffer", torch.tensor(0.)).item()
        ema = 1e-12 if case == "rank_local_spike" and accelerator.process_index == 0 else None
        for micro in range(args.accumulation):
            raw.failure = {"nonfinite_loss": "loss", "nonfinite_gradient": "gradient"}.get(case) \
                if accelerator.process_index == 0 and micro == 0 else None
            with accelerator.accumulate(model):
                _, metrics = train_step(model, batch, diffusion, optimizer, accelerator,
                                        grad_clip=None if case == "nonfinite_gradient" else 100.0,
                                        loss_ema=ema, loss_spike_threshold=2.0,
                                        grad_norm_spike_threshold=float("inf"))
            if micro < args.accumulation - 1:
                assert not metrics["sync_gradients"] and not metrics["optimizer_step"]
                assert raw.weight.item() == .25
        if metrics["optimizer_step"]:
            scheduler.step()
            shadow.update()
        scale = accelerator.scaler.get_scale() if accelerator.scaler is not None else 1.0
        payload = torch.tensor([raw.weight.item(), float(metrics["skipped"]), scale],
                               device=accelerator.device)
        gathered = [torch.empty_like(payload) for _ in range(2)]
        dist.all_gather(gathered, payload)
        values = [v.cpu().tolist() for v in gathered]
        expected_skip = not case.startswith("healthy")
        expected = .25 if expected_skip else .25 - .1 * (.9 * momentum - 1.5)
        passed = (all(math.isfinite(w) and abs(w - expected) < 1e-6 and bool(s) == expected_skip
                      for w, s, _ in values) and values[0][2] == values[1][2])
        if expected_skip:
            torch.testing.assert_close(optimizer.state_dict(), prior_optimizer, rtol=0, atol=0)
            torch.testing.assert_close(shadow.state_dict(), prior_shadow, rtol=0, atol=0)
            assert scheduler.state_dict() == prior_scheduler and shadow.metadata() == prior_meta
        else:
            assert shadow.updates == prior_meta["updates"] + 1
            assert scheduler.state_dict()["last_epoch"] == prior_scheduler["last_epoch"] + accelerator.num_processes
        records.append({"case": case, "passed": passed, "expected_weight": expected,
                        "ema_updates": shadow.updates, "scheduler_ticks": scheduler.state_dict()["last_epoch"],
                        "ranks": [{"weight": w if math.isfinite(w) else None, "skipped": bool(s),
                                   "loss_scale": scale} for w, s, scale in values]})
    # Different domains and unequal batch counts on each rank must not produce
    # a collective mismatch or an unweighted mean of rank means.
    rank = accelerator.process_index
    sums = {"loss": 2.0, "loss_mpm": 2.0} if rank == 0 else {"loss": 12.0, "loss_cloth": 12.0}
    counts = {key: 1 if rank == 0 else 3 for key in sums}
    values = _reduce_validation(sums, counts, accelerator, accelerator.device)
    expected = {"loss": 3.5, "loss_mpm": 2.0, "loss_cloth": 4.0}
    records.append(dict(case="unequal_validation_domains", passed=values == expected, metrics=values))
    if accelerator.device.type == "cuda":
        # Each process can see both GPUs. CPU validation and rank-local GPU
        # validation must leave ALL generators unchanged, including the other GPU.
        from train import validate, validate_sampler
        for on_cpu in (True, False):
            check_device = torch.device("cpu") if on_cpu else accelerator.device
            check_model = ConstantModel() if on_cpu else model
            check_batch = batch.to(check_device)
            reducer = SimpleNamespace(reduce=lambda value, reduction: value) if on_cpu else accelerator
            before_cpu = torch.random.get_rng_state()
            before_cuda = torch.cuda.get_rng_state_all()
            for function in (validate, validate_sampler):
                function(check_model, [check_batch], diffusion, None, check_device, reducer,
                         use_ema=False, num_batches=1)
            preserved = (torch.equal(before_cpu, torch.random.get_rng_state()) and
                         all(torch.equal(a, b) for a, b in zip(before_cuda, torch.cuda.get_rng_state_all())))
            agreement = accelerator.reduce(torch.tensor(int(preserved), device=accelerator.device), reduction="sum")
            records.append(dict(case=f"validation_rng_{check_device.type}", passed=int(agreement) == 2))
    from utils.checkpoint import atomic_torch_save, capture_runtime, restore_runtime
    random.seed(123 + rank); np.random.seed(123 + rank); torch.manual_seed(123 + rank)
    runtime = capture_runtime(accelerator, rank=rank, loss_ema=float(rank))
    assert runtime["version"] == 2 and runtime["mixed_precision"] == args.mixed_precision
    expected_python, expected_numpy = random.random(), np.random.rand()
    expected_cpu = torch.randn(4)
    expected_device = torch.randn(4, device=accelerator.device)
    expected_scale = accelerator.scaler.get_scale() if accelerator.scaler is not None else None
    runtime_path = args.output.with_suffix(".runtime.pt")
    if accelerator.is_main_process:
        atomic_torch_save(runtime, runtime_path)
    accelerator.wait_for_everyone()
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    if accelerator.scaler is not None:
        accelerator.scaler.update(new_scale=1.0)
    loop = restore_runtime(torch.load(runtime_path, map_location="cpu", weights_only=True), accelerator)
    replayed = (random.random() == expected_python and np.random.rand() == expected_numpy
                and torch.equal(torch.randn(4), expected_cpu)
                and torch.equal(torch.randn(4, device=accelerator.device), expected_device)
                and loop == dict(rank=rank, loss_ema=float(rank)))
    if accelerator.scaler is not None:
        replayed = replayed and accelerator.scaler.get_scale() == expected_scale
    agreement = accelerator.reduce(torch.tensor(int(replayed), device=accelerator.device), reduction="sum")
    records.append(dict(case="rank_local_runtime_replay", passed=int(agreement) == 2,
                        runtime_version=runtime["version"], mixed_precision=runtime["mixed_precision"]))
    if accelerator.is_main_process:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = {"device": str(accelerator.device), "backend": dist.get_backend(),
                  "torch": torch.__version__, "mixed_precision": args.mixed_precision,
                  "accumulation": args.accumulation, "cases": records,
                  "passed": all(r["passed"] for r in records)}
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    accelerator.wait_for_everyone()
    dist.destroy_process_group()
    if not all(r["passed"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
