"""Paired GPU timing of the geometry precision repair on one unchanged model.

Uses a synthetic mixed scene and seeded, full-size recipe weights in eager BF16.
Training timings include the objective, backward, clipping and AdamW with lr=0;
sampling timings include a complete diffusion window and materialization. This
is an engineering benchmark, without corpus access or a model-quality claim.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import subprocess
import time
import types

from omegaconf import OmegaConf
import torch

from diffusion.flow_matching import TrajectoryFlowMatching
from inference import run_diffusion_inference_df
from models.factory import build_particle_model
import tests.test_unet_std_pose_native as fixtures
from train import trajectory_loss_options
from utils.config import load_config
import utils.pose_native as geometry
from utils.reproducibility import code_revision, file_sha256
from utils.trajectory_objective import trajectory_objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unet.yaml"))
    parser.add_argument("--baseline-revision", default="73f7aa3")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--train-steps-per-repeat", type=int, default=1)
    parser.add_argument("--operations", nargs="+", choices=["train_step", "sample_window"],
                        default=["train_step", "sample_window"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required; CPU timing does not estimate the GPU tradeoff")
    if min(args.warmup, args.repeats, args.steps, args.train_steps_per_repeat) < 1:
        parser.error("warmup, repeats and step counts must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    cfg = load_config(args.config, ["train.compile=false", "model.dropout=0"])
    if cfg.model.architecture == "unet":
        cfg.model.unet_fps_random_start = False  # Paired calls use the same hierarchy.
    model = build_particle_model(cfg)
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.025)  # Exercise the trunk with active gates/readout.
    model = model.cuda()
    fixtures.B, fixtures.C, fixtures.F = 1, int(cfg.data.context_len), int(cfg.data.future_len)
    fixtures.T = fixtures.C + fixtures.F
    batch = fixtures.make_batch(deform_n=512, rigid=((1, 256), (1, 128), (1, 128), (2, 512))).to("cuda")
    diffusion = TrajectoryFlowMatching(
        num_train_timesteps=cfg.flow_matching.num_train_timesteps,
        shift=cfg.flow_matching.shift, prediction_type=cfg.flow_matching.prediction_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0, foreach=True)
    loss_options = trajectory_loss_options(cfg)

    baseline = types.ModuleType("_pose_precision_baseline")
    source = subprocess.run(["git", "show", f"{args.baseline_revision}:utils/pose_native.py"],
                            check=True, capture_output=True, text=True).stdout
    exec(compile(source, f"git:{args.baseline_revision}:utils/pose_native.py", "exec"), baseline.__dict__)
    names = ("pose_native_inputs", "context_pose_deltas", "materialize_pose_native")
    guarded = {name: getattr(geometry, name) for name in names}

    @contextmanager
    def version(policy):
        for name in names:
            setattr(geometry, name, guarded[name] if policy == "fp32_geometry" else getattr(baseline, name))
        try:
            yield
        finally:
            for name in names:
                setattr(geometry, name, guarded[name])

    info = dict(provenance=code_revision(), config=OmegaConf.to_container(cfg, resolve=True),
                baseline_revision=args.baseline_revision, seed=1729, checkpoint=None,
                geometry_sha256=file_sha256(Path("utils/pose_native.py")),
                benchmark_sha256=file_sha256(Path(__file__)),
                fixture_sha256=file_sha256(Path(fixtures.__file__)),
                parameters=sum(p.numel() for p in model.parameters()),
                batch_shape=list(batch.positions.shape), context=fixtures.C, future=fixtures.F,
                device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                cuda_version=torch.version.cuda, mixed_precision="bf16", compiled=False,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                schedule="pyramid", steps=args.steps, warmup=args.warmup, repeats=args.repeats,
                operations=args.operations, train_steps_per_repeat=args.train_steps_per_repeat,
                note="Synthetic mixed scene; eager full-size random model; no optimizer weight updates. "
                     "Both policies share weights, inputs and RNG. Shared-GPU timings are indicative.",
                precision={}, records=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n")

    # Physical errors against identical inputs evaluated in FP32, independently of the network.
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    pose = batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, ridx], clip=False)
    particle = torch.zeros_like(batch.future)
    expected = baseline.materialize_pose_native(batch, particle, pose)[0]
    context = baseline.context_pose_deltas(batch, ridx)
    rigid = batch.body_type[0, batch.particle_body[0]] == 1
    for policy in ("original_amp", "fp32_geometry"):
        with version(policy), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            patched, _ = geometry.pose_native_inputs(batch, particle, pose)
            reconstructed = batch.anchor[:, None] + batch.delta_stats.denormalize(patched).cumsum(1)
            materialized = geometry.materialize_pose_native(batch, particle, pose)[0]
            delta = geometry.context_pose_deltas(batch, ridx)
        info["precision"][policy] = dict(
            input_max_error_mm=float((reconstructed - expected).norm(dim=-1)[:, :, rigid].max() * 1000),
            output_max_error_mm=float((materialized - expected).norm(dim=-1)[:, :, rigid].max() * 1000),
            context_max_rotation_difference_rad=float((delta[..., 3:] - context[..., 3:]).abs().max()))
    print(json.dumps(dict(precision=info["precision"])), flush=True)
    save()

    calls = [0]
    hook = model.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))

    def operation(kind):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if kind == "train_step":
                loss, *_ = trajectory_objective(model, batch, diffusion, **loss_options)
            else:
                with torch.no_grad():
                    samples = run_diffusion_inference_df(model, batch, diffusion,
                                                         num_inference_steps=args.steps, schedule="pyramid")
                    result = geometry.materialize_pose_native(batch, *samples)[0]
                return result
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()  # lr=0: weights remain identical across policies.
        return loss.detach()

    for kind in args.operations:
        model.train(kind == "train_step")
        count = args.train_steps_per_repeat if kind == "train_step" else 1
        for iteration in range(-args.warmup, args.repeats):
            policies = ("original_amp", "fp32_geometry")
            if iteration % 2:
                policies = policies[::-1]
            for policy in policies:
                torch.manual_seed(1729)
                optimizer.zero_grad(set_to_none=True)
                calls[0] = 0
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                with version(policy):
                    start = time.perf_counter()
                    for _ in range(count):
                        optimizer.zero_grad(set_to_none=True)
                        result = operation(kind)
                    torch.cuda.synchronize()
                    elapsed_ms = (time.perf_counter() - start) * 1000 / count
                assert torch.isfinite(result).all(), (kind, policy)
                if iteration >= 0:
                    row = dict(kind=kind, policy=policy, iteration=iteration, milliseconds=elapsed_ms,
                               model_calls=calls[0], peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
                    info["records"].append(row)
                    print(json.dumps(row), flush=True)
                    save()
                del result
    hook.remove()
    info["summary"] = {}
    for kind in args.operations:
        summary = {}
        for policy in ("original_amp", "fp32_geometry"):
            rows = [r for r in info["records"] if r["kind"] == kind and r["policy"] == policy]
            summary[policy] = dict(median_ms=statistics.median(r["milliseconds"] for r in rows),
                                   min_ms=min(r["milliseconds"] for r in rows),
                                   max_ms=max(r["milliseconds"] for r in rows),
                                   peak_allocated_mib=max(r["peak_allocated_mib"] for r in rows))
        summary["median_change_percent"] = 100 * (summary["fp32_geometry"]["median_ms"] /
                                                   summary["original_amp"]["median_ms"] - 1)
        info["summary"][kind] = summary
    save()
    print(json.dumps(info["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
