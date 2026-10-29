"""Paired full-size eager BF16 allocation timing on synthetic scenes.

Same weights/inputs/RNG; objective + backward + clipping + AdamW(lr=0). No corpus
training or accuracy claim. Reports lower- and higher-particle-count scenes.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch

from diffusion import TrajectoryFlowMatching
from models.factory import build_particle_model, parse_model_config
from models.unet_std import build_unet_config
import tests.test_unet_std_pose_native as fixtures
from train import trajectory_loss_options
from utils.config import load_config
from utils.reproducibility import code_revision, config_for_json, file_sha256
from utils.trajectory_objective import trajectory_objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if min(args.warmup, args.repeats) < 1:
        parser.error("warmup and repeats must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(811)
    config = load_config("configs/unet.yaml", ["train.compile=false", "model.unet_fps_random_start=false"])
    model = build_particle_model(config)
    with torch.no_grad():
        for p in model.parameters():
            if torch.count_nonzero(p) == 0:
                p.normal_(0, .025)
    model.cuda().train()
    fixed = model.model.cfg
    adaptive_config = load_config("configs/unet_adaptive.yaml", ["train.compile=false",
                                 "model.unet_fps_random_start=false"])
    adaptive = build_unet_config(parse_model_config(adaptive_config.model)[1])
    assert fixed.level_sizes == adaptive.level_sizes
    policies = dict(fixed=fixed, adaptive=adaptive)
    # Only the graph's allocation policy changes. Blocks retain their identical
    # construction/config/parameters; no learned component reads these L1 fields.
    names = ("l1_by_mat", "l1_default", "l1_rule", "l1_rule_min", "l1_rule_max", "l1_scene_budget", "l1_rigid_max")
    settings = {name: {k: getattr(c, k) for k in names} for name, c in policies.items()}

    @contextmanager
    def policy(name):
        saved = {k: getattr(fixed, k) for k in names}
        for k, value in settings[name].items():
            setattr(fixed, k, value)
        try:
            yield
        finally:
            for k, value in saved.items():
                setattr(fixed, k, value)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0, foreach=True)
    options = trajectory_loss_options(config)
    diffusion = TrajectoryFlowMatching()
    fixtures.B, fixtures.C, fixtures.F = 1, int(config.data.context_len), int(config.data.future_len)
    fixtures.T = fixtures.C + fixtures.F
    report = dict(provenance=code_revision(), checkpoint=None, seed=811,
        protocol="Synthetic mixed native-pose scenes, unchanged active random weights; "
                 "full objective/backward/clipping/AdamW(lr=0), alternating policy order. "
                 "Shared GPU; timings are indicative, not accuracy results.",
        device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
        config=config_for_json(config), adaptive_config=config_for_json(adaptive_config),
        policies={k: asdict(v) for k, v in policies.items()},
        parameters=sum(p.numel() for p in model.parameters()), compiled=False, precision="bf16",
        warmup=args.warmup, repeats=args.repeats, records=[], scenes={},
        source_sha256={str(p): file_sha256(p) for p in (Path(__file__), Path(fixtures.__file__),
                                                       Path("models/unet_std.py"))})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    # Cloth-like deformable plus two actuated/passive rigid groups; these are
    # engineering shapes, not claims about the distribution of real scenes.
    for scene, deform_n, rigid_n in (("lower_count", 512, 512), ("large_objects", 2048, 2048)):
        batch = fixtures.make_batch(deform_n=deform_n, rigid=((1, rigid_n), (2, rigid_n)), seed=811).to("cuda")
        counts, mats = [deform_n, rigid_n, rigid_n], [5, 4, 3]
        allocations = {}
        for name in policies:
            with policy(name):
                allocations[name] = fixed.l1_for_scene(mats, counts, rigid=[False, True, True])
        report["scenes"][scene] = dict(batch_shape=list(batch.positions.shape), raw_counts=counts,
                                         allocations=allocations)
        for iteration in range(-args.warmup, args.repeats):
            order = ("fixed", "adaptive") if iteration % 2 == 0 else ("adaptive", "fixed")
            for name in order:
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(811)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                with policy(name):
                    start = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        # Drop diagnostic tensors immediately: retaining the previous
                        # draw across iterations would contaminate peak-memory readings.
                        loss = trajectory_objective(model, batch, diffusion, **options)[0]
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                    peak_before_optimizer = torch.cuda.max_memory_allocated() / 2**20
                    optimizer.step()
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - start) * 1000
                assert torch.isfinite(loss) and torch.isfinite(grad_norm), (scene, name)
                if iteration >= 0:
                    row = dict(scene=scene, policy=name, iteration=iteration, milliseconds=elapsed,
                               peak_before_optimizer_mib=peak_before_optimizer,
                               peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
                    report["records"].append(row)
                    print(json.dumps(row), flush=True)
                    save()
                del loss, grad_norm
        optimizer.zero_grad(set_to_none=True)
        del batch
    report["summary"] = {}
    for scene in report["scenes"]:
        report["summary"][scene] = {}
        for name in policies:
            rows = [r for r in report["records"] if r["scene"] == scene and r["policy"] == name]
            report["summary"][scene][name] = dict(
                median_ms=statistics.median(r["milliseconds"] for r in rows),
                min_ms=min(r["milliseconds"] for r in rows), max_ms=max(r["milliseconds"] for r in rows),
                peak_before_optimizer_mib=max(r["peak_before_optimizer_mib"] for r in rows),
                peak_allocated_mib=max(r["peak_allocated_mib"] for r in rows))
    report["complete"] = True
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
