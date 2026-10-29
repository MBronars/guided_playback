"""Compare a trusted local checkpoint on pinned real-domain batches.

The first invocation freezes validation batches and their source receipts. Later
invocations reuse those exact tensors, isolating code changes from dataset changes.
Outputs are an engineering regression sample, NOT a paper benchmark.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--batch-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare", type=Path)
    p.add_argument("--domains", default="mpm,pickup,cloth,pushing")
    p.add_argument("--batch_name", help="saved batch stem(s), comma list parallel to --domains; "
                   "defaults to domain names and never establishes a fixture's domain identity")
    p.add_argument("--rebind_normalization", action="store_true",
                   help="Explicitly attach checkpoint normalizers to the same raw fixture tensors; "
                   "required for a normalization ablation or a raw calibration fixture")
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--precision", choices=("fp32", "bf16", "bf16-forward"), default="bf16",
                   help="bf16 wraps the entire sampler; bf16-forward matches Accelerate-wrapped training evaluation")
    p.add_argument("--seeds", type=int, nargs="+", default=[1618])
    a = p.parse_args()
    for name in ("repo", "checkpoint", "batch_dir", "output", "compare"):
        value = getattr(a, name)
        if value is not None:
            setattr(a, name, value.resolve())
    sys.path.insert(0, str(a.repo))
    os.chdir(a.repo)
    import numpy as np
    import torch
    from data.normalization import NORMALIZATION_DOMAINS, prepare_fixture, resolve_normalization
    from inference import load_model_and_diffusion, run_diffusion_inference_df
    from scripts.unified.eval_unified import evaluation_dataset
    from utils.pose_native import materialize_pose_native
    from utils.reproducibility import code_revision, file_sha256

    domains = a.domains.split(",")
    names = a.batch_name.split(",") if a.batch_name is not None else domains
    if (not all(domain in NORMALIZATION_DOMAINS for domain in domains) or
            len(set(domains)) != len(domains)):
        p.error("--domains must contain distinct supported domain names")
    if len(names) != len(domains):
        p.error("Provide one --batch_name per domain")
    if (len(set(names)) != len(names) or any(not name or name in {".", ".."} or
            "/" in name or "\\" in name for name in names)):
        p.error("--batch_name requires distinct nonempty file stems without path traversal")

    torch.set_num_threads(1)
    a.batch_dir.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    model, diffusion, cfg, _ = load_model_and_diffusion(str(a.checkpoint), torch.device(a.device))
    normalization = resolve_normalization(cfg.normalization)
    if a.precision == "bf16-forward":
        original_forward = model.forward

        def forward_autocast(*args, **kwargs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                value = original_forward(*args, **kwargs)
            if isinstance(value, tuple):
                return tuple(v.float() if v is not None else None for v in value)
            return value.float()

        model.forward = forward_autocast
    results = {}
    metadata = {"checkpoint": str(a.checkpoint), "checkpoint_sha256": file_sha256(a.checkpoint),
                "repo": str(a.repo), **code_revision(),
                "normalization": normalization, "normalization_sha256": normalization["sha256"],
                "torch": torch.__version__, "steps": a.steps, "domains": {},
                "sampling_precision": a.precision, "seeds": a.seeds}
    for domain, batch_name in zip(domains, names):
        random.seed(2718); np.random.seed(2718); torch.manual_seed(2718)
        batch_path = a.batch_dir / (batch_name + ".pt")
        receipt_path = batch_path.with_suffix(".json")
        if not batch_path.exists():
            ds = evaluation_dataset(cfg, domain, int(cfg.data.context_len), int(cfg.data.future_len))
            ds.emit_sidecar = True
            ds.articulated = bool(cfg.get("model", {}).get("articulated_bodies", False))
            ds.batch_size = 1
            sources = []
            parse = ds._process_sample

            def recording_parse(sample):
                sources.append({k: str(sample.get(k, "")) for k in ("__url__", "__key__")})
                return parse(sample)

            ds._process_sample = recording_parse
            batch = next(iter(ds)).slice(1)
            batch.domain = domain
            torch.save(batch, batch_path)
            receipt = {"domain": domain, "sources": sources, "shape": list(batch.positions.shape),
                       "context_len": batch.context_len, "future_len": batch.num_future_frames,
                       "normalization": normalization, "normalization_sha256": normalization["sha256"],
                       "sha256": file_sha256(batch_path)}
            receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
            print(f"Pinned {domain}: {receipt}", flush=True)
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("sha256") != file_sha256(batch_path):
            raise ValueError(f"Batch receipt SHA256 mismatch: {batch_path}")
        batch = torch.load(batch_path, map_location="cpu", weights_only=False)
        batch, fixture = prepare_fixture(batch, receipt, domain, normalization, batch.batch_size,
            rebind=a.rebind_normalization, context_len=int(cfg.data.context_len),
            future_len=int(cfg.data.future_len))
        batch = batch.to(a.device)
        start = time.monotonic()
        predictions, metrics = [], []
        for seed in a.seeds:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
                x0 = run_diffusion_inference_df(
                    model, batch, diffusion, num_inference_steps=a.steps,
                    schedule=cfg.get("df_schedule", "pyramid"),
                    uncertainty_scale=float(cfg.get("df_uncertainty_scale", 1.0)))
                pred = (materialize_pose_native(batch, *x0)[0] if isinstance(x0, tuple)
                        else batch.predictions_to_positions(batch.prediction_stats.denormalize(x0)))
            assert torch.isfinite(pred).all(), f"Non-finite {domain} rollout"
            error = (pred.float() - batch.future.float()).norm(dim=-1) * 1000
            valid = batch.get_valid_mask()
            scores = {"seed": seed}
            for name, mask in (("all_valid", valid), ("passive", valid & ~batch.action_mask),
                               ("actuated", valid & batch.action_mask)):
                scores[name] = ({"particles": int(mask.sum()),
                                  "mean_mm": error[mask[:, None].expand_as(error)].mean().item(),
                                  "final_mm": error[:, -1][mask].mean().item()}
                                 if mask.any() else None)
            predictions.append(pred.detach().float().cpu())
            metrics.append(scores)
        torch.cuda.synchronize(a.device)
        results[domain] = predictions[0] if len(predictions) == 1 else torch.stack(predictions)
        info = {"shape": list(pred.shape), "seconds": time.monotonic() - start,
                "peak_allocated_gb": torch.cuda.max_memory_allocated(a.device) / 1e9,
                "receipt": str(receipt_path), "batch_name": batch_name, "fixture": fixture,
                "metrics_by_seed": metrics}
        metadata["domains"][domain] = info
        print(f"{domain}: { {key: value for key, value in info.items() if key != 'fixture'} }", flush=True)
        torch.save(results, a.output)
        a.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    if a.compare:
        reference = torch.load(a.compare, weights_only=True)
        assert reference.keys() == results.keys()
        for domain in reference:
            torch.testing.assert_close(results[domain], reference[domain], rtol=1e-5, atol=1e-6)
        metadata["comparison"] = {"passed": True, "reference": str(a.compare),
                                  "max_absolute_error": max((results[d] - reference[d]).abs().max().item()
                                                            for d in reference)}
        a.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(metadata["comparison"], flush=True)


if __name__ == "__main__":
    main()
