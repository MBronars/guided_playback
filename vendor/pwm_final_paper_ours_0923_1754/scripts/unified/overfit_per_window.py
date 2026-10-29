"""Per-window 8-seed scores of an overfit-harness ``final.pt`` on its saved fixture.

Mirrors overfit_prod's final evaluation exactly (raw weights, Accelerate autocast,
run_diffusion_inference_df with the recipe's schedule, ``torch.manual_seed(10000 + s)``, the
checkpoint's own normalization bound to the fixture) and prints the batch mean per seed next to
the harness's recorded values, which it must reproduce. The harness reports one batch mean;
this splits it per fixture row (window): mean and final-frame error over the row's valid
particles, and the ground-truth motion per frame.

    PYTHONPATH=. python scripts/unified/overfit_per_window.py --run <out_root>/<arm>/outputs \
        --fixture <batch_dir>/<batch_name> --out per_window.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from data.normalization import prepare_fixture
from diffusion.flow_matching import TrajectoryFlowMatching
from inference import run_diffusion_inference_df
from models.factory import build_particle_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="harness outputs directory holding final.pt and <domain>.json")
    ap.add_argument("--fixture", required=True, help="fixture path prefix (<prefix>.pt and <prefix>.json)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(f"{a.run}/final.pt", weights_only=False, map_location="cpu")
    cfg = OmegaConf.create(ck["cfg"])
    model = build_particle_model(cfg)
    model.load_state_dict(ck["model"])
    acc = Accelerator(mixed_precision=str(cfg.train.get("mixed_precision", "no")))
    model = acc.prepare(model)
    m = acc.unwrap_model(model)
    m.eval()
    fm = cfg.flow_matching
    diffusion = TrajectoryFlowMatching(
        num_train_timesteps=fm.num_train_timesteps, shift=fm.shift, prediction_type=fm.prediction_type,
        time_sampler=str(fm.get("time_sampler", "uniform")), beta_alpha=float(fm.get("beta_alpha", 1.5)),
        beta_beta=float(fm.get("beta_beta", 1.0)), beta_t_min=float(fm.get("beta_t_min", 0.001)))
    meta = json.load(open(f"{a.fixture}.json"))
    domain = meta["domain"]
    batch, _ = prepare_fixture(torch.load(f"{a.fixture}.pt", weights_only=False), meta, domain,
                               OmegaConf.to_container(cfg.normalization, resolve=True), int(cfg.data.batch_size),
                               rebind=True, context_len=int(cfg.data.context_len), future_len=int(cfg.data.future_len))
    batch.domain = domain
    batch = batch.to(acc.device)
    C, F = batch.context_len, batch.num_future_frames
    gt, vm = batch.positions[:, C:C + F], batch.get_valid_mask()
    motion = batch.prediction_x0.norm(dim=-1) * 1e3
    rows, means = [], []
    for s in range(a.seeds):
        torch.manual_seed(10_000 + s)
        with torch.no_grad():
            out = run_diffusion_inference_df(m, batch, diffusion, num_inference_steps=int(fm.inference_steps),
                                             schedule=cfg.get("df_schedule", "pyramid"),
                                             uncertainty_scale=float(cfg.get("df_uncertainty_scale", 1.0)))
        if isinstance(out, tuple):
            raise ValueError("per-window scoring covers particle-only (passive) fixtures")
        pred = batch.predictions_to_positions(batch.prediction_stats.denormalize(out)).float()
        e = (pred - gt).norm(dim=-1) * 1e3
        means.append(e[vm[:, None].expand_as(e)].mean().item())
        rows.append([(e[i][:, vm[i]].mean().item(), e[i, -1][vm[i]].mean().item()) for i in range(e.shape[0])])
    rows = np.array(rows)
    harness = json.load(open(f"{a.run}/{domain}.json")).get("final_multi_seed_mm", {}).get("values")
    res = {"run": a.run, "fixture": a.fixture, "domain": domain, "seeds": [10_000 + s for s in range(a.seeds)],
           "batch_mean_per_seed": means, "harness_final_values": harness,
           "windows": [{"row": i, "gt_motion_mm_per_frame": motion[i][:, vm[i]].mean().item(),
                        "mean_mm": float(rows[:, i, 0].mean()), "mean_std": float(rows[:, i, 0].std()),
                        "final_mm": float(rows[:, i, 1].mean())} for i in range(rows.shape[1])]}
    json.dump(res, open(a.out, "w"), indent=1)
    print("batch mean per seed:", np.round(means, 2), "| harness:", None if harness is None else np.round(harness, 2))
    for w in res["windows"]:
        print(f"window {w['row']}: GT {w['gt_motion_mm_per_frame']:5.1f} mm/frame | mean {w['mean_mm']:6.1f} "
              f"+- {w['mean_std']:4.1f} | final frame {w['final_mm']:6.1f}")


if __name__ == "__main__":
    main()
