"""One sampled prediction of an overfit-harness ``final.pt`` on its water fixture, every window, saved
for the renderers in raw Water-3D box units (y up): <out_prefix>_w<k>.npz with context [C, N, 3],
gt / pred [F, N, 3] (valid particles only) and the window's mean error in mm.

Runs in the training environment (torch); rendering runs in env/render_water3d.requirements.txt.
The draw is the harness's first final-evaluation seed by default (torch.manual_seed(10000)).

    PYTHONPATH=. python scripts/render_water3d/dump_overfit_windows.py --run <out_root>/<arm>/outputs \
        --fixture <batch_dir>/<batch_name> --out_prefix dumps/arm
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf

from data.normalization import prepare_fixture
from data.water3d import from_model_frame
from diffusion.flow_matching import TrajectoryFlowMatching
from inference import run_diffusion_inference_df
from models.factory import build_particle_model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="harness outputs directory holding final.pt")
    ap.add_argument("--fixture", required=True, help="water fixture path prefix (<prefix>.pt and <prefix>.json)")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--seed", type=int, default=10_000)
    a = ap.parse_args()
    meta = json.load(open(f"{a.fixture}.json"))
    if meta["domain"] != "water":
        raise ValueError(f"expected a water fixture, got domain {meta['domain']!r}")
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
    batch, _ = prepare_fixture(torch.load(f"{a.fixture}.pt", weights_only=False), meta, "water",
                               OmegaConf.to_container(cfg.normalization, resolve=True), int(cfg.data.batch_size),
                               rebind=True, context_len=int(cfg.data.context_len), future_len=int(cfg.data.future_len))
    batch.domain = "water"
    batch = batch.to(acc.device)
    C, F = batch.context_len, batch.num_future_frames
    torch.manual_seed(a.seed)
    with torch.no_grad():
        out = run_diffusion_inference_df(m, batch, diffusion, num_inference_steps=int(fm.inference_steps),
                                         schedule=cfg.get("df_schedule", "pyramid"),
                                         uncertainty_scale=float(cfg.get("df_uncertainty_scale", 1.0)))
    pred = batch.predictions_to_positions(batch.prediction_stats.denormalize(out)).float().cpu().numpy()
    pos, vm = batch.positions.float().cpu().numpy(), batch.get_valid_mask().cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(a.out_prefix)), exist_ok=True)
    for k in range(pos.shape[0]):
        v = vm[k]
        ctx, gt, pr = pos[k, :C][:, v], pos[k, C:C + F][:, v], pred[k][:, v]
        err = float(np.linalg.norm(pr - gt, axis=-1).mean() * 1e3)
        np.savez(f"{a.out_prefix}_w{k}.npz", context=from_model_frame(ctx), gt=from_model_frame(gt),
                 pred=from_model_frame(pr), window=np.array(k), seed=np.array(a.seed), mean_err_mm=np.array(err))
        print(f"window {k}: N={int(v.sum())} mean error {err:.1f} mm -> {a.out_prefix}_w{k}.npz")


if __name__ == "__main__":
    main()
