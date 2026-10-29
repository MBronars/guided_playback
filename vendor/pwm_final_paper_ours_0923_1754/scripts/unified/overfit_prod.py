"""Overfit ONE production batch per domain with the PRODUCTION model, and check the REAL
sampler reproduces it. Several --domain entries (comma list) interleave their batches
round-robin, one optimizer step per domain in turn, so ONE unified model must hold all
of them at once -- the same domain-pure alternation as real unified training.

Full-capacity engineering check (see explicit deviations below):
  model      built from --config exactly as train.py builds it (unet_std: 331.6M)
  batch      one draw from the production train stream (same dataset class, window
             sampler, sidecar, stats set, B, C, F); saved so every run shares it
  train step train.train_step with the config's own kwargs (bf16, DF, noise_context, ...)
  sampler    inference.run_diffusion_inference_df with the config's schedule and step
             count, output materialised exactly as scripts/unified/eval_unified does

Every --eval_every steps each batch is sampled from pure noise and scored in MILLIMETRES
against ground truth (particles; plus rigid-body pose error where bodies exist), next to
two baselines: predict no motion, and hold the last context velocity. Report both:
near-static windows can have an almost perfect trivial baseline. Falling training loss
alone does not establish memorization or a sampler bug; objective weighting, optimization,
parameterization and rollout errors need separate checks. Never treat this as held-out
generalization, even when a frozen validation batch is reused as a training fixture.

Deviations from production, on purpose: no torch.compile (irrelevant here), no EMA
(sampled with the raw weights; EMA at 0.99 lags ~100 steps), spike-skipping OFF so every
step counts, a short linear LR warmup instead of 5000 steps.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/unified/overfit_prod.py \
        --config configs/unet.yaml --domain mpm

    # joint: one model, four batches; 40K steps = 10K updates per domain, and
    # eval_every 2000 / warmup 800 keep the per-domain cadence of the 10K single runs
    ... --domain mpm,pushing,pickup,cloth --holdout_name mpm_hold,pushing_hold,pickup_hold,cloth_hold \
        --steps 40000 --eval_every 2000 --warmup 800
"""
import argparse, json, os, random, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.unified_dataset import build_domain_dataset
from data.normalization import (bind_batch_normalization as bind_batch_normalization,
                                fixture_normalization_label,
                                prepare_fixture, resolve_normalization)
from utils.config import load_config
from diffusion import TrajectoryFlowMatching
from inference import run_diffusion_inference_df
from train import train_step
from scripts.unified.eval_unified import materialize_pose_native
from utils.reproducibility import code_revision, config_for_json, file_sha256, seed_for_device, working_source_versions


def sample_for_eval(sample, batch, device, *, seed=4242):
    """Isolate diagnostic RNG, or explicitly reproduce the historical shared stream."""
    if seed is None:
        return sample(batch)
    device = torch.device(device)
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        seed_for_device(seed, device)
        return sample(batch)

def build_optimizer(model, train_config):
    """Use the same AdamW hyperparameters as the production optimizer."""
    return torch.optim.AdamW(model.parameters(), lr=float(train_config.lr),
        weight_decay=float(train_config.get("weight_decay", 0.0)),
        betas=tuple(train_config.get("adam_betas") or (0.9, 0.999)),
        eps=float(train_config.get("adam_eps", 1e-8)))


def vel_mm(pred, g, v):
    """Temporal coherence: mean |(pred[t]-pred[t-1]) - (gt[t]-gt[t-1])| over valid
    particles, mm/frame. Separates jitter-around-the-right-place from wrong-place."""
    if pred.shape[1] < 2:
        return None  # No adjacent predicted frames: the metric is undefined.
    dp = pred[:, 1:] - pred[:, :-1]
    dg = g[:, 1:pred.shape[1]] - g[:, :pred.shape[1] - 1]
    e = (dp - dg).norm(dim=-1) * 1e3
    return e[v[:, None].expand_as(e)].mean().item()


def part_mm(pred, g, v):
    """mean over valid particles and frames, mm; and final-frame mean, mm."""
    e = (pred - g[:, :pred.shape[1]]).norm(dim=-1) * 1e3     # [B, F, N]
    m = v[:, None].expand_as(e)
    return (e[m].mean().item(), e[:, -1][v].mean().item())


def primary_mask(batch):
    """Score predicted bodies, keeping all commanded deformable particles.

Rigid actuator/fixture poses are supplied inputs. Including their reconstructed
points can obscure errors on a small manipulated object (especially Dex).
Dragging commands do not prescribe the achieved deformable trajectory.
"""
    valid = batch.get_valid_mask()
    return valid if batch.body_type is None else valid & ~batch.action_mask



def apply_window_policy(ds, cfg, domain):
    """The recipe's training window policy (data.window_motion, data.batch_across_episodes) is part of the
    production stream, so a fixture is drawn the way train.py draws windows: motion-weighted starts and
    cross-scene batches where the recipe says so, uniform per-episode windows otherwise."""
    from data.unified_dataset import validate_window_sampling
    data = cfg.data
    weighting, across = validate_window_sampling(
        OmegaConf.to_container(data.window_motion, resolve=True) if data.get("window_motion") is not None else None,
        list(data.batch_across_episodes) if data.get("batch_across_episodes") is not None else None)
    if domain in weighting:
        ds.motion_alpha, ds.motion_floor_m = weighting[domain]["alpha"], weighting[domain]["floor_mm"] / 1000.0
    if domain in across:
        ds.cross_episode = True
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--domain", required=True,
                    help="one of mpm/pickup/cloth/pushing, or a comma list interleaved round-robin")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--fixed_probe", action="store_true", help="Measure fixed-t/noise production losses at step 0 and final")
    ap.add_argument("--save_final", action="store_true", help="Save weights/optimizer/RNG for this engineering run")
    ap.add_argument("--batch_dir", default="validation/overfit/batches")
    ap.add_argument("--require_saved_batches", action="store_true",
                    help="Require every train/holdout tensor and receipt before starting; never redraw")
    ap.add_argument("--eval_rng", choices=("isolated", "legacy"), default="isolated",
                    help="legacy reproduces overfit_v2: train-batch eval consumes training RNG; "
                         "holdout eval stays isolated with seed 4242")
    ap.add_argument("--tag", default=None, help="output subdir (default: config stem)")
    ap.add_argument("--out_root", default="validation/overfit")
    ap.add_argument("--override", nargs="*", default=[],
                    help="OmegaConf dotlist overrides, e.g. model.dropout=0.0 normalization.profile=xy_p99_per_domain")
    ap.add_argument("--batch_name", default=None,
                    help="saved batch file stem(s) (default: the domain); comma list parallel to --domain")
    ap.add_argument("--rebind_normalization", action="store_true",
                    help="Explicitly attach the recipe's normalizers to the same saved raw fixtures "
                         "for a matched normalization ablation; never rewrites the fixture")
    ap.add_argument("--holdout_name", default=None,
                    help="second saved-batch stem(s) (same domain): SAMPLED at every eval, never trained on "
                         "(scissor test); drawn+saved with --draw_skip if missing; comma list parallel to --domain")
    ap.add_argument("--draw_skip", type=int, default=0,
                    help="skip this many production-stream batches before drawing a fresh batch")
    ap.add_argument("--init_from", default=None,
                    help="initialize the model weights (EMA when present) from this checkpoint before the "
                         "fixed-batch run; a fine-tune probe, not the historical from-scratch recipe")
    a = ap.parse_args()
    if a.steps <= 0 or a.eval_every <= 0 or a.warmup < 0 or a.draw_skip < 0:
        ap.error("steps/eval_every must be positive; warmup/draw_skip must be nonnegative")

    domains = a.domain.split(",")
    from data.unified_dataset import SUPPORTED_DOMAINS
    assert all(d in SUPPORTED_DOMAINS for d in domains) and len(set(domains)) == len(domains), domains
    joint = len(domains) > 1
    bnames = a.batch_name.split(",") if a.batch_name else [None] * len(domains)
    hnames = a.holdout_name.split(",") if a.holdout_name else [None] * len(domains)
    assert len(bnames) == len(domains) and len(hnames) == len(domains), "one --batch_name/--holdout_name per domain"

    cfg = load_config(a.config, a.override)
    if int(cfg.train.get("grad_accum", 1)) != 1:
        ap.error("This single-batch diagnostic requires train.grad_accum=1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        ap.error("This diagnostic requires one process; use the dedicated DDP gates")
    if a.require_saved_batches:
        names = [name or domain for domain, name in zip(domains, bnames)]
        names += [name for name in hnames if name]
        missing = [str(Path(a.batch_dir) / f"{name}.{ext}")
                   for name in names for ext in ("pt", "json")
                   if not (Path(a.batch_dir) / f"{name}.{ext}").is_file()]
        if missing:
            ap.error("Pinned reproduction cannot redraw missing fixtures: " + ", ".join(missing))
    tag = a.tag or os.path.splitext(os.path.basename(a.config))[0]
    out_dir = f"{a.out_root}/{tag}"
    # A reused diagnostic tag must never overwrite an earlier run/checkpoint.
    Path(out_dir).mkdir(parents=True, exist_ok=False)
    os.makedirs(a.batch_dir, exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(int(cfg.get("seed", 42)))
    np.random.seed(int(cfg.get("seed", 42)))
    random.seed(int(cfg.get("seed", 42)))
    stats_set = cfg.normalization.profile
    provenance = {**code_revision(), "torch": str(torch.__version__), "numpy": str(np.__version__),
                  "source_versions": working_source_versions(), "arguments": vars(a),
                  "resolved_config": config_for_json(cfg),
                  "metric_population": "all valid particles, including actuated",
                  "sampling_precision": "Accelerate-wrapped forward; FP32 integration outside forward",
                  "sampling_seed": 4242 if a.eval_rng == "isolated" else None,
                  "eval_rng": a.eval_rng, "holdout_sampling_seed": 4242,
                  "final_sampling_seeds": list(range(10000, 10008)),
                  "final_holdout_sampling_seeds": list(range(20000, 20008))}


    # ---------------------------------------------------------------- batches
    def _stream(domain, sources):
        from data.unified_dataset import resolve_data_roots
        ds = build_domain_dataset(domain, split="train", training=True,
                                  future_len=cfg.data.future_len,
                                  context_len=cfg.data.context_len, normalization=cfg.normalization,
                                  roots=resolve_data_roots(OmegaConf.to_container(cfg.data.roots, resolve=True)
                                                          if cfg.data.get("roots") is not None else None))
        ds.batch_size = int(cfg.data.batch_size)
        ds.bucket_max_items = int(cfg.data.get("bucket_max_items", 0) or 0)
        ds.emit_sidecar = bool(cfg.data.get("emit_sidecar", False))
        ds.articulated = bool(cfg.model.get("articulated_bodies", False))
        apply_window_policy(ds, cfg, domain)
        if ds.emit_sidecar and domain != "mpm":
            ds._pose_stats()
        process = getattr(ds, "_process_sample", None)     # shard datasets; water reads .npy windows

        def record_sample(sample):
            sources.append({k: str(sample.get(k, "")) for k in ("__url__", "__key__")})
            return process(sample)

        if process is not None:
            ds._process_sample = record_sample
        return iter(DataLoader(ds, batch_size=None, num_workers=0))


    def load_batch(domain, name, skip, extra_meta):
        """Load {batch_dir}/{name}.pt, or draw it from the production stream (after `skip`
        batches) and save it so every run shares it. Returns (batch, meta)."""
        path, mpath = f"{a.batch_dir}/{name}.pt", f"{a.batch_dir}/{name}.json"
        if os.path.exists(path):
            meta = json.load(open(mpath))
            actual_digest = file_sha256(path)
            if meta.get("sha256") and meta["sha256"] != actual_digest:
                raise ValueError(f"Batch receipt SHA256 mismatch: {path}")
            if not meta.get("sha256"):
                meta["integrity_note"] = "Legacy receipt had no digest; recorded current bytes, not verified historical identity"
                meta["sha256"] = actual_digest
            b = torch.load(path, weights_only=False)
            print(f"[batch] loaded {path}  (drawn {meta.get('drawn', 'pinned receipt')}, "
                  f"stats {fixture_normalization_label(meta)})", flush=True)
            return b, meta
        if a.require_saved_batches:
            raise FileNotFoundError(f"Pinned reproduction cannot redraw missing fixture: {path}")
        sources = []
        it = _stream(domain, sources)
        for _ in range(skip):
            next(it)
        b = next(it)
        torch.save(b, path)
        meta = {"domain": domain, "stats_set": stats_set,
                "normalization": resolve_normalization(cfg.normalization),
                "normalization_sha256": cfg.normalization.sha256, "config": a.config, "drawn": time.ctime(),
                "sha256": file_sha256(path), "parsed_sources_before_draw": sources,
                "source_note": "Includes parsed sources consumed while skipping/bucketing; exact fixture is the hashed saved batch",
                "B": b.batch_size, "N": int(b.positions.shape[2]),
                "C": int(b.context_len), "F": int(b.num_future_frames),
                "draw_skip": skip, **extra_meta}
        json.dump(meta, open(mpath, "w"), indent=1)
        print(f"[batch] drew a fresh production-stream batch (skip {skip}) -> {path}", flush=True)
        return b, meta


    class Dom:
        """Everything one domain carries: its batch, the scissor holdout, baselines, history."""


    doms = []
    for d, bname, hname in zip(domains, bnames, hnames):
        D = Dom(); D.name, D.hname = d, hname
        batch, meta = load_batch(d, bname or d, a.draw_skip, {})
        batch, meta = prepare_fixture(batch, meta, d, cfg.normalization,
            int(cfg.data.batch_size), rebind=a.rebind_normalization,
            context_len=int(cfg.data.context_len), future_len=int(cfg.data.future_len))
        D.receipt = meta
        if bool(cfg.model.get("articulated_bodies", False)) and d in ("pickup", "cloth", "rl_push", "folding", "yikai", "dex") and batch.joint_type is None:
            raise ValueError(f"model.articulated_bodies is on but the {d} fixture has no joints; "
                             "use a fixture drawn with articulated=True")
        batch.domain = d
        D.batch = batch.to(dev)
        D.B, D.C, D.F, D.N = batch.batch_size, batch.context_len, batch.num_future_frames, batch.positions.shape[2]
        print(f"[batch] {d}: B={D.B} C={D.C} F={D.F} N={D.N} stats={stats_set}", flush=True)

        # ------------------------------------------------- holdout batch (scissor test)
        D.hbatch = None
        if hname:
            hb, hmeta = load_batch(d, hname, max(a.draw_skip, 3), {"holdout_for": d})
            hb, hmeta = prepare_fixture(hb, hmeta, d, cfg.normalization,
                int(cfg.data.batch_size), rebind=a.rebind_normalization,
                context_len=int(cfg.data.context_len), future_len=int(cfg.data.future_len))
            D.hreceipt = hmeta
            hb.domain = d
            hb = hb.to(dev)
            _same = (hb.positions.shape == D.batch.positions.shape
                     and torch.equal(hb.positions, D.batch.positions))
            print(f"[holdout] {hname}: B={hb.batch_size} N={hb.positions.shape[2]} "
                  f"identical_to_train={_same}", flush=True)
            assert not _same, "holdout batch is identical to the train batch; raise --draw_skip"
            D.hbatch = hb
        doms.append(D)

    # ---------------------------------------------------------------- shared production construction
    from models.factory import build_particle_model, validate_particle_training_config
    validate_particle_training_config(cfg)
    arch = cfg.model.get("architecture", "transformer")
    model = build_particle_model(cfg).to(dev)
    if a.init_from:
        from utils.checkpoint import read_checkpoint
        from utils.state_dict import load_model_state
        _ck = read_checkpoint(a.init_from)
        _key = "ema" if _ck.get("ema") else "model"
        load_model_state(model, _ck[_key])
        provenance["init_from"] = {"checkpoint": a.init_from, "weights": _key,
                                   "sha256": file_sha256(a.init_from)}
        print(f"[model] initialized from {a.init_from} ({_key} weights)", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {arch}  {n_params / 1e6:.1f}M params  ({a.config})", flush=True)

    diffusion = TrajectoryFlowMatching(
        num_train_timesteps=cfg.flow_matching.num_train_timesteps, shift=cfg.flow_matching.shift,
        prediction_type=cfg.flow_matching.prediction_type,
        time_sampler=str(cfg.flow_matching.get("time_sampler", "uniform")),
        beta_alpha=float(cfg.flow_matching.get("beta_alpha", 1.5)),
        beta_beta=float(cfg.flow_matching.get("beta_beta", 1.0)),
        beta_t_min=float(cfg.flow_matching.get("beta_t_min", 0.001)))
    opt = build_optimizer(model, cfg.train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(a.warmup, 1)))
    acc = Accelerator(mixed_precision=str(cfg.train.get("mixed_precision", "no")))
    model, opt = acc.prepare(model, opt)

    # Same resolved objective as training AND validation; finite spike thresholds disabled.
    from train import trajectory_loss_options
    OBJECTIVE_OPTIONS = trajectory_loss_options(cfg)
    TS_KW = dict(OBJECTIVE_OPTIONS, grad_clip=cfg.train.grad_clip,
                 loss_ema=None, loss_spike_threshold=float("inf"), grad_norm_spike_threshold=float("inf"))


    # ---------------------------------------------------------------- scoring


    for D in doms:
        b, C, F = D.batch, D.C, D.F
        D.vm = b.get_valid_mask()                                 # [B, N]
        D.primary = primary_mask(b)
        D.gt = b.positions[:, C:C + F]                            # [B, F, N, 3]
        D.has_rigid = b.body_type is not None and bool((b.body_type == 1).any())
        zero_pred = b.positions[:, C - 1:C].expand(-1, F, -1, -1)
        vel = b.positions[:, C - 1:C] - b.positions[:, C - 2:C - 1]
        k = torch.arange(1, F + 1, device=dev).view(1, -1, 1, 1)
        D.BASE = {"zero_motion": part_mm(zero_pred, D.gt, D.vm),
                  "hold_velocity": part_mm(b.positions[:, C - 1:C] + k * vel, D.gt, D.vm)}
        D.PRIMARY_BASE = {"zero_motion": part_mm(zero_pred, D.gt, D.primary),
                          "hold_velocity": part_mm(b.positions[:, C - 1:C] + k * vel, D.gt, D.primary)}
        D.gt_motion = b.prediction_x0.norm(dim=-1)[D.vm[:, None].expand(-1, F, -1)].mean().item() * 1e3
        D.HBASE = None
        if D.hbatch is not None:
            hb = D.hbatch
            D.hvm = hb.get_valid_mask()
            D.hprimary = primary_mask(hb)
            D.hgt = hb.positions[:, C:C + F]
            _hzero = hb.positions[:, C - 1:C].expand(-1, F, -1, -1)
            _hvel = hb.positions[:, C - 1:C] - hb.positions[:, C - 2:C - 1]
            D.HBASE = {"zero_motion": part_mm(_hzero, D.hgt, D.hvm),
                       "hold_velocity": part_mm(hb.positions[:, C - 1:C] + k * _hvel, D.hgt, D.hvm)}
        # episode to visualize: the one with the most GT motion, not blindly episode 0
        # (the production mpm batch's episode 0 moves 0.04 mm/frame -- a frozen GIF).
        show_motion = (D.gt[:, 1:] - D.gt[:, :-1]) if F > 1 else (D.gt - b.positions[:, C - 1:C])
        D.SHOW_EP = int(show_motion.norm(dim=-1).mean(dim=(1, 2)).argmax())
        D.evals, D.hist, D.gnorm = [], [], []


    def sample(bb):
        C, F = bb.context_len, bb.num_future_frames
        m = acc.unwrap_model(model)  # retains Accelerate's configured forward autocast
        was_training = m.training
        m.eval()
        try:
            with torch.no_grad():
                out = run_diffusion_inference_df(
                    m, bb, diffusion, num_inference_steps=int(cfg.flow_matching.inference_steps),
                    schedule=cfg.get("df_schedule", "pyramid"),
                    uncertainty_scale=float(cfg.get("df_uncertainty_scale", 1.0)))
        finally:
            m.train(was_training)
        pose = None
        if isinstance(out, tuple):
            pred, R, T, ridx = materialize_pose_native(bb, *out)
            Rg = bb.body_poses[:, C:C + F, ridx, :3, :3].float()
            Tg = bb.body_poses[:, C:C + F, ridx, :3, 3].float()
            tr = torch.einsum("bfkii->bfk", R.transpose(-1, -2) @ Rg)
            ang = torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))
            pose = {"trans_mm": (T - Tg).norm(dim=-1).mean().item() * 1e3, "rot_deg": ang.mean().item()}
        else:
            pred = bb.predictions_to_positions(bb.prediction_stats.denormalize(out))
        return pred.float(), pose


    def save_sample(D, pred, step):
        idx = D.vm[D.SHOW_EP].nonzero(as_tuple=True)[0]
        if idx.numel() > 4000:
            g = torch.Generator().manual_seed(0)
            idx = idx[torch.randperm(idx.numel(), generator=g)[:4000]].sort().values
        e, C = D.SHOW_EP, D.C
        np.savez_compressed(f"{out_dir}/{D.name}_sample_step{step:05d}.npz",
                            pred=pred[e, :, idx].cpu().numpy().astype("float32"),
                            gt=D.gt[e, :, idx].cpu().numpy().astype("float32"),
                            context=D.batch.positions[e, :C, idx].cpu().numpy().astype("float32"),
                            particle_indices=idx.cpu().numpy(),
                            action_mask=D.batch.action_mask[e, idx].cpu().numpy(),
                            object_ids=D.batch.object_ids[e, idx].cpu().numpy().astype("int32"),
                            episode=np.int32(e))


    def dump(D):
        json.dump({"domain": D.name, "domains": domains, "joint": joint, "config": a.config,
                   "provenance": provenance,
                   "arch": arch, "n_params": n_params, "batch_receipt": D.receipt,
                   "B": D.B, "C": D.C, "F": D.F, "N": D.N, "stats_set": stats_set, "steps": a.steps,
                   "holdout_name": D.hname, "holdout_baselines": D.HBASE,
                   "holdout_receipt": getattr(D, "hreceipt", None),
                   "gt_motion_mm": D.gt_motion, "baselines": D.BASE, "has_rigid": D.has_rigid,
                   "primary_baselines": D.PRIMARY_BASE,
                   "primary_population": "valid non-actuated particles for rigid/mixed scenes; all deformable particles for MPM",
                   "evals": D.evals, "history": D.hist, "grad_norm": D.gnorm},
                  open(f"{out_dir}/{D.name}.json", "w"), allow_nan=False)


    def evaluate(D, step):
        # Default: cadence cannot change training noise/dropout. Historical reruns
        # explicitly opt into overfit_v2's shared training/evaluation RNG stream.
        pred, pose = sample_for_eval(sample, D.batch, dev,
                                    seed=None if a.eval_rng == "legacy" else 4242)
        mean_mm, final_mm = part_mm(pred, D.gt, D.vm)
        primary_mean, primary_final = part_mm(pred, D.gt, D.primary)
        save_sample(D, pred, step)
        med = float(np.median(D.hist[-100:])) if D.hist else float("nan")
        entry = {"step": step, "domain_updates": len(D.hist), "loss_median100": med if D.hist else None,
                 "sample_mean_mm": mean_mm, "sample_final_mm": final_mm,
                 "sample_primary_mean_mm": primary_mean, "sample_primary_final_mm": primary_final,
                 "sample_vel_mm": vel_mm(pred, D.gt, D.vm), "pose": pose, "elapsed_s": time.time() - t0}
        if a.fixed_probe and step in (0, a.steps):
            from scripts.unified.lossprobe import fixed_t_loss, large_t_loss
            sweep = fixed_t_loss(model, D.batch, diffusion, draws=4, loss_options=OBJECTIVE_OPTIONS)
            entry["fixed_t_objective"] = sweep
            entry["fixed_large_t_objective"] = large_t_loss(sweep)
        hnote = ""
        if D.hbatch is not None:
            # fixed noise seed -> smooth scissor curve; fork_rng so the training
            # RNG stream is identical to a run without --holdout_name
            hpred, hpose = sample_for_eval(sample, D.hbatch, dev)
            hmean, hfinal = part_mm(hpred, D.hgt, D.hvm)
            hpmean, hpfinal = part_mm(hpred, D.hgt, D.hprimary)
            entry.update({"holdout_mean_mm": hmean, "holdout_final_mm": hfinal,
                          "holdout_primary_mean_mm": hpmean, "holdout_primary_final_mm": hpfinal,
                          "holdout_vel_mm": vel_mm(hpred, D.hgt, D.hvm),
                          "holdout_pose": hpose})
            hnote = f"   {hmean:8.2f}"
        D.evals.append(entry)
        ps = f"{pose['trans_mm']:6.2f} / {pose['rot_deg']:5.2f}" if pose else "      -      "
        sps = (time.time() - t0) / max(step, 1)
        vel_note = "n/a" if entry["sample_vel_mm"] is None else f"{entry['sample_vel_mm']:.2f}"
        print(f"{step:>5} {D.name:>7} {med:>9.4f} {np.mean(D.gnorm[-100:]) if D.gnorm else float('nan'):>6.2f} "
              f"{mean_mm:>10.2f} {final_mm:>8.2f}  {ps}   {sps:.2f}"
              f"  vel {vel_note} primary {primary_mean:.2f}" + hnote, flush=True)
        dump(D)


    for D in doms:
        print(f"[score] {D.name}: GT motion {D.gt_motion:.2f} mm/frame | BASELINES zero-motion "
              f"{D.BASE['zero_motion'][0]:.2f} mm hold-velocity {D.BASE['hold_velocity'][0]:.2f} mm  (mean over frames)",
              flush=True)
        if D.HBASE is not None:
            print(f"[score] {D.name}: HOLDOUT baselines: zero-motion {D.HBASE['zero_motion'][0]:.2f} mm "
                  f"hold-velocity {D.HBASE['hold_velocity'][0]:.2f} mm", flush=True)
    if joint:
        print(f"[joint] {len(domains)} batches interleaved round-robin {domains}: {a.steps} steps = "
              f"{a.steps // len(domains)} updates per domain", flush=True)
    print(f"{'step':>5} {'domain':>7} {'loss med':>9} {'grad':>6} {'SAMPLE mm':>10} {'final-f':>8}  pose(trans mm / rot deg)   s/step"
          + ("   HOLD mm" if any(D.hbatch is not None for D in doms) else ""), flush=True)
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            for D in doms:
                evaluate(D, step)
        if step < a.steps:
            D = doms[step % len(doms)]
            loss, m = train_step(model=model, batch=D.batch, diffusion=diffusion, optimizer=opt,
                                 accelerator=acc, **TS_KW)
            if m["optimizer_step"]:
                sched.step()
            D.hist.append(float(loss)); D.gnorm.append(float(m.get("grad_norm", float("nan"))))
            if m["skipped"]:
                dump(D)
                raise RuntimeError(f"Rejected update at step {step} ({D.name}): {m}")
    if a.save_final:
        from utils.checkpoint import atomic_torch_save, capture_runtime
        atomic_torch_save({"model": acc.unwrap_model(model).state_dict(), "optimizer": opt.state_dict(),
                           "scheduler": sched.state_dict(), "cfg": OmegaConf.to_container(cfg, resolve=True),
                           "step": a.steps, "next_step": a.steps, "runtime": capture_runtime(acc),
                           "engineering_run": vars(a), "provenance": provenance}, f"{out_dir}/final.pt")

    # ---- how much does the final sample depend on the noise draw? K seeds on the memorised batch
    for D in doms:
        multi, primary_multi = [], []
        for seed in range(8):
            torch.manual_seed(10_000 + seed)
            pm, _ = sample(D.batch)
            multi.append(part_mm(pm, D.gt, D.vm)[0])
            primary_multi.append(part_mm(pm, D.gt, D.primary)[0])
        res = json.load(open(f"{out_dir}/{D.name}.json"))
        res["final_multi_seed_mm"] = {"n": len(multi), "mean": float(np.mean(multi)), "std": float(np.std(multi)),
                                      "min": float(np.min(multi)), "max": float(np.max(multi)), "values": multi}
        res["final_multi_seed_primary_mm"] = {"n": len(primary_multi), "mean": float(np.mean(primary_multi)),
            "std": float(np.std(primary_multi)), "values": primary_multi}
        if D.hbatch is not None:
            hm8 = []
            for seed in range(8):
                torch.manual_seed(20_000 + seed)
                pm, _ = sample(D.hbatch)
                hm8.append(part_mm(pm, D.hgt, D.hvm)[0])
            res["final_multi_seed_holdout_mm"] = {"n": 8, "mean": float(np.mean(hm8)), "std": float(np.std(hm8)),
                                                  "min": float(np.min(hm8)), "max": float(np.max(hm8)), "values": hm8}
            print(f"[final] {D.name} HOLDOUT over 8 noise seeds: mean {np.mean(hm8):.2f} mm  std {np.std(hm8):.2f}", flush=True)
        json.dump(res, open(f"{out_dir}/{D.name}.json", "w"), allow_nan=False)
        print(f"[final] {D.name} sample over 8 noise seeds: mean {np.mean(multi):.2f} mm  std {np.std(multi):.2f}  "
              f"min {np.min(multi):.2f}  max {np.max(multi):.2f}", flush=True)
        print(f"[done] {out_dir}/{D.name}.json", flush=True)


if __name__ == "__main__":
    main()
