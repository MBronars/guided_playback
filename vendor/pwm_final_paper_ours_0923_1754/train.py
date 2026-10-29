"""
Train either unified particle/pose flow-matching backbone.

Usage:
    # Single GPU
    python train.py --config configs/unet.yaml

    # Multi-GPU with Accelerate
    accelerate launch --multi_gpu --num_processes=2 train.py --config configs/transformer.yaml

    # Resume from checkpoint
    python train.py --config configs/unet.yaml --resume runs/exp/checkpoints/step_10000.pt
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed as accelerate_set_seed

from data.dataset import create_dataloader
from diffusion import TrajectoryFlowMatching
from utils.ema import EMA
from utils.iteration import repeat_nonempty
from utils import init_wandb, log_metrics, log_video, finish as finish_wandb
from utils.visualization import create_trajectory_comparison_gif
# Public aliases preserve historical imports; implementation lives outside run I/O.
from utils.trajectory_objective import (  # noqa: F401
    compute_aux_losses, compute_mesh_pair_loss,
    trajectory_loss_options, trajectory_objective,
)




def _gif_job(log_key, gt_positions, pred_positions, ib, sample_idx, gif_path, cfg):
    """Build a CPU-only render-job dict for the parallel gif pool. All tensors are
    moved to CPU here — the workers run in forked processes where CUDA is unusable."""
    return {
        "log_key": log_key,
        "path": gif_path,
        "kwargs": dict(
            gt_positions=gt_positions.detach().cpu(),
            pred_positions=pred_positions.detach().cpu(),
            action_mask=ib.action_mask.detach().cpu(),
            save_path=gif_path,
            sample_idx=sample_idx,
            valid_mask=ib.get_valid_mask().detach().cpu(),
            object_ids=ib.object_ids.detach().cpu(),
            context_len=cfg.model.context_len,
            future_len=cfg.data.get('future_len', None),
            obj0_height_color=cfg.data.get('obj0_height_color', False),
        ),
    }


def _render_gif_worker(kwargs):
    try:
        create_trajectory_comparison_gif(**kwargs)
        return None
    except Exception as e:
        return str(e)


def _render_and_log_gifs(jobs, step):
    """Render all gifs of one sample_every round in a CPU process pool, then upload.
    Serial matplotlib rendering was the dominant cost of the viz round (other ranks
    sit at the DDP barrier the whole time); the pool collapses it to ~one-gif time.
    Falls back to serial rendering if the fork pool fails for any reason — a slow
    viz round beats a crashed run."""
    if not jobs:
        return
    try:
        import multiprocessing as mp
        n = min(len(jobs), max(1, (os.cpu_count() or 8) - 2))
        with mp.get_context("fork").Pool(n) as pool:
            errs = pool.map(_render_gif_worker, [j["kwargs"] for j in jobs])
    except Exception as e:
        print(f"  gif render pool failed ({e}); falling back to serial rendering")
        errs = [_render_gif_worker(j["kwargs"]) for j in jobs]
    for job, err in zip(jobs, errs):
        if err is not None:
            print(f"  gif render failed ({job['log_key']}): {err}")
        else:
            log_video(job["log_key"], job["path"], step)


def train(cfg):
    """Main training function with multi-GPU support via Accelerate."""
    from utils.config import resolve_config
    from utils.checkpoint import (checkpoint_config, corpus_identity, read_checkpoint,
                                  validate_resume_configuration)
    cfg = resolve_config(cfg)
    corpus = corpus_identity(cfg)   # the exclusion contents its datasets load; saved in every checkpoint
    resume_state = None
    if cfg.resume:
        resume_state = read_checkpoint(cfg.resume, weights_only=False)
        validate_resume_configuration(resume_state, cfg, corpus=corpus)
        saved_precision = checkpoint_config(resume_state).train.get("mixed_precision", "no")
        if saved_precision != cfg.train.mixed_precision:
            raise ValueError("Resume mixed_precision differs from the checkpoint; exact continuation requires the same precision")

    def create_run_loader(*, training, num_workers=None, shuffle=True,
                          no_shard_split=False, first_window=False):
        """Every stream gets the same domain, geometry and numerical contract."""
        return create_dataloader(
            wds_location=cfg.data.wds_location, format=cfg.data.format,
            normalization=cfg.normalization, prediction_target=cfg.prediction_target,
            context_len=cfg.data.context_len, future_len=cfg.data.future_len,
            batch_size=cfg.data.batch_size, num_workers=(cfg.data.num_workers if num_workers is None else num_workers),
            emit_sidecar=cfg.data.emit_sidecar, bucket_max_items=cfg.data.bucket_max_items,
            articulated=bool(cfg.model.get("articulated_bodies", False)),
            domains=cfg.data.domains, eval_domains=cfg.data.get("eval_domains"),
            domain_weights=cfg.data.get("domain_weights"), seed=int(cfg.data.get("seed", 0)),
            roots=(OmegaConf.to_container(cfg.data.roots, resolve=True)
                   if cfg.data.get("roots") is not None else None),
            window_motion=(OmegaConf.to_container(cfg.data.window_motion, resolve=True)
                           if cfg.data.get("window_motion") is not None else None),
            batch_across_episodes=(OmegaConf.to_container(cfg.data.batch_across_episodes, resolve=True)
                                   if cfg.data.get("batch_across_episodes") is not None else None),
            training=training, shuffle=shuffle, no_shard_split=no_shard_split,
            first_window=first_window)

    from models.factory import validate_particle_training_config
    validate_particle_training_config(cfg)
    from data.dataset import validate_validation_config
    validate_validation_config(cfg)
    objective_options = trajectory_loss_options(cfg)
    # log.stream_rollout_keep: the logged rollouts and the sampler validation walk the STREAMING
    # staircase with c rows kept clean, the schedule a stream-trained checkpoint is meant for,
    # instead of the legacy pyramid. None keeps df_schedule.
    _stream_keep = cfg.log.get("stream_rollout_keep")
    # Initialize Accelerator
    gradient_accumulation_steps = cfg.train.grad_accum
    mixed_precision = cfg.train.mixed_precision  # "no", "fp16", "bf16"

    # Mixed domains activate different particle/native-pose and conditioning
    # branches. DDP must account for their changing used-parameter sets.
    # train.ddp_static_graph (opt-in): static_graph=True records the true
    # used-parameter set on iteration 1 — handles unused modules AND skips the
    # per-step graph walk find_unused_parameters does, which is faster when the
    # set of used params never changes across steps.
    # broadcast_buffers=False (ported from the unet branch, 2026-08-18): DDP's
    # default buffer broadcast fires on the FIRST forward after a backward —
    # during a rank0-only sample/viz round that forward emits a broadcast the
    # other ranks never join -> collective mispair -> NCCL deadlock at the
    # round-end barrier. Every registered buffer here is a deterministic
    # config constant (identical on all ranks); nothing needs the sync.
    if bool(cfg.train.get("ddp_static_graph", False)):
        ddp_kwargs = DistributedDataParallelKwargs(static_graph=True, broadcast_buffers=False)
    else:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
    # train.nccl_timeout_min: raise NCCL's collective timeout (default 10 min). The periodic
    # sample_every viz/inference block runs on rank 0 ONLY — other ranks block at the next
    # collective until it finishes, and a slow viz round (gif rendering + wandb video upload)
    # can exceed 10 min and get the job killed by the NCCL watchdog.
    from datetime import timedelta
    from accelerate import InitProcessGroupKwargs
    _pg_kwargs = InitProcessGroupKwargs(
        timeout=timedelta(minutes=int(cfg.train.get("nccl_timeout_min", 10))))
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs, _pg_kwargs],
    )

    # Set seed (Accelerate handles distributed seeding)
    accelerate_set_seed(cfg.seed)

    # Device is managed by accelerator
    device = accelerator.device

    # Only main process creates directories and saves config
    run_dir = Path(cfg.log.dir) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    if accelerator.is_main_process:
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)
        OmegaConf.save(cfg, run_dir / "config.yaml")
        print(f"Run directory: {run_dir}")
        print(f"Distributed: {accelerator.num_processes} process(es), mixed_precision={mixed_precision}")

    # Wait for main process to create directories
    accelerator.wait_for_everyone()

    # Initialize wandb (only on main process)
    if accelerator.is_main_process:
        init_wandb(
            project=cfg.log.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            run_name=cfg.run_name,
            entity=cfg.log.entity,
        )

    # Check for overfit mode
    overfit_batches = cfg.train.overfit_batches

    if accelerator.is_main_process:
        print(f"Creating unified dataloaders; prediction target={cfg.prediction_target}")
    train_loader = create_run_loader(
        training=True, shuffle=(overfit_batches <= 0),
        first_window=bool(cfg.data.get("first_window", False)))

    cached_batches = None
    if overfit_batches > 0:
        print(f"OVERFIT MODE: Caching {overfit_batches} batch(es)...")
        cached_batches = []
        for i, batch in enumerate(train_loader):
            if i == 0:
                print(f"[DataLoader] batch_size={batch.batch_size}, positions={list(batch.positions.shape)}")
            cached_batches.append(batch)
            if len(cached_batches) >= overfit_batches:
                break
        if not cached_batches:
            raise ValueError("Empty training loader: cannot cache overfit batches")
        print(f"Cached {len(cached_batches)} batch(es) for overfitting")
        val_loader = cached_batches
    else:
        # Disabled diagnostics must not open otherwise-unused datasets.
        val_loader = (create_run_loader(training=False)
                      if cfg.log.val_every > 0 or cfg.log.sample_every > 0 else None)

    # Training-population diagnostics use the same selected domains, objective,
    # normalization and weights as validation; only the source population differs.
    train_eval_loader = (create_run_loader(training=True, shuffle=True)
                         if overfit_batches <= 0 and cfg.log.val_every > 0 else None)

    # Model
    if accelerator.is_main_process:
        print("Creating model...")
    from models.factory import build_particle_model
    architecture = cfg.model.get("architecture", "transformer")
    model = build_particle_model(cfg)
    init_from = cfg.train.get("init_from")
    if init_from and not cfg.resume:
        # train.init_from starts a NEW experiment from another run's weights: fresh
        # optimizer, schedule, EMA and step 0 (the EMA below is built from the loaded
        # weights). Resume continues a run; on a resubmission its state supersedes this.
        from utils.checkpoint import validate_initialization
        from utils.reproducibility import file_sha256
        from utils.state_dict import load_model_state
        source = read_checkpoint(init_from, weights_only=False)
        validate_initialization(source, cfg)
        weights = "ema" if source.get("ema") else "model"
        load_model_state(model, source[weights])
        if accelerator.is_main_process:
            print(f"Initialized {weights} weights from {init_from} (step {source.get('step')}, "
                  f"sha256 {file_sha256(init_from)[:12]})")
        del source
    if accelerator.is_main_process:
        print(f"Architecture: {architecture}  ({type(model).__name__})")

    num_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process:
        print(f"Model parameters: {num_params:,}")

    diffusion = TrajectoryFlowMatching(
        num_train_timesteps=cfg.flow_matching.num_train_timesteps,
        shift=cfg.flow_matching.shift,
        prediction_type=cfg.flow_matching.prediction_type,
        time_sampler=str(cfg.flow_matching.get("time_sampler", "uniform")),
        beta_alpha=float(cfg.flow_matching.get("beta_alpha", 1.5)),
        beta_beta=float(cfg.flow_matching.get("beta_beta", 1.0)),
        beta_t_min=float(cfg.flow_matching.get("beta_t_min", 0.001)),
    )
    inference_steps = cfg.flow_matching.inference_steps
    if accelerator.is_main_process:
        print(f"Flow matching: prediction_type={diffusion.prediction_type}, inference_steps={inference_steps}")

    # Optimizer and scheduler
    # adam_betas / adam_eps are opt-in; unset => torch defaults (0.9, 0.999) / 1e-8,
    # i.e. byte-identical to every previous production run. pwm_fresh runs
    # betas (0.9, 0.95) + eps 1e-6 (shorter second-moment memory, larger eps floor).
    _betas = cfg.train.get("adam_betas", None)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(float(b) for b in _betas) if _betas is not None else (0.9, 0.999),
        eps=float(cfg.train.get("adam_eps", 1e-8)),
    )

    from utils.checkpoint import capture_runtime, load_training_state, read_checkpoint, restore_runtime
    from utils.lr_schedule import make_scheduler
    scheduler = make_scheduler(optimizer, cfg.train, accelerator.num_processes)
    lr_schedule = str(cfg.train.get("lr_schedule", "cosine"))
    wsd_decay_frac = float(cfg.train.get("wsd_decay_frac", 0.1))

    start_step = 0
    if cfg.resume:
        start_step = load_training_state(resume_state, model, optimizer, scheduler,
                                         normalization=OmegaConf.to_container(cfg.normalization, resolve=True))
        if accelerator.is_main_process:
            print(f"Resumed at next_step={start_step}, scheduler.last_epoch={scheduler.last_epoch}, "
                  f"current_lr={scheduler.get_last_lr()[0]:.3e}")

    # Prepare model, optimizer, scheduler with Accelerator
    # NOTE: We do NOT prepare dataloaders because:
    # 1. WebDataset already handles distributed sampling via split_by_node/split_by_worker
    # 2. Accelerate can't concatenate our custom TrajectoryBatch dataclass
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    # EMA tracks the unwrapped model for clean state dicts
    unwrapped_model = accelerator.unwrap_model(model)
    ema = EMA(unwrapped_model, decay=cfg.train.ema_decay,
              warmup=bool(cfg.train.get("ema_warmup", False)))

    from utils.compilation import prepare_compilation
    model = prepare_compilation(model, cfg.train, is_main_process=accelerator.is_main_process)

    # If resuming, load EMA state after creating EMA
    if resume_state is not None:
        ema.load_state_dict(resume_state["ema"], metadata=resume_state.get("ema_meta"))

    # Training loop
    step = start_step
    spike_saved = 0   # count of spike dumps written (bounded, see save block)
    model.train()

    if accelerator.is_main_process:
        print(f"Starting training for {cfg.train.steps} steps...")

    batch_iter = repeat_nonempty(cached_batches if cached_batches is not None else train_loader)

    # Get max batch size config (-1 means use full batch)
    max_batch_size = cfg.train.max_batch_size

    # Loss spike detection — overridable via config (defaults match train_step's defaults)
    loss_spike_threshold = cfg.train.get('loss_spike_threshold', 10.0)
    grad_norm_spike_threshold = cfg.train.get('grad_norm_spike_threshold', 50.0)
    loss_ema_value = None
    loss_ema_decay = 0.99
    best_val_loss = float('inf')
    best_metric = str(cfg.log.get("best_metric", "sample_mm"))
    last_log_time = None  # wall-clock of the previous periodic log, for it/s

    first_batch_logged = False

    # Sampling diagnostics have their own rank-zero draws. Opening these is
    # conditional: sample_every=0 performs no visualization-corpus I/O. All
    # draws retain the shared sidecar, domain, batch and normalization contract.
    pinned_viz_batches, pinned_train_batches = [], []
    num_vis = int(cfg.log.get("num_vis", 4))
    val_draw_loader = viz_val_draw_loader = viz_train_draw_loader = None
    if accelerator.is_main_process and cfg.log.sample_every > 0 and cached_batches is None:
        val_draw_loader = create_run_loader(training=False, num_workers=0, no_shard_split=True)
        viz_val_draw_loader = create_run_loader(training=False, num_workers=0, no_shard_split=True,
            first_window=bool(cfg.log.get("viz_first_window", False)))
        viz_train_draw_loader = create_run_loader(training=True, num_workers=0, no_shard_split=True,
            first_window=bool(cfg.log.get("viz_first_window", False)))

    def _make_draw(loader, label):
        if loader is None:
            return lambda: None  # explicitly disabled, or deliberate overfit cache
        batches = repeat_nonempty(loader)

        def draw():
            try:
                return next(batches).to(device)
            except Exception as error:
                raise RuntimeError(f"{label} data draw failed; evaluation cannot silently omit this split") from error
        return draw

    _draw_val_batch = _make_draw(val_draw_loader, "sample validation")
    _draw_val_viz_batch = _make_draw(viz_val_draw_loader, "visualization validation")
    _draw_train_viz_batch = _make_draw(viz_train_draw_loader, "visualization training")

    def _random_singles(_draw, _n):
        """_n single-episode batches from a draw function (splits pre-batched draws)."""
        from data.dataset import BaseGraspDataset
        out = []
        while len(out) < _n:
            b = _draw()
            if b is None:
                break
            for _i in range(b.batch_size):
                out.append(BaseGraspDataset._split_batch(b, _i, _i + 1))
                if len(out) >= _n:
                    break
        return out

    # Yaw (about-Z) rotation augmentation (see data/rotation_aug.py): fresh
    # random XY rotation per sample per step about the batch centroid; Z is
    # gravity and never rotates. All derived deltas/offsets follow positions.
    rot_aug = bool(cfg.train.get('rot_aug', False))
    _rot_aug_warned = False
    if rot_aug and accelerator.is_main_process:
        print("[ROT-AUG] Yaw-about-Z rotation augmentation ENABLED (XY-plane only; Z/gravity fixed)")

    # train.stop_after_step (optional, default off): leave the loop after the iteration
    # that writes step_<N>.pt, with the LR schedule still spanning cfg.train.steps —
    # for matched-step arms scored against an existing run's snapshots.
    stop_after = int(cfg.train.get("stop_after_step", 0) or 0)
    if stop_after and accelerator.is_main_process:
        print(f"train.stop_after_step={stop_after}: schedule spans {cfg.train.steps} steps, "
              f"run stops after step {stop_after}")
    if resume_state is not None:
        if "runtime" in resume_state:
            loop = restore_runtime(resume_state["runtime"], accelerator,
                                   legacy_precision=saved_precision)
            loss_ema_value = loop.get("loss_ema")
            # best_val is in the units of the metric that set it; a changed
            # log.best_metric (or a checkpoint written before the metric was
            # named) would compare millimetres against a loss, so the search
            # restarts instead.
            saved_metric = loop.get("best_metric")
            best_val_loss = loop.get("best_val", float("inf")) if saved_metric == best_metric else float("inf")
            if saved_metric != best_metric and accelerator.is_main_process:
                print(f"log.best_metric={best_metric} differs from the checkpoint's {saved_metric!r}: "
                      f"best.pt selection restarts from this step")
            spike_saved = int(loop.get("spike_saved", 0))
        elif accelerator.is_main_process:
            print("WARNING: legacy checkpoint lacks rank RNG/scaler/loop state and an unambiguous next_step; "
                  "this is not an exact continuation.")
        del resume_state
    while step < cfg.train.steps and not (stop_after and step > stop_after):
        checkpoint_due = False
        checkpoint_paths = []
        batch = next(batch_iter).to(device)
        if rot_aug:
            from data.rotation_aug import apply_yaw_aug, xy_centroid
            if not _rot_aug_warned and accelerator.is_main_process:
                _rot_aug_warned = True
                _d = batch.delta_stats
                if _d is not None and abs(float(_d.max_val[0]) - float(_d.max_val[1])) > 0.01 * abs(float(_d.max_val[0])):
                    print(f"[ROT-AUG] WARNING: delta stats are XY-anisotropic "
                          f"(x={float(_d.max_val[0]):.4f}, y={float(_d.max_val[1]):.4f}); yaw mixes "
                          f"X/Y so expect slightly more delta clipping than the un-rotated audit. "
                          f"For exact yaw-invariance use XY-isotropic stats (max of the two bounds).")
            apply_yaw_aug(batch, xy_centroid(batch.positions, batch.valid_lengths))

        # Log first batch info
        if not first_batch_logged and accelerator.is_main_process:
            print(f"[DataLoader] batch_size={batch.batch_size}, positions={list(batch.positions.shape)}")
            first_batch_logged = True

        # Optionally slice batch to reduce memory usage
        if max_batch_size > 0:
            batch = batch.slice(max_batch_size)

        # Training step with gradient accumulation support
        with accelerator.accumulate(model):
            in_grace = step < int(cfg.train.get("spike_grace_steps", 0))
            loss, metrics = train_step(
                model, batch, diffusion, optimizer, accelerator,
                grad_clip=cfg.train.grad_clip,
                loss_ema=None if in_grace else loss_ema_value,
                loss_spike_threshold=loss_spike_threshold,
                grad_norm_spike_threshold=float("inf") if in_grace else grad_norm_spike_threshold,
                **objective_options)

        # A step counts a synchronized accumulation group, not each microbatch.
        if not metrics["sync_gradients"]:
            continue
        skipped = metrics.get("skipped", False)

        # Save batch + draw forensics on the first few spikes (bounded count)
        if skipped and accelerator.is_main_process and spike_saved < 8:
            spike_path = os.path.join(cfg.log.dir, cfg.run_name,
                                      f"spike_batch_step{step}.pt")
            torch.save({"positions": batch.positions.cpu(), "actions": batch.actions.cpu(),
                        "action_mask": batch.action_mask.cpu(), "object_ids": batch.object_ids.cpu(),
                        "step": step, "loss": loss, "grad_norm": metrics.get("grad_norm"),
                        **{k: v for k, v in metrics.items() if k.startswith("spike_")}},
                       spike_path)
            print(f"  Saved spike batch to {spike_path}")
            spike_saved += 1

        if metrics["optimizer_step"]:
            scheduler.step()
            ema.update()

        # Update loss EMA (only from non-skipped steps)
        if not skipped:
            if loss_ema_value is None:
                loss_ema_value = loss
            else:
                loss_ema_value = loss_ema_decay * loss_ema_value + (1 - loss_ema_decay) * loss

        # Logging (only main process)
        if accelerator.is_main_process and step % cfg.log.every == 0:
            lr = scheduler.get_last_lr()[0]
            metrics["lr"] = lr
            now = time.time()
            if last_log_time is not None:
                metrics["steps_per_sec"] = cfg.log.every / (now - last_log_time)
            last_log_time = now
            if torch.cuda.is_available():
                # peak allocated since process start — the memory-profiling number
                metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            log_metrics({"train/" + k: v for k, v in metrics.items()}, step)

            msg = f"step {step:06d} | loss: {loss:.4f} | lr: {lr:.2e}"
            if "x0_mse_raw" in metrics:
                msg += f" | x0_raw: {metrics['x0_mse_raw']:.6f}"
            if "steps_per_sec" in metrics:
                msg += f" | {metrics['steps_per_sec']:.2f} it/s"
            if "gpu_mem_gb" in metrics:
                msg += f" | peak {metrics['gpu_mem_gb']:.1f} GiB"
            if "grad_norm" in metrics:
                msg += f" | grad_norm: {metrics['grad_norm']:.2f}"
            for name in ("loss_pos_cum", "loss_accel", "loss_com", "loss_spread", "loss_spread_axes", "loss_mesh",
                         "loss_strain_inc", "loss_strain_cum", "loss_floor", "loss_vel", "loss_accel_f0", "loss_accel"):
                if name in metrics:
                    msg += f" | {name[5:]}: {metrics[name]:.4f}"
            if skipped:
                msg += " | SKIPPED (spike)"
            print(msg)

        # Validation (all processes run validation, but only main logs)
        if cfg.log.val_every > 0 and step > 0 and step % cfg.log.val_every == 0:
            checkpoint_due = True
            torch.cuda.empty_cache()
            _val_kwargs = dict(
                model=unwrapped_model, val_loader=val_loader, diffusion=diffusion, ema=ema,
                device=device, accelerator=accelerator, num_batches=cfg.log.val_batches,
                **objective_options)
            # Twin validation passes: RAW weights are the PRIMARY val/* (pairs
            # with train/*); log.best_gate selects the best.pt criterion below.
            # The EMA shadow is
            # logged as val/*_ema (the deployable artifact; scalars only — all
            # visualizations in the sample_every block use EMA). The raw-vs-ema
            # difference doubles as a continuous EMA-health/lag curve.
            val_metrics = validate(use_ema=False, **_val_kwargs)
            val_metrics_ema = validate(use_ema=True, **_val_kwargs)
            # Full-sampler validation: the number that actually tracks generation.
            # log.sampler_batches=0 disables this production DF sampling pass.
            sampler_metrics = None
            _sb = int(cfg.log.get("sampler_batches", 8))
            if _sb > 0 and isinstance(diffusion, TrajectoryFlowMatching):
                sampler_metrics = validate_sampler(
                    model=model, val_loader=val_loader, diffusion=diffusion, ema=ema,
                    device=device, accelerator=accelerator, num_batches=_sb,
                    num_inference_steps=int(cfg.flow_matching.inference_steps),
                    schedule=("stream" if _stream_keep is not None
                              else cfg.get("df_schedule", "pyramid")),
                    uncertainty_scale=float(cfg.get("df_uncertainty_scale", 1.0)),
                    stream_keep=_stream_keep,
                    action_dropout=float(cfg.train.get("action_dropout", 0.0)))
            # EMA on a random TRAIN subset (same batch count): train/*_ema —
            # vs val/*_ema this is the deployed model's train/val gap.
            train_metrics_ema = None
            if train_eval_loader is not None:
                train_metrics_ema = validate(
                    use_ema=True, **{**_val_kwargs, "val_loader": train_eval_loader})
            # Return the val/inference rounds' transient blocks to CUDA: at
            # bs4 the 166M model trains at 36-38 GiB of a 39.5 GiB card, and
            # allocator fragmentation accumulated across val rounds is what
            # tipped the long split runs into OOM (~every 60k steps).
            torch.cuda.empty_cache()
            if accelerator.is_main_process:
                log_metrics({"val/" + k: v for k, v in val_metrics.items()}, step)
                log_metrics({f"val/{k}_ema": v for k, v in val_metrics_ema.items()}, step)
                if sampler_metrics is not None:
                    log_metrics({"val/" + k: v for k, v in sampler_metrics.items()}, step)
                if train_metrics_ema is not None:
                    log_metrics({f"train/{k}_ema": v for k, v in train_metrics_ema.items()}, step)
                val_msg = (f"step {step:06d} | val_loss: {val_metrics['loss']:.4f} "
                           f"(ema: {val_metrics_ema['loss']:.4f}"
                           + (f", train_ema: {train_metrics_ema['loss']:.4f})"
                              if train_metrics_ema is not None else ")"))
                # Per-domain val (unified loader) — the blended number can hide a
                # collapsing domain; keep the split visible in the console log.
                _doms = sorted(k[5:] for k in val_metrics if k.startswith("loss_"))
                if _doms:
                    val_msg += " | " + " ".join(
                        f"{d}: {val_metrics[f'loss_{d}']:.4f}/{val_metrics_ema.get(f'loss_{d}', float('nan')):.4f}"
                        for d in _doms)
                if "x0_mse_raw" in val_metrics:
                    val_msg += (f" | val_x0_raw: {val_metrics['x0_mse_raw']:.6f} "
                                f"(ema: {val_metrics_ema.get('x0_mse_raw', float('nan')):.6f})")
                if "x0_mm_t09" in val_metrics:
                    val_msg += " | x0mm@t.9: " + " ".join(
                        f"{k[len('x0_mm_t09_'):]}:{val_metrics[k]:.2f}"
                        for k in sorted(val_metrics) if k.startswith("x0_mm_t09_"))
                for name in ("loss_pos_cum", "loss_accel", "loss_com", "loss_spread", "loss_spread_axes",
                             "loss_strain_inc", "loss_strain_cum", "loss_floor", "loss_vel", "loss_accel_f0"):
                    if name in val_metrics:
                        val_msg += f" | val_{name[5:]}: {val_metrics[name]:.4f}"
                print(val_msg)
                if sampler_metrics is not None:
                    print("         | SAMPLER (EMA, fixed seeds, mm): " + "  ".join(
                        f"{k[len('sample_mm_'):]}: {sampler_metrics[k]:.2f}"
                        for k in sorted(sampler_metrics) if k.startswith("sample_mm_"))
                        + f"  | overall {sampler_metrics['sample_mm']:.2f}")

                # Save best checkpoint — log.best_metric ("sample_mm"|"loss",
                # default "sample_mm"): WHICH number crowns best.pt.
                # sample_mm is the full-sampler error in millimetres from
                # validate_sampler (EMA weights, fixed per-batch seeds): it
                # measures generation, which the one-pass loss does not track —
                # at 40K the uniform-t loss read 2.6 mm while the sampler
                # produced 26 mm on the same batches, and across normalizations
                # the loss ranked arms backwards. "loss" is the historical gate
                # and is not comparable across recipes that see different
                # information (bidirectional vs causal, different context).
                # log.best_gate ("ema"|"raw") then selects which weights' loss;
                # ema by default because rollouts and offline eval deploy the
                # EMA weights (load_model_and_diffusion prefers them), so "best"
                # means "the step where the DEPLOYED artifact was best". During
                # the early EMA warmup the shadow is init-contaminated and
                # scores badly — correctly producing no best checkpoints until
                # it heals. (val/* raw stays the primary generalization curve.)
                # log.save_best: false -> never write best.pt (debug runs keep latest.pt only)
                if bool(cfg.log.get("save_best", True)):
                    _gate = str(cfg.log.get("best_gate", "ema"))
                    if best_metric == "sample_mm":
                        if sampler_metrics is None:
                            raise ValueError("log.best_metric=sample_mm requires log.sampler_batches > 0")
                        gate_value, gate_name = sampler_metrics["sample_mm"], "sample_mm"
                    else:
                        gate_value = val_metrics_ema['loss'] if _gate == "ema" else val_metrics['loss']
                        gate_name = f"val_loss ({_gate}-gated)"
                    if gate_value < best_val_loss:
                        best_val_loss = gate_value
                        checkpoint_paths.append(ckpt_dir / "best.pt")
                        print(f"  New best {gate_name}: {gate_value:.4f}")

                # Always save latest checkpoint (overwritten each time)
                checkpoint_paths.append(ckpt_dir / "latest.pt")

        # Periodic step snapshots (log.save_every, 0 = off). Needed for rollout-based checkpoint
        # selection: one-step val loss does not track rollout quality, so best.pt alone is not
        # enough to pick the deployable checkpoint.
        save_every = int(cfg.log.get("save_every", 0))
        if save_every > 0 and step > 0 and step % save_every == 0:
            checkpoint_due = True
            if accelerator.is_main_process:
                checkpoint_paths.append(ckpt_dir / f"step_{step}.pt")

        # PRE-DECAY snapshot (log.save_pre_decay, wsd only; ported from the unet
        # branch): one extra checkpoint at the last step of the STABLE phase,
        # just before the WSD decay ramp. The decay anneals lr to 0 — a one-way
        # door — so this is the only way to branch a different decay
        # (shorter/longer, different data) without re-running the stable phase.
        # Written ONCE per run as checkpoints/pre_decay.pt (idempotent across
        # requeues via the exists() check).
        if bool(cfg.log.get("save_pre_decay", False)) and lr_schedule == "wsd":
            _pre_decay_step = int(round(cfg.train.steps * (1.0 - wsd_decay_frac)))
            _pre_decay_path = ckpt_dir / "pre_decay.pt"
            if step == max(start_step, _pre_decay_step):
                checkpoint_due = True
                if accelerator.is_main_process and not _pre_decay_path.exists():
                    checkpoint_paths.append(_pre_decay_path)

        # Periodic inference (full sampling) - only on main process, but all must sync
        sample_every = cfg.log.sample_every
        if sample_every > 0 and step > start_step and step % sample_every == 0:  # resume-step guard
            # Only main process runs inference, but ALL processes must enter this block
            if accelerator.is_main_process:
                torch.cuda.empty_cache()
                # log.viz_weights ("ema"|"raw", default "ema"): which weights the
                # sample_every eval family (inference/onestep scalars + ALL gifs)
                # runs on. Default ema = the deployable artifact.
                _viz_ema = str(cfg.log.get("viz_weights", "ema")) == "ema"
                if _viz_ema:
                    ema.apply_shadow()
                model.eval()

                try:
                    viz_dir = run_dir / "visualizations"
                    viz_dir.mkdir(exist_ok=True)
                    gif_jobs = []

                    # Eval data for this round: the live TRAIN batch + a FRESH random
                    # held-out VAL draw (scalars on both -> train/val gap curves).
                    val_batch = _draw_val_batch()
                    if val_batch is not None and max_batch_size > 0:
                        val_batch = val_batch.slice(max_batch_size)
                    eval_sets = [("train", batch)]
                    if val_batch is not None:
                        eval_sets.append(("val", val_batch))

                    if cfg.diffusion_forcing:
                        # Assemble this round's viz slots ONCE (shared by both
                        # schedules so they render the same episodes): pinned
                        # prefix, then fresh random draws for the remainder.
                        round_viz = {
                            "train": list(pinned_train_batches),
                            "val": list(pinned_viz_batches),
                        }
                        round_viz["train"] += _random_singles(
                            _draw_train_viz_batch, num_vis - len(round_viz["train"]))
                        round_viz["val"] += _random_singles(
                            _draw_val_viz_batch, num_vis - len(round_viz["val"]))
                        # OVERFIT MODE (and any run without viz draw loaders) leaves both
                        # slot lists empty, which silently renders ONLY the onestep gifs
                        # and no full_sequence/pyramid rollouts. Fall back to the live
                        # batch itself -- in overfit mode that IS the cached batch, which
                        # is exactly the episode you want to watch being fitted.
                        from data.dataset import BaseGraspDataset as _BGD
                        if not round_viz["train"]:
                            round_viz["train"] = [
                                _BGD._split_batch(batch, _i, _i + 1)
                                for _i in range(min(num_vis, batch.batch_size))]
                        if not round_viz["val"] and val_batch is not None:
                            round_viz["val"] = [
                                _BGD._split_batch(val_batch, _i, _i + 1)
                                for _i in range(min(num_vis, val_batch.batch_size))]

                        # DF: both schedules; scalars per data split; gifs from
                        # the round's viz slots.
                        # log.stream_rollout_keep replaces the pyramid slot with the streaming
                        # staircase a stream-trained checkpoint is actually meant to walk.
                        for schedule in ["full_sequence",
                                         "stream" if _stream_keep is not None else "pyramid"]:
                            for tag, ib in eval_sets:
                                inference_metrics = run_inference_df(
                                    model=model,
                                    batch=ib,
                                    diffusion=diffusion,
                                    num_inference_steps=inference_steps,
                                    schedule=schedule,
                                    uncertainty_scale=cfg.df_uncertainty_scale,
                                    stream_keep=_stream_keep if schedule == "stream" else None,
                                )
                                log_metrics({
                                    f"inference_{schedule}/position_mse_{tag}": inference_metrics["position_mse"],
                                    f"inference_{schedule}/delta_mse_{tag}": inference_metrics["delta_mse"],
                                }, step)
                                print(f"step {step:06d} | inference ({schedule}, {tag}) "
                                      f"pos_mse: {inference_metrics['position_mse']:.6f} | "
                                      f"delta_mse: {inference_metrics['delta_mse']:.6f}")
                            # Every slot i renders a TRAIN/VAL PAIR under matching
                            # keys. Slots beyond
                            # the pinned sets are fresh random draws this round
                            # (round_viz, assembled once so both schedules render
                            # the same episodes).
                            for pi in range(num_vis):
                                pairs = []
                                for ptag in ("train", "val"):
                                    if pi < len(round_viz[ptag]):
                                        pairs.append((ptag, round_viz[ptag][pi]))
                                for ptag, pb in pairs:
                                    pm = run_inference_df(
                                        model=model, batch=pb, diffusion=diffusion,
                                        num_inference_steps=inference_steps,
                                        schedule=schedule,
                                        uncertainty_scale=cfg.df_uncertainty_scale,
                                        stream_keep=_stream_keep if schedule == "stream" else None,
                                    )
                                    gif_jobs.append(_gif_job(
                                        f"inference_{schedule}/viz{pi}_{ptag}",
                                        pm["gt_positions"], pm["predicted_positions"],
                                        pb, 0,
                                        str(viz_dir / f"step_{step:07d}_{schedule}_viz{pi}_{ptag}.gif"),
                                        cfg))

                    # One-step denoising: fixed low/mid/high noise levels + the
                    # x0_recon-style "random" variant (per-frame random t, matching
                    # the training draws — one render spans the whole noise range).
                    # Scalars AND gifs on both splits: every gif has a train/val pair.
                    for test_timestep in [100, 500, 900, "random"]:
                        label = "trand" if test_timestep == "random" else f"t{test_timestep}"
                        for tag, ib in eval_sets:
                            onestep_result = run_one_step_denoising(
                                model=model,
                                batch=ib,
                                diffusion=diffusion,
                                timestep=test_timestep,
                                diffusion_forcing=cfg.diffusion_forcing
                            )
                            log_metrics({f"onestep/mse_{label}_{tag}": onestep_result["mse"]}, step)
                            print(f"  one-step {label} ({tag}): mse={onestep_result['mse']:.6f}")
                            gif_jobs.append(_gif_job(
                                f"onestep/{label}_{tag}",
                                onestep_result["gt_positions"],
                                onestep_result["est_positions"],
                                ib, 0,
                                str(viz_dir / f"step_{step:07d}_onestep_{label}_{tag}.gif"),
                                cfg))

                    # Render every gif of this round in a CPU pool, then upload.
                    _render_and_log_gifs(gif_jobs, step)
                finally:
                    # Always restore training weights, even if exception occurs
                    if _viz_ema:
                        ema.restore()
                    model.train()

            # ALL processes must sync here (outside the is_main_process block)
            accelerator.wait_for_everyone()

        if checkpoint_due:
            runtime = capture_runtime(accelerator, next_step=step + 1, loss_ema=loss_ema_value,
                                      best_val=best_val_loss, best_metric=best_metric,
                                      spike_saved=spike_saved)
            if accelerator.is_main_process:
                for path in checkpoint_paths:
                    save_checkpoint(path, unwrapped_model, ema, optimizer, scheduler, step, cfg,
                                    corpus=corpus, next_step=step + 1, runtime=runtime)
        step += 1

    # Final checkpoint
    accelerator.wait_for_everyone()
    final_runtime = (capture_runtime(accelerator, next_step=step, loss_ema=loss_ema_value,
                                     best_val=best_val_loss, best_metric=best_metric,
                                     spike_saved=spike_saved)
                     if bool(cfg.log.get("save_final", True)) else None)
    if accelerator.is_main_process:
        if bool(cfg.log.get("save_final", True)):   # false -> latest.pt only
            ckpt_path = ckpt_dir / "final.pt"
            save_checkpoint(ckpt_path, unwrapped_model, ema, optimizer, scheduler, step, cfg,
                            corpus=corpus, next_step=step, runtime=final_runtime)
        print("Training complete!")
        finish_wandb()


def train_step(model, batch, diffusion, optimizer, accelerator, grad_clip=None, *,
               loss_ema=None, loss_spike_threshold=10.0,
               grad_norm_spike_threshold=50.0, **loss_options):
    """Production backward/update boundary; loss options go to trajectory_objective."""
    from utils.optimization import backward_step
    loss, metrics, draw = trajectory_objective(model, batch, diffusion, **loss_options)
    step = backward_step(
        loss, model, optimizer, accelerator, grad_clip=grad_clip,
        loss_ema=loss_ema, loss_spike_threshold=loss_spike_threshold,
        grad_norm_spike_threshold=grad_norm_spike_threshold)
    metrics.update(step)
    if step["skipped"]:
        with torch.no_grad():
            prediction, target = draw["prediction"], draw["target"]
            if prediction is None:
                prediction, target = draw["pose_prediction"], draw["pose_target"]
                metrics["spike_space"] = "pose"
            else:
                metrics["spike_space"] = "particle"
            per = (prediction.float() - target.float()).square()
            if metrics["spike_space"] == "pose" and draw.get("pose_mask") is not None:
                per = per * draw["pose_mask"]           # jointed bodies: only their joint channel
            metrics["spike_timesteps"] = draw["timesteps"].detach().cpu()
            metrics["spike_frame_mse"] = per.mean((0, 2, 3)).cpu()
            metrics["spike_elem_mse"] = per.mean((1, 2, 3)).cpu()
            flat = per.mean(-1).flatten()
            top_values, top_indices = flat.topk(min(10, flat.numel()))
            metrics["spike_top_errs"] = top_values.cpu()
            metrics["spike_top_flat_idx"] = top_indices.cpu()
            if draw["pose_native"] and draw["prediction"] is not None:
                pose_per = (draw["pose_prediction"].float() - draw["pose_target"].float()).square()
                if draw.get("pose_mask") is not None:
                    pose_per = pose_per * draw["pose_mask"]
                metrics["spike_pose_frame_mse"] = pose_per.mean((0, 2, 3)).cpu()
    return loss.item(), metrics

def _prediction_metrics(batch, predicted_positions):
    """Physical valid-particle metrics shared by training-time diagnostics."""
    from utils.evaluation import validate_prediction
    valid, gt = batch.get_valid_mask(), batch.future
    validate_prediction(predicted_positions, gt, valid)
    predicted_deltas = torch.diff(torch.cat([batch.anchor[:, None], predicted_positions], dim=1), dim=1)
    mask = valid[:, None, :, None].expand_as(gt)
    error = (predicted_positions - gt).square()
    # Mask by selection rather than multiplication: NaN padding must not leak.
    per_frame = error.permute(1, 0, 2, 3)[:, valid].mean(dim=(1, 2))
    return dict(position_mse=error[mask].mean().item(),
                delta_mse=(predicted_deltas - batch.future_deltas).square()[mask].mean().item(),
                per_frame_mse=per_frame.cpu().tolist(), predicted_positions=predicted_positions,
                predicted_deltas=predicted_deltas, gt_positions=gt)


@torch.no_grad()
def run_one_step_denoising(model, batch, diffusion, timestep=500, diffusion_forcing=True):
    """Fixed-noise diagnostic for deformable particles and native rigid poses.

    Reports physical position MSE after production output materialization.
    This deliberately replaces the old particle-only normalized-space diagnostic.
    """
    from utils.conditioning import clean_context, model_metadata
    from utils.pose_native import materialize_pose_native
    if not diffusion_forcing:
        raise ValueError("Unified diagnostics require diffusion_forcing")
    B, F = batch.batch_size, batch.num_future_frames
    if timestep == "random":
        t = torch.randint(1, diffusion.num_train_timesteps, (B, F), device=batch.device)
    else:
        if not 0 < timestep < diffusion.num_train_timesteps:
            raise ValueError("Diagnostic timestep must lie inside the training interval")
        t = torch.full((B, F), timestep, device=batch.device, dtype=torch.long)
    x0 = batch.prediction_stats.normalize(batch.prediction_x0, clip=True)
    noisy = diffusion.add_noise(x0, torch.randn_like(x0), t)
    context, ctx_t = clean_context(model, batch)
    kwargs = dict(noisy_context=context, context_timesteps=ctx_t)
    base = model_metadata(model)
    native = bool(getattr(getattr(base, "cfg", None), "pose_native", False)
                  and batch.pose_deltas_gt is not None)
    if native:
        ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
        pose_x0 = batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, ridx], clip=True)
        pose_noise = torch.randn_like(pose_x0)
        from utils.pose_native import pose_channel_mask
        mask = pose_channel_mask(batch, ridx)
        if mask is not None:                            # jointed bodies: only their joint channel
            mask = mask.to(pose_x0)
            pose_x0, pose_noise = pose_x0 * mask, pose_noise * mask
        noisy_pose = diffusion.add_noise(pose_x0, pose_noise, t)
        pred, pose_pred = model(batch, noisy, t, noisy_pose=noisy_pose, **kwargs)
        estimated = None if pred is None else diffusion.get_x0_est(pred, t, noisy)
        pose_est = diffusion.get_x0_est(pose_pred, t, noisy_pose)
        positions = materialize_pose_native(batch, estimated, pose_est)[0]
    else:
        pred = model(batch, noisy, t, **kwargs)
        estimated = diffusion.get_x0_est(pred, t, noisy)
        positions = batch.predictions_to_positions(batch.prediction_stats.denormalize(estimated))
    metrics = _prediction_metrics(batch, positions)
    return dict(gt_positions=batch.future, est_positions=positions,
                mse=metrics["position_mse"], timestep=timestep,
                metric_units="m^2; valid particles after pose materialization")


@torch.no_grad()
def run_inference_df(model, batch, diffusion, num_inference_steps=50,
                     schedule="full_sequence", uncertainty_scale=1.0, stream_keep=None):
    """Training diagnostics use the exact same sampler as offline evaluation."""
    from inference import run_diffusion_inference_df
    from utils.pose_native import materialize_pose_native
    samples = run_diffusion_inference_df(model, batch, diffusion, num_inference_steps,
                                        schedule, uncertainty_scale, stream_keep=stream_keep)
    if isinstance(samples, tuple):
        positions = materialize_pose_native(batch, *samples)[0]
    else:
        positions = batch.predictions_to_positions(batch.prediction_stats.denormalize(samples))
    return _prediction_metrics(batch, positions)


def _large_t_error(model, batch, diffusion, draw, seed):
    """Fixed t≈0.9 particle error in mm, on a forked RNG stream."""
    if (not isinstance(diffusion, TrajectoryFlowMatching)
            or draw["prediction"] is None or not bool(draw["valid_mask"].any())):
        return None
    device = batch.device
    from utils.reproducibility import seed_for_device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        seed_for_device(seed, device)
        timestep = 0.9 * diffusion.num_train_timesteps
        times = torch.full_like(draw["timesteps"], timestep)
        noisy = diffusion.add_noise(draw["x0"], torch.randn_like(draw["x0"]), times)
        kwargs = dict(noisy_context=draw["context"], context_timesteps=draw["context_timesteps"])
        if draw["pose_native"]:
            pose_x0 = draw["pose_x0"]
            pose_noise = torch.randn_like(pose_x0)
            if draw.get("pose_mask") is not None:       # jointed bodies: only their joint channel
                pose_noise = pose_noise * draw["pose_mask"]
            kwargs["noisy_pose"] = diffusion.add_noise(pose_x0, pose_noise, times)
            prediction, _ = model(batch, noisy, times, **kwargs)
        else:
            prediction = model(batch, noisy, times, **kwargs)
        if prediction is None:
            return None
        estimate = diffusion.get_x0_est(prediction, times, noisy)
        error = (batch.prediction_stats.denormalize(estimate) - batch.prediction_x0).norm(dim=-1) * 1000
        return error[draw["valid_mask"][..., 0]].mean().item()


def _reduce_validation(sums, counts, accelerator, device, required="loss"):
    """Reduce sufficient statistics, including ranks with missing domain keys."""
    keys = set(sums) | {required}
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, sorted(keys))
        keys = set().union(*gathered)
    names = sorted(keys)
    values = torch.tensor([[sums.get(k, 0.0), counts.get(k, 0)] for k in names],
                          device=device, dtype=torch.float64)
    values = accelerator.reduce(values, reduction="sum").cpu()
    if values[names.index(required), 1] == 0:
        raise ValueError("Validation produced no batches; refusing to report a zero loss")
    result = {name: float(total / count) for name, (total, count) in zip(names, values) if count > 0}
    if not all(math.isfinite(v) for v in result.values()):
        raise RuntimeError(f"Non-finite validation metrics: {result}")
    return result


@torch.no_grad()
def validate(
    model, val_loader, diffusion, ema, device, accelerator, num_batches=10,
    diffusion_forcing=True, loss_weighting="uniform",
    aux_loss_weights=None, noise_context=None, pose_noise_context=False, deformable_context=False,
    light_context_noise_std=0.0, lambda_pose_native=1.0,
    clean_ctx_timesteps=False, use_ema=True, *, clip_deltas=True, clip_pose_deltas=True,
    lambda_mesh=0.0, mesh_pair_K=8, supervise_context=False, action_dropout=0.0,
    light_pose_noise_std=0.0, light_pose_noise_anchor=False,
    light_pose_noise_target="restore", light_pose_noise_scope="auto", light_pose_noise_type="iid",
    pose_loss_space="velocity",
    timestep_schedule="independent", ramp_slope=None, ramp_jitter=None, stream_options=None,
    velocity_weight_cap=0.0, pose_action_mode="skip",
):
    """Draw-matched validation of the SAME objective as train_step.

    Model mode and ambient RNG are restored, even on error. Losses are averaged
    over domain-pure batches, matching the training stream's sampling unit.
    Empty/non-finite evaluations are errors, never candidate best checkpoints.
    """
    from itertools import islice
    from utils.conditioning import model_metadata
    from utils.reproducibility import seed_for_device
    was_training = model.training
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    base = model_metadata(model)
    model_cfg = getattr(base, "cfg", None)
    options = dict(
        diffusion_forcing=diffusion_forcing, loss_weighting=loss_weighting,
        aux_loss_weights=aux_loss_weights,
        noise_context=(bool(diffusion_forcing and getattr(model_cfg, "all_relative", False))
                       if noise_context is None else noise_context),
        pose_noise_context=pose_noise_context,
        deformable_context=deformable_context,
        light_context_noise_std=light_context_noise_std,
        lambda_pose_native=lambda_pose_native, clean_ctx_timesteps=clean_ctx_timesteps,
        clip_deltas=clip_deltas, clip_pose_deltas=clip_pose_deltas, lambda_mesh=lambda_mesh,
        mesh_pair_K=mesh_pair_K, supervise_context=supervise_context, action_dropout=action_dropout,
        light_pose_noise_std=light_pose_noise_std, light_pose_noise_anchor=light_pose_noise_anchor,
        light_pose_noise_target=light_pose_noise_target, light_pose_noise_scope=light_pose_noise_scope,
        light_pose_noise_type=light_pose_noise_type, pose_loss_space=pose_loss_space,
        timestep_schedule=timestep_schedule, ramp_slope=ramp_slope, ramp_jitter=ramp_jitter,
        stream_options=stream_options, velocity_weight_cap=velocity_weight_cap,
        pose_action_mode=pose_action_mode)
    sums, counts = {}, {}

    def add(name, value):
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1

    if use_ema:
        ema.apply_shadow()
    model.eval()
    try:
        seed_for_device(12345, device)
        for index, batch in enumerate(islice(val_loader, num_batches)):
            batch = batch.to(device)
            _, metrics, draw = trajectory_objective(model, batch, diffusion, **options)
            for name, value in metrics.items():
                if name.startswith("loss") or name == "x0_mse_raw":
                    add(name, value)
            domain = getattr(batch, "domain", None)
            if domain is not None:
                add(f"loss_{domain}", metrics["loss"])
            # Reuse the actual augmented batch, including sampled partial action dropout.
            error = _large_t_error(model, draw["batch"], diffusion, draw, 777 + index)
            if error is not None:
                add("x0_mm_t09", error)
                if domain is not None:
                    add(f"x0_mm_t09_{domain}", error)
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        if use_ema:
            ema.restore()
        model.train(was_training)
    result = _reduce_validation(sums, counts, accelerator, device)
    if isinstance(diffusion, TrajectoryFlowMatching):
        result.setdefault("x0_mse_raw", 0.0)  # no particle head on pure-rigid batches
    return result


@torch.no_grad()
def validate_sampler(
    model: nn.Module,
    val_loader,
    diffusion: TrajectoryFlowMatching,
    ema: EMA,
    device: torch.device,
    accelerator: Accelerator,
    num_batches: int = 8,
    num_inference_steps: int = 25,
    schedule: str = "pyramid",
    uncertainty_scale: float = 1.0,
    stream_keep: int | None = None,
    use_ema: bool = True,
    action_dropout: float = 0.0,
) -> dict:
    """Full-sampler validation (2026-08-30): generate with the REAL DF sampler from pure
    noise and score the particles in millimetres against ground truth.

    Exists because the one-pass val loss cannot see sample quality: at 40K the uniform-t
    metric read 2.6 mm while the sampler produced 26 mm on the same batches, and across
    normalizations the loss ranked arms backwards. Fixed per-batch seeds make this a
    deterministic curve; EMA weights by default (the deployable artifact). Cost is
    num_batches x num_inference_steps forwards -- keep num_batches small (the unified val
    round-robin gives one batch per domain per round, so 8 = 2 per domain).
    """
    from inference import run_diffusion_inference_df           # lazy: inference imports models, not train
    from utils.pose_native import materialize_pose_native
    from utils.conditioning import hide_actions
    from itertools import islice
    from utils.reproducibility import seed_for_device
    was_training = model.training
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    base = getattr(model, "module", model)
    sums, counts = {}, {}

    def add(name, value):
        sums[name] = sums.get(name, 0.0) + value
        counts[name] = counts.get(name, 0) + 1

    if use_ema:
        ema.apply_shadow()
    model.eval()
    try:
        # model.unet_pose_action_mode=clean with 0 < action_dropout < 1 trains ONE model for two tasks
        # (actuator given / actuator generated). Score both on the SAME noise, as a paired comparison:
        # sample_mm* keeps the given-actuator meaning, uncond_sample_mm* is every action hidden.
        # Other recipes score exactly as before -- one conditioning, one seed.
        _clean = str(getattr(getattr(base, "unet_cfg", None), "pose_action_mode", "skip")) == "clean"
        _both = _clean and 0.0 < float(action_dropout) < 1.0
        for index, batch in enumerate(islice(val_loader, num_batches)):
            batch = batch.to(device)
            # A recipe trained with every action hidden (train.action_dropout=1) is scored as
            # it deploys, generating the actuator; ground truth still scores every body.
            conds = [("", hide_actions(batch, 1.0) if action_dropout >= 1 else batch)]
            if _both:
                conds.append(("uncond_", hide_actions(batch, 1.0)))
            for prefix, sampled in conds:
                seed_for_device(54321 + index, device)
                out = run_diffusion_inference_df(
                    base, sampled, diffusion, num_inference_steps=num_inference_steps,
                    schedule=schedule, uncertainty_scale=uncertainty_scale, stream_keep=stream_keep)
                if isinstance(out, tuple):
                    pred, _, _, _ = materialize_pose_native(batch, *out)
                else:
                    pred = batch.predictions_to_positions(
                        batch.prediction_stats.denormalize(out))
                gt = batch.positions[:, batch.context_len:batch.context_len + pred.shape[1]]
                vm = batch.get_valid_mask()
                err = (pred.float() - gt).norm(dim=-1) * 1e3
                mean = err[vm[:, None].expand_as(err)].mean().item()
                final = err[:, -1][vm].mean().item()
                add(f"{prefix}sample_mm", mean)
                add(f"{prefix}sample_final_mm", final)
                domain = getattr(batch, "domain", None)
                if domain is not None:
                    add(f"{prefix}sample_mm_{domain}", mean)
                    add(f"{prefix}sample_final_mm_{domain}", final)
    finally:
        torch.random.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        if use_ema:
            ema.restore()
        model.train(was_training)
    return _reduce_validation(sums, counts, accelerator, device, required="sample_mm")


def save_checkpoint(path, model, ema, optimizer, scheduler, step, cfg, *, corpus, next_step=None, runtime=None):
    """Always atomic; step labels the artifact, next_step identifies continuation and corpus the
    exclusion contents the run loaded (utils.checkpoint.corpus_identity)."""
    from utils.checkpoint import atomic_torch_save, training_checkpoint
    atomic_torch_save(training_checkpoint(model, ema, optimizer, scheduler, step, cfg, corpus=corpus,
                                          next_step=next_step, runtime=runtime), path)
    print(f"Saved: {path}")


def load_checkpoint(path, model, optimizer, scheduler, accelerator):
    from utils.checkpoint import load_training_state, read_checkpoint
    state = read_checkpoint(path, weights_only=False)
    return load_training_state(state, model, optimizer, scheduler)


def load_ema_checkpoint(path, ema, device):
    """Compatibility entrypoint; EMA moves weights onto its own model's device."""
    from utils.checkpoint import read_checkpoint
    state = read_checkpoint(path, weights_only=False)
    ema.load_state_dict(state["ema"], metadata=state.get("ema_meta"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train particle trajectory diffusion model")
    parser.add_argument("--config", type=str, required=True, help="Path to a supported config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format (e.g., model.dim=512 train.lr=1e-4)",
    )
    args = parser.parse_args()

    # Load base config
    from utils.config import load_config
    cfg = load_config(args.config, args.overrides)

    if args.resume:
        cfg.resume = args.resume

    # Print final config
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    train(cfg)
