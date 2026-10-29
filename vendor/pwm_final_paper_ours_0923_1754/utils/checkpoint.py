"""Atomic checkpoint writes and rank-local training runtime state.

Runtime state covers RNGs, mixed-precision scaler and caller-provided loop state.
It does NOT capture WebDataset/worker iterator positions; exact corpus replay
requires a separately resumable input stream or pinned batches.
"""
from pathlib import Path
import os
import copy
import random
import tempfile
import warnings
from numbers import Integral
from zipfile import is_zipfile

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf


def read_checkpoint(path, *, weights_only=True):
    """Load on CPU, mapping modern tensor storage lazily instead of copying Adam
    state just to run inference. Legacy serialization remains readable. PyTorch's
    default private mapping does not write through to the original checkpoint.
    Pass weights_only=False only for trusted legacy training metadata.
    """
    # Historical engineering provenance stored torch.__version__ as this known
    # str subclass. Keep restricted loading; never retry with weights_only=False.
    from torch.torch_version import TorchVersion
    already_allowed = torch.serialization.get_safe_globals()
    allowed = [TorchVersion] if weights_only and TorchVersion not in already_allowed else []
    # Do not remove a caller's pre-existing allowance when this context exits.
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=weights_only, mmap=is_zipfile(path))


def checkpoint_config(state):
    """Resolve checkpoint-owned normalization without applying fresh run defaults.

    Only the two frozen xy-p99 profiles can migrate from a historical URI. Older
    specialist/baseline normalizers remain at the pre-unified-only revision.
    A modern snapshot is self-contained and must agree with its config copy.
    """
    from data.normalization import resolve_normalization

    if not isinstance(state.get("cfg"), dict):
        raise ValueError("Checkpoint has no resolved configuration")
    cfg = OmegaConf.create(state["cfg"])
    if cfg.get("data", {}).get("format") != "unified":
        raise ValueError("Only unified checkpoints are supported; historical specialists are "
                         "reproducible at robert/pre-unified-only-2026-09-06")
    configured = cfg.get("normalization")
    embedded = state.get("normalization")
    if configured is not None:
        configured = resolve_normalization(OmegaConf.to_container(configured, resolve=True))
    if embedded is not None:
        embedded = resolve_normalization(embedded)
        if configured is not None and configured["sha256"] != embedded["sha256"]:
            raise ValueError("Checkpoint normalization disagrees with its saved configuration")
    if configured is None and embedded is None:
        legacy = cfg.data.get("stats_location")
        if legacy not in ("unified://xy_p99", "xy_p99", "unified://xy_p99_global", "xy_p99_global"):
            raise ValueError("Historical normalization is not an active xy-p99 profile; use "
                             "robert/pre-unified-only-2026-09-06 to reproduce this checkpoint")
        configured = resolve_normalization(legacy)
    cfg.normalization = embedded or configured
    return cfg


def validate_resume_normalization(state, normalization):
    """Reject changed coordinates before loading any learned or optimizer state."""
    from data.normalization import resolve_normalization

    saved = checkpoint_config(state).normalization
    requested = resolve_normalization(normalization)
    if saved.sha256 != requested["sha256"]:
        raise ValueError("Resume normalization differs from the checkpoint; choose its exact "
                         "profile/snapshot, or start a separately named experiment")


def _model_contract(cfg):
    """Every model field, weightless ones included, with historical aliases canonicalized."""
    from dataclasses import asdict
    from models.factory import parse_model_config
    architecture, model = parse_model_config(cfg.model, checkpoint=True)
    return OmegaConf.to_container(OmegaConf.create({"architecture": architecture, **asdict(model)}), resolve=True)


def _data_roots(data):
    """The canonical data.roots selection: release paths, exclusion files and task per domain."""
    from data.blocklists import manifest_path
    from data.unified_dataset import resolve_data_roots
    spec = data.get("roots")
    roots = resolve_data_roots(OmegaConf.to_container(spec, resolve=True) if OmegaConf.is_config(spec) else spec)
    for entry in roots.values():
        entry["exclusions"] = {split: [str(manifest_path(p).resolve()) for p in files]
                               for split, files in entry["exclusions"].items()}
    return roots


def corpus_identity(cfg):
    """sha256 of exclusions and frozen rl_push manifests, {split: {domain: {path: sha256}}}.

    Equal settings and paths can still exclude different episodes after a list is edited in
    place. train.py records this once at startup, when its datasets load the same files."""
    from data.blocklists import manifest_path
    from data.unified_dataset import RL_PUSH_ROOT, exclusion_files, selected_domains
    from utils.reproducibility import file_sha256
    data = cfg.data
    domains = selected_domains(data.get("domains"), data.get("domain_weights"))[0]
    eval_domains = (selected_domains(data.eval_domains)[0] if data.get("eval_domains") is not None
                    else domains)
    roots = _data_roots(data)
    identity = {split: {domain: {str(manifest_path(p).resolve()): file_sha256(manifest_path(p))
                             for p in exclusion_files(domain, split, roots.get(domain))}
                    for domain in selected}
            for split, selected in (("train", domains), ("val", eval_domains))}
    verified = {}
    for split, selected in (("train", domains), ("val", eval_domains)):
        from data.unified_dataset import INDEXED_DOMAINS
        from data.indexed_episodes import load_index
        for domain in set(selected) & set(INDEXED_DOMAINS):
            for root in roots[domain][split]:
                path = str(Path(root).resolve() / "INDEX.json")
                # A release's test folder doubles as the validation root (validate_index_pair).
                accepted = "train" if split == "train" else ("val", "test")
                identity[split][domain][path] = load_index(root, domain, accepted)["index_sha256"]
        if "rl_push" not in selected:
            continue
        spec = roots.get("rl_push", {"train": [RL_PUSH_ROOT], "val": []})
        for root in spec["train"] if split == "train" else (spec["val"] or spec["train"]):
            path = Path(root).resolve() / "RELEASE.json"
            if path.exists():
                if str(path) not in verified:
                    from utils.rl_push_release import verify_rl_push_release
                    verified[str(path)] = verify_rl_push_release(root)
                identity[split]["rl_push"][str(path)] = verified[str(path)]
    return identity


# Settings that change how a step is COMPUTED or FED, never what it trains on: activation
# checkpointing (recompute vs store) and loader workers. A resume may change them freely.
_RESOURCE_MODEL_FIELDS = ("unet_ckpt_from_level", "unet_ckpt_ingraph", "unet_ckpt_seams")


def _continuation_contract(cfg, world_size=1):
    """Scientific settings, excluding output names and logging/stop controls.

    Canonicalize historical aliases/defaults, not the learned weights. Changing
    a head's parameterization is not resume even when every tensor shape matches.
    """
    from data.unified_dataset import selected_domains
    from utils.trajectory_objective import trajectory_loss_options
    from utils.config import DEFAULTS

    data = cfg.data
    domains, weights = selected_domains(data.get("domains"), data.get("domain_weights"))
    total = sum(weights.values())
    fm_defaults = {**DEFAULTS["flow_matching"], "time_sampler": "uniform",
                   "beta_alpha": 1.5, "beta_beta": 1.0, "beta_t_min": 0.001}
    train_defaults = {**DEFAULTS["train"], "adam_betas": [0.9, 0.999], "adam_eps": 1e-8,
                      "ema_warmup": False, "rot_aug": False}
    # grad_accum and data.batch_size are NOT pinned separately: only their product with the GPU count,
    # the windows per optimizer step, is (effective_batch below). A run may resume on different
    # hardware with a different split -- e.g. 16 x 1 on two PRO 6000 == 4 x 4 on two L40S.
    train_names = ("lr", "weight_decay", "adam_betas", "adam_eps",
                   "mixed_precision", "grad_clip", "ema_decay", "ema_warmup", "rot_aug",
                   "loss_spike_threshold", "grad_norm_spike_threshold", "spike_grace_steps",
                   "overfit_batches", "max_batch_size")
    selected_data = {name: data.get(name, DEFAULTS["data"].get(name)) for name in
                     ("format", "wds_location", "context_len", "future_len",
                      "emit_sidecar", "bucket_max_items")}
    effective_batch = (int(data.get("batch_size", DEFAULTS["data"]["batch_size"]))
                       * int(cfg.train.get("grad_accum", train_defaults["grad_accum"])) * int(world_size))
    model_contract = {k: v for k, v in _model_contract(cfg).items() if k not in _RESOURCE_MODEL_FIELDS}
    selected_data.update(domains=list(domains), domain_probabilities={d: weights[d] / total for d in domains},
                         seed=data.get("seed", 0), first_window=data.get("first_window", False),
                         eval_domains=data.get("eval_domains"), roots=_data_roots(data))
    value = dict(model=model_contract, data=selected_data, effective_batch=effective_batch,
                 prediction_target=cfg.get("prediction_target", "delta"),
                 denoise_method=cfg.get("denoise_method"), seed=cfg.get("seed", 42),
                 flow_matching={k: cfg.flow_matching.get(k, v) for k, v in fm_defaults.items()
                                if k != "inference_steps"},
                 objective=trajectory_loss_options(cfg),
                 train={k: cfg.train.get(k, train_defaults[k]) for k in train_names})
    if value["train"]["adam_betas"] is None:
        value["train"]["adam_betas"] = [0.9, 0.999]
    return OmegaConf.to_container(OmegaConf.create(value), resolve=True)


def validate_resume_configuration(state, cfg, *, corpus=None, world_size=None):
    """Resume continues the same run: settings, dataset releases and exclusion contents.
    `corpus` is corpus_identity(cfg) when the caller has already computed it.

    The batch is pinned as EFFECTIVE batch -- data.batch_size x train.grad_accum x GPUs, the windows
    per optimizer step -- so a run can resume with a different per-GPU batch, accumulation or
    activation checkpointing. The saved GPU count is the checkpoint's runtime.world_size; the current
    one is `world_size`, else WORLD_SIZE (set by accelerate launch), else 1."""
    validate_resume_normalization(state, OmegaConf.to_container(cfg.normalization, resolve=True))
    saved_ws = int((state.get("runtime") or {}).get("world_size", 1))
    req_ws = int(world_size if world_size is not None else os.environ.get("WORLD_SIZE", 1))
    saved = _continuation_contract(checkpoint_config(state), saved_ws)
    requested = _continuation_contract(cfg, req_ws)
    different = [name for name in saved if saved[name] != requested.get(name)]
    if different:
        detail = ""
        if "effective_batch" in different:
            scfg = checkpoint_config(state)
            detail = (f" Effective batch (batch_size x grad_accum x GPUs): saved "
                      f"{scfg.data.get('batch_size')} x {scfg.train.get('grad_accum', 1)} x {saved_ws} = "
                      f"{saved['effective_batch']}, requested {cfg.data.get('batch_size')} x "
                      f"{cfg.train.get('grad_accum', 1)} x {req_ws} = {requested['effective_batch']}.")
        raise ValueError(f"Resume configuration changes the training contract: {different}.{detail} "
                         "Use the saved settings; changed experiments need separate initialization.")
    if "corpus" not in state:
        warnings.warn("Checkpoint predates exclusion-file hashes; resuming without verifying that "
                      "its exclusion files are unchanged", stacklevel=2)
        return
    current = corpus_identity(cfg) if corpus is None else corpus
    changed = []
    legacy_releases = set()
    for split, domains in current.items():
        for domain, files in domains.items():
            before = state["corpus"].get(split, {}).get(domain, {})
            # Old checkpoints pin exclusions only. Verify today's release fully, warn about
            # the historical gap, and bind its digest in subsequent checkpoints.
            legacy = {p for p in files if domain == "rl_push" and Path(p).name == "RELEASE.json" and p not in before}
            legacy_releases.update(legacy)
            changed += [f"{split}/{domain} {path}" for path in sorted(set(before) | set(files))
                        if path not in legacy and before.get(path) != files.get(path)]
    if legacy_releases:
        warnings.warn("Checkpoint predates frozen-release hashes; current rl_push files were verified, "
                      "but historical release identity cannot be established: " + str(sorted(legacy_releases)), stacklevel=2)
    if changed:
        raise ValueError(f"Resume corpus files differ from the checkpoint's: {changed}. "
                         "Restore them, or start a new experiment with train.init_from.")


def validate_initialization(state, cfg):
    """A weight-only start (train.init_from) must describe the same network in the same coordinates.

    Optimizer, schedule, objective and data belong to the new experiment. Every model
    field, including weightless ones such as the temporal attention direction, and the
    normalization snapshot must match, or the weights would be read under another meaning.
    """
    validate_resume_normalization(state, OmegaConf.to_container(cfg.normalization, resolve=True))
    saved, requested = _model_contract(checkpoint_config(state)), _model_contract(cfg)
    different = sorted(name for name in set(saved) | set(requested) if saved.get(name) != requested.get(name))
    if different:
        raise ValueError(f"train.init_from describes a different model: {different}")


def _validate_optimizer_state(optimizer, state):
    """Catch malformed Adam moments before loading weights or optimizer state."""
    groups = state.get("param_groups", [])
    if len(groups) != len(optimizer.param_groups):
        raise RuntimeError("Optimizer parameter group count differs from checkpoint")
    for saved, current in zip(groups, optimizer.param_groups):
        if len(saved["params"]) != len(current["params"]):
            raise RuntimeError("Optimizer parameter group size differs from checkpoint")
        for index, parameter in zip(saved["params"], current["params"]):
            for name, value in state.get("state", {}).get(index, {}).items():
                if name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"):
                    if not torch.is_tensor(value) or value.shape != parameter.shape:
                        raise RuntimeError(f"Optimizer {name} shape differs from parameter {index}")
                elif name == "step" and torch.is_tensor(value) and value.numel() != 1:
                    raise RuntimeError("Optimizer step must be scalar")


def atomic_torch_save(state, path):
    """Replace only after a complete, fsynced write in the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def training_checkpoint(model, ema, optimizer, scheduler, step, cfg, *, next_step=None, runtime=None, **extra):
    """Legacy inference fields plus an unambiguous continuation boundary."""
    if not bool(cfg.get("log", {}).get("atomic_save", True)):
        raise ValueError("log.atomic_save=false is no longer supported; training checkpoints are always atomic")
    state = dict(checkpoint_version=2, step=int(step), next_step=int(step if next_step is None else next_step),
                 model=model.state_dict(), ema=ema.state_dict(), ema_meta=ema.metadata(),
                 optimizer=optimizer.state_dict(),
                 cfg=OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else copy.deepcopy(cfg))
    if cfg.get("data", {}).get("format") == "unified":
        from data.normalization import resolve_normalization
        spec = cfg.get("normalization")
        if OmegaConf.is_config(spec):
            spec = OmegaConf.to_container(spec, resolve=True)
        if spec is None:
            raise ValueError("New unified checkpoints require resolved normalization")
        state["normalization"] = resolve_normalization(spec)
        state["cfg"]["normalization"] = state["normalization"]
        state["checkpoint_version"] = 3
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if runtime is not None:
        state["runtime"] = runtime
    if state.keys() & extra.keys():
        raise ValueError("Extra checkpoint metadata cannot replace training state")
    return dict(state, **extra)


def load_training_state(state, model, optimizer=None, scheduler=None, ema=None, *, normalization=None):
    """Restore model/optimizer/scheduler/EMA; runtime restoration is deferred."""
    from utils.state_dict import load_model_state
    from utils.lr_schedule import validate_scheduler_state
    if normalization is not None:
        validate_resume_normalization(state, normalization)
    if scheduler is not None:
        validate_scheduler_state(scheduler, state["scheduler"])
    if optimizer is not None:
        _validate_optimizer_state(optimizer, state["optimizer"])
    if "ema" in state:
        from utils.ema import validate_ema_state
        validate_ema_state(model, state["ema"], metadata=state.get("ema_meta"))
    next_step = state.get("next_step", state.get("step"))
    if isinstance(next_step, bool) or not isinstance(next_step, Integral) or next_step < 0:
        raise ValueError("Checkpoint continuation step must be a nonnegative integer")
    load_model_state(model, state["model"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema is not None:
        ema.load_state_dict(state["ema"], metadata=state.get("ema_meta"))
    return int(next_step)


def capture_runtime(accelerator, **loop_state):
    """Call on EVERY rank at the same synchronized update boundary."""
    if not accelerator.sync_gradients:
        raise RuntimeError("Cannot checkpoint inside a gradient accumulation group")
    numpy_state = np.random.get_state()
    local = dict(
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=(torch.cuda.get_rng_state(accelerator.device) if accelerator.device.type == "cuda" else None),
        python=random.getstate(),
        # Primitive lists preserve compatibility with weights_only=True readers.
        numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        scaler=accelerator.scaler.state_dict() if accelerator.scaler is not None else None,
        loop=loop_state,
    )
    ranks = [local]
    if accelerator.num_processes > 1:
        if not dist.is_initialized() or dist.get_world_size() != accelerator.num_processes:
            raise RuntimeError("Checkpoint runtime requires the active distributed process group")
        ranks = [None] * accelerator.num_processes
        dist.all_gather_object(ranks, local)
    return dict(version=2, world_size=accelerator.num_processes,
                mixed_precision=accelerator.mixed_precision, ranks=ranks)


def restore_runtime(runtime, accelerator, *, legacy_precision=None):
    """Restore this rank AFTER construction/caching and immediately before training.

    A different world size or precision is not an exact continuation. Reject it
    here; weight-only initialization is a separate, explicit workflow.
    """
    if runtime.get("version") not in (1, 2) or runtime.get("world_size") != accelerator.num_processes:
        raise RuntimeError("Checkpoint runtime version/world size differs; use explicit weight-only initialization")
    precision = runtime.get("mixed_precision") if runtime["version"] == 2 else legacy_precision
    if precision is None or precision != accelerator.mixed_precision:
        raise RuntimeError("Checkpoint precision differs or is unknown; legacy runtime requires its saved config precision")
    if len(runtime["ranks"]) != accelerator.num_processes:
        raise RuntimeError("Checkpoint runtime is missing rank state")
    local = runtime["ranks"][accelerator.process_index]
    scaler = local["scaler"]
    if (scaler is None) != (accelerator.scaler is None):
        raise RuntimeError("Checkpoint precision/scaler differs from the current training run")
    if (local["torch_cuda"] is None) != (accelerator.device.type != "cuda"):
        raise RuntimeError("Checkpoint RNG device type differs from the current training run")
    if scaler is not None:
        accelerator.scaler.load_state_dict(scaler)
    torch.random.set_rng_state(local["torch_cpu"])
    if local["torch_cuda"] is not None:
        torch.cuda.set_rng_state(local["torch_cuda"], accelerator.device)
    random.setstate(local["python"])
    np.random.set_state(local["numpy"])
    return dict(local["loop"])
