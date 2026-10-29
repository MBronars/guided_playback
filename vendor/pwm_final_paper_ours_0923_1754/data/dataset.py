"""
Trajectory datasets: corpus adapters returning TrajectoryBatch.

All preprocessing happens here - training loop just iterates.
"""

import glob
import inspect
import os
import re
from typing import Optional, Iterator

import numpy as np
import torch
import webdataset as wds

from data.batch import TrajectoryBatch
# Re-export for historical imports and pickled TrajectoryBatch fixtures.
from data.normalization import NormalizationStats  # noqa: F401


def _with_source_rows(samples):
    """Tag original pre-batch indices before exclusions slice any .pyd fields."""
    for sample in samples:
        positions = sample["particle_pos.pyd"]
        rows = int(positions.shape[0]) if positions.ndim == 4 else 1
        yield dict(sample, **{"calibration_rows.pyd": np.arange(rows, dtype=np.int64)})


def stratified_shard_groups(shards, roots):
    """Shards nested by dataset root, then by directory below that root.

    Groups keep the listed order, both of the shards inside a directory and of
    the groups themselves (first appearance), so the nesting is deterministic."""
    ordered_roots = sorted((str(root).rstrip("/") for root in roots), key=len, reverse=True)
    nested: dict = {}
    for shard in shards:
        root = next((r for r in ordered_roots if shard.startswith(r + "/")), os.path.dirname(shard))
        nested.setdefault(root, {}).setdefault(os.path.dirname(shard), []).append(shard)
    return [list(directories.values()) for directories in nested.values()]


def _round_robin(iterables):
    """One item from each iterator in turn, dropping iterators as they end."""
    live = [iter(iterable) for iterable in iterables]
    while live:
        for iterator in list(live):
            try:
                yield next(iterator)
            except StopIteration:
                live.remove(iterator)


class _RoundRobinSamples(torch.utils.data.IterableDataset):
    """Undecoded records alternating across roots, then across directories within a root."""

    def __init__(self, groups, split_stages, handler):
        self.groups, self.split_stages, self.handler = groups, list(split_stages), handler

    def _directory(self, shards):
        return wds.DataPipeline(wds.SimpleShardList(shards), *self.split_stages,
                                wds.tarfile_to_samples(handler=self.handler))

    def __iter__(self):
        return _round_robin(_round_robin(self._directory(d) for d in root) for root in self.groups)


def _drain_exact(buf: list, bs: int):
    """Yield EXACT bs-sample batches from a list of TrajectoryBatch items (mutating
    buf in place; the boundary item is split and its remainder pushed back). Items
    may be pre-batched (B > 1) — batch sizes are counted in SAMPLES, so the knob is
    independent of how the shards are packaged."""
    total = sum(b.positions.shape[0] for b in buf)
    while total >= bs:
        take, got = [], 0
        while got < bs:
            item = buf.pop(0)
            ib = item.positions.shape[0]
            need = bs - got
            if ib <= need:
                take.append(item)
                got += ib
            else:
                take.append(BaseGraspDataset._split_batch(item, 0, need))
                buf.insert(0, BaseGraspDataset._split_batch(item, need, ib))
                got = bs
        yield take[0] if len(take) == 1 else TrajectoryBatch.collate(take)
        total = sum(b.positions.shape[0] for b in buf)


class SkipSample(ValueError):
    """An explicit adapter exclusion, not a malformed-batch/programming error."""


class BaseGraspDataset(torch.utils.data.IterableDataset):
    """Base class with shared WebDataset pipeline logic."""

    def __init__(
        self,
        wds_location: str,
        stats_location: str,
        context_len: int = 2,
        future_len: int = -1,
        training: bool = True,
        shuffle: bool = True,
        num_collate: int = 1,
        prediction_target: str = "delta",
        window_movement_min_mm: float = 0.0,
        window_movement_power: float = 0.0,
        normalization=None,
        raw: bool = False,
    ):
        """
        Args:
            wds_location: Path to WebDataset shards (directory with .tar files)
            stats_location: Optional legacy native-MPM .npz path; unified uses normalization
            context_len: Number of context frames
            future_len: Number of future frames to predict (-1 = all)
            training: Whether in training mode
            shuffle: Whether to shuffle shards (set False for deterministic iteration)
            num_collate: Number of samples to collate into one batch (default 1 = no collation)
            prediction_target: "delta" (physical displacements between adjacent frames)
        """
        self.wds_location = wds_location
        self.context_len = context_len
        self.future_len = future_len
        self.training = training
        self.shuffle = shuffle
        self.num_collate = num_collate
        self.prediction_target = prediction_target
        self.raw = bool(raw)
        self.normalization = None
        # MOVEMENT-AWARE window sampling (cfg.train.window_movement_*): the
        # random sub-window draw accepts only windows whose cloud-mean
        # per-frame displacement clears window_movement_min_mm (mm/frame; a
        # clip with NO such window contributes its max-movement window — every
        # clip always trains), and draws among the eligible windows with
        # probability ∝ movement^window_movement_power (0 = uniform among
        # eligible = the pure gate). Closed-form, no rejection loop.
        self.window_movement_min_mm = float(window_movement_min_mm or 0.0)
        self.window_movement_power = float(window_movement_power or 0.0)
        
        # Load normalization stats (delta, position, offset if available)
        if raw:
            self.delta_stats = NormalizationStats(-torch.ones(3), torch.ones(3))
            self.position_stats = None
        elif normalization is not None:
            from data.normalization import resolve_normalization, domain_normalizers
            self.normalization = resolve_normalization(normalization)
            self.delta_stats, self.position_stats, _ = domain_normalizers(self.normalization, "mpm")
        else:
            self.delta_stats, self.position_stats, _ = NormalizationStats.load_all(stats_location)
        print(f"Loaded delta stats: {self.delta_stats}")
        if self.position_stats is not None:
            print(f"Loaded position stats: {self.position_stats}")
        if prediction_target != "delta":
            raise ValueError("Canonical MPM readers support prediction_target=delta only")

        # Find shards
        if not wds_location.endswith('/'):
            wds_location += '/'
        # dict.fromkeys de-duplicates: for a LEAF shard dir both globs return the same
        # files ('**' matches zero directories), which otherwise lists every shard twice.
        self.shards = sorted(dict.fromkeys(
            glob.glob(f"{wds_location}*.tar") +
            glob.glob(f"{wds_location}**/*.tar", recursive=True)
        ))
        if not self.shards:
            raise ValueError(f"No .tar files found in {wds_location} (checked subdirectories too)")
        print(f"Found {len(self.shards)} shards in {wds_location}")

    def _create_pipeline(self) -> wds.DataPipeline:
        """Create WebDataset pipeline.

        no_shard_split (attribute, set by create_dataloader): skip the
        split_by_node/split_by_worker stages so a SINGLE process sees EVERY
        shard. Required for rank-0-only eval loaders under DDP (pinned viz,
        sample_every val draws) — with splitting, rank 0 silently sees only
        its 1/world_size share of the shards.

        Evaluation reads (not training, not shuffled) alternate records across
        shard groups (stratified_shard_groups: each dataset root, then each
        directory below it), so every prefix of the stream covers every pickup
        task root and every MPM object-count directory. In sorted order each
        group was one contiguous block: a 40-record prefix of pickup validation
        was a single place shard, of MPM validation only 1- and 2-object
        scenes. Membership and each group's own order are unchanged. Setting
        the stratify_shards attribute to False restores sorted order, e.g. to
        reproduce a receipt made before 2026-09-14.
        """
        handler = (wds.reraise_exception if getattr(self, "strict_read", False)
                   else wds.warn_and_continue)
        split = [] if getattr(self, 'no_shard_split', False) else [wds.split_by_node, wds.split_by_worker]
        stratify = getattr(self, "stratify_shards", None)
        if self.shuffle:
            pipeline = [wds.ResampledShards(self.shards), *split, wds.tarfile_to_samples(handler=handler)]
        elif (not self.training) if stratify is None else stratify:
            roots = getattr(self, "roots", None) or [self.wds_location]
            pipeline = [_RoundRobinSamples(stratified_shard_groups(self.shards, roots), split, handler)]
        else:
            pipeline = [wds.SimpleShardList(self.shards), *split, wds.tarfile_to_samples(handler=handler)]
        pipeline.append(wds.decode("torch"))
        if getattr(self, "record_source_rows", False):
            pipeline.append(_with_source_rows)
        return wds.DataPipeline(*pipeline)

    def _process_sample(self, sample: dict) -> TrajectoryBatch:
        """Process a raw WebDataset sample into TrajectoryBatch. Override in subclass."""
        raise NotImplementedError

    @staticmethod
    def _split_batch(b: TrajectoryBatch, start: int, end: int) -> TrajectoryBatch:
        """Slice every batch tensor; share the contract's metadata-preserving path."""
        return b._map_tensors(lambda value: value[start:end])

    def __iter__(self) -> Iterator[TrajectoryBatch]:
        """Iterate over dataset, yielding TrajectoryBatch objects.

        Batching semantics (per shape bucket, see below):
          - self.batch_size > 0 (set by create_dataloader from data.batch_size):
            accumulate until the bucket holds >= batch_size SAMPLES (counting
            each item's actual batch dim, so pre-batched formats count their
            inner samples), then collate+yield. This is the direct "how many
            samples per step" knob, decoupled from how the data is packaged.
          - else num_collate > 1 (legacy): collate every num_collate ITEMS,
            i.e. the effective sample count depends on the format's packaging.

        Buffering is PER SHAPE BUCKET (particle count N): with mixed
        object-count shards (1obj..4obj have N = 1024*O, no padding), consecutive
        samples can differ in N and must never be collated together. Each yielded
        batch is therefore homogeneous in N / object count (the uniform-layout
        invariant the batched model paths rely on); buckets fill at rates
        proportional to the data mix.
        """
        pipeline = self._create_pipeline()
        buffers: dict = {}
        bs = int(getattr(self, 'batch_size', 0) or 0)

        for sample in pipeline:
            try:
                batch = self._process_sample(sample)

                # Skip corrupted samples with NaN/Inf in positions
                if torch.isnan(batch.positions).any() or torch.isinf(batch.positions).any():
                    print(f"[DataLoader] Skipping sample with NaN/Inf positions (shape={list(batch.positions.shape)})")
                    continue

                if bs <= 0 and self.num_collate <= 1:
                    # An uncollated record keeps its shard, so evaluators can record
                    # which shards the clips they scored came from.
                    batch.source_shard = str(sample.get("__url__", ""))
                    yield batch
                else:
                    key = batch.positions.shape[2]          # N (particle count)
                    buf = buffers.setdefault(key, [])
                    buf.append(batch)
                    if bs > 0:
                        yield from _drain_exact(buf, bs)    # exact-size batches, remainder carried
                        # bucket_max_items (docs/CODE_WALKTHROUGH.md): with
                        # many distinct N (cloth: 280 values) partial buckets
                        # would park hundreds of windows per worker. Past the
                        # cap, flush the OLDEST partial bucket, padded to bs by
                        # repeating its own samples so batch shapes stay static
                        # (rare-N episodes still train; nothing is dropped).
                        _cap = int(getattr(self, 'bucket_max_items', 0) or 0)
                        if _cap > 0 and sum(len(b) for b in buffers.values()) > _cap:
                            _k = next(k for k, b in buffers.items() if b)
                            _old = TrajectoryBatch.collate(buffers.pop(_k))
                            # Repeat SAMPLES, not pre-batched records (a record's
                            # B can shrink after episode-level exclusions).
                            _reps = [self._split_batch(_old, i % _old.batch_size,
                                                       i % _old.batch_size + 1) for i in range(bs)]
                            yield TrajectoryBatch.collate(_reps)
                    elif len(buf) >= self.num_collate:
                        yield TrajectoryBatch.collate(buf)
                        buffers[key] = []
            except SkipSample as e:
                print(f"[DataLoader] Skipping {sample.get('__url__', '<unknown shard>')}:"
                      f"{sample.get('__key__', '<unknown sample>')}: {e}")
            except Exception as e:
                raise RuntimeError(
                    f"Error processing {sample.get('__url__', '<unknown shard>')}:"
                    f"{sample.get('__key__', '<unknown sample>')}: {e}") from e

        # Yield remaining samples in the buckets
        for buf in buffers.values():
            if buf:
                yield buf[0] if len(buf) == 1 else TrajectoryBatch.collate(buf)


    


class MPMMultiObjectDataset(BaseGraspDataset):
    """
    WebDataset loader for multi-object MPM simulation data.

    Data format (pre-batched in tar files):
        particle_pos.pyd: [B, T, N, 3] - particle positions over time
        valid_particles.pyd: [B, T, N] - validity mask
        material_properties.pyd: [B, N, 4] - (E, nu, rho, mat_type) per particle
        object_ids.pyd: [B, N] - object membership per particle

    Output TrajectoryBatch:
        positions: [B, T, N, 3] - batched trajectories
        actions: [B, F, N, 3] - zeros (no external actions)
        action_mask: [B, N] - all False (no actuated particles)
        object_ids: [B, N] - per-particle object assignment
    """

    def _validate_command_input(self,sample):
        from collection.mpm.contracts import require_passive_consumer_input
        require_passive_consumer_input(sample)

    def _process_sample(self, sample: dict) -> TrajectoryBatch:
        """Process a pre-batched sample into TrajectoryBatch.

        If self.future_len > 0 and the stored clip is longer than
        context_len + future_len, sample a random sub-window per call.
        This gives the model more variety in (context, future) splits than
        the fixed pre-chunking allows (avoids the AR-boundary distribution
        gap diagnosed in the mpm_training_audit).
        """
        self._validate_command_input(sample)
        def to_tensor(x):
            return torch.from_numpy(x) if isinstance(x, np.ndarray) else x

        positions = to_tensor(sample["particle_pos.pyd"]).float()  # [B, T, N, 3]
        valid_mask = to_tensor(sample["valid_particles.pyd"]).bool()  # [B, T, N]
        material_props = to_tensor(sample["material_properties.pyd"]).float()  # [B, N, 4]
        object_ids = to_tensor(sample["object_ids.pyd"]).long()  # [B, N]

        B, T, N, _ = positions.shape
        self._window_starts = [0] * B

        # Random sub-window slicing (only meaningful if the stored clip is
        # longer than the requested window).
        if self.future_len > 0:
            window = self.context_len + self.future_len
            if T > window:
                thr = getattr(self, 'window_movement_min_mm', 0.0) / 1000.0  # m/frame
                pw = float(getattr(self, 'window_movement_power', 0.0) or 0.0)
                if getattr(self, 'first_window', False):
                    start = 0  # viz loaders: first window = the drop + impact
                    positions = positions[:, start : start + window]
                    valid_mask = valid_mask[:, start : start + window]
                elif thr > 0 or pw > 0:
                    # MOVEMENT-GATED start, PER TRAJECTORY in the pre-batched
                    # slab (the uniform path shares one start across the slab).
                    # Uniform draw among starts whose window cloud-mean
                    # per-frame displacement >= thr == redraw-until-accept
                    # without the loop; a trajectory with no eligible window
                    # falls back to its max-movement window.
                    B_s = positions.shape[0]
                    d_ = (positions[:, 1:] - positions[:, :-1]).norm(dim=-1)   # [B,T-1,N]
                    vm_ = (valid_mask[:, 1:] & valid_mask[:, :-1]).to(d_.dtype)
                    m_ = (d_ * vm_).sum(-1) / vm_.sum(-1).clamp(min=1.0)        # [B,T-1]
                    cs = torch.cat([torch.zeros(B_s, 1), m_.cumsum(1)], dim=1)  # [B,T]
                    n_s = T - window + 1
                    Wm = (cs[:, window - 1 : window - 1 + n_s] - cs[:, :n_s]) \
                        / (window - 1)                                          # [B,n_s]
                    pos_out, vm_out = [], []
                    self._window_starts = []
                    for b_ in range(B_s):
                        elig = (Wm[b_] >= thr).nonzero(as_tuple=True)[0]
                        if len(elig) > 0:
                            if pw > 0:
                                w_ = Wm[b_][elig].clamp(min=1e-12) ** pw
                                s_ = int(elig[int(torch.multinomial(w_ / w_.sum(), 1).item())])
                            else:
                                s_ = int(elig[int(torch.randint(0, len(elig), (1,)).item())])
                        else:
                            s_ = int(Wm[b_].argmax().item())
                        pos_out.append(positions[b_, s_ : s_ + window])
                        vm_out.append(valid_mask[b_, s_ : s_ + window])
                        self._window_starts.append(s_)
                    positions = torch.stack(pos_out)
                    valid_mask = torch.stack(vm_out)
                else:
                    start = int(torch.randint(0, T - window + 1, (1,)).item())
                    positions = positions[:, start : start + window]
                    valid_mask = valid_mask[:, start : start + window]
                if not (thr > 0 or pw > 0) or getattr(self, 'first_window', False):
                    self._window_starts = [start] * B
                T = window

        F = T - self.context_len

        # No actions for passive simulation
        actions = torch.zeros(B, F, N, 3, dtype=positions.dtype)

        # No actuated particles
        action_mask = torch.zeros(B, N, dtype=torch.bool)

        # Valid lengths from mask (use first frame)
        valid_lengths = valid_mask[:, 0, :].sum(dim=1)  # [B]

        # Material properties: [B, N, 4] -> E, nu, rho, mat_type
        E = material_props[:, :, 0]
        nu = material_props[:, :, 1]
        rho = material_props[:, :, 2]
        mat_type = material_props[:, :, 3].long()

        return TrajectoryBatch(
            positions=positions,
            actions=actions,
            action_mask=action_mask,
            object_ids=object_ids,
            valid_lengths=valid_lengths.long(),
            E=E,
            nu=nu,
            rho=rho,
            mat_type=mat_type,
            context_len=self.context_len,
            delta_stats=self.delta_stats,
            position_stats=self.position_stats,
            prediction_target=self.prediction_target,
        )

    def __len__(self) -> int:
        """Approximate dataset size."""
        return len(self.shards) * 8  # ~8 batches per shard


# Dataset registry
DATASET_REGISTRY = {
    "mpm_multi": MPMMultiObjectDataset,
}


def validate_validation_config(cfg):
    """Preflight held-out selection before Accelerator, logging or corpus I/O.

    Returns the separate validation root, or None for an adapter-owned split.
    This checks configuration, not episode identity: symlink checks and corpus
    provenance audits remain necessary after constructing the datasets.
    """
    if cfg.data.format == "unified":
        from data.unified_dataset import selected_domains
        selected_domains(cfg.data.get("domains"), cfg.data.get("domain_weights"))
        if cfg.data.get("eval_domains") is not None:
            selected_domains(cfg.data.eval_domains)
    if int(cfg.train.get("overfit_batches", -1)) > 0:
        return None  # explicit memorization mode; trainer reuses cached batches
    data = cfg.data
    val_root = data.get("val_wds_location")
    if val_root is not None and os.path.realpath(val_root) == os.path.realpath(data.wds_location):
        raise ValueError("data.val_wds_location resolves to training; a val set must be held out")
    fmt = data.format
    accepted = set() if fmt == "unified" else set(inspect.signature(DATASET_REGISTRY[fmt].__init__).parameters)
    requested = {"val_files": bool(data.get("val_files")),
                 "num_val_trajs": int(data.get("num_val_trajs", 0)) > 0}
    for key, enabled in requested.items():
        if enabled and key not in accepted:
            raise ValueError(f"data.{key} is set but format '{fmt}' does not accept it; "
                             "use data.val_wds_location for a held-out shard directory")
    if not (val_root or fmt == "unified" or any(requested.values())):
        if not data.get("allow_val_on_train", False):
            raise ValueError("No held-out validation set configured: set data.val_wds_location "
                             "to a verified held-out shard directory, or use val_files / "
                             "num_val_trajs on formats that support them. For deliberate "
                             "memorization only, data.allow_val_on_train=true is available.")
        print("[VAL] WARNING: data.allow_val_on_train=true — validation uses TRAINING data")
    return val_root


def create_dataloader(
    wds_location: str,
    stats_location=None,
    num_workers: int = 4,
    format: Optional[str] = None,
    num_collate: int = 1,
    batch_size: int = 0,
    no_shard_split: bool = False,
    first_window: bool = False,
    prediction_target: str = "delta",
    **dataset_kwargs,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for a supported corpus (format must be explicit).

    Args:
        wds_location: Path to WebDataset shards
        stats_location: Path to normalization stats
        num_workers: Number of data loading workers
        format: unified, or mpm_multi for native read-only evaluation populations
        num_collate: LEGACY: number of yielded ITEMS to collate into one batch —
                     the effective sample count depends on the format's packaging
                     (num_collate=4 with B=4 pre-batched data yields B=16 per step)
        batch_size: Direct SAMPLES-per-batch knob (per process). When > 0 it takes
                    precedence over num_collate: the dataset accumulates until the
                    (shape-bucketed) buffer holds >= batch_size samples, counting
                    each item's actual batch dim — independent of packaging.
        prediction_target: "delta" or "position" — what the diffusion model predicts
        **dataset_kwargs: Additional args for the dataset class

    Returns:
        DataLoader yielding TrajectoryBatch objects
    """
    # Pop filtering options before passing to dataset constructor
    max_objects = dataset_kwargs.pop('max_objects', -1)

    # Unified mixed-corpus loader (docs/unified_dataset_spec.md). Not a registry
    # class: roots/stats/windows are per-domain and live in data/unified_dataset.py.
    # training=True -> infinite weighted domain mix; training=False -> strict
    # round-robin over the four finite val splits (num_batches=4k => k per domain).
    # Lazy import: unified_dataset imports from this module.
    if format == "unified":
        from data.unified_dataset import (UnifiedTrainStream, UnifiedValRoundRobin,
                                          selected_domains, _validate_window,
                                          resolve_data_roots)
        from data.normalization import resolve_normalization
        if stats_location is not None:
            raise ValueError("Unified loaders no longer accept stats_location; use normalization= "
                             "with a profile, artifact selector, or resolved snapshot")
        if prediction_target != "delta":
            raise ValueError("Unified datasets support prediction_target='delta' only")
        if wds_location != "unified://v1":
            raise ValueError("Unified corpus roots are registered under unified://v1")
        if max_objects > 0 or num_collate != 1:
            raise ValueError("Unified loaders require max_objects=-1 and num_collate=1; use batch_size")
        if dataset_kwargs.get("allow_missing_blocklists", False):
            raise ValueError("Blocklists are required; allow_missing_blocklists is no longer supported")
        accepted = {"training", "shuffle", "context_len", "future_len", "emit_sidecar", "articulated",
                    "bucket_max_items", "domain_weights", "seed", "domains", "eval_domains",
                    "normalization", "roots", "window_motion", "batch_across_episodes"}
        inert_defaults = {"num_val_trajs": 0, "eager_load": False, "val_files": None,
                          "temporal_stride": 1, "window_movement_min_mm": 0.0,
                          "window_movement_power": 0.0, "allow_missing_blocklists": False}
        unsupported = [name for name, value in dataset_kwargs.items()
                       if name not in accepted and
                       (name not in inert_defaults or value != inert_defaults[name])]
        if unsupported:
            raise ValueError(f"Unsupported unified dataset options: {sorted(unsupported)}")
        _sidecar = bool(dataset_kwargs.get("emit_sidecar", False))
        _articulated = bool(dataset_kwargs.get("articulated", False))
        domains, weights = selected_domains(dataset_kwargs.get("domains"), dataset_kwargs.get("domain_weights"))
        eval_domains = dataset_kwargs.get("eval_domains")
        eval_domains = selected_domains(eval_domains)[0] if eval_domains is not None else domains
        # Checkpoints own the fully resolved normalization snapshot, independent
        # of cwd or external stats paths.
        _norm = resolve_normalization(dataset_kwargs.get("normalization"))
        # future_len <= 0 keeps the per-domain DOMAIN_FUTURE_LEN; a positive int
        # makes the window uniform across domains. context_len threads through.
        _fl = dataset_kwargs.get("future_len", -1)
        _validate_window(dataset_kwargs.get("context_len", 2), _fl)
        _fl = int(_fl) if _fl is not None and int(_fl) > 0 else None
        _cl = int(dataset_kwargs.get("context_len", 2))
        _bmi = int(dataset_kwargs.get("bucket_max_items", 0) or 0)
        # Explicit dataset releases (data.roots): validated here, before any shard
        # is opened; domains not named keep the registered unified://v1 roots.
        _roots = resolve_data_roots(dataset_kwargs.get("roots"))
        if dataset_kwargs.get("training", True):
            return UnifiedTrainStream(
                domains=domains, weights=weights,
                batch_size=batch_size or 4, num_workers=num_workers,
                seed=int(dataset_kwargs.get("seed", 0)), emit_sidecar=_sidecar,
                future_len=_fl, context_len=_cl, bucket_max_items=_bmi,
                normalization=_norm, first_window=first_window, no_shard_split=no_shard_split,
                shuffle=bool(dataset_kwargs.get("shuffle", True)), roots=_roots,
                articulated=_articulated, window_motion=dataset_kwargs.get("window_motion"),
                batch_across_episodes=dataset_kwargs.get("batch_across_episodes"))
        return UnifiedValRoundRobin(batch_size=batch_size or 4, num_workers=0,
                                    emit_sidecar=_sidecar,
                                    future_len=_fl, context_len=_cl,
                                    bucket_max_items=_bmi, normalization=_norm,
                                    first_window=first_window, domains=eval_domains, roots=_roots,
                                    articulated=_articulated)

    if format not in DATASET_REGISTRY:
        raise ValueError(f"Unknown format '{format}'. Available: {list(DATASET_REGISTRY.keys())}")

    dataset_cls = DATASET_REGISTRY[format]
    # Drop kwargs the specific dataset class doesn't accept (e.g. sim1_cloth's
    # num_val_trajs) so callers can pass a uniform set of args across formats.
    accepted = set(inspect.signature(dataset_cls.__init__).parameters)
    dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if k in accepted}
    dataset = dataset_cls(
        wds_location=wds_location,
        stats_location=stats_location,
        num_collate=num_collate,
        prediction_target=prediction_target,
        **dataset_kwargs,
    )
    # Direct samples-per-batch knob: set as an attribute (not a ctor arg) so every
    # dataset class gets it without signature churn; both __iter__ implementations
    # read it via getattr and it takes precedence over num_collate.
    if no_shard_split:
        # Rank-0-only eval loaders (pinned viz, sample_every val draws) must see
        # EVERY shard, not this rank's 1/world_size share.
        dataset.no_shard_split = True
    if first_window:
        # Visualization loaders: slice the FIRST context+future window of each
        # episode (the drop + impact) instead of a random sub-window, so gifs
        # show maximum motion. Read by _process_sample via getattr.
        dataset.first_window = True
    if batch_size and batch_size > 0:
        dataset.batch_size = int(batch_size)
    elif num_collate > 1:
        print(f"[DataLoader] DEPRECATED: num_collate={num_collate} counts yielded ITEMS, "
              f"so the effective batch depends on the shard packaging. Set data.batch_size "
              f"(exact SAMPLES per step) instead; num_collate will be removed once the "
              f"remaining configs are migrated.")

    # Filter shards by object count (for datasets with {n}obj/ subdirectories)
    if max_objects > 0:
        filtered = [
            s for s in dataset.shards
            if (m := re.search(r'/(\d+)obj/', s)) is None or int(m.group(1)) <= max_objects
        ]
        print(f"Filtered shards by max_objects={max_objects}: {len(dataset.shards)} → {len(filtered)}")
        dataset.shards = filtered

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
    )
