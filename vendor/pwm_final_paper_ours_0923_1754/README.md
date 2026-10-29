# Particle World Model

One unified pose-native world model with two backbone choices:
TrajectoryTransformer and hierarchical UNet. Train all four domains—passive
MPM, pickup, cloth, pushing—or select any subset using the same trainer.

Start with the [supported scope](docs/SUPPORTED.md), the
[five-part walkthrough](docs/CODE_WALKTHROUGH.md), and the
[normalization contract](stats/README.md). For the completed historical overfit
checks, read the [reproduction guide](docs/OVERFIT_REPRO.md). The dated reports of
the 2026-09 cleanup (overfit retrospective, round reports, parity audit) left the
tree on 2026-09-19; git history keeps them, last present at `79a3c84`.

**Training and evaluating on the pwm-data-v1 release.** Start with [TRAIN.md](TRAIN.md)
and [EVAL.md](EVAL.md): the exact protocol of every dataset, with amounts, paths, metrics
and what is saved. Eight datasets, one script per training job in
[launch/](launch/README.md) (`bash launch/submit_<job>.sh --print`), and one command that
scores a checkpoint under the frozen evaluation protocol, whole-clip rollouts included:
`PYTHONPATH=. python scripts/unified/eval_protocol.py --checkpoint <run>/checkpoints/final.pt`.
The datasets are described in [docs/DATASETS.md](docs/DATASETS.md) and the reasoning behind
the protocol in [docs/EVAL_PROTOCOLS.md](docs/EVAL_PROTOCOLS.md).

For the new cloth collection, see the [SIM1 folding dataset format and training
handoff](docs/SIM1_FOLDING_DATASET.md), including retained split indices, NPZ
fields, units/clocks, and the remaining adapter compatibility checks.
The completed collection's [test GIF gallery](docs/SIM1_FOLDING_GALLERY.md)
shows six garment/action examples rendered from retained data.

Same-weight forwards/samplers, single updates, deterministic multi-step loops,
the compiled UNet and two-rank DDP agree bitwise with the pre-cleanup code; retraining
differs only by GPU non-determinism (the parity audit of 2026-09-09, `docs/cleanup/PARITY_AUDIT_2026-09-09.md` in git history at `79a3c84`).
Transformer MPM and global-cloth drift remain open scientific issues. Do not interpret a completed
cleanup as successful memorization or generalization on every domain.

## Training

Use the tested CUDA-matched PyTorch/PyTorch3D environment ([exact record and
rebuild steps](docs/ENVIRONMENT.md)), then:

```bash
python -m pip install --no-deps -e .
PYTHONPATH=. python -m pytest tests -q -m 'not integration'
python train.py --config configs/transformer.yaml
python train.py --config configs/unet.yaml 'data.domains=[mpm]'
```

These training commands launch research runs, not smoke tests. Set a distinct
`run_name`, output directory and compute budget first.
External-corpus contract checks are separate: run `pytest tests -m integration`
only where the documented research datasets are available.
Use `WANDB_MODE=disabled` for local diagnostics. Shared defaults live in
`utils/config.py`; the fully resolved config is saved with each run.

The opt-in [adaptive UNet allocation](docs/UNET_ADAPTIVE_ALLOCATION.md) recipe,
`configs/unet_adaptive.yaml`, caps coarse particle tokens by object and scene.
Its prediction-quality benefit has not been established.

Normalization is explicit: `normalization.profile=xy_p99_per_domain` (20 scales)
or `xy_p99_global` (4 scales). Both are frozen in one versioned JSON artifact.
Neither is declared the scientific winner. Checkpoints embed the resolved
numeric snapshot; selecting a specialist never refits its scales.

Overfit plots compare loss heights only within matching coordinates. Use
`scripts/unified/plot_overfit.py --physical_only` to compare physical rollout
errors across normalization profiles or prediction parameterizations.

## Source layout

- `configs/`: two primary backbone recipes and an experimental adaptive UNet recipe.
- `models/`: Transformer, hierarchical UNet, shared geometry/attention components.
- `data/`: unified adapters, batch contracts, exclusions and raw calibration.
- `diffusion/`: flow-matching algebra and per-frame schedules.
- `train.py`, `inference.py`: one training loop and shared sampling.
- `scripts/compute_stats.py`: deterministic collect/fit/audit pipeline.
- `scripts/unified/`: named evaluations, corpus preparation/audits and overfit plots.
- `scripts/eval_object_rollout.py`: passive MPM population/protocol selection.
- `stats/normalization.json`: both normalization profiles and provenance.

## Reproducibility and limits

Standalone joint-MPM, pushing-pose, pickup-pose and historical guidance workflows
are recoverable at `c5fac334540abec89c62824a528e60985e75a7c7`, a durable ancestor
of the cleanup branch. `robert/pre-unified-only-2026-09-06` is a local convenience
branch pointing to that commit.
The earlier broad experiment tree remains at `robert/pre-cleanup-2026-09-05`.
No original datasets, checkpoints or experiment outputs are deleted.

Active MPM drag is a deferred capability, not supported by removing its
historical code. Guidance on the unified model is Layer 0 of
[docs/guidance/DESIGN.md](docs/guidance/DESIGN.md) (`guidance/`), not the
historical specialist guidance workflows. Numerical parity and memorization diagnostics
are engineering checks, not held-out benchmark results. Report checkpoint,
normalization, exact population/exclusions and rollout protocol with every score.

Generated receipts and diagnostics belong under ignored `validation/`.
Existing `runs/` and `reports/` contain research artifacts, not active model
implementations; do not delete or relocate them without preserving their
provenance and updating consumers. Linked verification GIFs/checkpoints and
prepared input symlinks are local artifacts, not supplied by a fresh checkout.
Small CPU fixed-input tests cover optimizer/EMA/RNG continuation, not exact
full-size stochastic GPU training replay. Streaming loader position and worker
state are not saved, so exact subsequent corpus replay is not claimed.
