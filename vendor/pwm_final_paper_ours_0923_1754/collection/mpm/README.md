# MPM collection

All maintained MPM-specific collection code, launch configuration, tests and
documentation live here on `robert/clean`. Shared helpers remain one directory up.

For training on collected data, start with [Dataset format and training handoff](docs/DATASET_FORMAT.md):
current dataset roots, exact TAR schema/units, loader smoke test, root assembly,
training configuration, and passive-versus-dragging differences.

For the frozen 100-episode-per-category test suite, see
[Evaluation collection readiness](docs/EVALUATION_COLLECTION.md). Evaluation
uses `evaluation_suite.py`, with its own validation gate and generated Slurm
scripts; it is separate from the training-only production launch approval.

```text
collection/mpm/
  *.py             # asset preparation, simulation, filtering and export
  launch/          # public operator, one sbatch, runtime and publishing helpers
  drag/            # separate active-command qualification and bounded collection
  tests/           # MPM contracts, CPU physics, operator and publication tests
  docs/
    RUNBOOK.md     # collection protocol, launch, status and recovery
    ASSETS.md      # shared train/validation/test assets and lab access
    DRAG_DATASET_FORMAT.md # dragging fields, paths, actions and training setup
    INTEGRATION.md # provenance, tests and committed-source publication
    SOURCE_PARITY.json
```

From the repository root:

```bash
python3 collection/mpm/launch/mpm_collect.py assets
python3 collection/mpm/launch/mpm_collect.py preflight
python3 collection/mpm/launch/mpm_collect.py plan --name alice-night-one
# Submits four PRO6000 GPUs for 48 hours:
python3 collection/mpm/launch/mpm_collect.py launch --name alice-night-one
```

`launch/mpm_collect.py` is the supported passive-training production operator. It handles
disjoint sequence reservations and per-run directories on the shared SSD. The
other launcher modules are frozen runtime/support dependencies, not alternative
passive-training workflows. Do not submit the training sbatch directly or use historical helper
`freeze`/`submit` paths documented as unsupported in the [integration](docs/INTEGRATION.md).

The settings-enabled, qualified-cache shared entry point is SSD `tools/mpm-v4/mpm_collect.py`.
The previous `tools/mpm-v1`, `tools/mpm-v2`, and `tools/mpm-v3` deployments remain unchanged.
Assets, environments and generated datasets are external;
this folder is not a standalone copy of those assets or the shared helpers.

The passive operator's scope is training, **1–4 objects**; startup GPU
qualification remains mandatory. Fast 5–7-object evaluation is not approved.
Dragging has a separate bounded [gentle-v4 release](docs/DRAG_GENTLE_V4_RELEASE.md)
and [recovery/restart](docs/DRAG_RECOVERY_20260915.md); do not use the passive
launcher for it. Historical faster-command proposals are not its current defaults.

For training on dragging, start with **[dataset format and training guide](docs/DRAG_DATASET_FORMAT.md)**:
verified roots, TAR fields, commanded actions versus prediction targets,
hold/release gates, loader smoke check, Transformer/UNet settings and the remaining
validation/test split requirements. That bounded dragging launcher remains
train-only; evaluation uses the separate frozen worker and qualification above.

See the [runbook](docs/RUNBOOK.md), [shared assets](docs/ASSETS.md), and
[source integration/publication](docs/INTEGRATION.md).
For object counts, weighted actions, budgets, runtime and evaluation previews,
see [collection settings](docs/CONFIG.md).

Measured bottlenecks, cache A/B results and pending CPU/RAM optimizations are
tracked in [performance work](docs/PERFORMANCE.md).

Tests, from the repository root in the collection environment:

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q collection/mpm/tests
```

Set `PWM_MPM_REFERENCE_ROOT` to the explicit historical fixture root to include
reference-dependent tests; otherwise those tests report skips. The repository's
default pytest discovery also includes this directory.
