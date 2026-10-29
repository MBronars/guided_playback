# Dataset collection

Run collection from the repository root. `data/` consumes datasets;
`collection/` prepares, simulates, validates and exports them. Model training
does not import a simulator. Integration does not approve a dataset for training.

## Domain layout convention

Keep domain-specific collection code, launch/config files, tests, and operational
documentation under `collection/<domain>/`. Use `launch/`, `tests/`, and `docs/`
subdirectories plus a domain-root `README.md`; keep core simulator modules at the
domain root. No duplicate domain launchers/configs belong in the repository's
top-level `launch/`, `tests/`, or `docs/` directories. Cross-domain integration
tests, project-wide documentation, and training-side data adapters remain shared.

Only genuinely reusable helpers belong directly in `collection/`. Dataset assets,
simulator environments, generated outputs and frozen deployed bundles stay on
external storage. Freeze only the explicit runtime closure, never recursively
include a domain's tests/docs or development launch utilities.

[MPM](mpm/README.md) follows this convention now. Apply it to subsequent domain
integrations; this change does not move or redesign SIM1 or other collectors.
Default pytest discovery includes both `tests/` and `collection/`.

## Supported development interfaces

| Domain | Entry point | Purpose |
| --- | --- | --- |
| MPM production | `python3 collection/mpm/launch/mpm_collect.py` | shared assets, preflight, unique sequences, four-PRO training launch/status |
| MPM | `python -m collection.mpm` | prepare, inspect, bounded collect, export |
| MPM | `python -m collection.mpm.review` | integrity, interactions, loader checks, actual-state GIFs |
| Folding | `python -m collection.sim1.replay` | bounded joint-command simulation, optional slower final return |
| Folding | `python -m collection.sim1.replay_review` | independently check completed raw replay |
| Folding | `python -m collection.sim1.actual_export` | actual cloth/gripper states at 30 Hz |
| Folding | `python -m collection.sim1.replay_visualize` | fixed-camera, actual-state GIFs |
| Folding, experimental | `python -m collection.sim1.fold_plan` | geometry-targeted IK command plans |

SIM1 also retains `qualify`, `compare`, and `review` for matched fixed-home
diagnostics. These are not alternative production folding collectors. Geometry,
input, rest-state, placement, recording and export helpers are package-local.
The external simulator and its assets are not vendored.

## Current capability boundary

MPM production uses the explicit `fresh-shapes-v1` recipe, frozen 10K training
and held-out banks, bounded cycle namespaces, four-world simulation, saved-frame
journals, contraction/collapse checks, and certified sand-only wall-contact
selection. Only passive 1–4-object training is approved by the production launcher;
5–7-object fast evaluation and active dragging are not approved. Legacy recipe
versions remain for reproducibility, not as alternative production defaults.
See the [operator runbook](mpm/docs/RUNBOOK.md) and
[MPM source integration](mpm/docs/INTEGRATION.md).

Folding replay, source-rest handling, dynamic jaw-contact filtering, actual-state
export, integrity review and visualization are integrated. The final-return
retimer is opt-in. It changes only the last fully open return, not earlier
repositioning. No teleport, cloth attachment or hidden state reset is introduced.

The new fold planner is **experimental**, not three qualified action policies.
Its nine-case preview produced two completed inward folds and seven IK-planning
rejections before simulation. Those results do not establish physical failure
for the rejected paths, nor a successful 10-garment x 3-action collection.
A quiet initial cloth state is diagnostic rather than a prerequisite to replay.
Numerical completion does not prove a successful grasp or fold.

Model guidance is separate from these collection interfaces. Integrating this
package neither removes nor changes the clean branch's guidance implementation.

## Environments and input paths

Keep simulator environments separate from model training:

- MPM: verified Warp 1.10.1 environment; optional `collection-mpm` dependencies.
- SIM1: pinned external SIM1/Newton checkout and matching simulator environment.
  Pass `--sim1-source` and `--assets-root` explicitly. Do not install an arbitrary
  package named Newton or replace the pinned simulator while replaying evidence.
- Open3D geometry regressions: separate SIM1-environment process, not after
  initializing Genesis Warp in the same process.

MPM uses `PWM_MPM_REFERENCE_ROOT` for frozen control/reference data, with no
machine-local default. An optional `--bank` requires the matching `--pool`.
Folding requires explicit source-rest input, home/command NPZs and external assets.
These are data/dependency inputs, not Python imports from the old collection worktree.
The planner also requires an explicit targeting reference and measured pad audit.

Review resolves this checkout's model adapter by default. Set `PWM_MODEL_ROOT`
only when deliberately using another model checkout. Importing conflicting
already-loaded model packages fails closed.

## Execution and evidence

GPU execution requires the currently allocated UUID and explicit budgets.
The low-level serial MPM interface requires an idle GPU; the production controller
owns its Slurm allocation and explicitly manages its four-world workers.
SIM1 `replay --shared-gpu` opts into observed free-memory
and own-memory checks; these are not allocator reservations. The lease coordinates
cooperating same-user processes on one node, not Slurm or other users' jobs.

Use new, immutable run directories. Preserve source hashes, parameters, failed
attempts, exclusion decisions and explicit infrastructure retries. Do not resume
old prepared jobs with newly edited code or rewrite their source hashes. Journals
and replay chunks preserve committed prefixes, not mid-episode physical resume.
Freeze the source used by a running job before continuing development.

Use SSD `staging/` for unapproved attempts, `evidence/` for retained audits and
`releases/` only after explicit approval. Raw arrays, checkpoints, GIFs, snapshots
and one-off historical launchers do not belong in tracked source. There is no
automatic training-promotion command.

The earlier clean-branch integration notes, the original migration handoff and the source extraction map
(`docs/collection/`) were historical provenance and left the tree on 2026-09-19; they are in git history, last present at `79a3c84`.
