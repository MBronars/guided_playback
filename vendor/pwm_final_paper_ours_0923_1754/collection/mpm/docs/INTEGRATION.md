# MPM production source on `robert/clean`

## Ownership and scope

`robert/clean` is the maintained MPM source. The collection research worktree is
retained for historical evidence and unfinished folding experiments; it is not
an import dependency of the integrated MPM package or launcher. This is a
selective source integration, not a merge of the older branch's model tree.

The integration includes fresh-shape preparation/auditing, split-aware banks,
cycle sampling, collapse/contraction checks, material-aware wall selection,
four-world simulation, export/consumer contracts, campaign qualification,
bounded orchestration, shared access and cross-user sequence reservations.
No model, training configuration, normalization file, SIM1 implementation,
dataset payload or existing GPU job is changed.

## Maintained layout

```text
collection/mpm/                  # asset preparation through reviewed export
collection/mpm/launch/mpm_collect.py            # sole supported operator front end
collection/mpm/launch/mpm_collect.json          # pinned deployment and shared registry paths
collection/mpm/launch/mpm_training_campaign.sbatch
collection/mpm/launch/mpm_training_campaign.py  # training-only allocation controller
collection/mpm/launch/mpm_{training,ada,four,campaign,scale,handoff}.py
                                # preserved runtime/support closure
collection/mpm/launch/mpm_publish_shared.py     # no-clobber shared asset publication
collection/mpm/launch/mpm_shared_runtime.py     # committed-source deployment publication
collection/mpm/tests/test_mpm*.py    # physical/data/orchestration regressions
collection/mpm/tests/test_mpm_collect.py        # operator and reservation regressions
collection/mpm/tests/test_mpm_publication.py    # source provenance and publication regressions
```

The existing supported deployment is still SSD `tools/mpm-v1`; its snapshots
are immutable runtime artifacts, not another editable source tree. Git does not
contain the shared 10K bank, Python environment, simulation raws or GIFs.

Only **one MPM sbatch** is retained in the main checkout. Old serial campaigns,
Ada pilots, full-scope qualification jobs and benchmark shells were not imported.
The former untracked `mpm_collect_shared.sbatch` was renamed to the controller's
canonical `mpm_training_campaign.sbatch`; its contents are unchanged.

The support modules still contain some historical functions. They are preserved
to isolate this integration from numerical/orchestration refactoring, not offered
as alternative public workflows. Some legacy `freeze`/`submit` helpers reference
archived shells intentionally absent here. Use `mpm_collect.py`; do not directly
invoke those old command paths. Further extraction of reusable helpers is a
separate tested refactor, not a claim that all historical code has been removed.

## Verified source provenance

[SOURCE_PARITY.json](SOURCE_PARITY.json) records **36 byte-identical
collection/common and MPM package files** against the deployed template:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/tools/mpm-v1/template-v1
BUNDLE.json: 3acfb65a94a3b82ecad0e2acf2e6096896f9921bb8ce066515c00da538d4f0ea
```

The research worktree base was `6e3b6e3`; its newer uncommitted MPM files were
compared against the deployed snapshot, rather than assumed to belong to that
commit. The numerical package is imported exactly. The runtime drivers are
preserved except that `mpm_ada.freeze` no longer requires/copies an obsolete Ada
pilot sbatch. The production wrapper selects the explicit shared interpreter.
No solver, timestep, constitutive law, quality threshold or output selection
rule was relaxed during integration.

The parity manifest is an integration checkpoint. An intentional later MPM
source change needs a reviewed new reference and corresponding tests; do not
silently rewrite qualification evidence to make a changed simulator pass.

## Operators: unchanged workflow

```bash
python3 collection/mpm/launch/mpm_collect.py assets
python3 collection/mpm/launch/mpm_collect.py preflight
python3 collection/mpm/launch/mpm_collect.py plan --name alice-night-one
# The following command submits a real four-PRO, 48-hour job:
python3 collection/mpm/launch/mpm_collect.py launch --name alice-night-one
```

Both the checkout and the shared launcher use the same shared asset paths and
sequence registry. Each run writes into its own directory under
`staging/mpm-operator-campaigns-v1/`; it does not append into somebody else's
shards. See the [runbook](RUNBOOK.md) and
[shared assets/access](ASSETS.md).

The current pinned `mpm-v1` release remains deployed. Source integration does
not hot-update it, restart a job, or reserve new cycles. Each newly allocated
GPU must still pass startup qualification. Fast 5–7-object evaluation and active
dragging remain outside the approved production scope.

## Maintainers: publish from committed source

Future versions must be staged from a committed checkout using
`mpm_shared_runtime.py stage --project ...`. The publisher checks Git status
and each committed blob, copies the canonical MPM source into the new baseline,
and writes `REPOSITORY.json` with the commit and per-file hashes. Uncommitted,
untracked or racing source changes are rejected. Frozen consumer `data/` and
`stats/` compatibility inputs are preserved from the selected template; they
never overwrite the current training checkout.

Example for a **new, nonexistent** deployment root (not an operator command):

```bash
MPM_SHARED=/data/group_data/katefgroup-ssd/world_model/particle_world_model
python3 collection/mpm/launch/mpm_shared_runtime.py stage \
  --project /data/user_data/robertre/particle_world_model \
  --root "$MPM_SHARED/tools/mpm-v2" \
  --template "$MPM_SHARED/tools/mpm-v1/template-v1" \
  --evidence "$MPM_SHARED/tools/mpm-v1/qualification-reference" \
  --assets "$MPM_SHARED/assets/mpm/fresh-10k-v1" \
  --python "$MPM_SHARED/environments/mpm-genesis-v1/bin/python"
python3 collection/mpm/launch/mpm_shared_runtime.py finish \
  --project /data/user_data/robertre/particle_world_model \
  --root "$MPM_SHARED/tools/mpm-v2" \
  --environment-root "$MPM_SHARED/environments/mpm-genesis-v1" \
  --group kfragki2_group
```

These commands publish files, not GPU jobs. `finish` freezes/verifies the new
controller and sets lab-group modes. Preserve the common registry and campaign
roots across versions. Changing the checkout's pinned template path/SHA or
instructing operators to adopt a new release is a separate explicit decision.
Do not overwrite `mpm-v1`, modify active bundles or replace qualification receipts.

## Verification

- MPM-specific suite: **134 tests pass**, with the explicit historical reference
  root available (including CPU serial/grouped checks and both loader profiles).
- Full existing collection suite: **305 tests discovered; 303 pass, 2 skipped**.
- Operator/publication suite: **18 tests pass**.
- All 36 collection package files match the deployed source hashes.
- Shell syntax and whitespace checks pass.
- Post-commit smoke test: staged and froze a fresh training bundle from commit
  `1fd24056d638cea6b042f20ed78413fbd8daf901`; full controller verification passes
  against the shared banks and preserved qualification references. All **44**
  maintained source/wrapper files match that commit's recorded hashes. The
  temporary test bundle SHA-256 is
  `38a4e198e338d764167bacc5f91870ff82db45ba58e6b8ff360d0b854a1b8ffc`.
  This was a CPU preparation/verification test, not a GPU job or a new shared
  release. The existing shared deployment and production registry were untouched.

Reproduction commands (hide GPUs for these tests):

```bash
MPM_PYTHON=/data/group_data/katefgroup-ssd/world_model/particle_world_model/environments/mpm-genesis-v1/bin/python
CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  PWM_MPM_REFERENCE_ROOT=/data/user_data/robertre/particle_world_model/reports/mpm_correctness_20260907 \
  "$MPM_PYTHON" -m unittest discover -s collection/mpm/tests -q
python3 collection/mpm/tests/test_mpm_collect.py
python3 collection/mpm/tests/test_mpm_publication.py
```

The historical-reference path above is for Robert's full regression fixtures;
normal shared operators need only the minimal shared runtime/reference closure.
Dependency-specific skips and existing unclosed-resource warnings are not new GPU
qualification. This integration does not claim a fresh GPU throughput benchmark.

## Domain-local layout follow-up

MPM-specific launch/config files, tests (including wall and journal fixtures), and
these docs now live under `collection/mpm/`. Root `launch/` contains only a pointer;
the previous MPM files are moved, not duplicated or kept as compatibility shims.
The former core filenames and all 36 numerical/shared package bytes are unchanged.

The publisher maps `collection/mpm/launch/` into the same flat frozen driver layout,
and selects only root-level collection/MPM modules for `source/`. Domain tests,
docs and operator/publishing utilities do not enter the frozen numerical closure.
Editable-checkout `runtime.sources()` conservatively also hashes domain-local
Python support/tests; frozen production bundles contain only the explicit runtime
closure. Existing prepared plans remain bound to their original source snapshots.

The default pytest configuration includes domain-local tests. Shared collection
tests remain in `tests/collection/`; run those separately with unittest discovery
or use the repository's default pytest discovery to include both locations.
Existing SSD operator paths, asset paths, registries and published bundles are
unchanged. Historical verification counts above describe the integration before
relocation, not a new GPU qualification.

After relocation, combined MPM and remaining collection tests pass with **323
passed and 2 skipped**. Default repository pytest discovery also succeeds in the
model environment; missing optional Warp/trimesh tests report skips rather than
import errors. Collection-environment tests still execute those checks. Pytest's
unrelated auto-loaded rerun plugin requires a socket, so sandboxed verification
used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (no assertions or quality gates disabled).
Relative documentation links resolve, the relocated operator's `assets`/`plan`
commands pass without reserving cycles, and the 36-file parity check still passes.

Post-commit relocation smoke test also passes: a new temporary bundle was staged
from `7ab2ce05a1e42404975f592139f5bccfc2c17434`, frozen and fully verified against
the shared banks and retained qualification inputs. All 44 recorded runtime
source/wrapper hashes match the commit. The frozen package reports exactly 36
source files, with no domain tests/docs/launch directories copied under `source/`.
Bundle SHA-256:
`2357a1468952664afe478b1602dc54d1f6d78621975380f90f9864e03db8ebab`.
The closure probe explicitly checked the imported frozen `ROOT`, avoiding the
editable checkout taking precedence in a `python -c` process. No GPU job, new
shared deployment, or production sequence reservation was created by this test.

## Configurable collection release (2026-09-13)

The maintained operator now accepts one validated `collection` object; see
[CONFIG.md](CONFIG.md). Implementation commit `3382760` adds frozen settings,
weighted count/action sampling, exact attempt budgets and CPU evaluation
previews. Follow-up `88639e1` ensures custom material and stiffness offsets span
all nine pairings. The original full default specification stream is unchanged.

This is an intentional extension, not a claim that every collection file is
still byte-identical to the original integration. Of the 36 shared/core source
files, only `collection/mpm/collect.py` changes: an optional validated fresh-scene
specification factory. `SOURCE_PARITY.json` records that delta. The solver,
guarded fast backend, adaptive clock, quality filters, and export contract have
not changed. The runtime closure now contains 45 maintained files: the same
36 shared/core modules and nine frozen launch dependencies, including settings.

The new immutable shared release is:

```
/data/group_data/katefgroup-ssd/world_model/particle_world_model/tools/mpm-v2
```

Its template `BUNDLE.json` SHA-256 is
`cb64bfdb2dd0db3df499f5ffda7c466ed4bb7e1202d80d79d7d08eaf2122c323`.
`template-v1/REPOSITORY.json` binds the maintained runtime to commit
`88639e162df730b844c621a0dbabdabf71b0b51a`.
`READY.json` records successful shared permission, frozen source, asset/reference
and common-registry lock checks. The previous `mpm-v1` deployment is unchanged.
Both deployments use the same shared assets, registry and campaign roots;
publication does not create another sequence namespace or submit collection.

Verification:

- Combined domain/shared collection suite: **333 passed, 2 skipped**.
- Local committed-source staging, freeze and verification pass, including a
  nine-attempt / four-cycle / 24-hour custom training configuration.
- Real CPU preparation passes for custom 1/3-object training, 2/4-object held-out
  validation and 5/7-object held-out test previews. Assertions check actual bank
  UID membership, object counts, actions, unique sequence keys, matched controls,
  and correct frozen-package import root. Local receipt:
  `/tmp/mpm-config-check-25m1fp8t/deployment/CONFIG_CHECK.json` (temporary artifact).
- The final shared release independently passes configured freeze/verify and
  nine-state preparation, with all 45 maintained runtime hashes matching the
  canonical checkout. Receipt: `/tmp/mpm-config-release-jb4eaqmv/CHECK.json`.
- The public shared operator's held-out validation `preview` succeeds using an
  external JSON config. Main/shared config template, asset and coordination
  paths agree; production registry still proposes cycle 84000 without reserving
  it. Preview: `/tmp/mpm-public-preview-dfazmlfg/preview`.
- Slurm accepts the configured four-PRO, 24-hour command with `sbatch --test-only`
  (exit 0). This is scheduler syntax/access validation, not a real job, GPU
  allocation or runtime correctness test. Receipt:
  `/tmp/mpm-config-release-jb4eaqmv/SCHEDULER_CHECK.json`.
- Production evaluation remains blocked. Preview receipts explicitly deny
  collection approval; no new GPU correctness or throughput claim is made.

Only 1–4-object training configurations can be submitted by this launcher.
The normal startup GPU qualification and automated collection reviews still
run on each new allocation. No existing collection job was changed or launched
as part of publishing this configuration interface.

## Qualification-seeded cache release (`mpm-v3`)

This supersedes the operator default above; v1/v2 and every existing campaign
bundle remain unchanged. The shared entry point is now:

```
/data/group_data/katefgroup-ssd/world_model/particle_world_model/tools/mpm-v3/mpm_collect.py
```

The v3 template `BUNDLE.json` SHA-256 is
`d92509b0fe87e1a7fa6fc65d40513d22a3c3bf5b8f3b1e9db9023b41940f2d0b`.
Its `REPOSITORY.json` binds maintained runtime source to commit `6d2c695`.
There are still 45 maintained files: 36 shared/core modules and nine launch
dependencies. This release changes three launch drivers (`mpm_four.py`,
`mpm_ada.py`, `mpm_training_campaign.py`), not the solver, grouped kernels,
physical constants, consumer, dataset clocks or collection protocol.

Each passing qualification lane now records hashes of the compiled artifacts
that actually passed. The controller binds that manifest into its approval and
passes it to every worker on the same GPU. Workers verify/copy the artifacts
into a private node-local cache and verify them again after simulation. This
removes repeated cold compilation; it does not bypass qualification. The
asynchronous writer and diagnostic benchmark helpers are not frozen production
dependencies and asynchronous writing is not enabled.

See [PERFORMANCE.md](PERFORMANCE.md) for full measurements and preserved failed
exploratory tests. Production-sized worker timings improved by 24–26%; replay
using the exact compiled reference passed all 59 simulated coverage cases on
each of two GPUs. The final seed-argument worker CLI additionally passed a
four-case/two-control smoke test on both GPUs with zero compilation. These are
bounded tests, not a new four-GPU production approval or a sustained end-to-end
collection-rate measurement. Every new allocation still requires startup and
first-wave approval under the unchanged tolerances.

Publication finished with successful shared source/asset/reference/permission
and registry preflight. It neither reserves sequences nor submits a job. Main
and shared launchers keep the same assets, registry and campaign roots.

## v4: two/four-GPU controller and qualified background writing

The next immutable release adds `mpm_async_io.py` to the frozen launch closure
(46 maintained files). Numerical simulation and consumer source remain unchanged.
Both candidate qualification and production use the same private journal-binding
adapter: four writer threads, 64 MiB owned-copy limit, 32 pending writes, ordered
per-case writes and mandatory drain before journal verification/completion.
Serial qualification references retain synchronous writing. Errors still fail
the run; neither qualification nor first-wave review is waived.

The controller supports exactly two or four PRO GPUs. All allocated lanes must
pass startup and first-wave checks; the config controls the Slurm GPU request,
cycle enumeration and consumer count. Defaults remain four GPUs. The intended
two-GPU continuation in allocation 10432785 uses a new common-registry seed
reservation and output directory, preserving the original interrupted chunks.
Publication and actual runtime receipts, not this design note, establish launch
status. See PERFORMANCE.md and the allocation handoff for the measured evidence.
