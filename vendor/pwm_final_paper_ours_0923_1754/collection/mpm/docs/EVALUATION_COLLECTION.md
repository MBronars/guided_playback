# MPM evaluation collection: 100 episodes per category

## Morning recovery update (September 18, 11:05 EDT)

At 11:03 EDT there were 471 retained episodes: passive seen 100, passive unseen
12, passive count-seen 100, passive count-unseen 100, drag seen 81, drag unseen
78. The three complete categories had passed indexed/raw-consumer checks; both
drag categories were still progressing.

Passive-unseen candidate 12 had stopped because a GPU-conflict exception was
wrapped by the passive collector in `Simulation failed; preserve and diagnose`.
The first supervisor recognized direct resource errors but not this wrapper, so
it blocked this category instead of retrying it. This was a recovery-classifier
gap, not a physical-quality rejection.

`operator-supervision-v2` now also recognizes the wrapper **only when every
reason is a known resource error and the underlying failed `RESULT.json`
corroborates those exact reasons**. Mixed, numerical, missing-result, or unknown
errors still block. Seven regression cases were added. No frozen simulator,
test candidate, or filter was changed.

Only the two idle supervisors were replaced: `10484354.97` and `10457623.295`
now use v2 and prioritize passive unseen. Their budgets are 9.5 hours, with
10-hour Slurm step limits. Active dragging steps `10484354.86` and
`10457623.288` were left undisturbed. The old blocker is preserved as
`supervision-v1/RESOLVED-passive_unseen-v1.json`; its diagnosis and new operator
hash are bound in `RECOVERY-passive-unseen-v2.json`. The original failed raw
attempt is preserved, and the retry must match its initial-state hash.

## Current overnight recovery (September 18, 04:20 EDT)

The original one-shot lanes below are historical. Three stopped on GPU resource
guards when other workloads appeared; the unseen 5–7-object passive lane
completed all 100 episodes. A tested recovery supervisor now schedules the
unchanged frozen driver one candidate at a time:

| Supervisor slot | Allocation / step | Device | Initial role |
|---|---|---|---|
| `b-idle` | `10484354.86` | `ec4125d0-5595-2372-f0a9-18864219d774` | Resume passive; other pending categories afterward |
| `h-idle` | `10457623.288` | `2d4235b3-7d14-03c6-024c-5758f8cf3422` | Resume unseen dragging |
| `b-opportunistic` | `10484354.87` | `6140b8fb-5dc7-9f93-4b30-24fd978074bf` | Wait for foreign workload; prefer seen dragging |
| `h-opportunistic` | `10457623.289` | `5d583653-cd9c-ffe4-7a0f-800b9f757e51` | Wait for foreign workload; prefer seen 5–7-object passive |

Every slot can take other unfinished categories when its preferred category is
already owned or complete. Cross-node dispatch locks and the original category
locks prevent duplicate work. Each slot has a **16-hour worker budget**, a
17-hour step limit, and a 75-minute admission reserve. These controllers do not
renew allocations or promise completion within ten hours.

- Preserved GPU-conflict / idle-guard failures are retried with the exact same
  candidate and the frozen driver's initial-state hash check. Old attempts stay.
- An occupied GPU is polled every 30 seconds without starting more simulations.
  The exclusive GPU guard remains unchanged; shared-GPU execution is **not yet
  qualified**. No foreign process is killed or GPU reset performed.
- At most 12 attempts are allowed per candidate. Unknown failures, numerical
  issues, integrity mismatches, review errors, and exhausted candidate/retry
  budgets produce `supervision-v1/BLOCKED-<category>.json`. Other categories can
  continue. Such blocks need inspection; there is no claim of unattended fixes
  for arbitrary code defects.
- Completed categories are published, indexed against the unchanged training
  release, and checked against every retained raw episode under both statistics
  profiles and both rollout lengths. `supervision-v1/READY-<category>.json`
  binds the successful index and integration-check receipt. Check all six READY
  files before calling the dataset handoff finished, not merely episode counts.

The operator code is sealed separately at `operator-supervision-v1/SEAL.json`;
the simulation driver, numerical bundles, test plan, and physics filters were
not edited. Event records are under `supervision-v1/<slot>/`; current stdout and
stderr are `logs/supervise-<slot>-<job.step>.out` / `.err`. The detached tmux
sessions are `supervise-<slot>` on the existing `mpm-evaluation-20260918` server
on `babel-l9-20`. Keep both allocations available. These processes survive chat
closure, but not arbitrary allocation cancellation or host loss.

Verification: **526 MPM CPU tests passed** after adding the supervisor, including
23 new recovery/locking/GPU-admission tests. The deployment test caught the new
operator accidentally entering the training runtime's source closure; it is now
explicitly excluded, preserving that 46-file training closure. The real
`passive_seen` candidate 114 retry completed and retained with the same initial
state hash as its interrupted attempt. The 100-episode `passive_count_unseen`
publication has already passed the automatic index and raw-consumer handoff.

## Initial launch history

September 18, 2026 (EDT). **The frozen 600-episode suite launched at
02:23 EDT in `dev-blackwellb` (allocation `10484354`, node `babel-l9-16`).**
Passive step `10484354.51` runs the four passive categories on GPU
`ec4125d0-5595-2372-f0a9-18864219d774`; drag step `10484354.50` runs the two
drag categories on GPU `6140b8fb-5dc7-9f93-4b30-24fd978074bf`.
Both use the frozen driver, sources, candidates and filters below. Both passive
and drag qualification passed. The final
worker retained 24 validation episodes from 29 candidates, and all six indexed
categories passed raw-to-loader checks under both normalization profiles and
both rollout lengths. **503 MPM CPU tests pass.** Both production launch gates
were exercised successfully before starting test simulation. These are steps
inside the existing allocation, not two newly submitted batch jobs;
training/acceleration gates remain unchanged.

### Live launch and lifetime

At **02:54 EDT**, two more workers started on the previously idle GPUs in
`dev-blackwellh` (allocation `10457623`, node `babel-l9-20`):

| Step | Category | GPU UUID suffix |
|---|---|---|
| `10457623.263` | `drag_unseen` | `5d583653-cd9c-ffe4-7a0f-800b9f757e51` |
| `10457623.264` | `passive_count_unseen` | `2d4235b3-7d14-03c6-024c-5758f8cf3422` |

These have a 15.75-hour worker budget and 16-hour step limit. Their detached
tmux sessions are `h-drag-unseen` and `h-passive-count-unseen` on the same
controller server below. No existing worker was interrupted and no candidate
plan or physical filter changed. The old `dev-blackwellb` lanes still contain
these categories at the end of their original queues. A completed category is
verified without recollection; if an old lane reaches a category while its new
worker still owns it, the exclusive category lock stops the old lane with
`Category already has a worker`. That is a prevented duplicate dispatch, not
a failed simulated episode. Check that the designated new worker remains live
and the preceding category reached its quota before treating that exit as
expected. Never launch another worker or relax the lock to bypass it.

Additional logs in the same test-root `logs/` directory:

- `dev-blackwellh-drag-unseen-10457623.263.out` and `.err`
- `dev-blackwellh-passive-count-unseen-10457623.264.out` and `.err`

`LAUNCH-dev-blackwellh-20260918.json` records this additive dispatch. The added
allocation expires September 19 at 16:03 EDT. The original launch below and its
immutable receipt remain preserved.

The worker budget is 18.75 hours, with a 19-hour Slurm step limit and the frozen
shutdown reserve. Completion of all 600 episodes in that interval is **not
guaranteed**. Each lane advances through its categories sequentially; incomplete
quotas must resume from the same frozen plan. Do not launch the standalone
`collect-*.sbatch` scripts concurrently with these active lanes.

Controllers run in detached tmux sessions `passive` and `drag`, server
`mpm-evaluation-20260918`, on `babel-l9-20`. They survive closing this chat, but
must not be assumed to survive cancellation/replacement of their controller
host's allocation. The simulator steps require `dev-blackwellb` to remain
allocated (scheduled end September 18, 22:31 EDT).

Logs are in the test root's `logs/` directory:

- `dev-blackwellb-passive-10484354.51.out` and `.err`
- `dev-blackwellb-drag-10484354.50.out` and `.err`

The prelaunch `READINESS.json` intentionally remains an immutable historical
receipt with `production_submitted=false`; it is not a live status file.
Use the frozen driver's `status` command, Slurm step state, and per-candidate
`OUTCOME.json`/failure receipts for current progress. The second GPU's initial
utilization reading was anomalous (100% with no process or allocated memory);
its Warp memory probe passed and its production simulation advanced through
settling. No GPU reset or other workload cancellation was performed.

## Final qualification results

| Category | Candidates completed | Retained | Retained count coverage |
|---|---:|---:|---|
| passive_seen | 4 | 4 | 1, 2, 3, 4 |
| passive_unseen | 4 | 4 | 1, 2, 3, 4 |
| passive_count_seen | 5 | 4 | 5, 5, 6, 7 |
| passive_count_unseen | 4 | 4 | 5, 5, 6, 7 |
| drag_seen | 5 | 4 | 1, 2, 3, 4 |
| drag_unseen | 7 | 4 | 1, 2, 3, 4 |

Five candidates were rejected under unchanged filters: three drag settling
failures, one drag fragmentation failure, and one non-sand wall-contact failure.
One additional attempt was interrupted by a foreign GPU process before
simulation and retried explicitly on another GPU with an identical initial-state
hash. It is not another independent candidate or a physical rejection.

Final evidence under the shared roots below:

- Validation: `QUALIFIED-passive.json`, `QUALIFIED-drag.json`, six separate
  category manifests and indices, and `INDEXED-CHECK-all.json`.
- Test plan: `READINESS.json`, `CPU-TESTS.xml`, sealed candidate plans, and the
  two unsubmitted `collect-*.sbatch` scripts.
- The indexed check covers 24 category × profile × horizon combinations and
  all 24 retained episodes; positions, targets, material fields, commanded drag
  displacement and hold/release gates match raw state exactly.

Actual final-worker held-out drag examples (3, 4 and 1 objects, all three-event
sequences; saved states, not model predictions):

![Final held-out drag validation](../../../runs/mpm-evaluation-readiness-20260917/worker-v3-gallery/heldout-drag-three-event.gif)

This is reference-execution and data-interface qualification, **not a new
physical-convergence claim** or approval of the accelerated 5–7-object backend.

## Earlier readiness smoke (before worker qualification)

- All bank payloads loaded and hash-checked: 10,000 train, 258 validation, 262
  test shapes; UID cohorts disjoint.
- Manifest geometry hashes are also unique across all 10,520 prepared shapes;
  no exact geometry-hash duplicate crosses splits. This is not a near-duplicate
  or semantic-similarity audit.
- Actual passive exports: 47,288 unique episodes across 3,488 shards, covering
  all 10,000 training shapes, with no held-out UIDs in training provenance.
- The specific drag export at `staging/mpm-drag-blackwellh-20260916/data/train`
  contains 154 unique episodes across 24 shards and **364 seen shapes**. This is
  an inventory of that export, not a claim about all historical dragging data.
- 125 targeted CPU tests pass, including the new inventory/failure-publication
  tests and existing split, operator, preparation, export and consumer tests.
- Initial passive validation smoke: both pinned wall controls behaved correctly;
  all five coverage scenes completed with integrity passing. One was retained
  and matched raw positions/targets exactly through both normalization profiles.
  The four scatter scenes were rejected by the unchanged material-aware wall
  policy; the 6/7-object cases also failed the scatter-motion semantic check.
  Thus this small smoke **does not establish retained 5–7-object coverage**.
- The separately declared random-scene follow-up retained **5/5** scenes under
  the existing sand-only wall policy (counts 1,4,5,6,7). All five passed exact
  raw-to-loader positions, identities, targets, full-trajectory and C4/F16 window
  checks under both normalization profiles. The 5–7-object examples have allowed
  sand wall contact; they are not strict-wall-free examples.
- The first review helper tried to combine two evaluation categories; the export
  isolation guard rejected it. The helper now exports each category separately.
  The failed receipt is preserved, not rewritten as a success.
- Two serial drag attempts on allocation 10483072 stopped on the foreign-GPU-
  process guard. Neither is a simulation-quality verdict or a retained episode.
  The same input completed in allocation 10484354 on a different node, with the
  guard still enabled: the one-object/one-event case retained, while the declared
  four-object/three-event case rejected settling. The retained clip passed both
  current model-loader profiles, including exact command/window checks. No
  rejected trajectory was repaired or quietly replaced.
  The rejected case's final maximum particle displacement was 0.6436 mm against
  the unchanged 0.6000 mm settling threshold; it was not treated as a pass.

These are **12 independent validation candidates** (10 passive, 2 drag), not
production test data. There are 6 retained passive clips and 1 retained drag
clip. Repeated controls and the two infrastructure-interrupted drag attempts do
not add independent episodes. The four-object drag rejection means this pass
does not demonstrate retained multi-object/multi-event drag coverage.

Machine-readable inventory:
[`INVENTORY.json`](../../../runs/mpm-evaluation-readiness-20260917/INVENTORY.json).
Passive export/consumer check:
[`REVIEW.json`](../../../runs/mpm-evaluation-readiness-20260917/passive-review-v2/REVIEW.json).
Random-scene follow-up:
[`REVIEW.json`](../../../runs/mpm-evaluation-readiness-20260917/passive-random-review/REVIEW.json).
Drag export/current-consumer check:
[`REVIEW.json`](../../../runs/mpm-evaluation-readiness-20260917/drag-review/REVIEW.json).

Actual passive smoke visuals (diagnostic scenes, including rejected examples):

![Positive and negative wall controls](../../../runs/mpm-evaluation-readiness-20260917/passive-gallery/unseen/controls.gif)

![One-object held-out smoke](../../../runs/mpm-evaluation-readiness-20260917/passive-gallery/unseen/1obj-coverage-00.gif)

![Four-object scatter, rejected](../../../runs/mpm-evaluation-readiness-20260917/passive-gallery/unseen/4obj-coverage-00.gif)

Follow-up gallery uses the actual material-aware export verdict (unlike the
legacy strict-wall-only diagnostic renderer):

![Retained held-out 5-, 6-, and 7-object random scenes](../../../runs/mpm-evaluation-readiness-20260917/selection-gallery/count_unseen.gif)

![Retained held-out 1- and 4-object follow-up scenes](../../../runs/mpm-evaluation-readiness-20260917/selection-gallery/unseen.gif)

Retained one-object drag validation example (black points are the commanded
particle patch, not a robot gripper):

![Held-out drag validation example](../../../runs/mpm-evaluation-readiness-20260917/drag-gallery/drag-000.gif)

## Scope

Collect **600 retained independent episodes**, not 600 overlapping windows:

| Category | Task | Shape membership | Objects | Retained episodes |
|---|---|---|---|---:|
| passive_seen | Passive | Actually exported training shapes | 1–4 | 100 |
| passive_unseen | Passive | Frozen test bank | 1–4 | 100 |
| passive_count_seen | Passive | Actually exported training shapes | 5–7 | 100 |
| passive_count_unseen | Passive | Frozen test bank | 5–7 | 100 |
| drag_seen | Drag | Actually exported drag-training shapes | 1–4 | 100 |
| drag_unseen | Drag | Frozen test bank | 1–4 | 100 |

Count quotas are 25 each for 1/2/3/4 and 34/33/33 for 5/6/7. The identities of
primary candidates, reserve candidates, random namespaces, factor assignments,
and finite attempt budget must be frozen **before** test simulation starts.
Within a quota, retain the first candidates passing the unchanged physical and
integrity filters in declared order, independent of worker completion order.
Record every rejection. A shortage is an incomplete set, not permission to
change a filter or quietly draw additional candidates.

The suite is an initial diagnostic suite. A hundred episodes does not support
precise conclusions for every object-count × material × stiffness × action cell.
Report these marginals and uncertainty; do not describe it as a full factorial.

“Seen” is relative to a frozen exported training inventory. A later model must
also record which training exports it actually consumed: export membership alone
does not prove the model has seen a shape. Keep passive and drag inventories
separate. If evaluating a joint model with another definition of seen, make that
an explicit versioned evaluation protocol, not an implicit union of banks.

The newer `pwm-data-v1` training release was also checked during integration:
its passive index contains all 47,288 episodes; its drag index contains **2,333
episodes using 4,724 shapes**. The fixed 364-shape drag-seen roster is a verified
subset of those shapes, not a uniform sample from all 4,724. All 154 episodes in
the original drag inventory are present in that training release. Neither the
258-shape validation bank nor the 262-shape test bank overlaps the full drag
training release. The frozen plan's seen-roster restriction has not been silently
expanded in response to this newer inventory.

## Frozen inputs and numerical policy

Shared root:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model
  assets/mpm/fresh-10k-v1/
    SPLITS.json
    pool/
    train/          # 10,000 shapes
    validation/     # 258 shapes
    test/           # 262 shapes
  tools/mpm-v4/template-v1/source/     # frozen passive collector
  tools/mpm-v4/reference-assets/      # passive control inputs
  staging/mpm-drag-blackwellh-20260916/source/  # frozen drag collector
```

- Use validation shapes for engineering pilots; do not inspect test results to
  tune the collection settings, filters, model, normalization, or sampling policy.
- Verify payload hashes, split manifest, UID disjointness, and bundle source
  hashes. A plan must bind the specific training inventory and asset release.
- Passive uses the existing g100 simulator and material-dependent preselected
  clock, 49 states at 0.0417 s intervals. Do not substitute a changed checkout.
- Passive material range remains 10 kPa, 100 kPa, and 1 MPa with the original
  elastic/plasticine/sand policy. No 10 MPa extension in this suite.
- Use the production sand-only wall-contact exception for passive exports.
  Sand contact is **not** proof of a wall-free trajectory. Non-sand contact,
  ambiguous contact attribution, collapse, and other integrity failures reject.
- Drag uses the frozen g256 gentle-v4 protocol, homogeneous elastic/plasticine
  only, 0.05–0.25 m/s commanded speeds, one to three events, and its unchanged
  settling, strict-wall, collapse, fragmentation and interaction checks.
- Preserve the distinction between requested and actual command families: the
  interaction sampler may replace a lift with a pull. Measure actual coverage.
- The drag approval explicitly does **not** claim numerical convergence. A
  successful loader or smoke test cannot establish physical accuracy.
- Drag 5–7 objects, new stiffness ranges, and mixed-material drag are outside
  this collection request.

## Implemented verification helpers

`python -m collection.mpm.evaluation --help` provides:

1. `inventory`: verifies all three shape banks and scans actual exported JSON
   provenance, rejects overlapping/aliased roots, duplicate episodes, wrong
   splits, held-out training assets, and mixed passive/drag inputs. It does not
   deserialize pickle or hash entire training tar payloads. The receipt clearly
   records this limit. Supply immutable completed export roots, not raw pilots.
2. `prepare-smoke`: verifies the selected bundle and asset release, then invokes
   its frozen source in a separate process to prepare **validation-only** inputs.
   Passive covers counts 1,4,5,6,7 plus the pinned positive/negative controls.
   Drag covers a one-object/one-event case and a four-object/three-event case.
   Prepared inputs are not a simulation pass or a production gate.
   `--passive-family-index 0` prepares the separately declared random-scene
   follow-up (index 4 is the initial scatter smoke). This does not change any
   filter, replace the earlier scenes, or generate test episodes.
3. `review-smoke`: uses the frozen exporter, keeps evaluation categories separate,
   and compares retained raw trajectories with the current model adapter under
   both existing stats profiles. It preserves no-retained-example categories as
   incomplete coverage, not as a loader pass. `--drag-execution` explicitly names
   an infrastructure retry; interrupted directories are never automatically used.
4. `render`: plots reviewed passive raw states in XY/XZ with fixed metric axes and
   the actual filter verdict. A sand-contact allowance must not be relabeled as
   a blanket wall-contact rejection. GIFs are evidence for visual inspection, not
   a substitute for numerical validation.

Example, from the repository root (shell variables below are task-specific):

```bash
MPM_SHARED=/data/group_data/katefgroup-ssd/world_model/particle_world_model
MPM_PYTHON="$MPM_SHARED/environments/mpm-genesis-v1/bin/python"
CUDA_VISIBLE_DEVICES='' "$MPM_PYTHON" -m collection.mpm.evaluation prepare-smoke \
  --task passive \
  --bundle "$MPM_SHARED/tools/mpm-v4/template-v1" \
  --assets "$MPM_SHARED/assets/mpm/fresh-10k-v1" \
  --reference "$MPM_SHARED/tools/mpm-v4/reference-assets" \
  --python "$MPM_PYTHON" \
  --output runs/mpm-evaluation-validation-NEW/passive
```

Outputs are exclusive/no-clobber. Use a new named run for an intentional repeat;
do not overwrite a failed diagnostic or its prepared inputs.

## Frozen runner and launch procedure

The existing fast passive operator rejects non-training collections; the frozen
drag gate authorizes training 1–4 only. These restrictions are intentional.

`collection/mpm/evaluation_suite.py` is the evaluation-specific orchestration
entry point. It uses the pinned serial numerical implementations, including for
5–7-object passive scenes; it does not waive the failed accelerated 5–7 gate.
Its frozen copy runs independently of subsequent edits to this checkout.

The current shared release is:

```text
staging/mpm-evaluation-suites-v1/
  validation-20260917-v3/
    driver.py, PLAN.json, SEAL.json, INVENTORY.json
    model/                       # exact consumer + normalization snapshot
    work/<category>/candidate-*/  # inputs, logs, raw, review, outcome
    datasets/validation/<category>/MANIFEST.json + *.tar
    QUALIFIED-passive.json        # appears only after qualification
    QUALIFIED-drag.json           # appears only after qualification
  test-600-20260917-v3/
    driver.py, PLAN.json, SEAL.json, INVENTORY.json, model/
    collect-passive.sbatch        # 2 PRO 6000, 16 CPUs, 128 GiB, 12 hours
    collect-drag.sbatch           # separate 2-GPU request, same resources
    datasets/test/<category>/     # created only after complete category quotas
```

The test plan is already frozen: 100 retained episodes and 800 predeclared
candidates per category. The two scripts passed `sbatch --test-only`; this
checks Slurm request validity but does **not** reserve resources or submit jobs.
An actual invocation with incomplete validation was also checked: it refused
before generating test data. The original validation v1 directory is a preserved
prototype, not the qualification source for these scripts. Version 2 additionally
exposed an access issue: atomic temporary files/directories defaulted to 0600
and 0700 even with a shared umask. Version 3 explicitly sets receipts to 0640
and published category directories to 2750. Its numerical sources, candidate
lists and consumers are unchanged; its own validation has now passed. The older
version-2 drag lane was deliberately stopped once version 3 was available;
`SUPERSEDED.json` records that action, and all its results remain preserved.

Implemented protections:

- Seen eligibility uses the frozen task-specific exported UID roster. Only
  selected, hash-verified NPZs are loaded; points and original sampler order are
  unchanged. Passive and drag prepared-state hashes matched the original eager
  loader in direct v1/v2 checks.
- Exact source, inventory, asset release and current consumer/stat snapshots
  are bound by hashes. Validation and test have distinct namespaces. The test
  launcher recomputes qualification from the validation outcomes; a hand-edited
  success flag is insufficient.
- Each attempt has exclusive inputs, logs and receipts. Unexpected errors stop
  work. Known placement, settling and physical filter failures remain recorded
  rejections. Reserves are used in declared order within object-count quotas.
- Category locks prevent duplicate workers. Completed reviews resume without
  resimulation. An interrupted simulation requires inspection and an explicit
  retry, which must regenerate the same initial-state hash. Old attempts stay.
- Publication checks the entire exact manifest and shard set on resume, not
  merely the files a possibly edited manifest happens to list. Outputs are
  category-separated and marked `accepted_for_training=false`.
- Every retained candidate passes the raw-to-current-loader check under both
  existing normalization profiles before it can count toward a quota.
- A validation task is approved only after every category reaches its declared
  quotas. Drag additionally requires a retained multi-object/multi-event case.
  This checks reference execution and data plumbing, not physical convergence.

The qualification used four retained episodes per category: one at each count
1–4 and 2/1/1 episodes at counts 5/6/7. Its GPU workers used existing
allocations; they did not consume production test candidates. Validation is
complete and its workers have exited.

Inspect and qualify (qualification refuses incomplete results):

```bash
MPM_SHARED=/data/group_data/katefgroup-ssd/world_model/particle_world_model
MPM_PYTHON="$MPM_SHARED/environments/mpm-genesis-v1/bin/python"
MPM_SUITES="$MPM_SHARED/staging/mpm-evaluation-suites-v1"
MPM_VALIDATION="$MPM_SUITES/validation-20260917-v3"
MPM_TEST="$MPM_SUITES/test-600-20260917-v3"
"$MPM_PYTHON" "$MPM_VALIDATION/driver.py" status --root "$MPM_VALIDATION"
"$MPM_PYTHON" "$MPM_VALIDATION/driver.py" qualify --root "$MPM_VALIDATION" --task passive
"$MPM_PYTHON" "$MPM_VALIDATION/driver.py" qualify --root "$MPM_VALIDATION" --task drag
```

After qualification and the decision to start test collection:

```bash
sbatch "$MPM_TEST/collect-passive.sbatch"
sbatch "$MPM_TEST/collect-drag.sbatch"
"$MPM_PYTHON" "$MPM_TEST/driver.py" status --root "$MPM_TEST"
```

These are two separate jobs requesting four GPUs in total if run concurrently.
Run them sequentially if collection must stay within a two-GPU limit.
Twelve hours is a bounded allocation request, **not a measured completion-time
promise**. The worker reserves shutdown time and can finish incomplete. Resume
the same frozen suite rather than creating new candidate lists. Do not change
the filters if the finite candidate budget is exhausted.

This named suite belongs to `robertre`; submit/resume it as its owner. Published
categories and their manifests are lab-group-readable. That is not a claim that
another UID can resume owner-private intermediate files or use this account's
Slurm association. Other collectors need their own named plans and permitted
scheduler account; do not independently regenerate this frozen test suite.

For an interrupted candidate, inspect `FAILURE-*.json` and the corresponding
logs, resolve the infrastructure problem, then invoke the frozen driver's `run`
command with the same category, an allocated `--gpu GPU-...`, adequate bounded
`--hours`, and `--retry-interrupted`. This flag is deliberately absent from the
unattended production scripts. Never delete an attempt to make a rerun look new.

Regression command (the older recipe tests require the full legacy fixture,
not just the shared wall-control subset):

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
PWM_MPM_REFERENCE_ROOT="$PWD/reports/mpm_correctness_20260907" \
"$MPM_PYTHON" -m pytest -q collection/mpm/tests
```

The 503-test result includes 31 new evaluation-suite tests and 12 index-bridge
tests. The training runtime
publication closure explicitly excludes the two evaluation operators, which
snapshot their own source/consumer closure instead. No simulator equations,
timesteps, wall thresholds, settling thresholds or training approvals were
changed by this evaluation-runner work.

Actual version-2 frozen-worker drag examples (retained one-object/one-event and
two-object/two-event clips; the latter is a pull followed by a lift):

![Frozen worker drag validation](../../../runs/mpm-evaluation-readiness-20260917/worker-v2-gallery/drag-seen-1-and-2-objects.gif)

This is saved particle state, not model prediction. Colors identify materials,
black points identify the commanded material patch. Playback is approximately
2× slow, with fixed metric axes per clip. The rendering receipt beside the GIF
binds both raw hashes; this visual alone does not establish convergence.

Retained version-2 passive examples at counts 5, 6 and 7 (seen shapes):

![Frozen worker passive count validation](../../../runs/mpm-evaluation-readiness-20260917/worker-v2-passive-gallery/count_seen.gif)

These galleries remain labeled version 2 and are not substituted for version-3
qualification. The permission-only version change repeats the same validation
candidates; repeated runs are not additional independent examples.

## Training/evaluation consumer

### Current indexed domains

The newer unified training route uses **`mpm_fresh` and `mpm_drag` with
`INDEX.json`**, not the legacy `mpm` slot. The collector still publishes the same
verified WebDataset bytes. Convert each completed category with:

```bash
"$MPM_PYTHON" -m collection.mpm.evaluation index-export \
  --suite-root "$MPM_TEST" --category passive_unseen \
  --training-index "$MPM_SHARED/releases/pwm-data-v1/mpm_fresh/train" \
  --output "$MPM_TEST/indices/test/passive_unseen"
```

Use the corresponding `mpm_drag/train` directory for `drag_seen` and
`drag_unseen`; preserve all six categories as separate index roots. Run from the
repository root. This command verifies the frozen driver's gate and exact
published membership, hashes all shards, checks actual training episode/shape
membership, and creates an exclusive new index. It does **not** apply the generic
index builder's random train/validation split, modify shards, fit statistics,
or add these engineering episodes to an existing training release.

For the validation pilot, the analogous roots are
`validation-20260917-v3/indices/validation/<category>`; the index schema calls
their split `val`, and marks them **engineering validation only**. They are not
the production test set and should not replace the training release's pending
test folders. Only real test collection outputs get `split: test`.

All 16 final passive validation episodes were additionally verified through the
frozen `IndexedEpisodeDataset` using the training release's unchanged
`releases/pwm-data-v1/stats/normalization_collections_20260916_v2.json`, both
profiles, full trajectories and C4/F16 windows. Physical positions and targets
match raw data exactly; all normalized targets are finite. The existing
train/index pairing checks pass for each category. This supplements, rather than
replaces, the per-candidate legacy-adapter checks inside the numerical runner.

The same check has also passed on all eight retained drag episodes, including
exact raw commanded displacement, selected-patch masks, future gates and context
hold/release gates. The final machine-readable receipt is
`validation-20260917-v3/INDEXED-CHECK-all.json` (24 category × profile × horizon
combinations); the earlier first-five-category receipt is also preserved. The
reproducible CPU checker is:

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
"$MPM_PYTHON" collection/mpm/tests/check_evaluation_index.py \
  --root "$MPM_VALIDATION" \
  --training-release "$MPM_SHARED/releases/pwm-data-v1" \
  --output "$MPM_VALIDATION/INDEXED-CHECK-NEW.json" \
  --categories passive_seen passive_unseen passive_count_seen passive_count_unseen drag_seen drag_unseen
```

Choose a fresh receipt filename; existing reports are not overwritten.

The generic `scripts/unified/prepare_indexed_datasets.py` training index recipe
is **not** the command to run on these test categories: it repartitions groups.

Example deterministic full-trajectory read after indexing:

```python
from data.indexed_episodes import IndexedEpisodeDataset

dataset = IndexedEpisodeDataset(
    "/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/"
    "mpm-evaluation-suites-v1/test-600-20260917-v3/indices/test/passive_unseen",
    domain="mpm_fresh", context_len=4, future_len=-1,
    training=False, shuffle=False,
    normalization={
        "profile": "xy_p99_per_domain",
        "artifact": "/data/group_data/katefgroup-ssd/world_model/particle_world_model/"
                    "releases/pwm-data-v1/stats/normalization_collections_20260916_v2.json",
    },
)
batch = dataset.window(dataset.records[0], 0)  # C4, then all 45 future states
```

Use `domain="mpm_drag"` and the corresponding indexed drag category for active
sequences. Do not use the default four-domain stats artifact for these indexed
domains: it lacks their domain-specific entries. Existing default-profile MPM
checks are separate legacy-adapter compatibility checks, not a substitute for
this collections-artifact check.

### WebDataset contract and evaluation

See [DATASET_FORMAT.md](DATASET_FORMAT.md) for actual WebDataset batch fields,
units, coordinate transform, timesteps, passive/drag action contracts and root
configuration. Do not recompute normalization on validation or test data.

Use frame-0 C4-conditioned full autoregressive rollouts as the primary passive
evaluation (45 predicted states), with C4/F16 windows reported separately. For
drag, supply the recorded future commands, not achieved particle deltas; full
episodes contain 49/81/113 states. Report results separately by sequence length
as well as on a common horizon. All valid MPM material particles are prediction
targets; selected/unselected drag-patch errors can be additional diagnostics.

Report episode-level and per-object position error versus time, rollout failure
rates, and the last-position baseline. Average at episode level so longer drag
episodes do not silently get more weight. Sample uncertainty over episodes, not
overlapping windows. Keep all six categories separate before any aggregate.

The repository's newer training handoff records a decision to use test indices
for training-time validation/checkpoint selection. This collector does not
change that policy. If these test episodes are used to select checkpoints or
tune settings, disclose that use; their resulting score is not an untouched
final-test estimate. Engineering validation produced here must remain labeled
as such regardless of the later model-selection policy.
