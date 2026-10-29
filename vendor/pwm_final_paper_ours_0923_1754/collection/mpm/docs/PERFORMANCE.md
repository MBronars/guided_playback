# MPM performance work: physics-preserving changes

## Current recovery notice — 2026-09-14

The original v4 run stopped at 19:00 UTC on one contraction exclusion. A serial
replay reproduced it. The clip remains excluded; review-only recovery restarted
at 20:25 UTC with the same qualified GPU code and speedups. See
[RECOVERY_10432785.md](RECOVERY_10432785.md) for the cause, tests and current
paths. The operator's `active_recovery` field supersedes historical status.


## First change: reuse compiled kernels

The production worker previously set Warp's cache to a fresh per-chunk directory.
In three completed two-PRO chunks (40080, 40090, 40096), logs record 75.9–77.2 s
of compilation per chunk. Consecutive worker completions were 344.5–352.9 s
apart. The summed grouped stepping/frame-journal times were 222.7–228.5 s for
60 coverage attempts, plus 9.1–9.6 s for the two reference controls. Review
overlapped subsequent workers; its tail must not be added as a serial cost.

These are observed intervals, not a complete GPU-kernel profile. The grouped
timer includes synchronous saved-frame compression and durable writes. Removing
compilation suggests roughly 1.28x in this particular steady-state workload;
the smaller benchmark below will have a different overhead fraction.

The production candidate now reuses a **private node-local** cache under
`/tmp/pwm-mpm-kernel-cache-<uid>/<identity SHA>/`. It holds the existing GPU lease
while using the cache. The key includes the frozen bundle SHA, worker source SHA,
Python version/interpreter, Warp version, physical GPU UUID/architecture and
execution mode and the qualification's cache-seed SHA. Unsafe directories or
changed identity receipts fail closed. Each chunk records `KERNEL_CACHE.json`.
Nothing clears an existing user's cache.

The final implementation does **not** compile a second, independent warm cache.
Each passing qualification lane records `CACHE_SEED.json`, binding the exact
compiler artifacts used by that lane to its GPU, frozen bundle and execution
mode. The startup certificate binds this manifest. The controller passes its
path and SHA to every chunk on that GPU; the worker copies missing artifacts
into the private cache and checks every artifact before and after simulation.
Changed, missing, symlinked or unexpected artifacts fail closed. Production
workers require this seed; legacy workers retain their original per-chunk default.
An unseeded reusable cache is available only through an explicit diagnostic
argument. Compiled artifacts are not a substitute for passing numerical checks.

No solver, grouped-kernel implementation, precision, timestep, scene, safety
guard, wall policy, frame cadence, filter, journal or model-consumer code changes.
All normal startup and first-wave gates still apply. This does not resolve the
separate historical four-GPU qualification failure (one 3-step wall timestamp
difference against the unchanged 2-step tolerance).

## Bounded cold/warm benchmark

Driver: `collection/mpm/launch/mpm_cache_benchmark.py`. It submits no Slurm job.
Run it inside an allocated GPU context, with an explicit physical GPU UUID and
a new output directory. It verifies a frozen training bundle and prepared inputs.
Each arm uses a new process, as the production chunk worker does:

1. Original per-chunk cache (cold baseline).
2. Reusable cache, first use.
3. Reusable cache, warm process restart.
4. A second warm process restart.

All arms use identical initial particle files. The first bounded test selects
five scenes per object count (1–4), spanning all three stiffness phases, and
keeps both historical reference controls. Both current MPM GPUs run independently.
Every attempt's journal/raw integrity is checked; the positive/negative controls
must preserve their expected outcome. Every candidate coverage trajectory is
compared with the baseline using existing position RMS, wall-event, contact
presence and filter-selection checks. No thresholds are relaxed after results.

Initial test artifacts:

```
runs/mpm-cache-mshcgc9x/
  deployment/template-v1/   # frozen candidate, maintained-source commit 4c06480
  prepared/                # full CPU-prepared default inputs
  benchmark-plan/          # twenty preselected coverage cases + two controls
  lane-0/                  # fresh-process arms and result receipts
  lane-1/
```

The benchmark is diagnostic: no production sequence reservation, training
promotion, campaign replacement or scheduler submission occurs. Its elapsed time
includes worker startup and verification, but not input copying before launch
or post-run comparison. They are worker-stage timings, not sustained retained
clips/hour measurements.

## Results and numerical repeatability

Recorded under `runs/mpm-cache-mshcgc9x/`; all initial failed verdicts remain intact.
The final maintained runtime source is commit `6d2c695`; the exploratory frozen
candidate was `4c06480`. No numerical source changed between these candidates.

| Workload | GPU 0 | GPU 1 | Meaning |
| --- | --- | --- | --- |
| 20 coverage cases + 2 controls | 186.14 s cold / 108.23 s mean warm | 188.07 / 111.61 s | 1.72x / 1.69x; compilation-heavy small chunks |
| 60 coverage attempts + 2 controls | 347.66 s cold / 276.03 s warm | 346.85 / 279.24 s | **1.26x / 1.24x** worker throughput; production-sized chunks |

Warm runs logged zero compilation. The full plan has one predeclared placement
rejection, hence **59 simulated coverage trajectories**, not 60. The full-chunk
gain is the useful estimate: about 24–26% faster workers, not a 5x gain and not
yet a measured end-to-end collection-rate gain.

The initially unseeded cache experiment was **not an unconditional parity pass**:

- The 20-case test exceeded the existing 0.100 mm trajectory RMS limit on one
  four-object collision in three candidate arms (0.111–0.145 mm). Two GPU-0
  warm arms passed. Running the unchanged cold baseline again also failed on
  this collision on GPU 0 (0.138 mm), while GPU 1 passed.
- The full 60-attempt test passed all comparisons on GPU 0. On GPU 1, one
  one-object translation measured 0.115 mm RMS and failed. Its integrity,
  filtering and wall-event comparisons still matched. That failed verdict was
  not removed or converted into approval.
- Generated CUDA kernel bodies were identical across the inspected cold builds,
  but declaration order and some compiler artifacts differed. This is not a
  numerical certificate. A post-hoc same-cache comparison also retained an
  outlying collision, so compiler order alone does not explain all variation.
- A separate full-chunk replay copied the **exact original compiled reference
  artifacts** before restarting each worker. All **59 trajectory comparisons on
  each GPU passed**, and all copied artifacts remained unchanged. This motivated
  the final qualification-seeded design; it does not erase the earlier failures
  or establish universal deterministic replay.

Key receipts are `full-lane-{0,1}/RESULT.json`,
`full-lane-{0,1}/compiled-reference-repeat/RESULT.json`,
`lane-{0,1}/baseline-repeat/RESULT.json`, `COMPILED_EQUIVALENCE.json` and
`SAME_CACHE_DIAGNOSTIC.json`. The diagnostic driver now exits nonzero on a failed
comparison. Its first exploratory version wrote `passed: false` correctly but
returned exit zero; use the recorded verdict, not that original supervisor code.

[Repeatability plot](../../../runs/mpm-cache-mshcgc9x/repeatability.png)
shows frame-wise differences for the problematic collision and translation.
These are particle-error curves, not throughput or visual-quality certification.

The final `mpm_training_campaign.py worker` CLI was then exercised with the new
seed descriptor/hash arguments on both GPUs: the first four coverage cases from
the full plan (preserving their original group) and both controls. Both lanes
passed all four trajectory comparisons and all six integrity/control checks,
logged zero compilation, and completed the before/after seed inventory checks.
Worst trajectory RMS was 0.00331 mm. Receipts:
`seeded-smoke-{0,1}/RESULT.json`. This smoke test uses diagnostic descriptors
for the original reference artifacts; it is **not** a production qualification
certificate and cannot authorize a collection through the controller.

CPU regression after the final runtime change: **345 passed, 2 skipped** in
`collection/mpm/tests tests/collection`, with the explicit historical reference
root and GPUs disabled. Coverage includes private cache identity/permissions,
seed tampering, unexpected artifacts, path traversal, wrong-GPU rejection,
controller propagation and the separately disabled asynchronous writer.

Reproduce the combined CPU check from the canonical checkout:

```bash
CUDA_VISIBLE_DEVICES='' \
PWM_MPM_REFERENCE_ROOT=/data/user_data/robertre/particle_world_model/reports/mpm_correctness_20260907 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/data/group_data/katefgroup-ssd/world_model/particle_world_model/environments/mpm-genesis-v1/bin/python \
  -m pytest collection/mpm/tests tests/collection -q -rs
```

The two skips are SIM1 geometry-environment tests, outside this MPM change.
Without the explicit historical reference root, ten additional historical-bank
tests skip (335 pass). The smaller public deployment intentionally contains only
the reference closure needed by production; these extra historical bank tests
use the local audit archive. Disable automatic third-party pytest plugins in the
sandbox, and explicitly set the empty GPU mask for CPU observer tests.

The numerical tolerance remains **0.100 mm trajectory RMS**, the wall-event
tolerance remains **two native steps**, and contact presence/filter selection
remain exact. The previously failed four-GPU startup is still failed. A new
production allocation must pass its own startup qualification and first-wave
quality/consumer gate before continued collection.

## Larger CPU/RAM allocation: measured capacity sweep

Allocation `10432785` on `babel-l9-16` supplies **two PRO 6000s, 128 allocated
CPUs and 256 GiB RAM**. The batch script initially resumed the historical serial
collector, not the optimized v3 launcher. With explicit user approval, the
original supervisor was parked and its two simulation workers stopped. Completed
files and partial journals remain; no incomplete wave was counted or deleted.
See `runs/mpm-capacity-10432785/ALLOCATION_HANDOFF.md` for the allocation's parked
state and why blindly resuming that supervisor can release the allocation.

The diagnostic sweep ran both GPUs concurrently, using exactly the same 20
preselected coverage scenes plus two controls, with identical cached compiler
artifacts. Affinity limited workers to a total of 8 or 32 logical CPUs. These
are affinity counts, not a separate Slurm RAM-limit experiment or physical-core
count; masks are recorded in each receipt. All arms had the same 256 GiB job
memory allowance. BLAS/OpenMP thread settings remained at one. No production
configuration was modified.

| Total CPU affinity | Scenes per group | GPU 0 seconds | GPU 1 seconds | Comparison against first arm |
| --- | --- | --- | --- | --- |
| 8 | 4 | 103.95 | 104.08 | Baseline; verified journals and controls |
| 32 | 4 | 101.63 | 102.67 | All 20 coverage comparisons pass on both GPUs |
| 32 | 8 | 101.95 | 102.45 | All 20 pass on both GPUs |
| 32 | 16 | 103.17 | 103.96 | **GPU 0 fails one collision**; GPU 1 passes |

The failing case is `case-45-4obj-collision`, with 0.15203 mm trajectory RMS
against the unchanged 0.100 mm limit. Wall events, contact presence, filtering
and integrity still match. This failure remains a failure; the sweep exits
nonzero and records `passed: false`. It is not evidence for deploying group 16.

All arms logged zero compilation. Worker host peak RSS was only **0.78–0.86 GiB**,
and CPU time was roughly 94–96 CPU-seconds per 102–104 elapsed seconds. This
shows no worker RAM-capacity bottleneck in these cases; it does not bound the
whole campaign's preparation/review/page-cache memory. Extra CPUs and larger
groups gave **no material worker-stage gain** in this unreplicated sweep. The
1–2% differences are insufficient to justify a production change.

Evidence: `runs/mpm-capacity-10432785/RESULT.json`, per-arm/lane receipts and
worker logs. Driver source for this four-arm sweep is commit `4fb9a6e`.
It excludes CPU input preparation, final review/export/rendering and comparison
time from its worker metric. It is not a sustained collection-rate measurement.

### CPU-only compression/writing replay

On the same allocation, four real recorded scenes (one each with 1/2/3/4
objects) were replayed through the existing journal serializer: **196 frames,
396 files per arm**. Every arm's payloads, headers and chained receipts were
byte-for-byte identical to the synchronous baseline; full journal verification
also passed. The serializer, compression settings, fsync and clocks were kept.

| Background writer threads | Elapsed seconds |
| --- | --- |
| 0, synchronous first | 1.825 |
| 1 | 1.820 |
| 2 | 0.947 |
| 4 | 0.723 |
| 8 | 0.691 |
| 0, synchronous repeat | 1.875 |

Four threads are about **2.5–2.6x faster for this writing component**, not the
entire simulator. Eight threads give little additional benefit. The writer
used at most 7.51 MiB of owned queued copies under its 64 MiB budget. This does
not require hundreds of GiB of RAM. Evidence:
`runs/mpm-capacity-10432785/io-replay/RESULT.json` and `io_replay.py`.
Timing excludes loading the common inputs, creating journals and checking output
hashes afterward. It includes queue drain and all durable frame writes.

### Asynchronous writer inside the GPU worker: bounded candidate

A separate pair used the same 32-CPU affinity, four-scene grouping, exact prepared
20-scene plan and copied compiler artifacts on both GPUs. Only journal writing
changed. The diagnostic adapter gives each saved frame an immutable owned copy,
uses four writer threads and a 64 MiB/32-frame queue bound, and drains the queue
before the grouped simulator verifies journals and marks a trajectory complete.
Worker exit also drains the writer and propagates errors. The adapter restores
the original module entry points on success or failure; this restoration and
mandatory flush have dedicated tests.

| Worker | GPU 0 seconds | GPU 1 seconds |
| --- | --- | --- |
| Synchronous, fresh baseline | 102.092 | 102.870 |
| Four background writer threads | 92.553 | 93.566 |

This is **1.103x / 1.099x worker throughput**, with all 20 coverage trajectory
comparisons passing on both GPUs. Journal/raw/control checks pass; no compilation
occurred. Peak owned queued copies were 1.50 MiB per worker. This is a useful
software-overlap result, not proof that 128 CPUs or 256 GiB RAM are needed; worker
CPU use still averaged only about one core. The numerical solver, native guards,
precision, clocks, compression and durability semantics remain unchanged.

Evidence: `runs/mpm-capacity-10432785-async/RESULT.json`. Diagnostic adapter source
is commit `08be92c`, plus the separately hash-recorded `mpm_async_io.py`. The
adapter is additional execution provenance: an unchanged solver source hash
alone does not describe the modified I/O path. The frozen production worker
does not use this adapter, and no production release has enabled it.

The small tests above write to the user's project NFS export, not the production
shared-SSD export. A first full-chunk attempt on that user export was interrupted
before obtaining results so **both arms** could be rerun on production storage;
its partial files and failure receipt remain in
`runs/mpm-capacity-10432785-async-full/`. The new shared-SSD test root is
`/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-capacity-10432785-async-full`.
Do not extrapolate the small-test writing gain to sustained collection without
the production-storage/full-chunk confirmation.

CPU regression with the adapter tests: **348 passed, 2 skipped** in the combined
MPM/shared collection suite; the skips are the unrelated SIM1 geometry tests.

### Full production-sized chunks on the shared SSD: paired pass

The shared-SSD pair completed with **all 59 simulated coverage trajectory
comparisons passing on each GPU**, plus both matched controls and journal/raw
checks. Each plan contains 60 coverage attempts, including one predeclared
placement rejection. There was zero compilation in either arm.

| Worker | GPU 0 seconds | GPU 1 seconds |
| --- | --- | --- |
| Synchronous cached baseline | 266.591 | 271.238 |
| Four background writer threads | 237.422 | 242.624 |
| Baseline / candidate throughput ratio | 1.123x | 1.118x |

This is approximately **12% higher worker-stage throughput** on the actual
production output filesystem. Maximum trajectory RMS was 0.07381 mm on GPU 0
and 0.02257 mm on GPU 1, below the unchanged 0.100 mm gate. Filtering, integrity
and wall/contact-event checks also pass. These are bounded comparisons, not a
claim of bitwise simulation determinism or a sustained four-GPU collection rate.

The candidate's peak worker host RSS was 825/840 MiB; peak owned queued copies
were 1.50 MiB. CPU time was 251/257 CPU-seconds across 237/243 elapsed seconds:
about **1.06 CPU cores per worker on average**. This software overlap does not
depend on having 128 CPUs or 256 GiB RAM, and does not show that more RAM alone
improves throughput. The published v3 collector remains unchanged; the writer
is still a separately tested diagnostic candidate awaiting explicit production
integration and the normal new-runtime startup checks.

Full paired receipt:
`/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-capacity-10432785-async-full/RESULT.json`.

Both asynchronous full chunks then passed the unchanged production review path:
quality filtering, WebDataset export, and the actual loader under
`xy_p99_per_domain` and `xy_p99_global`. Each retained the same **52 of 60
attempted cases** and checked every retained episode under both profiles.
Positions, material fields, object IDs, clocks, passive-action contracts,
future deltas and finite normalized targets passed. The reviewed chunks have
`worker/COMPLETE.json` and a generated `worker/samples.gif` under each candidate
lane. These are two repetitions of the same cases, **not 104 distinct new
training clips**. No benchmark output was promoted into a production dataset.

Recommendation from these tests: keep four-scene grouping; do not request 128
CPUs/256 GiB solely for these workers. The useful next change is explicit
production integration of the bounded writer, followed by its normal frozen
runtime qualification. Group 16 stays rejected. The current allocation remains
held with the old collector paused; no collection restart or production writer
enablement occurred during the capacity tests.

## v4 integration (after the capacity tests)

The user authorized production integration and restart after the tests above.
Runtime commit `13390b6` adds the bounded writer to the frozen launch closure
and generalizes the controller to exactly two or four PRO GPUs. Both startup
candidate qualification and production workers execute the same adapter. It
uses a private copy of the simulator's function namespace for journal bindings;
it does not mutate module globals or change numerical source. Serial startup
references still use the original synchronous writer. All original thresholds,
controls, first-wave checks and per-chunk review remain mandatory.

The immutable shared release is `tools/mpm-v4`, template SHA-256
`88c84aa824a687393731c07e3bb638f13b050c401ea35fa75b098de95e87a184`.
Shared publication/source/asset/registry preflight passed. CPU integration tests:
354 passed, two unrelated SIM1 geometry tests skipped. Qualification/runtime
receipts in the new campaign determine actual execution status; publication is
not GPU approval.

The two-GPU continuation is registered as `fresh-10432785-v4`, cycles
84000–85999 (new namespace), under the common operator campaign root. It keeps
the original balanced 1–4-object passive training recipe and 10K training bank.
120,000 attempts is a ceiling, not a promised yield; the actual allocation
deadline and five-hour shutdown reserve can stop it earlier. Original campaign
files and interrupted chunks are preserved; benchmark clips are not promoted.

Startup qualification for this exact two-GPU frozen runtime passed on both
devices in allocation 10432785. Each lane passed 104 numerical comparisons
(candidate/serial, repeats, and both preserved serial references), isolation,
failure injection, and both actual-loader profiles on all 20 retained reference
episodes. Maximum RMS across **all** comparisons was 0.011632 mm / 0.009793 mm;
maximum wall-event timing difference was two native steps on both, inside the
unchanged gates. Direct optimized-versus-serial maxima were 0.009410 mm /
0.005464 mm. The startup grouped-versus-serial speed ratio was 6.240 / 6.169;
this is **not** the incremental writer speedup or sustained campaign rate.
Passing compiled kernels are bound to production workers by the approval.

Receipt: `staging/mpm-operator-campaigns-v1/fresh-10432785-v4/data/qualification/TRAINING_PASS.json`.
The production step is `10432785.43`, with detached launcher PID 3232524 in the
target allocation's extern lifetime group. No additional sbatch was submitted.

At 2026-09-14 04:24 UTC the first production pair passed review and automatic
first-wave approval: `train-84000` retained 55/60 attempts and `train-84001`
retained 56/60, **111 distinct fresh clips total**. Both checked every retained
episode under both loader profiles and generated `samples.gif`. There was no
new manual visual-review claim. Both worker logs contain zero new compilations;
each records qualified-kernel reuse in `KERNEL_CACHE.json`. Wave 1 (the second
pair, cycles 84002/84003) started automatically. The first-wave receipt is
`data/collection/PILOT_APPROVED.json`; current counts come from completed chunk
receipts, not this dated snapshot. There is no claim of a sustained new rate yet.

Retained dataset root:
`staging/mpm-operator-campaigns-v1/fresh-10432785-v4/data/collection/datasets/train`.
Full root prefix is `/data/group_data/katefgroup-ssd/world_model/particle_world_model/`.
New work admission stops by **2026-09-15 21:18:32 UTC** (five hours before this
allocation's deadline), or earlier for its attempt/target/disk budget or a failed
check. The parked historical batch must not be resumed/killed while the new
step uses the allocation. Detailed operational handoff remains in
`runs/mpm-capacity-10432785/ALLOCATION_HANDOFF.md` in the project checkout.

## Remaining possible optimizations

- Persistent GPU processes, with bounded lifetime and explicit memory monitoring.
- Separate review capacity and independent per-GPU scheduling while preserving
  first-wave approval, failure propagation, seed isolation and budget limits.

These need separate tests and profiling. Additional CPU/RAM alone does not enable
them. One-wave CPU preparation look-ahead and prior-wave review overlap already
exist in the production runner; the bounded writer is now integrated in v4.
