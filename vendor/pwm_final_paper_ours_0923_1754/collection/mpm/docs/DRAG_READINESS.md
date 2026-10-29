# Active MPM dragging: readiness gates

Current release: [gentle-v4 qualification and production collection](DRAG_GENTLE_V4_RELEASE.md)
passed its fresh pilot (24/36 retained; all measured checks passed). The original
step **10447018.28** stopped on a validation assertion; it is not still running.
[Recovery and restart](DRAG_RECOVERY_20260915.md) correct that check, recover
334 verified clips, and start a new 36-hour run on dev-blackwellf, step
**10447018.60** (September 15, 18:30 UTC). Scope: g256, train-only, homogeneous
elastic/plasticine, 1–4 objects, 10 kPa–1 MPa; no sand. The frozen bundles use
versioned gentler commands and mandatory fragmentation rejection. Shared-grid
contact and point-connectivity limitations remain explicit. The statuses below
are preserved historical work, not the current v4 launch verdict.

Particle-detachment update: [mitigation and new fragmentation screening](DRAG_FRAGMENTATION_FIX.md).
Seven of the previous 31 retained clips flag under the new conservative screen;
"retained" below describes the historical filter, not intact-object approval.

Previous v3: [native-g256 qualification](DRAG_G256_NATIVE_RELEASE.md) completed on
both dev-blackwelld GPUs: 31/36 attempts retained, with 2,255 recorded states.
Two serial/batched checks failed; unchanged serial repeats also fail the
trajectory or settling-selection criteria. That v3 release remains blocked;
the historical statuses below are not a g256 certificate.

## Historical protocol update (before gentle-v4 qualification)

**Interaction-rich scaling is on hold for contact accuracy.** The 24-scene
targeted pilot retained 20 scenes and passed four serial/grouped comparisons,
but those checks do not qualify the shared-grid contact model. Controlled
comparisons confirm premature interaction and artificial sticking. A separate
geometry-gated prototype addresses these effects on aligned blocks but has not
been generalized to the real shape bank. See the measurements and GIFs in
[DRAG_CONTACT_ACCURACY.md](DRAG_CONTACT_ACCURACY.md).

The interaction sampler adds 30% separated / 50% pair / 20% crowded layouts
among multi-object scenes, with single-object controls kept separate. Physical
filters and the passive clock are unchanged. See
[DRAG_SEQUENCE_PROTOCOL.md](DRAG_SEQUENCE_PROTOCOL.md). No interaction scale
gate has been issued. Historical separated-layout receipts below do not qualify
collision-rich collection.

New collection preparation now uses **sand-free sequence v3**: 1–3 separated
events, continuous 0.05–0.60 m/s diagnostic commands, longer recordings and
strict side/ceiling rejection. See [DRAG_SEQUENCE_PROTOCOL.md](DRAG_SEQUENCE_PROTOCOL.md)
for exact variation and current qualification. Historical v1 sand results below
remain diagnostic history, not the proposed dataset or a v3 scale certificate.

Work started 2026-09-14. **Not approved for scaling yet.** The v4 passive
campaign and its immutable bundle are not modified by this work.

## Protocol to qualify

- Same 1,024 surface particles/object, shape banks and UID-disjoint splits,
  material laws, physical walls, native guards and material-dependent timestep
  budget as passive collection. No physics threshold is loosened for dragging.
- Bounded measured settling before each recorded 49-state, 2.0016-second clip.
  Preserve settling evidence and full deformation/momentum state; never rebuild
  a settled object from positions alone or zero its velocities to manufacture rest.
- Object-specific local particle patch selected from the settled state.
  Commands are prescribed **pre-P2G particle velocities**, not a physical robot
  or guaranteed achieved particle displacement. Record commanded and observed
  motion separately. Keep selection, per-transition activity and velocity separate:
  active zero velocity means hold; inactive means release.
- Initial families: horizontal pull/release, lift/hold/release, reverse/release.
  Explicit bounded speeds, start/stop indices, direction and selected-object
  variations. No retries/top-up samples hidden from retention accounting.
- New task/sequence namespace and output roots. Training starts with 1–4 objects;
  held-out shapes and 5–7-object evaluation require their own passing pilot.

## Gates, in order

1. CPU contracts: schedule, zero hold, release, object-local selection, units,
   off-by-one windows, bounded settling, mutation/corruption rejection.
2. Real serial GPU pilot: all materials and training counts; every native guard,
   settling history, wall evidence, collapse checks, durable journals and GIFs.
3. Serial/grouped parity on exact inputs, including command application and
   isolation; unchanged 0.1 mm RMS and two-native-step wall timing limits.
   Reuse/writer performance changes require explicit qualification, not inheritance
   from passive-only results.
4. Raw → export → unified batch round trip, both normalization profiles; commanded
   displacement = velocity × actual transition duration. No achieved-future
   motion in action conditioning. Passive loaders must reject active samples.
5. Both retained architectures: forward/backward and hold-versus-release
   conditioning tests. Temporal activity must survive windowing/device transfer/
   batch transforms. All deformable particles remain prediction targets.
6. Split-aware immutable launch bundle, disjoint reservations, no-clobber outputs,
   failure/retention accounting, first-wave gates, budget/deadline handling,
   sustained pilot and operator runbook. No scale-ready claim before these pass.

Existing historical drag datasets/checkpoints are reference material only.
Their existence does not establish current generator or unified-model support.

## Measured progress, 2026-09-14

Evidence root: `runs/mpm-drag-20260914/` (local qualification artifacts, not a shared
training release). Source snapshots are copied before execution; trajectory collection
runs never import changing working-tree code. Passive production remains on its own
previously frozen bundle and allocation.

- Six grouped smoke scenes completed and retained: 1 and 4 objects, each with
  elastic / plasticine / sand. This small smoke uses E=10 kPa only.
- The three 1-object scenes also completed serially with exactly the same
  prepared inputs and command schedules. Vector position RMS over dragging:
  elastic 0.0000299 mm, plasticine 0.0000467 mm, sand 0.008945 mm. Settling RMS
  also passes the 0.1 mm gate; native contact presence/timing and retention agree.
  These are measured smoke comparisons, not approval of the entire material range.
- All six have 49 motion states, 97 settling states, measured quiet tails,
  guarded native stepping, full native wall observation in both phases, collapse
  checks including settling deformation, and verified motion journals.
- Raw → filtered diagnostic TAR → actual drag adapter has been checked for all
  six under `xy_p99_per_domain` and `xy_p99_global`. Commands use the recorded
  0.0417 s interval, not rounded 1/24 s. MPM bounds happen to coincide in these
  two frozen profiles; both resolver paths are nevertheless exercised.
- Synthetic tests cover both models' forward/backward, hold versus release,
  temporal/static all-active equivalence, batch transforms, long/AR window
  boundaries, commanded rather than observed conditioning, and passive rejection.
- The existing passive graph file is byte-for-byte unchanged. Drag uses a new
  `CommandGroup` subclass and does not inherit its production certificate.

### Actual-state galleries

One-object smoke (XY above, XZ below; selected material particles black):

![Single-object drag smoke](../../../runs/mpm-drag-20260914/review-smoke-v2/drag-000.gif)

Four-object smoke:

![Four-object drag smoke](../../../runs/mpm-drag-20260914/review-four-v2/drag-000.gif)

Both are 49 actual saved states at 2× slow playback. No interpolated physics,
learned predictions, mesh reconstruction or physical robot is shown. Initial
multi-object layouts contain separated distractors; this is **not** a qualified
object-to-object manipulation/contact curriculum. Sand naturally loses its
original shape while settling; we select a patch on the resulting material cloud.

### Failures retained, not hidden

`matrix-v3` completed its first four scenes, then stopped before the second
group simulated: the idle preflight still saw the previous worker's recent GPU
utilization. The new preflight waits up to 30 s for driver sampling to clear;
foreign processes still fail immediately. It never retries a simulated scene.
The failed run and all its receipts remain. `snapshot-v4` / `inputs-matrix-v4`
are the next immutable 36-scene matrix: counts 1–4 × materials 3 × E values
10 kPa / 100 kPa / 1 MPa. Each sees grouped and serial execution. This 36-scene
matrix does not fully cross every family with every material/stiffness;
108 prepared cases provide that larger crossing.

## Experimental training interface

Keep the standard Transformer or UNet recipe and both existing stats profiles.
Use explicit disjoint prefiltered drag roots:

```yaml
data:
  domains: [mpm]
  roots:
    mpm:
      task: drag
      train: /path/to/qualified-drag/train
      val: /path/to/qualified-drag/validation
```

The paths above are placeholders: no shared drag release exists yet.
Particle action conditioning must be enabled (`unet_actions: true` for UNet).
Pose-only action dropout/hiding is rejected. All deformable particles, including
the driven patch, remain prediction/evaluation targets. Commands are not clamped
onto predicted positions. Standard unified windowed rollout preserves the temporal
command gates. Historical passive H5/paper/object-rollout readers reject active
inputs instead of quietly replacing their commands with zeros.

## Detailed checkpoint: 2026-09-14, 20:27 UTC

**Implementation is experimental; scaling remains blocked by measured numerical
checks.** This is not a declaration that the existing passive collection is bad:
it is a separate active-manipulation qualification, and no passive physics or
running production bundle was modified.

### Finished evidence

| Check | Result | Artifact under the evidence root |
| --- | --- | --- |
| Full model/training CPU regression | 1,547 passed, 93 skipped, 10 deselected; includes the concurrent rollout-boundary fix | pytest run; GPU-marked skips are not GPU passes |
| Collection regression | 278 passed at last full run, including concurrent passive-collection additions | `collection/mpm/tests` |
| Same weights, pre-drag models | Exact bitwise forward agreement for both architectures, with zero-initialized heads/gates opened | `tests/test_mpm_drag_legacy_forward.py`, reference commit `7c2aa5b` |
| Real exported data → production optimizer and EMA | Both architectures × both profiles; each arm contains all three material episodes; three finite, unskipped updates/arm | `training-smoke-v2/RESULT.json` |
| GPU isolation/fault injection | Invalid commands leave all command buffers unchanged; NaN state stops before transfer; healthy neighbour unchanged | `PROBE.json` |
| 36-scene serial/grouped matrix | 32 completed pairs, **31 parity passes and one failure**; four rejections agree | `audit-v4-final.json` |

The optimizer smoke is deliberately small (CPU, reduced model configuration,
fixed corruption). It is **not** the old overfit-reproduction protocol and does
not establish learned rollout quality. All 1,024 particles in each real episode
were used. The frozen stats were reused, not refitted; their MPM bounds coincide.

### Numerical failures and the next discriminating tests

1. `drag-000028` in the v4 matrix: single-object plasticine, E=1 MPa,
   pull/release. Serial/grouped RMS is **0.117309 mm while settling** and
   **0.188183 mm during dragging**, above the predeclared 0.1 mm gate. Largest
   point differences are 0.388327 mm / 0.377199 mm. Contact presence and first
   timestamps match exactly, and retention agrees. The settled states already
   differ: the final metric alone does not isolate a command-application bug.
   `matrix-reference-v9` tests the original 256-thread particle layout and
   unfused walls on exact rebound v4 inputs, separately from the default 64-thread
   fused path. No material law, timestep or tolerance changes in that test.
   **That test has finished and does not fix the failure:** elastic and sand
   pass, but plasticine swaps two selected mask entries (one boundary point is
   replaced). Comparing the two independent *serial* executions already gives
   0.103671 mm settling RMS and the same two-entry membership difference;
   their subsequent motion differs by 0.850839 mm RMS. Thus independent warmups
   plus post-settling nearest-neighbour selection confound the active-path test.
   This is evidence of baseline variability, not permission to mark the failed
   original gate passed. The next discriminating test pins a common full settled
   checkpoint and command mask before comparing serial/grouped motion.
2. Independent timestep control, E=1 MPa: every branch starts from identical
   **full** settled physical state (including deformation, APIC state and plastic
   history), with fixed selected material indices. Same physical saved times.

   | Material | 50 vs 25 µs: whole-clip vector RMS | 25 vs 12.5 µs | Initial control |
   | --- | --- | --- | --- |
   | Elastic | 0.742 mm | 0.453 mm | Pass |
   | Plasticine | 2.517 mm | 1.821 mm | Fail |
   | Sand | 4.664 mm | 3.964 mm | Fail |

   Gate: whole-clip and selected-patch vector RMS ≤1 mm, largest particle
   difference ≤5 mm, with wall-free, collapse and floor checks. Sand's largest
   difference is 45.403 mm for 50 vs 25 µs. All branches pass the physical safety
   filters; stability and timestep accuracy are different tests. No gate is
   relaxed after seeing these results. `convergence-refinement-v7` continues
   with divisors 4, 8, 16, 32 relative to the original 50 µs step. Those are
   **diagnostic branches**, not newly approved collection settings.
   This refinement has now finished: elastic passes all adjacent comparisons;
   plasticine passes 8→16 and 16→32 (0.730 / 0.817 mm whole-clip RMS), but
   sand fails all three, including 16→32 at **7.045 mm** whole-clip RMS and
   24.955 mm maximum point difference. Thus an 8× smaller step is a local
   plasticine accuracy candidate, not a full-range certificate, and a blind
   timestep decrease is not a demonstrated fix for sand. Every refinement
   branch still passes the wall, floor and collapse checks.
3. One 100 kPa plasticine scene (`drag-000019`) failed the last-eight-quiet-
   transitions settling condition. The original group implementation also
   discarded its three independent healthy neighbours. The v5 recovery path
   preserves healthy full states, removes only failed worlds and rebuilds the
   graph; `inputs-recovery-v5` preserves the exact failing group for serial and
   grouped regression. No extra settling time or threshold relaxation is hidden.
   **Recovery is now GPU-verified:** both executions retain the same three
   healthy scenes, all three trajectory comparisons pass, and only case 19 is
   rejected. Independent journals, exports and both loader profiles pass in
   `queued-reviews-v8/recovery/REVIEW.json`.

![Measured timestep comparison](../../../runs/mpm-drag-20260914/convergence-view-v6.gif)

This GIF overlays actual 50 µs and 12.5 µs states. Curves show per-frame RMS;
the gate uses whole-clip RMS, not the maximum per-frame RMS. These are native
simulation outputs, not model predictions or interpolated frames.

### Bounded continuation already arranged

Only the isolated drag GPU in allocation `10414730` is used. The six-hour queue
does not submit a production job, retry rejected scenes, or touch passive MPM.
It runs finer-timestep control, exact rejection recovery, mixed-material train,
unseen-shape validation, 5–7-object validation, and the 108-scene training
family/material/stiffness/count crossing. Test-bank assets remain untouched by
this tuning queue. A bounded reference-graph interleave pauses only our waiting
queue supervisor; its `finally` block resumes it and writes
`INTERLEAVE-REFERENCE.json`.

- Execution receipts/logs: `queue-v7/`.
- Independent CPU journals/parity/exports/both-loader-profile checks and galleries:
  `queued-reviews-v8/`. These consume a separate immutable review snapshot.
- Any child/native failure stops the remaining execution queue and retains a
  failure receipt. A numerical comparison failing its tolerance remains in the
  reports; it never produces a scale approval.

The revised prepared sampler uses a namespace-specific cyclic permutation of
the 10K bank, rather than relying on random draws to eventually cover it.
Attempted coverage is guaranteed by that schedule; retained shape coverage must
still be counted after rejection. Selected-object indices rotate across all
objects. Homogeneous and mixed-material layouts are explicit, separate regimes.

At 20:22 UTC, the command-off control's supervisor received a termination signal.
Its cleanup resumed our main queue; Slurm step `10414730.352` was cancelled by
that cleanup. The signal's external origin is not established. This is **not**
a physics failure and is not a completed accuracy test. The dependent same-state
interleave stopped before launching anything. No diagnostic supervisor remains
paused by these controllers. Mixed-material collection resumed normally; the
held-out/count/diversity tasks and CPU review queue remain bounded continuations.

The interrupted command-off control preserves complete 50 and 25 µs branches
from identical full saved states. Their whole-clip RMS differences are 0.065 mm
elastic, 0.241 mm plasticine and 0.067 mm sand, much smaller than the active
control. These partial measurements suggest actuation contributes strongly to
the sensitivity; the missing third branch prevents a full control verdict.
Do not turn partial evidence into an approval or silently rerun into its output.

The latest source also persists `SETTLED_CHECKPOINT.json` and
`settled-full-state.npz` beside newly simulated raws, bound by hashes in the raw
manifest. This retains the full state needed for controlled continuation, beyond
the plotted x/v/F_trial/C journals. The running v5 matrix predates this extra
output; the new persistence path and same-state serial-reference diagnostic still
need their own GPU execution. Existing artifacts are not retroactively upgraded.

### Launcher implementation versus approval

`collection/mpm/drag/launch.py` and `collect.sbatch` exist, but **there is no real
approved drag gate yet**. The operator rejects a missing gate. It freezes both
generator and consumer code, checks evidence/source/bank hashes and explicit
split/count/material capabilities, reserves a unique sequence namespace in a
common registry, rejects copied bundles as new runs, and writes separate outputs
per block and per run. Group permissions are inherited from the chosen output
parent; cross-user access still needs a test in the actual shared publication.

A bounded operator pilot requires its own passing preliminary certificate
(parity, timestep, consumer, fault-isolation checks) and is capped at 108 attempts,
one GPU and two hours. A production gate additionally needs the measured sustained
operator pilot. This avoids the circular requirement of needing production
approval just to test the production execution path. Neither certificate exists
yet. CPU fixtures that test rejection logic are explicitly synthetic, not approvals.

The launcher clamps work to the actual Slurm allocation end, reserves an hour
for drain/export, enforces a free-disk floor and isolates worker process groups
for owned-child shutdown. Its multi-GPU sustained behavior, end-to-end shared
publication and target throughput are **not yet measured**. In particular, this
first drag path has not inherited passive writer/cache speed certificates.

Remaining release work: resolve the measured numerical failures; finish and
review the queued independent-domain/count evidence; run the real bounded
operator pilot; publish group-readable evidence/bundles and verify a second
operator's access; only then issue a source-bound scale gate and production
commands. No production sbatch is approved by this document.

### Operator interface prepared for the eventual release

Shared assets already exist at:
`/data/group_data/katefgroup-ssd/world_model/particle_world_model/assets/mpm/fresh-10k-v1`.
They contain 10,000 training, 258 validation and 262 test shape UIDs. The drag
experiments do not create another duplicate asset bank.

Proposed shared campaign parent (not yet a published drag release):
`/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-drag-campaigns-v1/`.
Each operator creates a new named bundle there and uses the same `drag-reservations`
directory. Each bundle receives a unique namespace; worker/GPU count does not
change the sequence identity for a given prepared case index.

The commands below describe the interface, **not permission to bypass the missing
certificate**. `/path/to/reviewed/DRAG_SCALE_GATE.json` is intentionally a
placeholder. `bundle` does not submit; `submit --test-only` requests Slurm's
non-submitting validation; only `submit` creates the job.

```bash
python -m collection.mpm.drag.launch bundle \
  --output /new/shared/campaign-path \
  --registry /shared/common/drag-reservations \
  --assets /data/group_data/katefgroup-ssd/world_model/particle_world_model/assets/mpm/fresh-10k-v1 \
  --gate /path/to/reviewed/DRAG_SCALE_GATE.json \
  --python /data/group_data/katefgroup-ssd/world_model/particle_world_model/environments/mpm-genesis-v1/bin/python \
  --name drag-train-001 --split train --evaluation train \
  --object-counts 1 2 3 4 --material-layout homogeneous \
  --attempts 20000 --gpus 2 --hours 48
python -m collection.mpm.drag.launch verify --bundle /new/shared/campaign-path
python -m collection.mpm.drag.launch submit --test-only --bundle /new/shared/campaign-path
python -m collection.mpm.drag.launch status --bundle /new/shared/campaign-path
```

A bounded operator pilot instead uses `--purpose pilot --attempts 108 --gpus 1
--hours 2` and the separately reviewed pilot certificate. Regimes supported by
the sampler are train/train; validation or test with seen/unseen/count_seen/
count_unseen. Counts 1–4 belong to ordinary regimes and 5–7 to count regimes.
Sampling support is **not** qualification: the certificate must explicitly list
the requested regime. Mixed-material scenes likewise need separate approval.

The eventual training root is
`<campaign>/data/<split>/<evaluation>/`; filtered TARs live below per-block
directories. `collection/block-*/` retains inputs, raw trajectories, journals and
rejection receipts. `status` distinguishes retained raw clips from exported clips
and does not mistake a started receipt for evidence of a currently live process.
