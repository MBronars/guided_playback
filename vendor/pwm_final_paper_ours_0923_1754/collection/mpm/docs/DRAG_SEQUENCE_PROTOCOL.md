# Sand-free dragging: sequence-v3 protocol and qualification history

Updated 2026-09-15. This supersedes the *proposed collection distribution* in
DRAG_READINESS.md, not its historical experimental evidence. **Experimental;
not scale-approved.** No v1 sand pilot qualifies this release.

**Latest user decision:** after viewing the timestep comparison GIFs, use the
same timestep policy as passive MPM and proceed **pilot, then scale if the pilot
passes**. This chooses a fixed-resolution empirical dataset; it does not certify
timestep convergence. The current range uses 50 µs native steps. The existing
velocity-override actuator stays unchanged. See the broader pilot below.

Sequence v3 changes reverse to a **single backward leg**, opposite the most
recent horizontal stage. There is no forward leg or direction flip within an
event. Intervening lifts preserve that horizontal reference; consecutive reverse
events alternate direction. If no horizontal stage precedes a reverse, it moves
opposite its sampled reference azimuth. A pull samples a new horizontal heading.
Existing v1/v2 recordings retain their original out-and-back semantics; readers
dispatch by schema. Previously prepared inputs, GIFs and GPU qualification
results below are **v2 history**, not evidence that v3 has passed physics checks.
New preparation emits v3, and the changed variation/source fingerprints require
new qualification. The velocity-override actuator itself is unchanged.

## Confirmed implementation

| Dimension | Sequence-v3 contract |
| --- | --- |
| Materials | Elastic and plasticine only, including every distractor; no sand |
| Young's modulus | 10 kPa, 100 kPa, 1 MPa; numerical qualification still required |
| Other material parameters | Poisson ratio 0.2, density 1000 kg/m³; plasticine yield 10 kPa |
| Objects | Train/ordinary validation: selectable subset of 1–4; count evaluation: 5–7 |
| Shape splits | Existing UID-disjoint 10,000 train / 258 validation / 262 test banks |
| Geometry | 1,024 surface points/object, 12–20 cm nominal maximum full-mesh extent, random yaw |
| Layout | `drag-interaction-mixture-v1`: single-object controls; multi-object 30% separated / 50% close pair / 20% crowded |
| Patch | Closest 10% of the driven object's particles to its settled topmost point (103/1,024) |
| Number of events | 1, 2 or 3, each separated by an inactive release interval |
| Event families | Horizontal pull/release; vertical lift/hold/release; single-leg reverse/release |
| Speed | Continuous uniform draw in 0.05–0.60 m/s, independently for each event |
| Direction | Separated controls: random azimuth; interaction layouts: first horizontal stage aims at a neighbour; reverse still opposes previous horizontal stage; lift upward |
| Path bound | At most 0.25 m of **commanded path per event** |
| Observation interval | Exactly 0.0417 s, unchanged from passive MPM |
| Recorded states | 49 / 81 / 113 for 1 / 2 / 3 events |
| Recorded duration | 2.0016 / 3.3360 / 4.6704 s, excluding initial settling |
| Walls | Strictly wall-free side walls and ceiling in both settling and recorded motion |
| Floor | Normal contact allowed; penetration/collapse checks retained |

This is **material-patch velocity actuation, not a physical gripper**. Commands
are applied before each native P2G. Achieved displacement can differ from the
integral of command velocity. Hold means selected/active with zero commanded
velocity; release means inactive, with no override.

Every event uses the same material particle indices, even after release. No
patch reselection, object switching, contact-based regrasp or simultaneous
multiple grippers is claimed. Curved paths and smooth speed ramps are not yet
included; direction changes, holds and releases are explicit piecewise-constant
segments. The new interaction sampler changes placement and action sampling, not
the shared-grid MPM contact law or actuator. See the sampling policy below.

## Interaction sampling (2026-09-15)

The user requested more inter-object collisions and accepted the 30/50/20
mixture. `drag/interactions.py` implements that mixture **per multi-object
count**, over each ten occurrences of that count. Single-object scenes are
their own control stratum. With uniformly cycling 1–4 counts, the attempted
mixture is 25% single / 22.5% separated / 37.5% pair / 15% crowded. Physical
rejections can change retained proportions; contact misses are labelled rather
than hidden or resampled.

- Pair: driven object and a neighbour share a horizontal axis, initially
  35–55 mm apart in projected surface-cloud extent; additional distractors are
  in a separate row. Crowded: all objects form a close chain along that axis.
  With only two objects, crowded is a close pair, **not a multi-target chain**.
- Original surface clouds are translated, not deformed, shrunk or welded.
  The existing 30 mm minimum nearest-point initial spacing, 60 mm initial box
  margin and 15 mm floor clearance checks remain. No initial overlap is needed
  to seek later collisions.
- The first horizontal event is aimed along the neighbour axis. Every horizontal
  event/reference angle receives the same rotation, preserving v3 reverse
  semantics. An all-lift targeted sequence gets a final pull; this also changes
  the first family for a one-event all-lift scene. Both requested and actual
  first families are recorded; pure lifts remain in separated controls.
- The aimed event's moving duration is extended toward a 14 cm **command path**
  when its speed permits, still at most 20 transitions / 25 cm. Speeds remain
  continuously sampled over 0.05–0.60 m/s. Slow commands or irregular shapes may
  miss; commanded reach is not achieved motion.
- Outcome reports use saved-particle proximity (at least two frames within
  20 mm) plus subsequent target COM motion (at least 5 mm) as a **transfer
  candidate**, not an exact collision/force measurement. Pre-drive motion,
  initial/final gap, closest gap and all requested pairs are also recorded.
  Objects share a grid inside one world: this is not a newly validated explicit
  multi-body friction/contact solver. Visual separation and sticking behaviour
  need review before release.

The immutable targeted pilot is in
`runs/mpm-drag-interactions-20260915`: 24 preselected attempts and four serial
spot checks, with unchanged physical filters and passive clock. It is not the
old separated pilot relabelled as collision evidence. The launch gate now binds
the interaction policy/source and requires explicit interaction review.

## Exact event timing and sampling

For event index `i` (zero-based), start is uniformly sampled from observation
transitions `4 + 32*i`, `5 + 32*i`, `6 + 32*i`. Each event stays within a
24-transition active budget. The clip has `16 + 32*event_count` transitions.
Thus initial free evolution is at least 0.1668 s, consecutive events have at
least 0.2502 s of release, and the final release lasts at least 0.7506 s.

The continuous speed is drawn first. Moving duration is then drawn in whole
observation transitions subject to the path bound:

- Pull/lift/reverse: 6–20 moving transitions in one direction, limited by speed
  and available hold time. Reverse no longer requires an even duration.
- Lift: an additional 4–8 active zero-velocity hold transitions.
- Duration is not continuously sampled: integer frames keep command boundaries
  unambiguous and preserve the existing observation clock.

The default homogeneous schedule crosses object count × material × stiffness ×
event count × **requested first** family in 216 consecutive attempts. Targeted
all-lift replacements mean actual first families are not a full crossing.
Preparation uses
blocks of at most 108 (`--start 0`, then `--start 108`) without resetting the
index. Subsequent event families are independent random draws. This is not a
full crossing of every possible three-family sequence, speed, direction and
shape. Smaller pilots must describe their actual coverage, not claim the full
crossing. In mixed scenes material types alternate elastic/plasticine and
stiffness phases cycle over objects; this does not enumerate all combinations.

Independent run namespaces change shape permutation and random draws. Reusing
the namespace, indices and immutable bank reproduces geometry. Each manifest
records the realized events, assets, source hashes and exact clock. Rejections
remain in attempted-case accounting; no hidden redraw/top-up is performed.

## Numerical and data safety

All existing deformation, nonfinite, stencil, speed, floor and object-collapse
guards remain. Settling lasts 96 observation transitions (4.0032 s), and the
quiet-tail criterion must pass. Full physical state, including plastic history
and APIC state, is preserved into actuation and saved for controlled comparisons.

The solver uses its material-dependent **fixed** native timestep chosen before
the scene, not runtime adaptive stepping. The CFL estimate is not an accuracy
certificate. Sequences v2/v3 change neither passive simulation kernels nor the
constitutive laws. Faster commands need measured refinement tests.

Wall evidence covers every native step, grid participation and particle-plane
checks. Missing or inconsistent evidence fails validation. V2/v3 raw validation
itself rejects sand or side/ceiling contact, so passing a sand-permissive export
option cannot bypass this contract. The collector and operator also explicitly
use the strict wall filter. Native wall observation remains separate from
permitted floor contact.

The raw validator reconstructs every command and active interval from the
manifest. The adapter accepts only the versioned lengths, exact observation
interval, bounded speed, sand-free material metadata and strict-wall provenance.
Commands become velocity × measured frame interval, never future target motion.
Export groups equal-length/object-count rows; it does not pad distinct clip
lengths together. Both normalization profiles and autoregressive boundary
slices must be checked. V1 inputs remain readable without relaxing their old
0.30 m/s limit or 49-state contract.

## Qualification and release gates

Recorded historical v1 failures remain relevant warnings: stiff plasticine
showed timestep sensitivity and independently settled comparisons can select
different patch particles. Removing sand does not resolve plasticine accuracy.

Required before production approval:

1. Synthetic malformed-input, no-sand, strict-wall, path/timing and loader tests.
2. Real serial/grouped sequence smoke test, including >0.30 m/s commands.
3. Full-state/fixed-patch timestep controls and serial/grouped controls, so
   settling/selection variability is not confused with command execution.
4. Coverage of all families, 1–3 events, 1–4 objects, both materials/stiffness
   ranges, continuous low/high speeds; mixed scenes and held-out validation.
5. Real GIF review with fixed metric axes and command activity labels.
6. Actual exporter → both normalization profiles → model/rollout checks.
7. Bounded sustained operator pilot with measured retention, throughput and
   complete failure accounting; shared publication/permissions verification.

Unchanged diagnostic thresholds: paired execution position-vector RMS ≤0.1 mm
in both settling and actuation; same commands/selection; native first wall-event
times within two steps and contact presence unchanged. Timestep refinement:
whole-clip and driven-patch vector RMS ≤1 mm and maximum particle difference
≤5 mm, with independent wall/floor/collapse checks. A failed cell cannot be
silently dropped from the claimed material/speed/event capability.

For the user's fixed-resolution release path, same-timestep execution agreement
and all physical/data-quality filters remain required. The failed timestep
comparisons are explicitly retained as an accepted limitation of the chosen
resolution, not converted to passes or deleted. A future fixed-resolution
production certificate must encode that distinction. The operator now supports
separate `mpm-drag-{pilot,scale}-fixed-resolution-gate-v1` schemas. They require
the accepted-resolution decision, the failed diagnostic results and clock-source
hash bound into the evidence, with `checks.convergence` explicitly **false**;
coverage, physical quality, visual review and fixed-resolution review must all
be true alongside the usual operational checks. No such real gate has been
issued. The original convergence-
requiring gate remains unchanged and cannot be satisfied by relabelling these
results; the broader qualification pilot does not issue a production gate.

`collection.mpm.drag.launch` still requires a measured, source/bank-bound gate.
The gate must now also match the complete `protocol.VARIATION` object; a v1 or
slow/single-event-only certificate cannot authorize this distribution. The
0.30–0.60 m/s band is implemented for diagnostics, **not yet certified**.

## V3 release work (2026-09-14)

Current immutable run root: `runs/mpm-drag-v3-20260914/`.

- `snapshot/`: frozen v3 collector and consumer; worktree edits cannot change
  the running experiment. Preflight passed **316 collection tests** and
  **24 model-adapter tests**. The original v2 `drag-000000/raw.h5` also still
  passes schema/integrity validation; its original command meaning is preserved.
- `inputs-smoke/`: six new shapes, one object and three events per scene,
  crossing elastic/plasticine and 10 kPa / 100 kPa / 1 MPa. This is **not the
  same geometry or action draw as identically numbered v2 samples**. Commands
  include single-leg reverse, lift and speeds above 0.3 m/s.
- `qualify.py`, `queue/`: bounded, detached diagnostic queue in allocation
  10414730 on one explicit PRO 6000. It completes the historical command-off
  control, then v3 raw smoke and same-state serial/grouped + 50/25/12.5 µs
  controls. It cannot submit production or author an approval gate. Inspect
  `*-START.json`, `*-EXIT.json`, `COMPLETE.json`, `STOP.json` or `FAILURE.json`;
  missing results are not passes. The earlier v2 accuracy Slurm step was
  cancelled before its command-off control completed, so that incomplete
  directory remains historical rather than being overwritten.
- `review_results.py`, `reviews/`: bounded CPU review generates actual-particle
  GIFs, verifies export/loading under both stats profiles, and consolidates
  results into `READINESS.json`. It records first timestep-error crossings as
  well as whole-clip errors, without changing acceptance thresholds.
- `legacy-command-off/RESULT.json`: **passed** both refinement pairs for both
  stiff historical scenes. Whole-clip RMS is 0.145 / 0.236 mm (elastic) and
  0.039 / 0.070 mm (plasticine), for 50→25 / 25→12.5 µs. Selected-patch and
  maximum-particle bounds, floor, walls and collapse checks also pass. This
  narrows the historical problem to actively driven/deformed motion, but does
  not by itself prove an actuator bug or qualify v3.
- `smoke-soft/COMPLETE.json`: **4/4 retained**, spanning elastic/plasticine at
  10 and 100 kPa. `reviews/soft/REVIEW.json` verifies all four actual raw windows
  and commands exactly through export and both normalization profiles.
- `training-probe/RESULT.json`: **12/12 real-data integration arms pass**:
  two models × two profiles × windows starting at states 4, 36 and 68; each
  contains four real 1,024-point clips and three finite production optimizer
  and EMA updates. This is not an overfit or learned-rollout accuracy result.
- Partial v3 accuracy analysis, from the completed divisor 1/2/4 NPZ branches
  in `convergence-soft/`: all four cases fail **both** refinement pairs. The
  serial branch and remaining stiff tests were still running when this entry
  was recorded. Do not confuse completed branch comparisons with completion
  of the entire diagnostic queue.

| V3 case | Material / E | 50→25 µs whole-clip RMS | 25→12.5 µs whole-clip RMS |
| --- | --- | --- | --- |
| 000000 | Elastic / 10 kPa | 12.787 mm | 11.294 mm |
| 000001 | Plasticine / 10 kPa | 8.314 mm | 5.821 mm |
| 000002 | Elastic / 100 kPa | 13.816 mm | 9.886 mm |
| 000003 | Plasticine / 100 kPa | 4.501 mm | 3.454 mm |

Error exceeds 1 mm in framewise RMS during the first pull (0.209–0.292 s),
before any reverse event. The reverse scheduling fix therefore does not resolve
the numerical discrepancy. A separate diagnostic rigid-pose alignment reduces
the first-pair shape RMS to 0.354–0.471 mm: position/orientation differences
dominate, but alignment is **not** an allowed substitute for the declared
trajectory-accuracy criterion. No gate is relaxed based on this observation.

In the four raw smoke clips, the patch centroid's travel projected along each
command is within 1.5 mm of the commanded path, with no moving observation
frame travelling more than 0.1 mm backward against its command. This confirms
the single-leg visible action behavior, not timestep convergence or exact
kinematic grip. The first three new clips can be inspected here:

![New v3 one-direction-per-stage diagnostic samples](../../../runs/mpm-drag-v3-20260914/reviews/soft/drag-000.gif)

Release order after these diagnostics:

1. Preserve the timestep failures under the explicit fixed-resolution decision;
   resolve any same-timestep paired-execution failure. No silent material, speed
   or action exclusions, and no relaxation of the physical/data-quality checks.
2. Verify the full proposed training crossing (216 attempts), mixed-material
   held-out validation, and the requested count-evaluation scope separately.
   Keep test-bank objects out of tuning. Review actual GIFs and retention by
   action/material/count rather than only aggregate acceptance.
3. Check real exporter → both profiles → both models/rollout, fault isolation,
   and every newly allocated GPU. Issue a **pilot-only** source/bank-bound gate
   only once these measured prerequisites pass.
4. Run the existing bounded one-GPU/two-hour operator pilot, including graceful
   stop, attempt accounting, loading and retained-throughput measurement.
5. Verify shared publication/access and freeze a production bundle with the
   measured capability gate. `drag/collect.sbatch` and `drag/launch.py` already
   exist, but their existence is not a release certificate. Passive collection
   is unchanged and must not be repurposed to bypass the drag gate.

**Scale approval remains false.** Exact-position (kinematic) grip is a possible
separate actuator candidate, not an implemented or approved replacement for the
current pre-P2G velocity override. Changing it needs an explicit semantic
decision and independent qualification.

## Broader fixed-resolution pilot: started 2026-09-14

Root: `runs/mpm-drag-v3-pilot-20260914/`. `REQUEST.json` is the immutable
predeclared plan, source/bank hashes, acceptance criteria and user resolution
decision. This pilot uses the existing frozen v3 source, not mutable worktree
imports. `pilot.py` runs under allocation 10414730 on the explicit PRO 6000,
with a four-hour internal budget; it does not request another allocation or
alter passive collection.

- **216 training attempts:** full 1–4 object × two materials × three stiffness
  levels × 1–3 stages × three first-action-family crossing.
- **72 mixed-material unseen-validation attempts:** stratified blocks cover
  all first families, stage counts and 1–4 object counts; stiffness phase rotates.
  This is not a full mixed-material combinatorial crossing.
- **540 distinct training shapes / 156 validation shapes**, UID-disjoint.
  No test-bank assets are used. The 5–7-object capability is not claimed here.
- **24 preselected serial-reference cases** cover both materials, every object
  count and event count, with varied first families/stiffness. Same-case
  independent settling and commands are checked, rather than choosing easy
  comparisons after seeing results.
- Native GPU command isolation/corruption preflight has passed. Every physical
  rejection and technical failure remains in the attempt accounting.
- Predeclared operational targets: no unexplained technical failure, complete
  accounting, ≥80% overall retention and ≥50% in each declared marginal
  count/material/event/family/split stratum, unchanged same-timestep parity
  limits, and exact retained-data loader checks under both stats profiles.
  These retention targets are operational pilot decisions, not physics claims.
- Concurrent `review.py` audits completed matrices, exports retained data,
  checks both profiles, compares serial references and generates a gallery.
  Inspect `review/RESULT.json`, per-block reports and `review/GALLERY.md`.
  `ready_for_manual_release_review` is deliberately distinct from
  `scaling_approved`; machine success cannot replace actual GIF review.

After a passing pilot: review the GIFs and throughput, verify shared output
access, and verify the explicit fixed-resolution production gate and
operator bundle. Only then submit collection for the tested capability. A
process exit of zero, a pre-existing sbatch, or the user accepting visual
similarity alone is not a production certificate. Existing strict-convergence
gates and historical failed verdicts remain truthful.

## Historical v2 evidence

Evidence root: `runs/mpm-drag-sequence-20260914/` in the canonical worktree.

- `snapshot-v2/`: frozen collector and consumer used for the new GPU tests.
- `inputs-smoke/`: six one-object, three-event scenes crossing two materials
  and three stiffness levels. The first event is pull; subsequent families vary.
- `matrix-smoke/`: paired grouped/serial executions; inspect actual receipts.
- `smoke-slurm.log`: initial launch failed before Python because `env` resolved
  to an inaccessible unrelated PATH entry. No simulation attempt occurred.
- `smoke-slurm-launch2.log`: corrected launch uses `/usr/bin/env` explicitly.

At implementation preflight: 309 collection tests and 26 focused model/adapter
tests passed (separate collector and model environments). These are software
checks, not physical accuracy or production approval. GPU results are reported
separately as they complete; a running job or an existing script is not a pass.

Completed since preflight:

- Full model suite: **1,557 passed, 93 skipped, 10 deselected**, 25 subtests;
  one known CPU-bf16 RMSNorm dispatch warning. Other sessions are editing this
  worktree: this is the suite result at execution time, not a promise about
  subsequent unrelated changes.
- Grouped smoke: **6/6 retained**, including both 1 MPa cases, all three-event
  clips. Native guards, measured settling, strict walls, floor and collapse
  filters passed. Serial parity and timestep controls are separate tests.
- Paired smoke subsequently completed: **6/6 retained in serial too; 5/6
  execution comparisons pass**. The 1 MPa elastic case (`drag-000004`) has
  actuation RMS 0.163130 mm versus the 0.1 mm limit (maximum individual-point
  difference 0.868469 mm). Settling RMS is 0.013165 mm. Commands, patch selection
  and wall evidence agree exactly. All other cases pass, including this
  particular 1 MPa plasticine scene. See `reviews/smoke/AUDIT.json`; do not
  reinterpret a physical-filter pass as a numerical-parity pass. The stiff
  same-state control is needed to separate settling from actuation differences.
- `review-first-four/REVIEW.json`: **4/4 exported and read exactly** under both
  stats profiles; real 113-state GIFs (`drag-000.gif`, `drag-003.gif`).
- `training-smoke/RESULT.json`: **12/12 integration arms**, each three production
  optimizer updates and three EMA updates, using four real 1,024-point clips.
  Two models × two stats profiles × windows starting at states 4, 36, 68.
  No nonfinite/skipped updates. This is not an overfit convergence experiment.
- `PREPARED_COVERAGE.json`: **216 prepared scenes, 540 distinct training shapes,
  432 events with 432 distinct speeds**, minimum 0.050002 and maximum 0.599346 m/s;
  236 events above 0.30 m/s. Exactly 72 scenes for each event count. Every allowed
  first-family/count/material/stiffness/event-count combination appears; zero
  sand objects. Preparation is not simulation or retained-data coverage.
- `inputs-validation-full/`: 72 mixed-material unseen-validation inputs,
  spanning 1–4 objects and 1–3 events. First event is pull; later families vary.
  Test-bank assets have not been used for qualification/tuning.

### Accuracy-first replan at approximately 21:51 UTC

The stiff same-state control completed:

| Material (1 MPa) | 50→25 µs whole-clip RMS | 25→12.5 µs whole-clip RMS | Same-state serial/grouped RMS |
| --- | --- | --- | --- |
| Elastic | **39.723 mm — fail** | **14.195 mm — fail** | 0.010833 mm — pass |
| Plasticine | **1.308 mm — fail** | 0.941 mm — pass | 0.002452 mm — pass |

Selected-patch/max-point thresholds are checked separately. All branches pass
native wall/floor/collapse checks. The 25→12.5 µs plasticine pair also passes
its selected-patch/max-point limits; this is one scene, not a material-wide
certificate. Elastic divergence grows strongly around landing after its lift.
See `convergence-stiff/RESULT.json` and `stiff-timestep-comparison.gif`/`.png`.

This separates two questions: identical-state parallel execution is accurate
on these cases, but the coarse timestep does **not** reproduce a refined
trajectory for the new actuation distribution. Neither a physical-filter pass
nor similarity to a same-timestep serial reference establishes timestep accuracy.

The original finite `qualification_queue.py` was deliberately paused while its
active soft-material control was allowed to finish. **The 216-scene training
sweep and 72-scene validation sweep are held, not running or approved.**
`accuracy_queue.py` records retirement of only that paused supervisor after its
child drains, then runs identical-state refinements at divisors 4/8/16, a
command-off control, and soft-material refinements if needed. It has a two-hour
internal budget and a 2h15m Slurm step limit in allocation 10414730.

The original CPU review queue already audited/exported the smoke matrix; it
may report the intentional old-supervisor stop when waiting for the held sweep.
This is a documented replan, not an unexplained simulator crash. See
`runs/mpm-drag-sequence-20260914/REPLAN.md` and `accuracy-queue/` receipts.
Inspect scheduler/process liveness as well as receipts. No queue issues a
pilot/production approval, even if every command exits zero.

The soft control also completed. All four identical-state serial/grouped
comparisons pass (RMS below 0.0004 mm), but neither refinement pair passes:

| Material | E | 50→25 µs RMS | 25→12.5 µs RMS |
| --- | --- | --- | --- |
| Elastic | 10 kPa | 4.719 mm | 4.783 mm |
| Plasticine | 10 kPa | 13.524 mm | 18.705 mm |
| Elastic | 100 kPa | 4.969 mm | 10.935 mm |
| Plasticine | 100 kPa | 5.961 mm | 5.878 mm |

All corresponding wall/floor/collapse checks pass. Therefore this is not only
a high-stiffness issue, and simply reducing the maximum material stiffness is
not a supported fix. The refined controls and command-off control are meant
to discriminate timestep/actuation effects; no root cause has yet been proven.
The source review confirms pre-P2G commands are followed by ordinary G2P
velocity reconstruction, so the patch is **not** a kinematic velocity boundary.
Changing this into a kinematic gripper or a finite-response actuator would be
a distinct control protocol, not an invisible accuracy optimization.

The protocol implementation and tests were committed on `robert/clean` as
`8c01ea4`. The running numerical snapshot predates only the later strengthening
of the launch gate's critical-file list; its simulation, protocol and consumer
code are the tested frozen versions. No production certificate was issued.

## Reproduction tools

Use `collection.mpm.drag.cli freeze --consumer --output <new snapshot>` first.
Run its modules from `/tmp` with `PYTHONPATH=<snapshot>` so the changing working
tree cannot override the snapshot. Use explicit interpreter and GPU UUID.

Prepare with `collection.mpm.drag.cli prepare`; new diagnostic controls include
`--event-counts 1 2 3` and `--speed-range 0.05 0.60`, in addition to object count,
split and material layout. Use separate immutable outputs for low-speed and
high-speed experiments. Do not overwrite a completed or failed run.

`collection.mpm.drag.matrix` runs serial/grouped comparisons;
`collection.mpm.drag.audit` checks complete case accounting and raw journals;
`collection.mpm.drag.review` produces GIFs and loader checks;
`collection.mpm.drag.convergence --serial-reference` compares identical full
settled states/commands as well as timestep refinement.

The production operator remains `collection.mpm.drag.launch` with
`collection/mpm/drag/collect.sbatch`. No production job has been authorized by
this document and no ready-to-use approval certificate exists.
