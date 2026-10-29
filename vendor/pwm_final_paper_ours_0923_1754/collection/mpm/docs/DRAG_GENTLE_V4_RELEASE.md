# Gentle-v4 dragging: qualification and bounded collection

September 15, 2026. User requested autonomous preparation and scaling on
**dev-blackwellf**. This is distinct from dev-blackwelld.

## September 15 recovery update

The original step **10447018.28 stopped at 13:17:24 UTC** on an incorrect
normalization round-trip assertion, not a simulator failure. Its early running
observations below are historical. The original bundle and failure are preserved.

All **334 retained clips** from its 444 completed attempts are now in a separate
verified recovery dataset, including **86 previously pending clips**. The fix
changes only validation and export orchestration, not physics, statistics, or
training normalization. **458 regression tests passed**, and the revised export
operator also published ten real clips successfully after consumer review.

A new bounded **36-hour** collection started at **18:30 UTC**, step
**10447018.60**, on the same two dev-blackwellf GPUs. See
[the recovery report](DRAG_RECOVERY_20260915.md) for live observations, exact
paths, evidence, and failure-handling behavior. At **18:39 UTC**, both lanes had
completed their first groups: **7/8 new raw clips retained, 471 driven states**,
with both GPUs continuing at **96% / 97% utilization**. New TAR blocks had not
completed yet. Submission is not completion.

## Qualification result and launch

Fresh-start qualification **passed**. The 48-hour production collection was
submitted at **07:38:59 UTC September 15** inside allocation **10447018**,
step **10447018.28**, node `babel-l9-16`, on its two allocated PRO6000 GPUs:

- `GPU-6140b8fb-5dc7-9f93-4b30-24fd978074bf`
- `GPU-ec4125d0-5595-2372-f0a9-18864219d774`

The allocation has 64 CPUs / 64 GiB; the pilot and collection use 16 CPUs /
48 GiB. The production supervisor and both worker lanes started; both startup
logs identify PRO6000 devices and the frozen shared bundle. **Both fresh-device
startup probes passed.** At **07:47 UTC**, the first two production groups had
completed **8 attempts, retaining 6 clips / 390 driven saved states**; the other
two attempts failed settling. Both workers continued to the next groups, with
no worker failure receipts. Both GPUs were measured at **97% utilization** during
the first groups. `runs/mpm-drag-gentle-v4-20260915/PRODUCTION_LIVENESS.json`
records the verified bundle and first-group evidence. These are time-stamped
observations, not a claim of continuous agent monitoring after handoff.

No production TAR block had completed at that observation; shards are published
after their 108-attempt block completes. The six raw clips above are distinct
from the 24 already exported pilot clips below.

The passive campaign and all other allocations are untouched. No failed v3
verdict was rewritten as a pass.

- Core: **16/24 retained**. Follow-on: **8/12 retained**.
- Total: **24/36 retained**, **1,560 driven saved states**, exported into ten
  diagnostic TAR shards. These are pilot outputs, not production counts.
- Ten settling exclusions; two persistent spatial-split exclusions. No attempt
  was replaced to raise the retention rate.
- Both retained serial/grouped comparisons passed: driven vector RMS
  **0.000188 mm** (000025) and **0.005639 mm** (000043), versus the unchanged
  0.1 mm limit. Settling RMS also passed, commands were exact, and wall-event
  arrays matched. The other two pairs agreed on exclusions; they are not
  retained-trajectory parity evidence.
- All 24 retained episodes loaded with exact raw-window/command agreement under
  both normalization profiles. Both GPU fault-isolation probes passed.
- **432 CPU tests passed** (406 collector + 26 adapter), zero skips/failures.
- All 24 retained clips received explicit sampled-keyframe inspection. All eight
  final measured checks passed before visual approval, gate publication or launch.

Evidence: `runs/mpm-drag-gentle-v4-20260915/review-final/RESULT.json`,
`VISUAL_REVIEW.json`, and the shared evidence/bundle paths below.

## Versioned protocol now implemented

`mpm-drag-command-sequence-v4` is the new preparation default. Readers retain
explicit support for v1, v2 and v3; their recorded command semantics are unchanged.

- Native **g256**, physical box 1.5625 m, original walls and floor.
- Same material-dependent clock; current 10 kPa / 100 kPa / 1 MPa range uses
  **50 microseconds × 834 native steps = 0.0417 s per recorded transition**.
- Elastic or plasticine objects only, including distractors; no sand.
- Train shapes from the existing immutable **10,000-shape training bank**.
- **1–4 objects**, homogeneous material layout for this qualification.
- **1–3 events**, with pull, lift/hold/release and single-leg reverse families.
- One fixed topmost spatial **10% particle patch**, no material-index reselection.
- Event peak speeds sampled **continuously uniformly from 0.05–0.25 m/s**.
  No clipping of a 0.6 m/s draw, hence no artificial probability spike at 0.25.
- Moving-stage frame multipliers rise `1/3, 2/3, 1` and fall symmetrically.
  Holds remain active with zero velocity; release is inactive. Commands still
  act as pre-P2G velocity overrides, not a physical force-limited robot.
- 49 / 81 / 113 recorded states, depending on event count. The ramp reduces
  displacement at fixed peak speed and duration; interaction is not guaranteed.
- Mandatory pinned `saved-point-fragmentation-review-v1`, applied during both
  settling and driven motion. Existing collapse, wall, nonfinite, stencil,
  advective-speed and export/consumer checks are retained.

The [actuation ablation](DRAG_FRAGMENTATION_FIX.md) motivates this candidate:
three tested lift failures stopped triggering fragmentation with gentler commands;
an elastic pull remained problematic. The new filter excludes remaining flagged
clips. Neither the ablation nor the filter establishes calibrated fracture physics.

## Predeclared fresh-start qualification

Evidence root: `runs/mpm-drag-gentle-v4-20260915/`.

The core has 24 declared attempts, crossing all four object counts, both
materials and all three stiffness levels. A further 12 attempts test continued
collection. Inputs, source hashes, exact labels and criteria are in `REQUEST.json`.
228 inputs are prepared for deterministic selection; only the declared 36 are
pilot attempts. Rejected scenes are not replaced or hidden.

Four predeclared scenes also run independently in serial. Retained paired
trajectories must satisfy the existing **0.1 mm vector RMS bound in both settling
and driven phases**, exact commands/selection, identical wall-contact presence,
and at most two native steps of wall-event timing difference. At least two
comparisons must contain retained trajectories. Matching exclusions only establish
selection agreement; they are not counted as retained-trajectory equivalence.

The prospective retention/admission floor is **60% separately for core and
follow-on**, because fragment rejection now removes scenes the old filter kept.
This is an efficiency criterion, not permission to accept physically rejected
clips or a retroactive change to the old v3 pilot's 80% criterion. Every retained
clip must pass all current filters; malformed/unknown failures block publication.

Also required: real-GPU command/fault isolation on both devices, exact export and
training-loader checks under both frozen normalization profiles, complete attempt
accounting, source/asset integrity, and review of retained-particle visualizations.

CPU results: **406 collection tests and 26 training-adapter tests pass**. The
operator negative test confirms the failed historical gate still cannot publish
a bundle. These tests do not replace GPU qualification.

## Actual pilot visualizations

[Complete gallery: all 24 retained and both fragmentation-rejected clips](../../../runs/mpm-drag-gentle-v4-20260915/GALLERY.md).

These are actual retained saved particles, not generated illustrations. Colors
identify objects; black is the commanded patch. XY and XZ views use fixed metric
axes, and GIF playback is 2× slow. Sampled keyframes are reviewed separately from
the automatic fragmentation check. They do not establish exact mesh contact.

- [Bulk lift and return, 000184](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-007/visuals/drag-000184.gif)
  ([keyframes](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-007/visuals/drag-000184.png)).
- [Approach and neighbor motion, 000018](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-001/visuals/drag-000018.gif)
  ([keyframes](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-001/visuals/drag-000018.png)).
- [Three-event branched object, 000068](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-003/visuals/drag-000068.gif)
  ([keyframes](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-003/visuals/drag-000068.png)).
- [Local soft-object lift, 000084](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-000/visuals/drag-000084.gif)
  ([keyframes](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-000/visuals/drag-000084.png)).

The last example illustrates an intentional limitation: lifting a patch does not
guarantee lifting the whole object. The filter rejects persistent newly separated
large point clusters, not all deformation or every small possible detachment.

[000200](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-004/visuals/drag-000200.gif)
([keyframes](../../../runs/mpm-drag-gentle-v4-20260915/previews/group-004/visuals/drag-000200.png))
is a retained but strongly stretched example: the last pull forms a narrow neck.
The nearest selected-to-unselected same-object gap stays at or below 5.161 mm
(5.153 mm initially), and the proximity-component filter passes. This does not
prove continuous mesh connectivity or calibrated elastic response. Do not describe
the release as eliminating all necking, small detachment, or material artifacts.

## Production location and operator

Published shared roots:

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/
  assets/mpm/fresh-10k-v1/                   # existing immutable shape banks
  evidence/mpm-drag-gentle-v4-20260915/      # passing gate + evidence
  staging/mpm-drag-gentle-v4-20260915/       # frozen run bundle
    source/                                # tested collector + consumer
    collection/block-*/                    # inputs, raw states, rejection receipts
    data/train/train/block-*/diagnostic-export/*.tar
    lane-0.log
    lane-1.log
```

The production cap is **48 hours on these two GPUs**, up to 20,000
attempts; it is a cap, **not a prediction of 20,000 clips**. A fresh namespace
reservation prevents duplicate sequence identities. The operator respects the
allocation deadline, shutdown reserve, disk admission and GPU leases, and records
each block's rejected/retained counts. It never cancels the allocation itself.

The existing `collection/mpm/drag/collect.sbatch` bootstrap is separately
syntax-checked and hash-bound. For this request `start.py` attaches that bootstrap
with `srun` inside allocation 10447018, rather than requesting another node.
`stage.py` refuses publication without all actual checks and visual approval.

The reserved production namespace is
`drag-gentle-v4-sep15-a6b8e6dc7bf842c69f023df5ff6c3a55`; the bundle SHA-256 is
`73c1ae62cd936ca10e052dd3475ea9de3ee92c5203061bd9f49614219dfaa6cb`.
Do not rerun or copy this consumed bundle as a new collection. The canonical
bundle command can create a new namespace against the same passing gate, only
within its qualified capabilities.

Read-only counts (raw retained clips are distinguished from exported clips):

```bash
python runs/mpm-drag-gentle-v4-20260915/status.py \
  /data/group_data/katefgroup-ssd/world_model/particle_world_model/staging/mpm-drag-gentle-v4-20260915
squeue --steps -j 10447018
```

TAR exports appear after a 108-attempt block finishes. A growing raw journal is
evidence of live generation, not yet a published training shard. The supervisor
stops both of its own worker process groups on a lane failure; it does not cancel
the allocation or unrelated jobs. Failure receipts and partial outputs are preserved.

The data adapter recursively discovers TAR shards under the train root; active
dragging must be selected explicitly (`task: drag`). Evaluation requires its own
disjoint held-out collection; this pilot does not qualify 5–7 objects, mixed
materials, 10 MPa, or use of training clips as validation/test data.

Shared-grid premature coupling/sticking and unestablished timestep/grid/particle
convergence remain disclosed limitations of the user's g256 method choice.
