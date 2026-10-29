# Native-g256 dragging collection

## Current status — September 15, 2026

Follow-up: [particle detachment mitigation and filtering](DRAG_FRAGMENTATION_FIX.md)
flags 7 of the 31 historically retained clips. The earlier filters did not screen
fragmentation. Retention numbers below preserve the original pilot, not a claim
that all 31 clips are acceptable intact-object examples.

The two-GPU qualification ran inside **Slurm allocation 10446119,
`dev-blackwelld`, node `babel-l9-20`**. `dev-blackwelld` is the job name, not an
SSH hostname. The passive campaign and other allocations are untouched.

**Scaling is currently blocked by two measured serial/batched disagreements.**
The core pilot retained 21/24 clips (87.5%); the follow-on retained 10/12 (83.3%),
for **31 retained clips from 36 attempts, 2,255 recorded states**. All 31 passed the
strict export and both normalization-profile loader checks. However:

| Paired scene | Outcome |
|---|---|
| 000025 | Both reject settling; agreement on rejection, not trajectory parity |
| 000175 | Pass: 0.00694 mm driven RMS; wall evidence identical |
| 000043 | Fail: grouped accepts settling; serial rejects the last transition |
| 000143 | Fail: 1.963 mm driven RMS, above the unchanged 0.1 mm bound |

For 000043, the final maximum displacement is 0.564 mm grouped versus 0.606 mm
serial, against the 0.600 mm settling limit. Settling trajectory RMS is only
0.0155 mm, but the retained/rejected decision differs and the gate remains failed.

For 000143, settling RMS is 0.00235 mm, but driven RMS reaches 1.963 mm across
the clip; the largest individual-particle difference is 97.3 mm. The first frame
with scene RMS over 0.1 mm is frame 41 (1.7097 seconds). Most error is on the
driven plasticine object, and later interaction also differs. Wall evidence and
retention agree. This is not explained away as a wall-timestamp discrepancy.

A bounded reference-block-size/unfused follow-up completed after the second
GPU's follow-on batch. Its 000143 comparison **also fails**, at 1.272 mm driven
RMS and 109.1 mm maximum individual-particle difference. Simply disabling the
optimizations has therefore not demonstrated a parity fix. An unchanged serial
repeat of 000143 on the other GPU **also fails**: 1.344 mm driven RMS, maximum
individual-particle difference 111.4 mm. Settling RMS is only 0.00330 mm.

An unchanged serial repeat of 000043 on the **same** GPU changes its settling
decision from rejected to retained: final maximum displacement 0.606 mm versus
0.544 mm. The reference-graph variant also retains it at 0.595 mm. Therefore the
fixed trajectory tolerance and exact discrete-selection criterion are not met
by repeated executions of the baseline itself, not just by the optimized path.
One repeat is not a distributional equivalence test and does not exonerate all
optimization changes or establish physical accuracy.
These diagnostics preserve the failed optimized results and change neither
material properties nor timestep nor physical filters. They are not a production
approval.

The user's choice to proceed with g256 accepts the documented shared-grid
approximation, not a fabricated passing numerical certificate. No production
gate or scale bundle has been published. The previous g100 gate cannot authorize
this resolution.

Completed diagnostics and limits:
[`DIAGNOSTIC_REVIEW.json`](../../../runs/mpm-drag-g256-native-20260915/DIAGNOSTIC_REVIEW.json).

## Next decision before scaling

Do not increase the 0.1 mm threshold merely to make this pilot pass. Preserve
the original failed result. The proposed next qualification should instead:

1. Compare serial and optimized updates from the **same complete physical
   checkpoint and commands**, over one native step and short horizons. Include
   pre-lift, stretched-neck and pre-interaction states, not only a quiet start.
2. Predeclare repeated serial and grouped full rollouts to measure baseline
   repeatability and any extra discrepancy introduced by optimization. Use
   independent, declared cases; do not set acceptance bounds from one observed
   1.344 mm repeat and immediately certify the 1.963 mm optimized result.
3. Treat near-threshold settling outcomes as a separately measured admission
   stability issue. Any conservative exclusion/margin policy must be explicit,
   versioned and tested; do not replace this pilot's rejection history.
4. Retain the same material range, native clock, wall/collapse checks, command
   protocol and exact export/loader contracts. Contact/fracture fidelity still
   needs to be distinguished from implementation equivalence.

This changes the **qualification design**, not a passed/failed flag after the
fact. It requires review before the two-day production launch. No production
gate, shared scale bundle or 48-hour job was created during this investigation.

Local evidence and scripts:

```text
runs/mpm-drag-g256-native-20260915/
  snapshot/           # frozen collection + consumer implementation
  REQUEST.json        # candidates, exact attempted labels, acceptance thresholds
  pilot.py            # bounded two-GPU supervisor
  grouped/            # grouped native-grid collection and rejected attempts
  paired/             # independent serial comparisons
  FAULT-{0,1}.json    # command/fault isolation checks on both GPUs
  review.py           # raw integrity, strict export, exact loader checks, parity
  visualize.py        # object-coloured actual-particle GIFs and keyframes
  stage.py            # publish only after actual completed and reviewed checks
  start.py            # existing-allocation launch, requires a real verified gate
  CPU_TESTS.xml       # 374 collection tests passed
  ADAPTER_TESTS.xml   # 24 training-adapter tests passed
  operator/           # separately frozen non-numerical sbatch entry point
```

## Protocol

- Start directly on g256, including all 96 settling transitions. Do not transplant
  a g100-settled checkpoint. Physical box stays 1.5625 m; cell width is 6.1035 mm.
  Original physical floor and wall locations are unchanged.
- The stability clock now uses actual cell width. The tested 10 kPa, 100 kPa and
  1 MPa regimes retain 50 microsecond native steps, 834 per saved transition,
  giving 0.0417 seconds between states. No claim that 10 MPa has been qualified.
- Use the existing immutable 10,000-training-shape bank. Train split only;
  1–4 objects; homogeneous elastic or plasticine; no sand. Held-out shape banks
  stay separate and are not qualified for dragging by this test.
- Keep sequence-v3 action sampling, continuous speeds, one direction per stage,
  interaction-seeking layouts, strict wall rejection, deformation/collapse
  checks and quiet-start requirements. See [sequence protocol](DRAG_SEQUENCE_PROTOCOL.md).
- Twenty-four predeclared core attempts cover every count × material × stiffness
  combination once. Twelve separately declared follow-on attempts check continued
  execution and export. Unused prepared candidates do not count as attempts.
- Four selected scenes also run independently, serially. Compare complete
  settling and driven position arrays: vector RMS ≤0.1 mm in each phase,
  unchanged retained/rejected decisions, and wall first-contact times within
  two native steps with unchanged contact presence. Matching settling rejections
  are reported separately, not counted as trajectory comparisons; at least two
  actual trajectory comparisons are required.
- Predeclared retention: at least 80% in both core and follow-on groups.
  Rejected attempts remain on disk; no seed replacement to improve the score.
- Export only strictly accepted clips. Both existing normalization profiles must
  reproduce exact raw position windows and patch commands through the active
  loader; the passive loader must reject active data.

The 12 follow-on attempts are a small operational check, not a statistically
precise retention estimate or a long-duration reliability benchmark.

## Contact limitations remain explicit

The [g256 comparison](DRAG_CONTACT_256.md) showed reduced premature coupling,
not elimination of sticking or proof of physically exact contact. Finer grid
cells with unchanged particle count also change particle occupancy. Neither
timestep nor joint particle/grid convergence is established. Saved particle
distance and shared-grid mass overlap are diagnostics, not mesh contact forces.

The [native pilot gallery](../../../runs/mpm-drag-g256-native-20260915/GALLERY.md)
includes scene 000143, where a driven plasticine patch separates from some of
its parent object. The numerical filters do not reject that behaviour, but this
is not a calibrated fracture model. Do not interpret a retention pass as proof
of validated contact, fracture, or permanent rest of every undriven particle.
All 31 retained clips have object-coloured GIFs and six sampled keyframes;
all keyframes were inspected. The recorded visual review remains unapproved for
production because numerical parity failed. Component separation also appears
in an elastic follow-on example (000226), so it is not solely a plasticine issue.

Final measured report:
[`review-final/RESULT.json`](../../../runs/mpm-drag-g256-native-20260915/review-final/RESULT.json).
The [actual failed-gate test](../../../runs/mpm-drag-g256-native-20260915/FAILED_GATE_TEST.json)
confirms that this report cannot publish a production certificate or bundle.

## Conditional production layout

Once the completed pilot and actual keyframes are reviewed, `stage.py` can
publish a **new** source-bound gate and immutable two-GPU, 48-hour bundle. It
refuses incomplete checks, changed sources, missing CPU regression, missing
visual inspection or changed assets. It does not submit anything itself.

Planned shared roots (do not assume they exist until publication succeeds):

```text
/data/group_data/katefgroup-ssd/world_model/particle_world_model/
  assets/mpm/fresh-10k-v1/
  evidence/mpm-drag-g256-20260915-v1/
  staging/mpm-drag-g256-20260915-v1/
    source/
    GATE.json
    BUNDLE.json
    collect.sbatch
    collection/block-*/
    data/train/train/block-*/diagnostic-export/
```

The 20,000-attempt cap is a ceiling, not a promised yield. Runtime is bounded by
48 hours, allocation expiry, disk headroom and shutdown reserve. Each GPU runs
one lane with groups of up to four independent scenes. Namespaces and block
indices prevent cross-lane sequence duplication. No existing campaign is
overwritten or silently changed from g100 to g256.

Completed optimized groups used 2,999.9 aggregate GPU-seconds for 36 attempts.
Ideal balanced two-GPU extrapolation is about 86 attempts/hour, or 74 retained
clips/hour at this small pilot's retention. That excludes export/review,
startup and load-imbalance overhead and is **not** a sustained throughput
qualification or guaranteed two-day yield. A few thousand retained clips is the
appropriate order of magnitude, not 20,000 promised clips.

The actual diagnostic training-format shards are currently local to this pilot:
`runs/mpm-drag-g256-native-20260915/review-final/filtered/diagnostic-export/`.
They are not a released production dataset. The planned shared paths above do
not exist yet.

For a published bundle, use its frozen source to verify or run it. The canonical
batch entry point remains `collection/mpm/drag/collect.sbatch`; inside the
existing `dev-blackwelld` allocation, a two-GPU `srun` can invoke that entry point
without submitting another node allocation. Do not run directly on unallocated
GPUs or launch a bundle before its real gate exists.

The exported `data/train/train` root is recursively shard-discoverable by the
unified **drag** adapter (`task: drag`). This does not add guidance support to
the passive model or train a model automatically.
