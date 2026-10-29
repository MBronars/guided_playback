# Dragging particle detachment: mitigation and quarantine

September 15, 2026. This investigation responds to visible particle clusters
being pulled off objects. **Not a production scaling certificate.**

Subsequent work: [gentle-v4 integration and qualification](DRAG_GENTLE_V4_RELEASE.md).
The statements below about production defaults describe this ablation's original
handoff; new preparation now emits explicitly versioned gentle v4.

## What changed

New dragging preparation now pins `saved-point-fragmentation-review-v1` in the
case configuration hash, manifest and plan. Validation screens both settling
and driven saved states. Persistent new spatial splits reject that clip;
export provenance records the policy. The check is additive: no wall, collapse,
material, timestep or command-validation threshold was relaxed. Old manifests,
raw files, original pilot receipts and passive collection are unchanged.

The default production action remains sequence-v3's 10% fixed material patch.
**Gentler commands and the 20% patch are diagnostic alternatives, not silently
installed production defaults or a physical force-limited gripper.**

Implementation:

- `collection/mpm/drag/fragmentation.py`: pure saved-point screening.
- `collection/mpm/drag/prepare.py`: policy/identity binding for new preparation.
- `collection/mpm/validate.py`: pinned-policy validation, rejection and provenance.
- `collection/mpm/drag/actuation_ablation.py`: reproducible exact-state GPU comparison.
- `collection/mpm/tests/test_mpm_drag_fragmentation.py`: synthetic and raw-filter tests.

## Screening rule and limitations

For each object independently, the proximity radius is the larger of twice the
grid spacing and 2.5 times the median initial third-nearest-neighbor distance.
At g256 the grid contribution is 12.207 mm. Actual per-object thresholds are
recorded, rather than using a single arbitrary distance across every shape.

Track each initially connected component separately. If a new non-largest
component contains at least `max(4, ceil(0.01 * object_particle_count))` particles,
track those same separated material points. At least that many points must
remain separated for three consecutive saved states to flag the object.
Existing disconnected source parts are not counted as *new* fragmentation;
another object's proximity cannot reconnect a fragment. This is a quarantine
heuristic, **not calibrated fracture or mesh-topology reconstruction**. Sparse
sampling and sufficiently large continuous stretch can still trigger it;
small fragments or short detachments can be missed. An unflagged clip is not
automatically physically correct.

The screen flags these **7/31 original retained clips**:
`000100, 000109, 000118, 000134, 000143, 000226, 000227`.
Their original historical "retained" labels are preserved. See
[the original scan](../../../runs/mpm-drag-fragmentation-20260915/ORIGINAL_SCAN.json).

The end-to-end counterfactual validator/export-selection check on temporary
copies retains exactly **24/31**, rejecting those same seven; no additional
settling fragmentation was found. Temporary test copies were removed and no
original data was deleted. See
[FILTER_RECHECK.json](../../../runs/mpm-drag-fragmentation-20260915/FILTER_RECHECK.json).

## Exact-state experiment on dev-blackwelld

Allocation 10446119, node `babel-l9-20`, two PRO6000 GPUs. Four previously
visually flagged scenes (`109, 134, 143, 226`) and one non-lift control (`175`)
were declared before this comparison. This is targeted diagnosis, **not a
random-sample retention or throughput estimate**.

Each scene runs four independent worlds from the exact same complete native-g256
settled checkpoint: positions, float64 master positions, velocities, F/F_trial,
C, plastic state and per-particle material coefficients. Every array is checked
equal before stepping. Geometry, materials, native timestep, grid, event timing,
directions and physical boundaries remain unchanged. Native guards and wall
observation remain enabled for all four arms.

| Variant | Changed from baseline |
|---|---|
| baseline | Nothing: original recorded patch and commands |
| patch20 | Topmost-point spatial patch expands from 10% to 20% |
| gentle | Original patch; speed capped at 0.25 m/s with three-transition ramps |
| patch20_gentle | Both changes |

The ramp multipliers rise `1/3, 2/3, 1` and fall symmetrically within each
moving stage; the hold remains active with zero velocity and release remains
inactive. **Gentle arms deliberately shorten the command path.** They are not
a same-task accuracy comparison, nor evidence that long aggressive lifts have
been repaired. Velocity overrides themselves remain unmodified.

## Measured lift-case results

| Scene | Baseline fragment flag | 20% patch only | Gentle | 20% + gentle | Ungrasped-body maximum lift, baseline → gentle → combined |
|---|---|---|---|---|---|
| 000109 | Yes | Yes | No | No | 19.6 → 5.1 → 10.7 mm |
| 000134 | Yes | Yes | No | No | 91.7 → 36.7 → 43.5 mm |
| 000143 | Yes | Yes | No | No | 107.1 → 61.2 → 64.1 mm |

The lift metric always measures the **same original unselected material points**
in all arms. It is maximum mean height change, not a guarantee that every point
clears the table. In 109, mitigation mostly gives modest local deformation;
the reduced motion is a real tradeoff. A larger patch alone does not fix tearing
and can pull off a larger cluster. No material or positional "glue" was added.

The elastic pulling case **000226 still flags in every arm**. Its no-command
g256 control does **not** flag (maximum individual-particle displacement about
9.1 mm over the clip). This supports an actuation-associated failure for this
case, but does not establish whether the culprit is contact discretization,
surface-sample support, shape topology, or the velocity override. Quarantine
remains necessary; the command mitigation is not universal.

## Visual comparisons

All **20 runs completed**. All passed the unchanged wall and collapse checks.
Control 000175 has no fragmentation flag in any arm; whole-object excursion is
337.5 / 420.8 / 145.0 / 175.3 mm in the four variants, again showing that the
gentle variants change the motion distribution. This is not a full simulation
accuracy or serial/batched equivalence certificate.

All images use actual simulated particles. Columns are baseline, larger patch,
gentle commands, and both changes. XY above / XZ below, fixed metric axes,
black marks the selected material patch, playback 2× slow. The flag shown in
each title is for the **whole clip**, not only that displayed frame.

- [000109 comparison GIF](../../../runs/mpm-drag-fragmentation-20260915/drag-000109/COMPARISON.gif)
- [000134 comparison GIF](../../../runs/mpm-drag-fragmentation-20260915/drag-000134/COMPARISON.gif)
- [000143 comparison GIF](../../../runs/mpm-drag-fragmentation-20260915/drag-000143/COMPARISON.gif)
- [000226 comparison GIF](../../../runs/mpm-drag-fragmentation-20260915/drag-000226/COMPARISON.gif)
- [000175 non-lift control GIF](../../../runs/mpm-drag-fragmentation-20260915/drag-000175/COMPARISON.gif)

## Evidence and readiness

Evidence root: `runs/mpm-drag-fragmentation-20260915/`.

[SUMMARY.json](../../../runs/mpm-drag-fragmentation-20260915/SUMMARY.json)
binds all 20 result/trajectory pairs, five GIFs, test receipts, filter recheck,
and final canonical filter hashes. All bounded GPU experiments have completed.

- `REQUEST.json` and `snapshot/SNAPSHOT.json`: declared cases, frozen source.
- `drag-*/REQUEST.json`: input/checkpoint hashes and exact modifications.
- `drag-*/{variant}/TRAJECTORY.npz`, `RESULT.json`: trajectories, wall evidence,
  screening, collapse diagnostics and achieved-motion measurements.
- `no-command-226/`: separately bounded residual-motion control; diagnostic
  path, not a production native-wall certificate.
- `FILTER_CPU_TESTS.xml`: 392 collection tests passed.
- `ADAPTER_TESTS.xml`: 24 training-adapter tests passed.

The experimental simulator source snapshot predates wiring the additive filter
into the canonical preparer/validator; its detector and stepping implementation
are unchanged. Numerical runs and CPU integration tests therefore have separate
explicit source bindings. No old pilot snapshot has been modified.

Before adopting gentler production actions, version their exact command contract
and review the reduced movement/interaction coverage. A production sampler
should sample continuous speeds in its chosen range, not accidentally create a
large point mass at the diagnostic cap. Qualify new native starts and loader
round-trips for that version. Keep the residual-failure filter and retain all
attempt/rejection accounting. The earlier serial/batched parity failure and
shared-grid contact limitations remain separate unresolved questions.
