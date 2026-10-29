# Goal and hold semantics in the September 15 guidance study

The primary experiment asks for a specific task, not merely reproduction of a
successful source clip. Picking targets the observed object template origin
plus10cm vertically, within20mm throughout the final six frames. Placing targets
the separately recorded requested pose, with rest, release and settling over
that interval. All primary trajectories, targets and criteria stay unchanged.

A reference audit of all16 pinned scenes finds **0/16 recorded clips pass this
complete task screen**. This does not establish that those recorded motions are
physically wrong. The recorded picks lift136.7–408.9mm and move34.2–74.3mm in XY;
they do not stop at the requested point10cm above the anchor. Five of eight
pass the interaction-only screen; three exceed the final relative-motion limits.
Every recorded placing clip retains mesh contact at the fingers during the final
interval, so every one fails the requested5mm release clearance. Some additionally
miss the requested final pose or settling limits.

Receipts: `validation/guidance_noise_2026-09-15/recorded_task_screen_v1/summary.json`
and `source_reward_and_motion.json`. These read the recorded reference only for
evaluation. They do not feed recorded future information into generation,
candidate selection or noise optimization. The replay-controller qualification
is separate: it asks whether an independent engine reproduces recorded motion,
not whether that motion fulfills our stricter requested task.

The currently available collector code filters windows by a peak reward, and
the current placing environment can count reaching the pose as success without
requiring release. These are clues to task semantics, not proof of the exact
configuration that produced the old shards. Pick rewards in these saved clips
peak near500, while placing rewards exceed4500; do not infer a common1000
threshold from the collector's current default. The exact original launch
configuration has not been recovered in this audit.

Interpret zero primary-task passes alongside continuous collision/contact and
goal metrics. The experiment demonstrates limitations on precise held-goal
guidance; it does not measure the source task's own success rate. A future
minimum-height lifting task or a placing task that allows retained contact would
be a different experiment, with a separately declared target and success rule.
Likewise, true released placing requires appropriate demonstrations or evidence
that planning can reliably produce that behavior from the existing model.

## Optional mean-held goal cost for picking

`--pick-goal-hold` changes only the physical picking objective's goal term from
the endpoint distance to the mean over `--hold-frames`. Placing already uses a
mean goal cost over its settle interval. The default pick behavior remains the
endpoint cost, and the independent task screen still checks the worst distance
over the complete hold interval. This option does not introduce a grasp proposal,
recorded future target, prescribed phase schedule, new checkpoint or threshold.

Tests cover earlier-frame gradients, endpoint-versus-mean diagnostics, invalid
and insufficient intervals, and unchanged independent constraint semantics.
The affected optimizer and physical-guidance suites pass39 tests. The real H16
one-update can checks pass for all-noise and actuator-only scope atc0b174b:
`hold_goal_gpu_gate_v1/report.json`. Endpoint and mean-held diagnostics agree
with direct measurements of saved poses. Longer development comparisons remain
necessary before adopting it.

## Separately declared minimum-height region experiment

The optional `--goal-region-lower=x,y,z --goal-region-upper=x,y,z` defines a
closed position box for physical picking. Infinite bounds permit an unbounded
axis. The goal cost is mean Euclidean distance to this region over the final
`--hold-frames`, and its independent goal criterion requires every held frame
to lie inside the region. Existing physical criteria remain unchanged. Signed
region violations also replace point-goal residuals if AL is explicitly used.
The nominal point-distance diagnostic remains distinct from the region distance.

The first exploratory lifting recipe pins XY bounds to the observed object
origin±100mm and requires height at least100mm above that observed origin,
with no upper height bound. It uses the same model, input, H36, seed8311,
24 updates and physical weights as the point-goal study. This is a minimum-height
lift within a200mm-wide square workspace, not precise point placement. No grasp
proposal, recorded future target or prescribed hand path is introduced.

All four existing development picks and then the four existing reserved picks
are included without replacements. This secondary task design followed an audit
of the reference motions, including reserved references; treat the region study
as exploratory, not an untouched task-design holdout. The primary precise-goal
comparison and its pinned scene inventory are preserved. A separately named
trust-bound sensitivity check compares0.25 and0.5 normalized-noise RMS on the
first two development picks. These are updated-noise bounds, not measurement
noise or physical distance bounds.

The affected region, physical-objective and optimizer suites pass50 tests.
Tests verify region distances and gradients, fixed bounds, earlier hold failures,
unchanged non-goal residuals and rejection of invalid/unbounded tasks. Region
sampling requires its own real-checkpoint CLI check before longer runs. It does
not by itself address force realism, actuator command conversion or execution.

## First region results and independent replay

The first two development picks (assets 5624 and 8026, seed 8311) pass the
region-specific geometric screen after 24 mesh-physical noise updates at RMS
bound 0.25. This screen is insufficient to establish execution: with the same
planned hand/finger motions in the qualified independent-engine diagnostic,
5624 rises only 0.82/0.22 mm at 0.5/0.25 ms steps, despite the predicted 232.57 mm
lift. Asset 8026 moves downward 23.57/24.39 mm. Both also fail timestep agreement.
Densely sampled articulated interpolation of the first plan preserves its
1.958 mm peak overlap; missing a collision between saved frames does not explain
that failure. Neither result is a physical task success.

`scripts/dev/audit_region_replays.py` replaces only the point-goal residual in
existing independent replay audits. It keeps every physical criterion,
controller qualification, hand tracking and timestep check. Original point-task
audits remain intact. A regression test covers inside/outside regions and
independent collision, controller and timestep failures.

The campaign contains `region_replay_report_v1/index.html` with model versus
both physics-step movies, and `task_reference_report_v1/index.html` with all
16 recorded task references. Their tracked renderers are
`scripts/dev/render_region_replays.py` and
`scripts/dev/render_guidance_task_reference.py`. These are exploratory results
on the single clean-context unconditional checkpoint pinned in the overnight
protocol, not a benchmark or evidence of unseen-training generalization.

## Broader placing supervision check

A separate seed19751 audit selects eight registered placing-training shards
uniformly without replacement and four stored batch keys per shard. Applying
the current registered training exclusions leaves106 stored40-frame segments
across85 assets. There is no outcome or asset filter in this initial selection.
These segments are correlated and are not full independent episodes.

Eight segments pass a sparse-point proxy requiring both fingers to stay at
least5mm from object points over the final six frames, after an earlier near
state. All eight receive collision geometry with verified native512-point
template correspondence. Seven still have finger mesh contact. One bowl
(asset4888) has sustained mesh clearance, including29.68mm to the palm and
7.06/9.18mm to the fingers. It passes rest, settling, collision, joint and
orientation checks but misses the precise held-position criterion; its final
point error is25.90mm. Thus the sampled data contains some released placing,
and the initial eight point-proxy positives substantially overstate it.

The other98 segments are not mesh audited, so one confirmed example is not a
full-corpus release rate or proof that the others cannot contain release. These
are recorded reference measurements, not model performance or input to planning.
The point audit also observes exported friction values from0.9 to1.1; this
supports the existing note that friction is not an explicit model input, without
establishing it as the cause of generated failures.

`scripts/dev/audit_pickup_release_support.py` retains the seeded selection,
exclusion hashes and every measured row. `place_release_support_v1/report_v2.json`
only corrects the original report's `episodes` field name to `segments`;
the original report and audit source are preserved. The task recheck's first
attempt stopped on an unbatched target shape; its corrected v2 uses the unchanged
validator with a batch dimension. No threshold or source motion changed.
`scripts/dev/render_pickup_release_support.py` shows all eight point/mesh
comparisons and the recorded mesh-clear example, explicitly labeled as data.
The main report links `place_release_report_v1/index.html`.
