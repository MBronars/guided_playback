# Guidance objectives and selected results

The initial-noise optimizer is reusable. The current physical objectives contain
handwritten assumptions about picking with a two-finger gripper and placing on a
table. We reuse them across objects; we do not design a grasp trajectory for each
bottle or cup. They are not yet a general physical-plausibility objective for
cloth, sand, water, arbitrary tools, or arbitrary manipulation tasks.

This gallery highlights promising **model predictions** from the September 15
experiments. None of these examples passes the complete qualified physics-execution
task. The continuous improvements are real, but a clean-looking movie is not
evidence that the generated gripper motion can execute the predicted motion.

## How to read the gallery

Each embedded GIF has three panels:

| Left | Middle | Right |
|---|---|---|
| Ordinary unconditional sample | Prediction after noise optimization | Recorded dataset reference |

The middle panel is the guidance result. The right panel provides context and is
not the target trajectory or a generated success. Orange points are the object;
blue, teal, and purple points are the gripper. These are subsampled point-cloud
renders; the reported overlap measurements use the collision meshes.

The first four frames are shared observations. The following 36 frames are model
predictions covering 1.8 seconds. The GIF playback is slower than physical time.

For lifting, the **red rectangle marks the bottom of the allowed region**:
the object's template origin must be at least 10 cm above its observed initial
origin and within ±10 cm of its initial x and y, throughout the last six predicted
frames. There is no upper height limit. The rectangle is not a platform, and the
goal applies to the object's origin, not every surface point.

For placing, the **red cross marks the requested position**. Orientation,
settling, table support, and release are also checked, although the cross alone
does not visualize those conditions.

The GIFs and measurements are existing local artifacts under `validation/`,
which is ignored by Git. Open this document in Markdown Preview in this workspace
to view the animations; a fresh clone will need those artifacts too.

## Selected gallery

The five lifts below are all five of the eight region-study predictions that
passed its geometric task screen. The teacup illustrates a different objective
and a remaining failure. This is a selected gallery, not a success-rate estimate.
All examples use seed 8311 and a budget of 24 noise updates; the saved result is
the best objective value encountered, which need not be the final update.

### 1. Syrup bottle · asset 8270

**Goal:** lift into the region and maintain a grasp near the end.
Peak measured object–gripper overlap decreases **5.54 → 0.48 mm**.
The predicted final lift is **21.20 cm**, and the region/geometry screen passes.

![Syrup bottle: ordinary prediction, noise-optimized prediction, and recorded reference](../../validation/guidance_noise_2026-09-15/g0v9_pick_reserved_00_asset_8270_region_mesh_physical24_r025_h36/comparison_8311.gif)

The model rises only 1.52 mm before any gripper link is within 2 mm at saved frames,
but rises 39.21 mm before both fingers are that close. Earlier palm or single-finger
contact can move an object, so this timing alone does not establish phantom motion.
In independent execution, the generated gripper leaves the object on the table;
the recorded gripper motion lifts it. This example's replay is timestep-stable and
its recorded-motion controller check passes.

[Recorded versus generated execution GIF](../../validation/guidance_noise_2026-09-15/qualified_control_report_v1/recorded_vs_guided_8270.gif)
· [Saved protocol](../../validation/guidance_noise_2026-09-15/g0v9_pick_reserved_00_asset_8270_region_mesh_physical24_r025_h36/protocol.json)

### 2. Energy-drink can · asset 5624

**Goal:** the same lift region, applied to a can initially lying on its side.
Peak overlap decreases **11.76 → 1.96 mm**. The predicted final lift is
**23.26 cm**, and the region/geometry screen passes.

![Can: ordinary prediction, noise-optimized prediction, and recorded reference](../../validation/guidance_noise_2026-09-15/g0v7_pick_development_00_asset_5624_region_mesh_physical24_r025_h36/comparison_8311.gif)

The selected prediction has much less overlap, but independent execution does not
produce a sustained lift and fails the timestep-stability check.

### 3. Spice jar · asset 8026

**Goal:** lift into the region. Peak overlap decreases **16.06 → 1.52 mm**.
The predicted final lift is **28.69 cm**, and the region/geometry screen passes.

![Spice jar 8026: ordinary prediction, noise-optimized prediction, and recorded reference](../../validation/guidance_noise_2026-09-15/g0v7_pick_development_01_asset_8026_region_mesh_physical24_r025_h36/comparison_8311.gif)

This is a large reduction in measured overlap. Independent execution still fails
the lift task and the timestep-stability check.

### 4. Jam jar · asset 6079

**Goal:** lift into the region. Peak overlap decreases **2.89 → 1.82 mm**.
The predicted final lift is **20.72 cm**, and the region/geometry screen passes.

![Jam jar: ordinary prediction, noise-optimized prediction, and recorded reference](../../validation/guidance_noise_2026-09-15/g0v9_pick_development_02_asset_6079_region_mesh_physical24_r025_h36/comparison_8311.gif)

The optimized lift is lower than the recorded reference; matching its height was
not the objective. Execution fails, but this scene also fails the recorded-motion
controller qualification, limiting attribution of that discrepancy to the model.

### 5. Second spice jar · asset 8043

**Goal:** lift into the region. The predicted final lift is **21.15 cm**, and the
region/geometry screen passes. Peak overlap changes **0.44 → 0.96 mm**: it remains
below the 2 mm screen tolerance but increases rather than improves.

![Spice jar 8043: ordinary prediction, noise-optimized prediction, and recorded reference](../../validation/guidance_noise_2026-09-15/g0v9_pick_reserved_01_asset_8043_region_mesh_physical24_r025_h36/comparison_8311.gif)

This demonstrates the tradeoff in a weighted objective: satisfying the goal and
contact terms does not guarantee improvement in every metric. Execution fails
the lift task and the timestep-stability check.

### 6. Teacup placement · asset 7803

**Goal:** reach the separately recorded requested pose, rest on the table, settle,
and release. The optimized final position error is **2.47 mm**. Peak mesh overlap
decreases **2.72 → 1.78 mm**. Position, orientation, support, and settling pass the
model's final task screen; **release still fails**.

![Teacup placing: ordinary prediction, noise-optimized prediction, and recorded reference](../../validation/guidance_noise_2026-09-15/g1v9_place_development_03_asset_7803_physical24_h36/comparison_8311.gif)

The final closest finger is about 2.14 mm from the object, short of the requested
5 mm release clearance. A cup near the cross is therefore not a completed released
placement. This is the primary physical-objective run, which used the earlier
sampled-geometry penalties; it is not the same mesh-cost recipe as the five lifts.
All reported overlaps above use the shared mesh audit. Its independent execution
is timestep-stable but still fails the complete requested task.

[All 16 primary scenes](../../validation/guidance_noise_2026-09-15/primary_overview_v1/index.html)
· [All eight region predictions and executions](../../validation/guidance_noise_2026-09-15/region_replay_report_v1/index.html)
· [Contact timing measurements](../../validation/guidance_noise_2026-09-15/region_contact_timing_v1/report.json)

## What the optimizer actually changes

In [`noise_opt`](../../guidance/unconditional.py), the model weights and observed
context stay fixed. The optimization variables are changes to the initial
**pose-noise channels**, including both the passive object and the articulated
gripper's valid channels (`scope="all"`). This is not model training or VAE-latent
optimization. Rigid particle positions follow the generated poses.

For each update, we generate the complete trajectory, compute a scalar cost,
backpropagate through the denoising computation, and apply Adam to the noise
changes. The 36-frame rollout uses 16 + 16 + 4 future-frame windows; gradients
also flow through the generated context passed between windows.

The bottle run uses learning rate 0.03, a mean-squared noise-change penalty with
weight 0.001, and a per-window RMS change cap of 0.25 in normalized noise units.
These limit departure from the initial draw; they do not certify that a result
is likely under the model or physically executable. The run selects by total
objective value; independent task screening is reported afterward.

## How the bottle's physical losses are implemented

The total objective is the goal cost plus these weighted terms:

```text
goal
+ 2 × finger-contact cost
+ 5 × unsupported-upward-motion cost
+ 1 × relative-translation cost
+ 5 × penetration cost
+ 5 × object/table-intrusion cost
+ 5 × gripper/table-intrusion cost
+ 10 × finger-opening-limit cost
+ 10 × finger-speed-limit cost
+ 0.001 × mean squared noise change
```

Geometry terms are expressed in metres. The weights and thresholds are manually
chosen. Mean/maximum reductions differ between terms, so a weight of 5 does not
mean five times the practical influence of every term weighted 1.

| Term | Actual calculation | Assumption or limitation |
|---|---|---|
| Goal region | Euclidean distance of the object origin to the allowed box, averaged over the final six frames; zero inside. | The region is an explicit task input. It is not derived from the GT future. |
| Finger proximity | Take the farther of the two finger-to-object surface gaps; penalize the amount beyond 2 mm when lifted, plus an end-of-trajectory contact cost over the last four frames. | Assumes a two-finger hold. Proximity does not measure grip force. The independent contact screen checks the final six frames. |
| Unsupported upward motion | Positive vertical displacement per transition, multiplied by a soft missing-contact factor from the preceding frame. | A pick-specific heuristic. It can penalize legitimate throwing, ballistic motion, or palm-supported movement. It does not constrain all physically unsupported motion. |
| Slip proxy | Change in the object's origin expressed in the hand's coordinate frame, weighted by lifted height and apparent contact. | Encourages the object to move with the hand. It does not enforce torque balance, friction, or rotational grasp stability. |
| Object–gripper penetration | Signed collision-component distances; penalize negative distances deeper than 2 mm using mean worst-link excess plus the global maximum excess. | Uses convex collision geometry. Zero cost permits up to 2 mm overlap and is not a proof of continuous nonpenetration. |
| Table intrusion | Object and gripper collision vertices below the support plane by more than 3 mm; mean plus maximum excess. | Assumes a plane inferred from observed object geometry, not a fully observed scene boundary. |
| Finger openings | Recover joint displacement along calibrated gripper axes and penalize positions outside the supplied limits. | Specific robot geometry/calibration is required. |
| Finger speeds | Penalize opening changes greater than the supplied velocity limit times the frame interval. | A kinematic limit, not an actuator torque or contact-force model. |

For the contact-related terms, let `gap[t]` be the larger of the two nonnegative
finger mesh gaps and `bottom[t]` the object's lowest collision vertex. The actual
soft factors are:

```text
lifted[t]  = clamp((bottom[t] - table_z - 0.003) / 0.020, 0, 1)
missing[t] = clamp((gap[t] - 0.002) / 0.010, 0, 1)

unsupported_rise = mean(max(z[t] - z[t-1], 0) * missing[t-1])
slip = mean(hand_relative_origin_step[t] * (1 - missing[t-1]) * lifted[t])
```

There is no hard grasp event: these factors turn on gradually. In particular,
`missing=0` means both fingers are within the chosen proximity tolerance, not that
they can apply forces sufficient to lift the object.

Implementation map:

| File | Responsibility |
|---|---|
| [`goal_region.py`](../../guidance/goal_region.py) | Fixed box bounds, distance cost, and held-region validation. |
| [`pick_objective.py`](../../guidance/pick_objective.py) | Base geometry, object motion, hand-relative translation, and support features. |
| [`physical_objective.py`](../../guidance/physical_objective.py) | Joint calibration, limits, peak penalties, and the held goal. |
| [`mesh_physical.py`](../../guidance/mesh_physical.py) | Replace sampled contact/penetration costs with collision-surface costs. |
| [`collision_distance.py`](../../guidance/collision_distance.py) | FCL signed distances and local first-order derivatives to object/gripper poses. |
| [`place_objective.py`](../../guidance/place_objective.py) | Placing-specific rest, release, settling, and orientation terms. |
| [`run_unconditional.py`](../../guidance/run_unconditional.py) | Compose the requested objective and optimization settings. |

FCL itself is not an autograd simulator. We query its distances and closest-point
witnesses, hold the witnesses fixed locally, and reconstruct a first-order pose
derivative in PyTorch. Nearest-feature changes are nonsmooth; overlapping convex
parts do not define an exact union penetration distance. No contact forces are
backpropagated through this loss.

## Do we need a different objective for each task?

**We need to specify what counts as achieving a task. We should be able to reuse
the optimizer and much of the objective machinery.** The current code already
shares primitives, but several physical terms are deliberately task-specific.

| Change | What needs to change today |
|---|---|
| Another bottle or can, same lifting task | Geometry and observed state; reuse the picking objective and weights. No object-specific grasp path is supplied. |
| Different lift height or horizontal destination | Goal point or region bounds. |
| Picking versus released placing | End conditions: retain contact for picking; request table support, settling, and finger clearance for placing. |
| A different robot/gripper | Kinematic calibration and contact-role assumptions. The current two-finger terms do not transfer automatically. |
| Cloth folding, sand, water, pushing, throwing | A suitable goal representation and compatible interaction assumptions. Rigid object-origin goals and two-finger attachment penalties are not sufficient. |

A useful separation is **task goal + reusable geometry/robot constraints + any
declared interaction assumptions**. Collision checks and joint limits can often
be reused. “Object must stay attached to these fingers” belongs to the task or
interaction description, not universal physics. Material properties condition the
model; these penalties do not independently verify material dynamics.

Even the current geometric task screen does not certify every intermediate
upward transition: its held-contact and relative-motion checks focus on the final
interval. Passing that screen is not a phantom-motion-free certificate. Because
both object and gripper noise can change, the optimizer can arrange a plausible
joint trajectory without establishing that the commanded gripper causes the
object to follow. The execution tests expose that unresolved issue.

## Experiment provenance

All clips use the unconditional bidirectional articulated UNet with clean context,
hidden future actions, EMA weights, FP32, and the pyramid schedule with 25 steps.
Checkpoint:
`/data/group_data/katefgroup/runs_pwm_robertre/pickup_jointed_unet_bidir_uncond_cleanctx20k_from40k_0914/checkpoints/final.pt`.
SHA256: `99ec6cc6102a0d654c375c9f7b9cc9c86edd0964acf9de1acb42d4fb578c9fba`.

Scene sources and exclusions are pinned in
[`scenes_v2.json`](../../validation/guidance_noise_2026-09-15/scenes_v2.json).
“Reserved” refers to guidance development, not necessarily unseen checkpoint
training data. Each GIF's directory contains its saved protocol, initial/selected
measurements, and shared mesh audit. Region task results are separately recorded
from the original precise-point audit; nominal point error must not be interpreted
as region failure.

Independent execution uses an approximate floating-hand MuJoCo setup at 0.5 and
0.25 ms integration steps, not the original source simulator or a real robot.
Collision geometry, inertia, support, and conversion from achieved gripper poses
to motor targets remain possible sources of discrepancy.

See the [complete review](REVIEW_2026-09-15.md) for the full experiment counts,
controls, remaining failures, and source-version qualifications.
