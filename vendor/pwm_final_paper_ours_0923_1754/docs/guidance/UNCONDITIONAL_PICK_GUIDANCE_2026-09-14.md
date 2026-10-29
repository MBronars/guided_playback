# Unconditional picking guidance: implementation and controlled diagnostic

## Decision

The existing differentiable sampler is a useful foundation for guidance. We now
have a separate unconditional entrypoint that preserves the generated hand,
finger and object trajectories, and optimizes a complete rollout through all
three prediction windows. The completed GPU experiment confirms that noise
optimization can move the object toward a goal. Goal accuracy alone does not
establish a grasp: the same model already produces unsupported lifting, and
goal-only optimization can preserve that behavior. Stronger interaction costs
improve proximity but do not demonstrate a reliable held pick. The next useful
step is explicit grasp formation/maintenance and simulator replay of the
generated actuator trajectory.

This is one pinned pick scene, two sampling seeds, one checkpoint, and a new
10 cm vertical lift goal. It is a development diagnostic, not a benchmark or a
real/simulated robot success rate. The implementation is in local commit
`54d79a1f872ae61f8daccaa75374f1abd74f3806`.

Progress and final artifact links: [UNCONDITIONAL_PICK_PROGRESS.md](UNCONDITIONAL_PICK_PROGRESS.md).

## What the code audit changed

The old planner interface takes an actuator plan, inserts it into the model's
conditioning, and overwrites the actuator track when carrying the result into
the next window. That is appropriate for a conditional dynamics model queried
with a proposed action. It is inappropriate for this checkpoint: the model was
trained with `action_dropout=1` and jointly generates the hand, fingers and
passive object. In addition, the old `apply_plan` interface rejects articulated
bodies. Simply invoking that entrypoint with this checkpoint would not test the
intended unconditional planning problem.

The new [unconditional sampler and optimizer](../../guidance/unconditional.py)
reuse the existing differentiable single-window sampler and the production
window-building helpers. They hide future actions at every model invocation,
keep the predicted actuator motion, and carry predicted context into subsequent
windows without breaking the gradient. The objective is evaluated once over
the complete 36-frame future. Applying the final target separately to every
16-frame window would demand premature completion of a reach–grasp–lift task.
The old CLI now rejects the fully unconditional checkpoint and directs the user
to the appropriate interface.

Two distinctions matter for interpreting this implementation:

* The optimizer adjusts the initial noise supplied to the model, not its weights.
  The sampler generates every candidate trajectory. This does not constrain every
  candidate to be physically valid.
* The actuator-only variant fixes the passive object's initial noise. The network
  still predicts object motion jointly with the actuator. This is an optimization
  restriction, not a causal intervention or a physics guarantee.

The UNet samples representative points using random FPS. Each ordinary rollout
records the per-window RNG state after its initial noise draw. Optimization
replays those states, so comparisons between iterates do not mix a noise update
with a fresh random choice of attention points. Spatial relationships can still
change with the generated geometry. A separate check changes FPS randomness
while holding the saved optimized noise fixed.

## How the literature informs the design

**Differentiating through the whole sampler is a reasonable first method.**
D-Flow optimizes a source sample through a frozen flow and a differentiable output
cost. Its argument for favorable gradient structure supports investigating this
approach; it does not certify physical constraints. Its discussion also makes
clear that invertible flows can map some source point to arbitrary outputs.
Our per-window RMS limit around the original noise draw and small displacement
regularizer limit optimization excursions; they do not constitute exact
posterior sampling or preserve the Gaussian prior distribution. [D-Flow](https://arxiv.org/html/2402.14017v2)

**Contact must participate in the objective.** DexHandDiff explicitly discusses
ghost object motion under task-only guidance. It combines contact-related stages
with task guidance and a separate learned dynamics consistency term. We borrow
the need to score interaction rather than just the object's endpoint. Our
two-finger proximity, relative-motion and collision costs are a different,
gripper-specific implementation. We have not reproduced its separate dynamics
model or established equivalent guarantees. [DexHandDiff](https://arxiv.org/html/2411.18562v6)

**Unsigned proximity and signed penetration answer different questions.** CHOIS
uses contact guidance, object-floor geometry, and signed-distance penetration
evaluation. Its contact guidance depends on predicted contact labels; its
guidance is applied late in denoising. We have no trained contact predictor.
Consequently, our smooth gates and sampled-point distances must be described as
proxies. We evaluate their behavior on held lifts, unsupported rises, falls and
known intersections before using them on model samples. [CHOIS](https://arxiv.org/html/2312.03913v2)

**Selection is a necessary baseline before more elaborate sampling.**
Feynman–Kac steering uses sequential Monte Carlo, intermediate potentials and
resampling to steer diffusion outputs. Best-of-eight selection is a simpler
baseline here. Implementing sequential resampling correctly would require
explicit proposals/diversification and the appropriate weights. Duplicating a
candidate and continuing the same deterministic Euler flow would otherwise
produce duplicate trajectories. The cited results concern image/text
generation, not reliable robotic contact. [Feynman–Kac steering](https://arxiv.org/html/2501.06848v5)

These papers support testing noise optimization, interaction objectives and
selection. None supplies a reason to equate an attractive generated pick with
successful execution.

## Experiment protocol and provenance

| Item | Fixed setting |
|---|---|
| Checkpoint | `pickup_jointed_unet_bidir_uncond/checkpoints/final.pt`, 40K, EMA |
| Checkpoint SHA256 | `f4c4d559d1ef520fa0465650e1fe2a48b6e4a4abe991cb84e159da8d68f8b65c` |
| Architecture | Bidirectional articulated pose-native UNet; `unet_actions=false` |
| Training context noise | Enabled in this old checkpoint; inference uses clean context |
| Input | `validation/phantom_regeneration_2026-09-14/pick_input.pt` |
| Input SHA256 | `ca66eff47d0902fc3444142cdf79ea13a5f13b551a0e5508674f185269855dbd` |
| Exact source | `robert_pickup_data_pick_success/gpu0/00044.tar`, `batch_00002829`, element 3 |
| Asset | 8059, `Kitchen_Table/spice_jar/asset_0062` |
| Sample selection | Previously pinned near-median-motion diagnostic from the first 512 stratified validation clips |
| Exclusions | Asset holdout via `holdout_TRAIN_block.json`; no combined pickup quality/place-strict filtering |
| Context / future | 4 observed, 36 predicted; future windows 16 + 16 + 4 |
| Time spacing | Pickup adapter: 0.05 s/frame |
| Schedule | Pyramid, 25 steps, uncertainty scale 1; 108 model calls per rollout |
| Seeds | 54321 and 54322, both on the same scene |
| Task | Final object body origin = observed anchor origin + (0, 0, 0.1) m |
| Precision | FP32, no autocast or compilation |
| Hardware | GPU 1, RTX PRO 6000 Blackwell; existing allocation 10434620 |

The checkpoint's normalization is retained: `xy_p99_per_domain`, profile hash
`2bc31504482f745a5bfce97558e7a366e1b6800d2bb149a8853e663d797587de`.
Full protocol receipts contain resolved paths, normalization values, hashes,
source location and device identity. No training or checkpoint comparison was
performed. A frozen, hash-verified copy of committed source insulates the run
from concurrent repository edits.

The six arms are ordinary sampling; best-of-eight by goal; best-of-eight by
interaction cost; goal-only noise optimization; interaction noise optimization;
and interaction optimization restricted to actuator noise. Optimization uses
six Adam updates, learning rate 0.03, a maximum RMS change of 0.25 per window,
and a 0.001 m coefficient on mean squared noise change. Only valid pose channels
are adjustable; unused finger channels and all particle noise remain fixed.
Each candidate retains its complete jointly generated hand/object trajectory.

The original draw is paired across arms. The optimizer evaluates the starting
point and every updated point, including the last update, then selects the
lowest total cost. Best-of-eight draws sequentially from the same seed, with
candidate zero matching ordinary sampling. Runtime includes the saved baseline
and selection overhead. Receipts distinguish forward NFE from actual model
invocations including gradient-checkpoint recomputation; backpropagation adds
cost beyond either count. We do not claim matched wall-clock compute.

## Geometry: what is measured and what is inferred

All 40 source frames match the fixture exactly after the canonical pickup
coordinate shift. Recreating the collector's object surface sample and aligning
the source gripper link points recovers body templates to approximately one
micrometre. The current source object has three closed, convex collision
components. Unit outward half-space planes are prepared from their geometry.

For a convex component with planes `(n, b)`, the margin is
`max_faces(n · point + b)`. A negative result proves that the tested point lies
inside that component. Taking the minimum over components tests membership in
their union. The magnitude inside one component is its nearest-face depth;
it is not an exact signed distance to a union of overlapping components.

We transform sampled gripper points into the object's body frame and query these
components. We also query sampled object points against per-link gripper convex
hulls. That reverse measurement is explicitly a **hull proxy**: the source
gripper meshes are not watertight as loaded. A detected intersection is useful
evidence; no sampled point inside a mesh does not prove collision clearance.
The source URDF geometry is verified, but the original simulation's USD/PhysX
collision cooking has not been independently reconstructed.

The shard does not save the table pose. The reference plane at z=0.0100 m is
inferred from the lowest object collision vertex in the observed context.
“Support intrusion” therefore means below this inferred reference, not a
measurement against a recovered table collider. Metadata and file hashes are in
`validation/guidance_pick_2026-09-14/collision_geometry.json`.

Finger slides are independently recovered using the observed source joint
positions and changes in hand-relative finger position along each joint axis.
This permits checking the URDF's 0–40 mm opening range and 0.2 m/s speed limit.
It does not establish full-arm inverse-kinematics feasibility. Finger asymmetry
is reported descriptively: the recorded trajectory itself is not perfectly
symmetric under contact.

## Picking costs and diagnostics

The [picking objective](../../guidance/pick_objective.py) sums the final goal
distance and the following costs in metres:

| Cost | Definition | Weight |
|---|---|---:|
| Contact | Farther-finger gap above 5 mm while lifted, plus its last-four-frame mean | 2 |
| Unsupported rise | Positive object vertical displacement weighted by missing prior contact | 5 |
| Relative motion | Object-centre displacement in the hand frame when near contact and lifted | 1 |
| Penetration | Mean excess above 1 mm in the larger of the two signed/proxy depth checks | 5 |
| Support | Mean object-bottom intrusion beyond 3 mm below inferred support | 5 |

Proximity and lift gates are continuous, and gradients include their dependence
on the trajectory. The objective does not use future recorded poses. The
recording is used only as a measurement reference; it ends above our requested
10 cm goal and is not an oracle trajectory for that new task.

The reporting diagnostics are deliberately separate from the smooth costs:

* Both-finger proximity requires **each** finger to have a sampled object point
  within 5 mm. This does not establish opposing contact normals, friction or
  sufficient grasp force.
* Unsupported rise sums upward increments above 0.5 mm when both endpoints lack
  the two-finger proximity condition. A free fall is not counted as an upward
  phantom move. A separate four-frame hover proxy detects near-stationary objects
  suspended above support without that proximity.
* An independent, stricter separation check sums rises whose endpoints have
  **every sampled gripper point**, including the palm, farther than 5 mm from
  every object point. Missing one finger is not equivalent to no robot contact.
  Neither measure rules out legitimate rising free flight after an impulse;
  these remain diagnostic flags rather than force-balance proofs.
* Signed/proxy penetration reports maximum depth and frames exceeding 1 mm,
  so a small mean objective cannot conceal a short, deep intersection.
* Hand-frame slip is reported only on transitions satisfying proximity at both
  ends while lifted. Missing such transitions yields `null`, not zero slip.

The weights and tolerances are initial engineering choices, not learned physical
parameters. Time-averaged penalties can trade away a brief bad contact, and
nearest sampled point distances can be inaccurate on sparse surfaces. The
diagnostics expose those limitations rather than treating a lower scalar cost
as a solved manipulation task.

## Verification

Six sampler/optimizer tests verify exact production parity with random FPS,
RNG replay, hidden future input corruption, preservation of generated actuator
motion, inactive pose channels, finite-difference agreement across carried
context, an actual actuator-only objective improvement, and the zero-update
baseline. Five geometry tests cover known held lifts, unsupported rises,
legitimate falls, signed 5 mm intersections and support/hover behavior.

The combined guidance, articulated-body and rollout-continuity regression run
passed 59 tests, skipped six GPU-dependent tests and deselected two tests. The
additional unconditional CLI guard test passed separately. The real-checkpoint
GPU smoke verified finite noise gradients through a 36-frame rollout and a
goal decrease after one update. CPU synthetic contracts and the actual GPU
experiment provide complementary evidence; neither is a simulator evaluation.

The saved-result audit checks identical initial samples, unchanged observed
context, matching checkpoint/input/geometry/normalization/sampling receipts, and
the expected 16+16+4 window boundaries across arms. The separate FPS check
replays the saved ordinary output before substituting new routing randomness.

## Results

All six initial arms completed. Each cell below gives **seed 54321 / seed 54322**
on the same scene. Distances are millimetres. “Separated rise” uses the stricter
all-gripper-points-over-5-mm diagnostic, not the two-finger objective gate.

| Method | Final target error | Final farther-finger gap | Separated rise | Max gripper-in-object depth |
|---|---:|---:|---:|---:|
| Ordinary | 92.7 / 99.7 | 13.7 / 15.3 | 0.0 / 175.7 | 8.56 / 0.00 |
| Best of 8, goal | 50.9 / 45.8 | 15.0 / 13.1 | 64.2 / 0.0 | 8.04 / 7.93 |
| Best of 8, interaction | 50.9 / 45.8 | 15.0 / 13.1 | 64.2 / 0.0 | 8.04 / 7.93 |
| Goal-only noise optimization | 22.2 / 11.7 | 35.1 / 11.4 | 73.1 / 102.8 | 7.96 / 0.00 |
| Interaction noise optimization | 34.1 / 26.0 | 11.4 / 8.8 | 48.3 / 91.2 | 1.11 / 0.00 |
| Actuator-only interaction optimization | 65.9 / 32.6 | 5.8 / 11.6 | 29.7 / 112.0 | 2.87 / 0.00 |

The two best-of-eight objectives selected the same draws: draw 3 for seed 54321,
draw 4 for 54322. Selection improved the endpoint but did not produce acceptable
interactions. In seed 54322 it introduced clear intersection that was absent
from the ordinary draw: 7.93 mm sampled gripper-in-object depth and 16.23 mm in
the reverse gripper-hull proxy.

Goal-only optimization reached the new target most closely, while both results
still rose with the entire sampled gripper separated from the object. The
interaction objective reduced penetration strongly in the first seed, but
traded away contact as the endpoint improved. Actuator-only optimization gave
four of 36 generated frames with two-finger proximity in the first seed and none
in the second. The other original arms had no generated frames satisfying that
condition. No original arm established a held pick.

The reference recording has a 1.89 mm final farther-finger gap, no sampled
gripper-in-object penetration, and 0.24 mm maximum reverse-hull depth. Its
hand-frame object displacement during held transitions averages 0.35 mm,
compared with 1.40 mm for the first actuator-only result. Its endpoint is 89.39 mm
from the requested target because it performs a different, higher lift; this
does not make the recording a failure at its original task.

| Method family | Seconds/draw including saved baseline | Forward NFE | Actual model calls including recomputation | Peak allocated memory |
|---|---:|---:|---:|---:|
| Ordinary | 5.9–6.1 | 108 | 108 | 1.68 GB |
| Best of 8 | 52.5–53.4 | 972 | 972 | 1.69 GB |
| Six-update noise optimization | 141.5–146.7 | 864 | 1512 | 7.64 GB |

The ordinary saved baseline adds one rollout to both selection/optimization
receipts. Best-of-eight therefore records nine forward rollouts, while six
updates evaluate seven optimizer iterates plus that baseline. There is no
equal-wall-clock comparison here. The interaction and actuator-only second
seeds selected iteration 5, not their final iteration 6; every iterate remains
in the trace.

The generated future spans 1.8 seconds, while the six-update optimizer takes
about 2.4 minutes excluding model loading. This implementation is an offline
diagnostic; the measured runtime does not support a real-time control claim.

The independent joint reconstruction agrees with recorded finger positions to
1.73 micrometres. Original generated results exceed the 40 mm opening limit by
up to 0.325 mm, versus 0.0012 mm in the recording. All checked generated finger
speeds remain below the URDF limit. These opening violations are small compared
with the contact failures, but the current articulated representation does not
itself impose absolute joint limits.

### Exploratory follow-up and routing check

Because the first objective visibly trades contact for endpoint improvement, a
separate bounded follow-up uses 12 updates at learning rate 0.02, contact weight
8, rise weight 10, penetration weight 20 and support weight 10. Scope remains all
pose noise; the checkpoint, input, seeds, sampler, prior and RMS limit remain
fixed. The update budget and multiple weights change together, so this checks
whether stronger interaction optimization is promising, not which individual
change causes an improvement. It is explicitly exploratory and retains the
original six-arm comparison.

The follow-up completed with the following paired results:

| Measurement | Seed 54321 | Seed 54322 |
|---|---:|---:|
| Final target error (mm) | 58.85 | 31.65 |
| Final farther-finger gap (mm) | 5.12 | 6.84 |
| Rise with every sampled gripper point >5 mm away (mm) | 20.71 | 73.51 |
| Max gripper-in-object depth (mm) | 1.73 | 0.00 |
| Generated frames with two-finger proximity / 36 | 3 | 0 |
| Selected iteration | 12 | 11 |
| Runtime including baseline (s) | 279.2 | 284.8 |

Each draw used 1512 forward NFE and 2808 actual model invocations including
recomputation, with peak allocated memory 7.64 GB. Contact gaps are smaller than
with the original interaction weights, but endpoint error is larger. In the
first seed, maximum detected penetration is also slightly larger than under
the original interaction objective. Increasing penalties is not a monotonic
repair. The second seed intrudes 4.57 mm below the inferred support reference
during its rollout. Neither result demonstrates the desired held lift.

The fresh-FPS check completed **20 rollouts**: five methods × two original noise
seeds × two fresh routing seeds (81001, 81002). Both saved ordinary samples first
replayed **bitwise exactly** on the actual GPU. New FPS randomness changed final
goal error by at most 0.91 mm across these checks, and the broad ordering and
failure conclusions persisted. The maximum change of any generated particle
position was 7.85 mm for an ordinary sample and 4.89 mm across guided samples.

Hard contact flags can still change near their threshold. For the first strong
interaction result, the final farther-finger gap changed from 5.12 mm to
4.59 / 5.20 mm under fresh FPS. Therefore report continuous gaps alongside
binary contact counts. These repeats do not establish population robustness,
uncertainty calibration, or robustness to corrupted physical state estimates.

### Visualizations

- [Six-arm comparison, seed 54321](../../validation/guidance_pick_2026-09-14/analysis/comparison_seed_54321.gif)
- [Six-arm comparison, seed 54322](../../validation/guidance_pick_2026-09-14/analysis/comparison_seed_54322.gif)
- [Per-frame contact, intersection and height, seed 54321](../../validation/guidance_pick_2026-09-14/analysis/traces_seed_54321.png)
- [Per-frame contact, intersection and height, seed 54322](../../validation/guidance_pick_2026-09-14/analysis/traces_seed_54322.png)
- [Stronger-contact follow-up, seed 54321](../../validation/guidance_pick_2026-09-14/analysis_followup/comparison_seed_54321.gif)
- [Stronger-contact follow-up, seed 54322](../../validation/guidance_pick_2026-09-14/analysis_followup/comparison_seed_54322.gif)
- [All original and follow-up measurements](../../validation/guidance_pick_2026-09-14/analysis_followup/metrics.csv)
- [Fresh-routing measurements and exact replay checks](../../validation/guidance_pick_2026-09-14/fresh_routing/results.json)

Blue is the object; red is the generated gripper; the green star is the target.
Every panel shares the same camera, bounds and playback speed. Numerical captions
show the current farther-finger gap and current sampled gripper-in-object depth;
the table above reports maximum depth over the complete future. The recording
is a reference panel, not another model. Best-of-eight goal and interaction
select identical trajectories, so the GIF shows that result once.

## Improvements to prioritize after this diagnostic

1. **Evaluate interaction quality before selecting by goal.** Keep goal accuracy,
   finger contact, signed geometry, joint limits and unsupported motion visible
   together. Retain an explicit “no acceptable candidate” outcome when all draws
   fail reasonable geometric checks. A weighted scalar can otherwise choose an
   attractive but invalid trajectory.
2. **Use a grasp phase that matches the object.** Sample candidate opposing
   contacts on the recovered object mesh, filter them by gripper opening and
   collision geometry, and derive a consistent hand pose plus finger opening.
   Guide reach/alignment, then closure, then lift with relative pose maintenance.
   This supplies an explicit sequence and inner-finger contact target rather
   than asking nearest-point costs to discover both. It is a proposed next
   experiment, not implemented here. Hard attachment before a valid grasp would
   hide the error by imposing the desired outcome.
3. **Separate controllable motion from imagined object motion during validation.**
   Replay the generated hand and finger trajectory in the source simulator from
   the observed initial state. Compare the resulting object trajectory with the
   model's predicted one. This directly tests whether the action causes the
   predicted lift. Fixing object noise in the optimizer is not a substitute.
4. **Increase search only after checking what the objective rewards.** More noise
   updates or resampled candidates can optimize the wrong behavior more strongly.
   Use paired seeds, fresh routing checks, runtime accounting and several pinned
   geometries before claiming that a method improves picking generally.

A separate conditional dynamics model could also supply an independent
consistency check, but this action-dropout-one checkpoint was not trained to
respond to externally supplied action commands. That capability should not be
assumed or retrofitted merely by setting an inference mask.

## Reproduction and artifacts

Entry point: [guidance/run_unconditional.py](../../guidance/run_unconditional.py).
Geometry preparation: [prepare_pick_guidance_geometry.py](../../scripts/dev/prepare_pick_guidance_geometry.py).
Analysis/GIFs: [analyze_pick_guidance.py](../../scripts/dev/analyze_pick_guidance.py).
Routing check: [check_pick_guidance_routing.py](../../scripts/dev/check_pick_guidance_routing.py).

The experiment directory `validation/guidance_pick_2026-09-14/` contains the
frozen source manifest, exact six-arm launch script, checkpoint/input hashes,
test receipts, source geometry, per-iteration optimizer traces, selected
trajectories and full per-frame diagnostics. Use a new output directory when
rerunning generation; existing results are never silently overwritten.

## Primary references

1. [D-Flow: Differentiating through Flows for Controlled Generation](https://arxiv.org/html/2402.14017v2), ICML 2024.
2. [DexHandDiff: Interaction-aware Diffusion Planning for Adaptive Dexterous Manipulation](https://arxiv.org/html/2411.18562v6), CVPR 2025.
3. [CHOIS: Controllable Human-Object Interaction Synthesis](https://arxiv.org/html/2312.03913v2), ECCV 2024.
4. [A General Framework for Inference-time Scaling and Steering of Diffusion Models](https://arxiv.org/html/2501.06848v5), ICML 2025.
