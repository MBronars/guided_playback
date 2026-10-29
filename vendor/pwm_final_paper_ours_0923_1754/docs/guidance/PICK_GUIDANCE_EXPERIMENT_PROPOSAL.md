# Experiment proposal: executable picking from an unconditional world model

Prepared 2026-09-14 for discussion. This document proposes experiments; it does
not record new runs or authorize a training launch. The immediate scope is the
unconditional, bidirectional pickup model. Later sections cover training,
imperfect observations and transfer to other materials.

## Recommendation

Make **successful execution of the generated gripper motion** the central
outcome. Develop an explicit reach–close–lift–hold objective, compare it against
the current interaction objective under the same compute budget, and validate
both by replaying the actuator motion in simulation. In parallel, measure how
often the unconditional prior already supplies a usable interaction. Those
experiments distinguish an inadequate objective from difficult search and an
inadequate learned prior.

Then compare constrained optimization and candidate selection. Training changes
become easier to interpret once the evaluation can distinguish good-looking
object motion from an executable grasp. The best next step is a sequence of
controlled comparisons, with independent branches for promising methods.

## What the completed evidence actually says

The [guidance diagnostic](UNCONDITIONAL_PICK_GUIDANCE_2026-09-14.md) used the
40K EMA checkpoint `pickup_jointed_unet_bidir_uncond/checkpoints/final.pt`,
SHA256 `f4c4d559d1ef520fa0465650e1fe2a48b6e4a4abe991cb84e159da8d68f8b65c`.
It was trained with noisy context and fully hidden future actions. Inference
used clean context, FP32, C4/F16/H36, pyramid/25 and checkpoint-owned
normalization. Both seeds, 54321 and 54322, share one pinned spice-jar scene,
asset 8059, from `robert_pickup_data_pick_success/gpu0/00044.tar`,
`batch_00002829`, element 3. The validation loader applied the registered asset
exclusion, without the combined training quality or separate place-strict list.
The report contains the exact input, source and normalization receipts.

For that diagnostic, ordinary final target errors of 92.7/99.7 mm became
22.2/11.7 mm with goal-only noise optimization. The generated object still rose
while separated from all sampled gripper points. Interaction costs improved
some geometry measurements, but the original and stronger follow-up objectives
did not establish a held pick. These are two draws from one context, with a new
10 cm lift target, not an execution success rate. Simulator replay is outstanding.

The [unguided sampler check](SAMPLER_CHECK_2026-09-14.md) evaluated four pinned
clips, eight draws and four settings: pyramid/25, pyramid/50 and two joint
schedules matched by model-call count. It found no consistent contact repair.
Retain pyramid/25 for the first comparisons. Higher-order integration or other
schedules remain hypotheses, but the evidence lowers their immediate priority.

The current code leaves several specific opportunities:

| Current implementation | Consequence to investigate |
|---|---|
| Nearest sampled-point distance from each finger | Nearby fingers need not present opposing inner-pad contacts |
| Smooth proximity/lift gates | The optimizer can reduce a gated cost by avoiding the gate |
| Object-centre motion in the hand frame | Rotation relative to the hand can change without this term detecting it |
| Mean collision/support penalties | A short, deep intersection can be inexpensive |
| Optimized initial noise for a complete joint trajectory | Moving the predicted object does not prove that the generated hand causes the motion |
| Generated achieved hand poses and finger motion | Full-arm commands, tracking and inverse kinematics still need reconstruction |

These observations come from [the objective](../../guidance/pick_objective.py)
and [the unconditional optimizer](../../guidance/unconditional.py). They identify
testable limitations; they do not establish which change will improve picking.

## Literature: what is useful and what transfers incompletely

| Source | Relevant evidence and implication |
|---|---|
| D-Flow, ICML 2024 [1] | Optimizes source noise through a frozen flow and differentiable output objective. Supports retaining our full-sampler gradient baseline. Its favorable bias does not certify a grasp or preserve the original source distribution after optimization. |
| DexHandDiff, CVPR 2025 [2] | Explicitly addresses ghost object motion using interaction stages and learned dynamics consistency. This is the closest motivation for grasp formation and maintenance. Our gripper geometry costs do not reproduce its separate dynamics model. |
| HardFlow, 2025 preprint [3] | Formulates constrained flow sampling as optimization with constraints on the final clean sample. Its manipulation experiment uses planar end-effector obstacle avoidance and learned linear dynamics. Contact-rich articulated picking requires a new adaptation and evaluation. |
| Feynman–Kac steering, ICML 2025 [4] | Uses sequential Monte Carlo with potentials, proposals and resampling. Motivates searching several candidates, but deterministic copies need meaningful diversification. Its image/text results do not establish manipulation performance. |
| PhysDiff, ICCV 2023 [5] | Integrates simulator-based human-motion projection into denoising. Supports investigating feedback from executed dynamics. It depends on a motion-imitation controller and simulator; a gripper implementation needs its own controller and contact validation. |
| Diffuser, ICML 2022 [6] | Uses a generative trajectory prior for planning with task guidance and conditioning. Motivates joint hand/object waypoint experiments. Conditioning our displacement-based, multiwindow flow requires an explicit adaptation. |
| Physics-Constrained Flow Matching, 2025 preprint [7] | Primarily studies nonlinear equality constraints and identifies inequalities as an extension. It is less directly applicable to unilateral contact and collision than its title alone might suggest. |

These are methodological starting points. The proposed experiments below are
our adaptations unless explicitly identified as existing code. Combining them
does not by itself establish algorithmic novelty.

## Experiment 0: establish an independent execution check

Reconstruct one replayable scene before interpreting model-generated success.
The collector saves `joint_pos_obs`, `joint_vel_obs`, `joint_pos_target`, link
poses, asset identity and material information. Observed state and commanded
target arrays have different temporal alignment. The collector documents these
contracts in
`/data/user_data/robertre/RLSkills_IsaacLab/scripts/collect_pickup_3dpcwm.py`.

Start by replaying the **recorded commanded joint targets** with the appropriate
initial state, controller, time spacing and collision assets. Match the table,
material properties and available velocities. The historical shard does not
save a complete simulator snapshot: table pose and original collision cooking
remain uncertain. If the recorded positive control fails, qualify the replay
setup before attributing generated failures to the model. A newly exported,
self-contained evaluation scene is a valid fallback, with its changed
provenance stated explicitly.

Then convert generated hand poses and finger openings into feasible arm joint
targets and execute them. Measure both hand tracking and resulting object
motion. The passive object must evolve under simulation. A moving-gripper-only
collision experiment is a useful intermediate diagnostic; full-arm execution
also requires inverse kinematics and controller feasibility.

Maintain three distinct outcomes:

1. Generated trajectory passes geometric checks.
2. Controller tracks the generated actuator trajectory acceptably.
3. The resulting simulated object reaches and holds the requested lift.

Keep continuous measurements alongside pass/fail labels: target error over the
hold, contact duration, full hand-relative object pose drift, maximum and tail
penetration, joint violations, tracking error and predicted-versus-executed
object error. Calibrate geometry tolerances using recordings and deliberately
invalid examples. Sampled distances, convex-hull proxies and inferred support
planes remain approximate. Rising free flight after an impulse can be physical;
our current separated-rise flag alone cannot adjudicate that case.

This infrastructure can proceed alongside development of the first guidance
objective, but execution claims depend on its qualification.

## Experiment 1: measure the prior and strengthen selection

Choose four development scenes and reserve four additional geometries from the
same eligible validation population. Select for observable geometry and initial
grasp difficulty before inspecting model results. Reserve the original spice
jar as a regression fixture. These additional scenes are withheld from guidance
tuning; they are not automatically out-of-training-distribution assets.

Generate an initial 16 draws per development scene. Inspect how many contain
plausible reach, closure and held motion even before applying the new goal.
Evaluate goal achievement separately from interaction validity. Reuse the
candidate bank for ordinary sampling and selection at K=1, 8 and 16; extend it
for compute-matched comparisons.

Compare selection by current scalar cost with selection that first filters
unacceptable geometry and then ranks remaining candidates by task performance.
Retain an explicit **no acceptable candidate** outcome. A lowest-cost result
from a wholly invalid bank must not count as success.

If usable interactions are common, selection and better search may suffice.
If they are rare, inspect whether guided search can recover them and prioritize
better checkpoints or conditioning. Sixteen failed samples do not prove that
the model cannot produce a valid grasp.

## Experiment 2: give guidance an explicit grasp sequence

This is the highest-priority implementation hypothesis.

From the observed object mesh, enumerate a small set of opposing contact pairs
that fit the gripper opening and permit an approach. Use the inner finger pads,
surface normals and hand orientation to score alignment. Check candidate palm
and finger geometry against the object and support. This is a geometric grasp
proposal, not a force-closure certificate.

Guide four stages: approach while open, close around the object, lift after
contact, and maintain a hold. Enumerate a few closure times and approach modes
so the method can adapt to different scenes. During the intended hold, penalize
changes in the full relative transform `T_hand^-1 T_object`, including rotation.
Keep absolute finger positions within calibrated joint limits. Use the observed
support until lift-off, with exceptions tied to the intended manipulation mode.

Make required closure and hold conditions explicit so avoiding contact cannot
silence the maintenance objective. Report worst-frame violations alongside
averages. Compare the current objective against this staged version using the
same optimizer, initial draws, trust bound, schedule and budget. Subsequently
remove individual additions to identify whether opposing contacts, timing or
relative-pose maintenance provided the improvement.

The goal and proposed grasps come from observed state and task geometry. Future
recorded contacts can serve as a separately labeled diagnostic upper bound, not
as hidden input to the deployable method. A geometric grasp proposal is also a
baseline in its own right: evaluate a conventional approach–close–lift trajectory
through the same controller to measure what the learned model adds.

## Experiment 3: improve search while holding the objective fixed

Our existing method differentiates the completed rollout through all windows.
Keep it as the full-chain reference [1]. Compare:

| Variant | Question it answers |
|---|---|
| One longer noise optimization | Is local refinement sufficient from one draw? |
| Several shorter optimizations | Is the main difficulty a bad initial grasp mode? |
| Candidate selection followed by refinement | Does a promising interaction provide a better starting point? |
| Coarse temporal noise updates plus a bounded residual | Does reducing update dimension improve stability and compute efficiency? |
| Actuator-only, all-body, and actuator-first then all-body updates | How much passive-object freedom helps task accuracy versus imagined motion? |

Optimize a perturbation of the same original Gaussian draw for the coarse-noise
variant. Replacing the initial Gaussian with smooth noise would change the prior
and confound that comparison. Preserve valid articulation channels and evaluate
accepted iterates, including the final one. A trust bound around a draw limits
the update; it is not an exact prior-density constraint.

Match wall-clock budgets, report actual model invocations and account for
backpropagation. Selection overhead counts against the hybrid's refinement
budget. Fix representative-point RNG for optimization and repeat final
candidates with fresh routing; geometric neighborhoods may still change.

Actuator-only noise optimization remains a restriction on search variables.
Because this model jointly generates the object, execution is still needed to
test whether the resulting actuator motion causes the predicted lift.

## Experiment 4: constraints and cheaper guidance during denoising

First compare the staged soft objective with an augmented-Lagrangian or
constrained-noise formulation for joint limits, collision bounds and required
grasp/hold conditions. Keep a separate task objective and report each constraint
residual. Limited optimization of nonconvex proxy constraints provides no blanket
feasibility guarantee. Apply the independent validator to the final result.

Next compare full-chain optimization with corrections computed from the local
clean-trajectory estimate during denoising. This could be substantially cheaper,
but may produce less accurate gradients. Under our linear corruption convention,
noise is at t=1 and data at t=0. For a velocity prediction, the existing
`get_x0_est` computes `x0_hat = x_t - t * v_hat`. These are normalized trajectory
deltas: physical geometry must be obtained through production integration and
articulation helpers before evaluating contact.

There are **two clocks**. A final denoised sample contains an entire physical
trajectory, so constraints on that sample can cover every physical frame [3].
Within our pyramid schedule, different frames have different noise levels.
Moreover, later absolute positions depend on preceding displacement estimates.
A single global "last ten denoising iterations" switch does not describe uniform
geometric reliability. Test clean-estimate corrections with explicit per-frame
and preceding-transition readiness, preserve observed context, and respect
already committed windows.

Treat a late-correction prototype as our own baseline. A faithful HardFlow
adaptation needs its constrained surrogate derived for this representation,
frame-wise schedule and multiwindow rollout; simply projecting positions would
not reproduce the method. Its simpler manipulation setting leaves this transfer
an open research question [3].

## Experiment 5: waypoints and sequential alternatives

Test joint hand/object waypoint guidance using the geometric grasp proposals:
align at pregrasp, close, then achieve the object lift with the hand following.
Begin with soft constraints. A later inpainting experiment must use the correct
noise marginal and masks. An absolute endpoint depends on integrated deltas;
it cannot be imposed by setting just the final displacement channel.

The unconditional model can potentially support approximate conditioning through
sampling, but it was trained with `action_dropout=1`. Supplying a new action mask
does not establish that its ordinary conditional-model interface works. Compare
waypoint conditioning against the existing joint-generation interface explicitly
[6].

For longer tasks, try a small beam over complete rollout windows. Branch with
fresh next-window noise, retain distinct grasp approaches and score stage-aware
progress. A branch that waits forever must not win merely because it avoids
contact violations. Copying a deterministic Euler state without changing its
continuation creates duplicates. Call this beam search; exact Feynman–Kac
steering additionally needs justified proposals, weights and resampling [4].

Once execution is qualified, compare open-loop plans with replanning from the
actual simulated observation after a short executed segment. Replanning from
the model's own prediction does not supply execution feedback.

## Experiment 6: improve the learned model

Evaluate better checkpoints with ordinary sampling and the strongest small set
of guidance methods before expanding the full grid. The prepared
[clean-context fine-tune](CLEAN_CONTEXT_FINETUNE_2026-09-14.md) is a starting
recipe; this proposal does not assert its current job status or results. A
fine-tune versus the old checkpoint measures the whole continuation recipe.
Attributing improvement specifically to context noising requires a matched
continuation control.

Further training candidates are contact-balanced window sampling; auxiliary
contact/phase prediction; and low-noise relative-pose/contact consistency losses.
Evaluate held contact and diverse valid futures together, since excessive
regularization could suppress legitimate release and regrasp. Geometry-derived
labels need their own quality audit.

A conditional dynamics model could independently predict object motion from
proposed actuator motion, or provide a consistency critic. Mixed action-dropout
training could support both conditional and unconditional queries. Neither
capability should be assumed for this checkpoint. A critic trained only on
successful recordings may fail on optimized out-of-distribution actions; include
executed failures and keep a separate execution test.

Audit missing physical inputs before adding channels. The collector exports
friction and joint state, whereas the current core batch prominently exposes
E, nu and density. Trace adapters and model paths to determine what actually
reaches the network. Support geometry, absolute joint state/limits, and friction
are candidates for that audit, not established root causes of the present error.

For any checkpoint comparison, pin corpus vintage, train/validation identities,
exclusions, normalization, evaluation episodes and compute. Training preflight
requirements are documented in [CLAUDE.md](../../CLAUDE.md). Prefer one change
per initial training comparison.

## Experiment 7: imperfect state estimates and cross-domain transfer

Once clean-state execution is credible, perturb the planner's observed state
while keeping a separate true simulator state. Initial pilot levels could be
1/3/5 mm translation bias, 1/3/5 degree orientation bias, one observation-frame
delay, and 10/25% physical-parameter mismatch. These are proposed stress levels,
to be adjusted against measurement scales and valid parameter ranges.

For rigid pose-estimation error, move the particle template coherently with the
corrupted pose. Independent point measurement noise is a different experiment;
it should not silently recreate the previous inconsistent rigid-context bug.
Persistent occlusion, dropped points and segmentation/tracking corruption need
explicit adapter/mask contracts and should be tested separately.

Compare clean-trained guidance, optimization over several perturbed estimates,
and observation-based replanning. Evaluate on perturbations not used during
optimization. Report actual execution success, prediction error and failure
detection coverage. Multiple generated samples alone do not establish calibrated
uncertainty.

Transfer the principles rather than a rigid grasp constraint to other domains.
For placing, test contact with support before release. For cloth, retain local
gripper attachment where appropriate while allowing folds and deformation. For
granular/MPM pushing, check tool contact and material response without attaching
the whole material to a hand. A shared method evaluated on several settings is
distinct from one checkpoint trained jointly across those settings; report the
actual arrangement.

## Proposed first pilot and decision gates

Use four development scenes, four independently seeded trials per scene, a
reachable lift goal, and a fixed horizon long enough for a hold. Lock four
additional evaluation geometries before tuning. Record exclusions and whether
their assets were seen during training. Calibrate a proposed 20 mm target
tolerance and 0.25 s hold against controller performance before treating them as
success thresholds.

| Arm | Purpose |
|---|---|
| A. Ordinary sample | Prior and regression reference |
| B. Feasibility-filtered best of K | Search-only baseline, plus K-dependent curves |
| C. Current interaction noise optimization | Existing method under the new evaluation |
| D. Staged grasp noise optimization | Isolate the new objective |
| E. D with explicit constraint handling | Test resistance to objective tradeoffs |
| F. D with selection then refinement | Test the value of a better starting interaction |

Use a provisional **150 seconds per planning trial** for B–F after timing
qualification. K and update counts must fit that budget; compare the best valid
result available by the deadline and retain failures. A can be the first draw
of B. Compare C/D first; E/F follow only after their correctness checks.

Previously, one ordinary draw took approximately 6 seconds and six-update
optimization approximately 142–147 seconds for the pinned scene on a PRO 6000.
At those rates, the development pilot's five budgeted arms × 16 trials need
roughly 3.3 GPU-hours. This is an execution estimate after implementations work,
excluding model loading, scene preparation, simulator qualification and rendering.
Geometry and simulator setup may dominate elapsed time. New methods must be
timed before making an overnight completion promise.

After development, lock hyperparameters and evaluate the strongest two methods
alongside current guidance and selection on the reserved scenes. Report
scene-level results and all seeds; draws from one scene are correlated. Eight
scenes support a pilot, not a broad robotics success claim.

If two GPUs are available, assign one to guidance experiments and one to replay
qualification and subsequent independent evaluation. Verify allocation and
occupancy before launch. Reserve a separate, explicit budget for training.

Decision gates:

- If recorded-command replay fails, resolve the execution setup first.
- If staged guidance improves geometric contact but execution still fails,
  inspect controller tracking and predicted-versus-executed dynamics.
- If filtered selection already matches optimized guidance at equal time,
  retain it as the stronger simple baseline.
- If optimization repeatedly finds invalid high-scoring trajectories, improve
  constraints and independent scoring before increasing search.
- If better checkpoints improve ordinary interaction quality, quantify how
  much guidance is still needed under the same task and execution protocol.

## Deliverables and research value

Produce synchronized views of ordinary generation, current guidance, the new
method and simulator execution. Overlay the intended and actual hand/object
trajectories, and display closure, lift-off and hold intervals with contact and
penetration traces. Use consistent cameras and include failures and every pilot
scene. Keep the recorded-command replay as a separately labeled positive control.

The quantitative result should connect planning budget, predicted plausibility,
executed success and sensitivity to imperfect state. That directly addresses
the paper's weaknesses in physical failure analysis and observation robustness.
An attractive generated rollout by itself does not answer those concerns.

The most promising research question is whether interaction-aware constraints
and selective search can make a joint generative world model produce executable
manipulation while retaining diverse solutions. Whether the eventual method is
novel enough requires comparison with the closest methods and evidence across
tasks; it is not established by this proposal.

## Primary references

[1] [D-Flow: Differentiating through Flows for Controlled Generation](https://arxiv.org/html/2402.14017v2).

[2] [DexHandDiff: Interaction-aware Diffusion Planning for Adaptive Dexterous Manipulation](https://arxiv.org/html/2411.18562v6).

[3] [HardFlow: Hard-Constrained Sampling for Flow-Matching Models via Trajectory Optimization](https://arxiv.org/html/2511.08425v1).

[4] [A General Framework for Inference-time Scaling and Steering of Diffusion Models](https://arxiv.org/html/2501.06848v5).

[5] [PhysDiff: Physics-Guided Human Motion Diffusion Model](https://arxiv.org/abs/2212.02500).

[6] [Planning with Diffusion for Flexible Behavior Synthesis](https://diffusion-planning.github.io/).

[7] [Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints](https://arxiv.org/html/2506.04171v1).
