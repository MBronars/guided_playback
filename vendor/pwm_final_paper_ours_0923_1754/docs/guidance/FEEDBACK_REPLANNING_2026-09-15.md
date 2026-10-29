# Feedback replanning qualification

Existing whole-horizon noise optimization differentiates through several model
windows. It does not observe execution between them. This continuation prepares
a separate experiment that executes a prefix, observes the resulting state and
replans toward the same fixed task goal.

`guidance.feedback.rigid_feedback_batch` rebuilds particle history from the latest
observed rigid poses and static templates. Future tensor slots repeat the anchor
and contain zero action/target deltas; they are sampler placeholders, not labels
for training or evaluation. It retains identities, materials, normalization and
articulation. `reanchor_finger_calibration` preserves absolute joint coordinates
while refreshing the observed opening, avoiding a false first-step joint velocity.
Six CPU checks cover coherent histories, poisoned recorded futures, articulation,
calibration and invalid rotations/context.

`replay_pickup_mujoco.py --save-frame-states` optionally saves the compiled model
and `mjSTATE_INTEGRATION` at every output frame, including solver warmstart and
user-controlled state. See the [MuJoCo state specification](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APItypes.html#mjtstate).
The compiled model includes the runtime asset mass/inertia adjustment that the
original XML alone does not preserve. The optional writer leaves the old replay
arrays bitwise unchanged on the generated-can gate.

`guidance.persistent_replay.PersistentPickupReplay` restores that exact model and
state, validates version and model checksum, and accepts only hand targets and
finger motor commands. It never prescribes object motion. The existing fixed
mesh-preload command conversion is retained; the optional damping-feedforward
ablation is not used by this interface.

The first real generated-can gate matches the original replay bitwise for all
poses and complete integration states: continuous 36-frame execution, chunks
of 8+8+16+4 frames, and restoration at frames 8, 16 and 32. Receipt:
`validation/guidance_noise_2026-09-15/replay_state_gate_v1/report.json`.
This establishes continuity for that fixture, not task success. Broader recorded
controls and a real-checkpoint feedback gate must pass before interpreting a
feedback planning trial.

The contemplated experiment keeps the initial goal/workspace and simulator
state fixed, refreshes all four observed context frames after an executed prefix,
and uses standard latent-noise optimization for each new plan. It must compare
against continuing the original plan from the identical prefix state. Every
replanning call and extra model evaluation counts toward compute. It remains
an independent MuJoCo diagnostic with a floating hand, approximate collision
properties and no robot arm; it is not source-PhysX or real-robot control.

## Completed integration checks and the pilot comparison

All 16 recorded controls at both 0.5 and 0.25 ms now preserve the original
trajectories bitwise when state saving is enabled. Executing in 8+8+16+4-frame
chunks, or restoring at frame16 and executing 16+4, preserves all poses and
complete integration states bitwise. See `persistent_replay_controls_v1_summary.json`.
The real-checkpoint `feedback_gpu_gate_v1/report.json` also verifies actual
frame16 observations, unchanged context, poisoned-future independence and a
valid gradient update. Its nearby endpoint target is a functional probe only.

`python -m guidance.run_feedback` implements one correction after a declared
executed prefix, using the saved base plan's goal, physical weights, checkpoint
and sampler. The remaining horizon ends at the original task deadline. It saves
three executed continuations from the identical prefix state: continue the old
plan, sample a fresh continuation, and optimize that same fresh sample. With
`--context-source predicted`, it instead uses the old model prediction as
context, withholding the observed correction. This ablation separates fresh
sampling from actual feedback. Both arms use identical fresh Gaussian noise and
routing RNG states in the first can gate.

Pick, place and predicted-context one-update integration gates complete, with
exact continuation of the old plan at both physics steps and frozen model
weights. The predicted-context update raises the objective, so best-so-far
selection correctly retains iteration0. The qualification receipt is
`feedback_driver_qualification_v2.json`. Its predecessor's meta-check incorrectly
required selecting iteration1; the optimizer behavior was correct.

The first can probe illustrates why the ablation matters. Continuing the old
region plan lifts only0.82/0.22mm at the two physics steps. Fresh sampling after
observed feedback lifts144.26/149.77mm; fresh sampling from the old predicted
context also lifts164.83/183.37mm. Thus improved lifting cannot yet be attributed
to observed feedback. All complete task checks still fail. The common executed
prefix already exceeds the simulator's finger travel limit at frames7–9 and11;
later replanning cannot repair that past violation. Interpret continuation
quality separately from complete-trajectory success, retaining both.

The later force audit shows that the observed-context fresh lift includes severe
hand/table crushing. Its greater final height must not be described as a better
grasp; see [executed forces](GRASP_EXECUTION_DIAGNOSTICS_2026-09-15.md). The complete
task failures above remain unchanged.

A supplied finer-step replay executes the same realized two-segment command
program. It does not repeat feedback optimization from finer-step observations.
The report states this distinction and records extra planning compute. Current
pilots use the unchanged fixed mesh-preload controller, whose recorded controls
qualify11/16 scenes under the earlier protocol; they do not use velocity
feedforward or recorded future commands.

`scripts/dev/audit_feedback_replays.py` independently rebuilds particles from
saved executed poses and reproduces every full-trajectory driver residual within
1 micrometre. It also reports the immutable prefix's hard constraints and the
remaining segment's full task screen separately. Full qualification retains the
complete task, recorded-control qualification, both integration steps, timestep
agreement, and hand tracking including the prefix. Saved state omits warning
counters, so the inherited prefix requires a warning-free original replay.

`scripts/dev/render_feedback_replays.py` presents three-column physics movies,
height/goal/timestep traces and the separate prefix/continuation/full-task table.
The main campaign page links `feedback_replay_report_v1/index.html` and displays
feedback optimization progress. Integration gates are labeled separately from
longer trials. When best-so-far selection keeps update zero, the page explicitly
states that the fresh and selected trajectories coincide.

Artifact convention in the first feedback runner: `<variant>_step<i>.npz` stores
full-horizon `R`/`T`, while `q`, `time`, and `integration_states` cover only the
continuation including its anchor. Use `prefix_frames` in `report.json` to align
them; do not interpret all arrays as having the same frame origin.

## Sample-and-select baseline

The feedback runner also accepts `--method best_of_k --iterations 0 --k 49`.
It draws a bank from the same fresh seed and selects by the same declared model
objective (with feasibility ranking only when the base protocol enables it).
Candidate zero must match the separately evaluated initial sample bitwise,
including poses, particles, Gaussian noise and routing RNG. Every candidate's
predicted poses are saved. The real-checkpoint two-candidate gate matches the
earlier noise-optimization initial draw exactly, selects the minimum objective,
and preserves both old-plan physics continuations exactly; 26 feedback and
unconditional-guidance CPU tests pass.

At H20, 49 candidates plus the separately evaluated paired initial draw use
3,400 model forward invocations. This equals 24 noise updates when counting
gradient recomputation, but does not equalize FLOPs or wall time: backward work
differs. Both actual calls and forward NFE are retained. New execution artifacts
also carry explicit pose/continuation frame origins. Receipt:
`feedback_bestk_qualification_v1.json`. The first gate attempt stopped before
model loading because its candidate provenance string was malformed; the next
snapshot corrected that metadata and passed.

## Early repeated feedback

`--frame 4 --replan-every 8 --iterations 6` starts correction after 0.2 seconds,
before the first can's original contact failure. It replans at global frames
4, 12, 20 and 28, with remaining horizons32,24,16,8. Each plan uses the latest
four observed frames, optimizes the same original goal, then executes eight
frames. The final deadline remains frame36. A shorter leftover interval is
executed from the preceding plan if it cannot contain the full goal hold.

`guidance.receding_feedback` compares the original continuation, fresh sampling
at every correction, and initial-noise optimization at every correction.
Gaussian noise and routing RNG are paired explicitly between the latter arms
at every stage; their later observed contexts can differ because their executed
controls differ. All physics state persists across command segments. The finer
step replays the same realized segmented program, including updated command
anchors; it does not replan a separate finer-step policy. Per-stage contexts,
initial draws, retained plans, commands, model calls and objective traces are
saved. The complete task screen includes the early prefix.

The4+8+8+8+8 partition and restoration at frame4 preserve recorded poses, joint
positions and complete integration states bitwise on all16 scenes at both
steps (32 checks). Ten feedback/replay CPU tests pass. A real can gate performs
one update at each correction: all observed context windows, zeroed future
labels, paired noise/routing, best-so-far choices and the unchanged original
continuation pass. A placing zero-update gate produces bitwise identical
sample/optimizer executions, including full physics states, at both steps.
Receipts: `receding_physics_qualification_v1.json` and
`receding_driver_qualification_v2.json`.

These qualify integration, not task success: the one-update can continuations
still fail physical checks and timestep agreement. Six updates at each of four
stages use3,136 model invocations for the optimized arm and224 for the fresh
arm (3,360 combined), including gradient recomputation. This is a different
information/compute allocation from a single late correction; all actual counts
remain in the report rather than asserting matched FLOPs or real-time control.
# Replaying the complete sample bank

`scripts/dev/audit_feedback_candidates.py` replays every saved candidate from a
completed `run_feedback.py --method best_of_k` run at both fixed physics steps.
It retains the original executed prefix, commands only the gripper, and reports
complete-task and continuation-only checks separately. Candidate 0 and the
model-selected candidate must reproduce the driver positions, joints, time, and
complete integration state bitwise; full-task residuals must agree within 1 µm.
The real two-candidate integration gate passes these four replay checks
(`feedback_candidate_audit_qualification_v1.json`).

This diagnostic adds simulator access. It distinguishes lack of usable sampled
continuations from poor selection by the learned cost; it is not a new
learned-model result or an independent validation set. No oracle-selected plan
is promoted to success, and earlier prefix failures remain disqualifying.
`feedback_candidates_report_v1/index.html` shows every candidate, including
failures, with model cost versus executed goal and complete-task violations.


## Pairing and inherited-prefix interpretation

`scripts/dev/audit_feedback_pairing.py` compares completed context, selection and
update-budget experiments. Context comparisons must share Gaussian noise/routing
and reproduce the original continuation identically. Noise optimization versus
best-of-K must also share the entire initial sampled trajectory and its physical
execution. Repeated-feedback budget comparisons must preserve all fresh-sampling
rollouts and executions, plus the optimizer's initial noise/routing at every
replan. A real one-versus-six-update can check passes all four stages and four
shared physics replays (`feedback_pairing_qualification_v1.json`).

Both49-candidate can banks are complete. Neither bank contains a candidate that
passes the remaining-segment task screens at both physics steps. Separately,
their fixed prefix already differs by4.13mm and10.90degrees between physics
steps, so no candidate can pass the full-rollout5degree stability criterion.
The candidate report explicitly separates that inherited failure from the
remaining segment. It is incorrect to attribute every full stability failure to
the newly sampled continuation. No acceptance thresholds were changed.

## Comparing only predictions that were actually executed

`scripts/dev/render_receding_diagnostics.py` compares each retained plan's next
eight predicted frames with the same eight frames of independent execution,
before the next replan changes the commands. Every predicted anchor must match
its observed physics state within the explicit FP32 conversion tolerance. Later
unexecuted parts of a replaced plan are excluded. The figures show object height
and position disagreement resetting at each new observation, then accumulating
through the committed segment. Full-task results remain separate.

Results: `receding_prediction_report_v1/index.html` under the campaign root.
This uses the0.5ms engine that supplies feedback. Its0.25ms companion is still
a replay of the realized commands, not another independently replanned policy.
The figures diagnose the prediction/execution interface; they do not isolate
world-model error from approximate actuation or engine mismatch.
