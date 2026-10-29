# Unconditional picking guidance

**Current work, September 15:** the broader clean-context checkpoint experiment
is documented in [the overnight report](NOISE_OPT_OVERNIGHT_2026-09-15.md).
Its [16-scene comparison](../../validation/guidance_noise_2026-09-15/primary_overview_v1/index.html)
is complete; feedback and candidate-selection experiments continue toward the
11 a.m. New York review. It uses a different checkpoint from the archived pilot
below. See [feedback integration](FEEDBACK_REPLANNING_2026-09-15.md) and
[executed contact diagnostics](GRASP_EXECUTION_DIAGNOSTICS_2026-09-15.md).

**September 14 pilot archive:** this investigation round is complete. All six original arms, the
stronger-contact follow-up and 20 fresh-FPS rollouts finished on GPU 1. The code
is committed locally; experiment receipts and visualizations are saved below.

- [Design, evidence and literature report](UNCONDITIONAL_PICK_GUIDANCE_2026-09-14.md)
- [Comparison GIF, seed 54321](../../validation/guidance_pick_2026-09-14/analysis/comparison_seed_54321.gif)
- [Comparison GIF, seed 54322](../../validation/guidance_pick_2026-09-14/analysis/comparison_seed_54322.gif)
- [Per-frame measurements, seed 54321](../../validation/guidance_pick_2026-09-14/analysis/traces_seed_54321.png)
- [Per-frame measurements, seed 54322](../../validation/guidance_pick_2026-09-14/analysis/traces_seed_54322.png)
- [All metrics and compute costs](../../validation/guidance_pick_2026-09-14/analysis/metrics.csv)
- [Stronger-contact GIF, seed 54321](../../validation/guidance_pick_2026-09-14/analysis_followup/comparison_seed_54321.gif)
- [Stronger-contact GIF, seed 54322](../../validation/guidance_pick_2026-09-14/analysis_followup/comparison_seed_54322.gif)
- [All seven configurations, metrics and costs](../../validation/guidance_pick_2026-09-14/analysis_followup/metrics.csv)
- [Fresh-FPS measurements](../../validation/guidance_pick_2026-09-14/fresh_routing/results.json)

In the GIFs, blue is the object, red is the generated gripper and the green star
is the requested 10 cm lift target. Watch whether the object stays between both
fingers as it approaches the target. The recording is a reference with a
different final height; all panels use the same camera and scale.

The six-arm run uses a hash-verified copy of the committed source in
`validation/guidance_pick_2026-09-14/source/`. All initial samples are bitwise
identical across methods, all observed context is unchanged, and the saved
protocol/window audit passed. GPU 0 belongs to another session.

The goal-only optimizer reduced target error from 92.68 to 22.24 mm (seed 54321)
and 99.68 to 11.68 mm (54322), but both selected trajectories still lack two-finger
proximity throughout the generated segment. These are not successful picks.

Default interaction optimization reaches 34.06 / 26.01 mm target error with
1.11 / 0.00 mm maximum sampled gripper-in-object depth. Its final farther-finger
gaps remain 11.42 / 8.83 mm. Actuator-only optimization improves contact in the
first seed but is inconsistent across the pair. The stronger follow-up uses
12 updates at learning rate 0.02, contact weight 8, rise weight 10, penetration
weight 20 and support weight 10; it is an exploratory follow-up, not part of the
original six-arm comparison. Both original and follow-up results are retained.

The stronger costs finish at 58.85 / 31.65 mm target error with 5.12 / 6.84 mm
farther-finger gaps. They improve proximity but do not demonstrate a held pick.
Both goal-only trajectories also contain substantial rises when **all** sampled
gripper points are farther than 5 mm from the object, so the issue is not merely
that one finger misses a contact threshold.

Ordinary sampling replays bitwise exactly on the real GPU. With fresh FPS,
endpoint errors change by at most 0.91 mm across the 20 checks, preserving the
main findings. Contact flags near 5 mm can change, so continuous gaps are saved.
My recommendation is to add explicit grasp formation/maintenance and replay
generated actuator trajectories in the source simulator. Increasing a scalar
guidance weight alone has not solved the interaction.

## Scope

Use the existing 40K unconditional bidirectional, articulated pickup UNet (EMA,
checkpoint SHA256 `f4c4d559d1ef520fa0465650e1fe2a48b6e4a4abe991cb84e159da8d68f8b65c`).
Start with the pinned pick fixture in
`validation/phantom_regeneration_2026-09-14/pick_input.pt`: four observed frames,
36 generated frames in windows of 16, 16 and 4; pyramid/25; no future actions.
The initial task is a 10 cm vertical object lift from the observed anchor.
This is a development diagnostic, not a success-rate benchmark. No model training
or comparison between different checkpoints is part of this task.

Use the idle GPU 1 in allocation 10434620 (RTX PRO 6000,
`GPU-59c00b08-da17-b2da-a6e8-bc1f65a99986`), with a bounded experiment budget after
CPU contract checks and a single-sample GPU forward/backward check.

## Findings from the code audit

- The existing differentiable sampler already supports articulated pose channels.
- Existing planners always apply a supplied actuator plan. `apply_plan` rejects
  articulated bodies; the receding-horizon driver also overwrites generated
  gripper motion with that plan. Unconditional planning must preserve generated
  hand and finger tracks across windows.
- The current noise optimizer changes all pose noise, including the passive
  object. Restricting adjustable noise to the actuator is an ablation, not a
  guarantee of physical consistency.
- The existing phantom penalty uses a hard nearest-point distance gate against
  any gripper point. It neither establishes a two-finger grasp nor gives a
  gradient encouraging contact. It also penalizes legitimate free motion.
- Nearest-particle distance is unsigned. It cannot certify nonpenetration. Mesh
  availability and the table reference must be established before labeling any
  measured distance as physical penetration.
- A terminal goal applied separately to every short window demands premature
  completion. The first implementation will optimize the complete rollout and
  differentiate through carried predicted context.

## Completed work

1. Added unconditional rollout/noise optimization using the existing sampler and
   shared production window helpers. Preserve random FPS, and replay its RNG for
   comparisons between optimizer iterates.
2. Tested ordinary-sampling equality, hidden future inputs, carried context,
   articulated constraints, finite gradients, optimizer masking and cost counts.
3. Added explicit picking diagnostics and differentiable interaction costs, with
   synthetic physical cases and recorded-trajectory checks.
4. Ran ordinary sampling, best-of-K, goal-only noise optimization, and interaction
   objectives with all-body versus actuator-only adjustable noise on paired seeds.
5. Saved trajectories, per-frame measurements, runtime/memory/cost receipts and
   comparison GIFs, including failures and the exploratory follow-up.
6. Committed implementation and findings locally; no push.

## Completed checks

- Six unconditional sampler/optimizer tests passed, including exact production
  parity with random FPS, replay, future-state/action corruption, masked channels,
  cross-window finite-difference gradients and an actual actuator-only objective
  reduction. Five geometry tests passed on known lift/fall/penetration cases.
- Combined regression run: 59 passed, 6 GPU-dependent cases skipped, 2 deselected.
  Receipt: `validation/guidance_pick_2026-09-14/tests.xml`.
- GPU smoke on the actual 40K checkpoint, pinned pick scene, seed 54321, H36,
  pyramid/25, hidden actions, FP32: finite gradients across all 108 forward calls;
  one goal-only update reduced target error from 92.68 to 78.61 mm. The check took
  36.8 seconds including its baseline; peak allocated memory was 7.64 GB. This is
  optimizer verification, not evidence of improved physical behavior.
- Recovered source episode `batch_00002829`, element 3, asset 8059
  (`Kitchen_Table/spice_jar/asset_0062`) from the pinned shard. All 40 frames match
  the saved fixture exactly after the canonical coordinate shift.
- Recreated the collector's object points and aligned source gripper link points:
  maximum template residuals are about 1 micrometre. Object collision components
  are closed and convex. Gripper collision hulls are explicitly labeled proxies.
- The support reference inferred from observed object collision geometry is
  z=0.0100 m. The shard did not record the table pose; it remains an inferred plane.
- On the same GPU-smoke sample, maximum sampled gripper depth inside the object's
  collision components is 8.56 mm; the recording has none detected. One goal-only
  update leaves 8.51 mm and increases the final farther-finger gap from 13.65 to
  16.14 mm. Target improvement alone therefore does not resolve the interaction.

All numeric statements above refer only to the pinned diagnostic and its saved
receipts in `validation/guidance_pick_2026-09-14/`; no population rates are claimed.
