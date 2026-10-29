# Staged picking implementation progress

Active work authorized on 2026-09-14. The full plan is
[PICK_GUIDANCE_EXPERIMENT_PROPOSAL.md](PICK_GUIDANCE_EXPERIMENT_PROPOSAL.md).

## Current stage

The main inference methods are implemented and real-checkpoint experiments are
running on both PRO 6000 GPUs. The focused suite passes **54 tests**, including
production parity, cross-window gradients, hidden future actions, constrained
selection, inpainting, beams, actuator-first/coarse-residual updates and replay
frame conversion. The receipt is
`validation/guidance_staged_2026-09-14/tests_routing_v15.xml`.

Start with the [combined progress and visualization page](../../validation/guidance_staged_2026-09-14/overview.html).
It refreshes job status and exposes new paired GIFs as experiments finish.
The completed jar experiments have not established a geometry-accepted held
pick; simulator replay remains pending NVIDIA EULA approval. These are recorded
simulation dataset scenes, not real-camera or real-robot trials.

## Initial implementation evidence

Eight opposing-contact proposals were prepared for the pinned spice-jar scene
using observed geometry and URDF finger limits. Absolute finger reconstruction
matches the recorded joint states within 1.73 micrometres. Proposal clearance is
a sampled geometry check, not simulator feasibility. Artifacts:
`validation/guidance_staged_2026-09-14/grasp_scene.pt` and `.json`.

The first GPU experiment used frozen commit `3453c1f`, the same old 40K checkpoint,
scene and paired seeds as the preceding diagnostic, and six updates at lr 0.03.
Both runs completed with finite gradients. Final target errors are 75.95/60.97 mm;
final farther-finger gaps are 21.54/16.33 mm. Neither has a held grasp. Sampled
intersection is zero in both directions, but contact remains insufficient.
Runtime is 140.4/139.8 seconds. The staged objective is implemented and exercised;
these results do not establish a quality improvement over current guidance.

- [Grasp candidates](../../validation/guidance_staged_2026-09-14/visuals/grasp_proposals.png)
- [Paired movie, seed 54321](../../validation/guidance_staged_2026-09-14/visuals/comparison_seed_54321.gif)
- [Per-frame traces](../../validation/guidance_staged_2026-09-14/visuals/traces_seed_54321.png)

Source-coordinate replay preparation matches recorded hand positions within
2.31 micrometres. Isaac Sim startup stopped at its explicit NVIDIA Omniverse
EULA prompt. Permission to accept that agreement has been requested; simulator
replay remains pending. The replay script has not yet passed simulator startup.
Table height and object velocity are reconstructed estimates, and controller/USD
vintage must still be qualified with recorded commands.

The next four paired methods finished on frozen source `1bb812e`. The independent
screen accepted none. Final target errors (seeds 54321/54322) are 55.4/61.9 mm for
screened best-of-16, 98.2/89.0 mm for constrained stages, 41.4/61.9 mm for selection
then refinement, and 66.7/73.5 mm for two shorter optimization starts. Runtime per
seed is approximately 100/141/122/114 seconds respectively, so these are exploratory
comparisons with reported compute, not exact matched-time arms. No draw in either
16-sample bank passed the interaction-only screen. All bank candidates are saved.

Eight new scene identities were pinned before model sampling in
`validation/guidance_pick_pilot_2026-09-14/manifest.json`: four development assets
(5624, 8026, 6079, 5173) and four reserved assets (8270, 8043, 6778, 6839). These
are held out from guidance tuning, not asserted to be absent from training. All
eight input tensors reproduce their exact source frames bitwise after the
canonical shift; `source_audit.json` records that check.

A newer clean-context continuation checkpoint was pinned at step 15,000 of the
existing 20K run. The corpus, split, exclusion and normalization audit is in
`validation/guidance_staged_2026-09-14/checkpoint_comparison_preflight.json`.
Checkpoint SHA-256: `dbb950fc6045a6fa4501f4079b7a94d232fe6c5d84e77f103a1965273c4998f1`.
Its two ordinary jar samples have final errors 106.6/95.1 mm and no held grasp;
the first also has roughly 9.5 mm sampled reverse-direction intersection.
This tiny diagnostic is not an aggregate checkpoint-quality comparison. The
continuation changes training duration, LR and context noise together; no new
training was launched by this work.

The prefix-aware local denoising method also completed both jar seeds on frozen
source `3e10d61`. It is substantially cheaper than full-chain optimization but
does not recover a held pick with the initial settings. Coarse temporal updates
and coherent observation biases have regression tests; real-sample experiments
for those additions are next.

Actuator waypoint inpainting now clamps normalized deltas to the appropriate
flow marginal at each frame's timestep. The entire reach/close/lift/hold path is
encoded through the articulation helpers, with window-aware indexing. Only
actuated, valid pose channels may be prescribed; passive object motion remains
generated. This is approximate conditioning of an unconditional model, with
desired achieved actuator poses, not a validated command sequence. Tests cover
cross-window pose recovery, unchanged empty-mask sampling, hidden future truth,
and rejection of passive-body or inactive-joint constraints. The focused suite
now has 40 passing tests, plus one mesh-preparation test.

All eight pilot scenes have collision geometry and joint calibration. Seven
have geometric grasp candidates. Reserved asset 6778 has no sampled clear
approach and remains an explicit planning failure. A can component failed the
original Boolean convexity test despite micrometre-scale plane deviations. The
preparer now requires closed, consistently wound meshes, supporting-plane error
at most 2 micrometres and relative hull-volume error at most 1e-5; it records both
errors. The original mesh is retained, and a concave-surface test guards against
accepting a meaningful dent. Reserved outcomes have not been used to tune grasp
search or model settings.

The recorded jar and three of the four development recordings pass the
interaction-only validator. The fourth has a maximum hand-relative translation
step of 3.58 mm and fails the provisional 3 mm threshold. All have both fingers
within the 5 mm proximity limit during the final six frames. This is a useful
positive-control check, with a remaining threshold-calibration issue; it is not
simulator replay. Recorded objects are still moving near the end, so they do not
provide a positive control for holding still at a newly requested goal. The
report is `validation/guidance_pick_pilot_2026-09-14/recorded_development_validation.json`.

Complete-window beam search now has tests for exact one-branch parity, fresh
continuation noise, complete-trajectory-only validation and replay of the selected
lineage. Optional pose and density input biases retain a separate true batch and
fixed task goal. These are input-sensitivity experiments until execution is
qualified. Grasp proposals remain fixed in object coordinates for these probes.

An expanded six-panel movie includes ordinary, current, staged, screened
selection, hybrid refinement and the recorded reference:
[search comparison](../../validation/guidance_staged_2026-09-14/visuals_search/comparison_seed_54321.gif).
Its captions use actual sampled finger-to-object gaps, rather than distance to
one prescribed contact pair. A close-looking image can still have substantial
intersection; continuous traces accompany the movies.

A later audit tightened fallback selection: an interaction-valid goal miss now
outranks an interaction-invalid goal hit, and wholly invalid banks are ranked by
maximum **interaction** violation before task loss. The earlier fallback used the
maximum residual including goal error. This did not change acceptance labels,
but could select the wrong diagnostic fallback. Existing frozen runs retain
their original rule and receipts; saved banks can be re-ranked without sampling.

## Remaining sequence

Status entering 2026-09-15:

- Implemented and exercised: staged objectives, independent screening,
  best-of-K/refinement/multistart, augmented-Lagrangian constraints, local
  denoising guidance, coarse noise updates, actuator inpainting and window beams.
- The completed jar checks still have no geometry-accepted held pick. Full
  actuator inpainting gives final errors 23.6/40.2 mm on old 40K and 23.8/43.4 mm
  on the 15K continuation, but fails sustained-contact and collision checks.
  Fingers-only inpainting, beam search and coarse updates also remain failures.
- GPU 1 is running a frozen development packet: four geometries, paired
  checkpoint sample banks, current/staged/constraint/hybrid arms, and four seeds
  per old-checkpoint method. These are fixed-work exploratory comparisons with
  measured runtime, not a completed matched-deadline benchmark.
- Pose/density sensitivity probes finished on the jar. For 5 degrees of bias,
  maximum object-path changes are 9.73/51.40 mm relative to clean-input planning;
  for 5 mm translation they are 5.00/8.11 mm. Same checkpoint, task goal and
  Gaussian arrays were verified. Pose bias also changes the constructed actuator
  path; these are end-to-end input-sensitivity results, not execution errors.
- The existing clean-context training finished at 20K; its final checkpoint
  SHA-256 is `99ec6cc6102a0d654c375c9f7b9cc9c86edd0964acf9de1acb42d4fb578c9fba`.
  Data/model/normalization/training configs match the pinned 15K continuation.
  `final_checkpoint_preflight.json` extends the corpus audit. GPU 0 is available
  for its checks and search across observed grasp candidates and physical timings.
- Waypoint search enumerates up to eight grasps and three approach/closure
  schedules, preserving the same Gaussian draws across actuator paths. Passive
  object motion remains generated and independent validation decides selection.
  The focused regression suite passes **46 tests**; results are recorded in
  `validation/guidance_staged_2026-09-14/tests_final.xml`.
- Simulator replay is pending the explicit NVIDIA EULA approval. Observed-state
  replanning and execution robustness depend on qualifying recorded-command and
  achieved-pose-controller positive controls first.
- A simulator-free URDF check reproduces the recorded future hand/finger
  positions within 4.84 micrometres. The native source frame matches the URDF
  link frame. Recorded final-grasp finger commands are approximately 6 mm farther
  closed than achieved q. Replay now distinguishes achieved-position tracking,
  recorded finger commands for the positive control, and an explicitly timed
  preload diagnostic. None of these controller variants has been qualified in
  Isaac yet. Requested goal error is measured at the model's template origin,
  with a tested inverse conversion from native simulator poses.
- All 24 grasp/timing combinations per seed failed interaction screening for
  both old 40K and final 20K-continuation jar models. The selected final goal errors
  are 26.0/23.9 mm and 20.1/37.6 mm respectively; improved endpoint proximity still
  does not establish a held grasp. All 96 candidate trajectories were retained.
  GPU 0 is now checking the final checkpoint on all four development scenes;
  GPU 1 continues the original development methods. The combined focused tests
  pass **50 tests**, including replay kinematics and command/frame conversion.
- Reserved evaluation, training changes, observation delays/occlusions and
  cross-domain variants follow the development results and execution checks.
  No additional training was launched by this work.

## Expanded search and evaluation controls

The old 40K prior banks now contain **256 draws**: four independently seeded
16-draw banks on each of the four development objects. None passes the held
interaction screen. Median worst-held finger gaps are 16.86, 19.31, 15.13 and
12.73 mm in development-scene order. The
[prior distributions](../../validation/guidance_staged_2026-09-14/visuals_prior/prior_geometry.png)
show contact gaps, sampled intersection and opening violations alongside recorded
interactions. A non-picking future can fail required contact without being
physically invalid. These counts are neither an execution success rate nor
evidence that the prior can never produce a usable grasp.

The [K-dependent selection curves](../../validation/guidance_staged_2026-09-14/visuals_prior/selection_k.png)
reuse the same ordered banks. Geometry-first ranking reduces its measured
interaction violations but often sacrifices target proximity. This describes
the tradeoff imposed by the selection rules; it does not independently validate
physical execution. Per-draw values and input/result hashes are retained in
`visuals_prior/report.json`. No reserved-scene outcomes were inspected.

Commit `64e3918` adds actuator-only versus actuator-first versus all-body
optimization and coarse updates with a separately bounded full-frame residual.
The original Gaussian arrays remain unchanged at zero updates; coarse and
residual components share a total trust bound. Actuator-first masks passive
gradients for three of six updates and uses all valid channels for RMS throughout.
Actuator-only uses its permitted channels for RMS. The frozen GPU packet
`search_variants_v14.json` compares five six-update variants on both jar seeds,
including matching all-body and coarse-only controls.

`long_refinement_v13.json` checks 24 updates on the old and final-continuation
checkpoints, plus a larger 0.5 RMS trust bound on the latter. Its compute is
intentionally larger than the initial six-update pilot. A matching final-model
six-update jar baseline is in `short_baseline_v16.json`. These are fixed-work
search diagnostics with measured runtimes; an exact 150-second deadline
comparison remains outstanding.

Commit `36cc2ca` adds two independent controls:

- **Representative-point RNG:** re-run each selected optimized Gaussian with
  the saved routing states first, requiring reproduction within 2 micrometres;
  then use three new per-window routing streams without re-optimizing or drawing
  new Gaussian noise. The sampler test verifies noise identity, caller-RNG
  preservation and reproducibility. Real-checkpoint results are queued in
  `routing_packet_v15.json`; `routing_v15/report.json` and its sensitivity plot
  are produced when that packet completes.
- **Constructed geometry:** prescribe a gripper path and deliberately translate
  the passive object with it after closure. This is a diagnostic positive
  control, not a generated or executed pick. Four of eight jar proposals pass
  all geometric limits. Counts on development scenes 0–3 are 2/8, 6/8, 3/8 and
  0/8. Failed controls exceed the sampled 1 mm intersection bound by 0.018–0.674
  mm on the jar and 0.116–0.670 mm on development scene 3. The default jar
  proposal exceeds it by 0.182 mm; proposal 3 passes and gets a separate paired
  guidance check. No tolerance was relaxed to make these paths pass. Results
  and plots are in `geometric_controls_v15/`.

The constructed controls reveal a limitation of the current geometric proposal
generator and straight approach/closure paths. They also show why small
threshold misses and large predicted separations need separate interpretation.
For example, the final checkpoint's current-guidance development-scene-2 seed
8312 passes contact and relative-pose limits, missing only the collision limit
by 0.093 mm, but its final goal error is still 183.76 mm. Continuous measurements
are retained instead of treating all failed draws as equally poor.

GPU 0 runs the final-checkpoint development packet, longer refinement, search
variants, routing/proposal controls and the short baseline in that order.
GPU 1 runs the original four-scene current/staged/constraint/hybrid packet.
Source snapshots, manifests and drivers are frozen per packet; output collisions
or failed commands stop dependent packets. CPU observers render completed trials
automatically. The strict deadline study, held-out method comparison, simulator
replanning, execution robustness, training changes and cross-domain extensions
remain subsequent stages, not completed work.

Review artifacts:

- [Combined live overview](../../validation/guidance_staged_2026-09-14/overview.html)
- [Progress page](../../validation/guidance_staged_2026-09-14/progress.html)
- [Completed-checkpoint development page](../../validation/guidance_staged_2026-09-14/final_checkpoint_progress.html)
- [Jar comparison table and plot](../../validation/guidance_staged_2026-09-14/report/index.html)
- [Fixed versus searched actuator paths](../../validation/guidance_staged_2026-09-14/visuals_control_search_v14/comparison_seed_54321.gif)
- [Actuator-conditioning GIF](../../validation/guidance_staged_2026-09-14/visuals_conditioning/comparison_seed_54321.gif)
- [State and density sensitivity](../../validation/guidance_staged_2026-09-14/visuals_sensitivity/sensitivity.png)
- [Exact development commands](../../validation/guidance_staged_2026-09-14/development_packet_v6.json)
- [Live worker status](../../validation/guidance_staged_2026-09-14/development_packet_v6_status.json)
- [Final-checkpoint development status](../../validation/guidance_staged_2026-09-14/final_development_v10_status.json)
- [Physical-input audit](PICK_PHYSICAL_INPUT_AUDIT.md)

Results and limitations will be appended as each stage completes. This file is
a progress record, not a claim that every planned experiment has run.
