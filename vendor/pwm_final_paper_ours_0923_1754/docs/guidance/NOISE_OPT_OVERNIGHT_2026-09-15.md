# Standard noise optimization: September 15, through 11 a.m. New York

Current results: [September15 review](REVIEW_2026-09-15.md) and the local
`validation/guidance_noise_2026-09-15/review_11am/index.html` entry page.
All priority GPU runs finished by10:24a.m. Eastern. The history below records
the original plan and subsequent protocol changes; it is not a list of runs
still pending.

The user authorized dev-blackwelle and requested strong results across different
pick and place samples. Review deadline: **2026-09-15 11:00 America/New_York
(EDT; 15:00 UTC)**. Stop starting expensive work by 10 a.m.; reserve the last hour
for reports and videos. The allocation is Slurm **10446392**, babel-l9-20, two
idle RTX PRO 6000 GPUs verified before launch. This experiment uses existing
weights; it launches no training or model-versus-model comparison.

## Fixed model and data

Primary EMA checkpoint:
`/data/group_data/katefgroup/runs_pwm_robertre/pickup_jointed_unet_bidir_uncond_cleanctx20k_from40k_0914/checkpoints/final.pt`

SHA-256: `99ec6cc6102a0d654c375c9f7b9cc9c86edd0964acf9de1acb42d4fb578c9fba`.

Unconditional bidirectional articulated UNet, action dropout 1, clean context,
checkpoint-owned normalization, FP32, no compilation or autocast. C4, internal
F16, pyramid25, uncertainty scale1, dt0.05s. Initial Gaussian pose/joint noise is
optimized; there is no VAE latent or prescribed gripper path. Model weights stay
frozen. Future recorded states/actions are unavailable to generation.

Corpus preflight receipts: `validation/guidance_staged_2026-09-14/` files
`checkpoint_comparison_preflight.json` and `final_checkpoint_preflight.json`.
Training corpus: four canonical pickup roots, 189 train /21 validation shards,
registered exclusions, sorted-shard validation split. Task subset roots are
`/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_pick_success` and
`/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_place_success`.

Pin **four development + four guidance-reserved scenes per task**, 16 total,
before looking at generated outcomes. Existing pick manifest seed831 lives at
`validation/guidance_pick_pilot_2026-09-14/manifest.json`; place seed915 manifest
is `validation/guidance_noise_2026-09-15/place_scenes_v1/manifest.json`.
Pick eligibility uses observed travel/gap/size as previously recorded. Place
eligibility uses a finite separately recorded task goal, with no size or future
motion filter. Distinct assets within each task; these are held out from guidance
tuning, not asserted unseen during model training. Geometry-preparation failures
stay in the inventory and denominators; never replace them with easier scenes.

Pick target: observed object template origin +10 cm vertical. Place target: the
segment's explicit requested goal pose, converted from native to model template
coordinates using verified surface correspondence. This is not an achieved
future endpoint. The collector's quaternion comment says xyzw incorrectly;
PlaceEnv stores/consumes wxyz. Place objectives include orientation, 0.04m/rad.
Task-specific geometric checks distinguish held picks from resting released places.

Joint calibration uses observed q, observed finger transforms, URDF limits and
dt; future q only audits calibration. No grasp proposals in the main comparison.
Place support height cannot be inferred from an airborne initial object. Its
reference comes from the requested resting pose and the collector's 2mm clearance;
the table pose itself was not saved. State that uncertainty in every report.
Object collision components may need explicit per-component convex-hull proxies;
record this choice, keep decompositions (e.g. cup holes), never silently label
proxies exact source or simulator geometry. Exact source/template correspondence
remains required even if asset catalog indices have changed.

## Qualification and experiment order

The real checkpoint qualification at `gpu_qualification_v2/report.json` passes:
production parity and exact replay at H16/H32/H36, articulation masks, no future
state/action leakage, zero-update parity, finite gradients across the H32 boundary,
identical forward trajectories with detached context, finite-difference agreement,
and one-update improvement. Its nearby generated-endpoint target is a functional
check, not task success. Version1 records an assertion-device error in the probe;
version2 corrects it. Initial focused CPU suite:59 passes. Add placing tests and a
real scene CLI smoke before broader placing runs.

| Device | Priority order |
|---|---|
| GPU0 | Coverage first: goal-only6 updates on development scenes, H16/H32/H36. H32 full-context versus detached-context gradient ablation. Paired ordinary samples are saved with each run. Then fixed H36 goal6 on reserved scenes; best-of-K13 development sampling controls and selected longer24-update checks as budget permits. |
| GPU1 | Proposal-free physical6 at H36 on development scenes, all-noise versus actuator-only scope. Fixed physical6 on reserved scenes. Then24 updates and actuator-first/coarse-update ablations on development scenes. Additional seeds and observation-bias/FPS sensitivity after main coverage. |

Primary seeds8311/8312, fixed recipes before reserved evaluation. Additional seeds
8313/8314 support confirmation; they do not create new independent scenes.
Default noise Adam lr0.03, update RMS cap0.25, prior weight0.001, six updates.
Long refinement changes only update count to24. Actuator-first uses half the
updates on actuators before releasing passive channels. Coarse variant uses four
temporal knots, preserving the original Gaussian draw. Report exact costs and
actual model calls, including backward recomputation. K13 is a sampling control;
claim matched compute only where the measured costs support it.

Any data/interface failure stops the affected comparison until diagnosed. A
failed physical task remains a result, not an infrastructure failure. Drivers
have per-run timeouts, a UTC deadline, immutable manifests/source, atomic status
files and preserved logs/partial results. They run under the allocation and do
not depend on the interactive GPU shell. New recipes use new immutable packets.

H16/H32/H36 differ in physical time and compute: do not interpret their difference
as purely a gradient-policy effect. H32 full-versus-detached holds seed, target,
horizon and forward sampler fixed. Detachment is a gradient ablation, not real
closed-loop replanning. Apply the terminal goal once at the whole-horizon end.
Physical variants additionally impose a final settle/hold interval explicitly.

## Evaluation and deliverables

Selection for the standard baseline uses its declared objective. Geometry audits
are independent; screening/reranking is a separately named option. Save initial
and selected trajectories/noise, per-iteration loss and timing, per-window gradient
norms and update RMS, sampler/RNG identities, source/checkpoint/input hashes.

Report all scenes and seeds, with goal error, orientation error for place,
both-finger distances, sampled penetration both directions, support intrusion,
joint range/speed, frame steps at window seams, unsupported-rise/hover proxies,
and final held-relative motion (pick) or rest/release/stability (place). Ground
truth is a recorded comparison, not a generated baseline. Geometric proxies do
not certify contact forces or executed success. Do not count ordinary non-picking
samples as physically impossible solely because they do not accomplish the task.

By 11 a.m.: a progress/report page, synchronized baseline-versus-guided GIFs for
every evaluated scene, horizon/gradient comparisons, continuous physics traces,
scene-level summaries and runtime/memory tables, explicit incomplete/failed runs,
a tested locally committed implementation, and recommendations based on results.
The method is implemented and being evaluated; improved generalization is a
hypothesis to test, not a promised overnight outcome.

Isaac Sim replay is independently pending the user's response to NVIDIA's EULA.
No acceptance or execution claim without that approval and a qualified recorded
controller replay. No push unless explicitly requested.

## Launch receipt, 3:18 a.m. New York

Both queues launched as detached Slurm steps at07:18:40 UTC on allocation10446392.
Authoritative manifests: `packet_gpu0_v4.json` (SHA-256
`51f031437a759d66ff5a9085c0e25a0632e4dc4b1d5b7aae8df98357a62910ca`)
and `packet_gpu1_v4.json` (`47ccd5eb1507b287333d6604fd2a5ca3bdc989e9925015e67e22139a31854713`),
under `validation/guidance_noise_2026-09-15/`. Model source is frozen at2d1c33d;
reporting source at57df222. The ordered backlog is deadline limited; do not treat
all448 planned seed trials as completed or guaranteed. Active trials stop by14:40
UTC, with automatic report updates throughout and a15:00 UTC review deadline.

All16 exact scenes qualified geometry/calibration preparation, without replacements.
Actual paths and recovered asset identities are in `scenes_v2.json`; original
selection manifests and failed preparation receipts are preserved. Focused suite:
61 passes; after diagnostic additions the affected21-test suite also passes.
The real-checkpoint gate passed, as did an H16 placing CLI smoke test.

`first_pair_identity.json` verifies bitwise-identical initial trajectories and
identical targets for goal6/physical6 on development asset5624, seed8311, H36.
That first pair reduced endpoint error160.4mm to65.0mm/85.5mm respectively; both
failed the geometric task screen. This is early development evidence and does
not establish robust picking, placing or executed success.

Live report: `validation/guidance_noise_2026-09-15/index.html`; paired movies and
plots live inside each trial directory. Worker status and the durable handoff are
in the same root. The detached jobs survive terminal refresh, but cancelling
allocation10446392 stops them. No weights or commits were pushed.

## Continued work after the session refresh

`--objective physical --constraints` now supports augmented-Lagrangian residuals
for both tasks without grasp proposals. `guidance/physical_constraints.py` mirrors
the independently evaluated pick/place thresholds, including peak collision and
support violations, joint limits, hold/release, and the goal over the final six
frames. The wrapper logs signed residuals and updates their nonnegative dual
weights. Candidate ranking remains based on the base objective; add `--screen`
explicitly for feasibility-first ranking. Finite optimization of these geometric
constraints cannot establish physical execution.

Long-run optimizer controls are optional: `--lr-schedule cosine`,
`--warmup-updates`, `--lr-min-ratio` (default0.1), and
`--normalize-noise-gradient`. Normalization uses one norm over the complete
trajectory, preserving relative gradient magnitude between windows. Existing
constant-rate, unnormalized defaults retain bitwise parity with2d1c33d on four
CPU optimizer configurations, including articulated scope and coarse residuals:
`optimizer_default_parity_v1.json`. The affected tests pass38/38, receipt
`tests_optimizer_schedule_v1.xml`. These additions are not part of the running
v4 snapshot and require a fresh packet for GPU experiments.

The [DNO paper](https://arxiv.org/html/2312.11994v2) and
[official optimizer](https://github.com/korrawe/Diffusion-Noise-Optimization/blob/main/dno.py)
motivate testing more than a handful of updates and controlled learning-rate
decay. This is not an exact port: our articulated channels, windowed flow sampler,
trust region and objective differ. Six updates establish functionality, not
convergence. Initial six-update noise audits show no obvious RMS or lag-one
correlation collapse (`noise_stats_v1.json`); prior regularization is therefore
not assumed to be the main problem.

An independent CPU MuJoCo3.3.7 replay diagnostic is being qualified while the GPU
queues run. Its scripts and complete receipts live in the validation root;
it does not depend on the pending Isaac license decision. A dynamic floating
gripper follows the generated/recorded hand path; the object advances under
gravity/contact without prescribed future object poses. The engine, approximate
inertia, collision proxies and floating-hand controller differ from the source
PhysX scene. Recorded controls are essential: some reproduce closely, others
are sensitive to timestep or differ substantially. No blanket execution pass is
justified. Recorded finger commands and achieved finger positions also give
very different pick outcomes; a separately labeled contact-preload controller
is under investigation. It uses planned proximity, never recorded future
commands for generated trajectories. Controller results are exploratory and
must not be presented as blind validation of source-simulator execution.

## Current continuation, 5:20 a.m. New York

The first64 trials are complete:16 scenes, two seeds, goal6 versus physical6.
All32 paired initial trajectories/noises/targets are bitwise identical. None
passes the shared full-mesh task screen. One sampled-point placing acceptance
was a false release: finger surfaces still overlapped. The optional FCL geometry
costs/screens and gripper-table support fix are documented in
`COLLISION_SURFACE_GUIDANCE_2026-09-15.md`; local source snapshots remain frozen.

The first can's100-update runs completed with selected final goal error1.03mm
(goal only) and0.42mm (physical), but maximum mesh overlap23.0mm and2.62mm.
Both fail the held task criterion: worst goal error over the final six frames
is41.7mm/53.8mm despite the accurate last frame. The existing pick physical
objective rewards the terminal target; placing already averages its goal over
six frames. Explicit AL constraints also bound the worst goal error over that interval.
No success is claimed from endpoint accuracy alone.

At09:13UTC the supervisors transitioned after clean sampler completion to
`packet_gpu0_v6.json` (SHA-256
`5fe1150f2b85c6b5a6f9dfbbd9d5555a1827a4aed29e1e0a2dec34b9092d126b`)
and `packet_gpu1_v7.json`
(`a6fa8e862dca495ec85d7d03045819556333cd61f0603eb270885a152deb08bc`).
The remaining primary comparison retains24 updates, seed8311, across all16
scenes.100-update convergence is limited to the first development pick and
place, freeing budget for early H16/H32 gradient-route checks and mesh/table
physical/AL comparisons. This changes the remaining compute allocation, not
the scene inventory or completed results. Pending arms remain deadline limited.
Standard baselines use source_v5=f47fc92; mesh/table arms use
source_v8=ae244ec. Two real pick/place GPU smoke runs pass the new geometry
path, in addition to29 relevant CPU tests. Default GPU backward reductions have
small repeat variation; `GRADIENT_REPEATABILITY_2026-09-15.md` records the probe
and optional deterministic CLI setting. Primary settings remain unchanged.

The independent replay now compares controllers at matched0.5ms integration
and repeats fixed mesh-preload controls at0.25ms.11/16 recorded scenes meet
the declared pose/tracking and timestep qualification; all16 remain reported.
Generated replays use only the generated gripper trajectory with the fixed
contact-preload rule. Poisoning recorded future states/actions with NaNs leaves
execution arrays bitwise identical. Many generated motions still diverge or
depend on timestep; these are local independent-engine diagnostics, not a
source-PhysX or hardware benchmark. Latest movies:
`validation/guidance_noise_2026-09-15/generated_replay_report_v2/index.html`.
The main page links these alongside recorded-control movies and uses a shared
v2 mesh task screen, including object and gripper support-plane checks.

## Additional diagnostics, 5:55 a.m. New York

Tolerance-scaled AL is an optional experiment, documented in
`CONSTRAINT_SCALING_2026-09-15.md`, implemented at641a016. The affected32 tests
and real H16 pick/all-noise plus place/actuator-only smoke runs pass:
`scaled_constraint_gpu_gate_v1/report.json`. At09:49:58UTC GPU1 transitioned
cleanly to `packet_gpu1_v8.json`, SHA-256
`ea2f346949f8ebbebd19baf0c90c4bed2919386276d2da5add4d53b86fffd153`.
Its first four runs compare scaled AL24 and actuator-only mesh physical24 on
the first can and bowl; the remaining primary24 coverage is preserved. These
four entries explicitly use source_v9=641a016; existing baseline entries retain
their earlier source. No successful task outcome has been established.

The first bowl's goal100 run finished with final position error0.177mm and
orientation error0.221degrees. Its shared mesh audit still fails: overlap4.92mm,
insufficient finger release, settling and held orientation. The first can and
bowl convergence figures are generated by `scripts/dev/render_noise_convergence.py`
into `convergence_report_v1/`. They distinguish last-frame goal accuracy from
the worst goal distance over the final six frames and plot collision/support
checks using unchanged thresholds. Prefixes are correlated saved candidates,
not additional independent trials.

`fixed_actuator_resampling_v2/index.html` shows one can scene with exactly the
same gripper path and different object noise. All8 cases preserve gripper poses
bitwise. For the physical100 plan, the original joint generation has0.42mm final
goal error; same-noise gripper inpainting has9.84mm and three fresh object-noise
draws have40.22/37.01/5.47mm. All fail the shared mesh screen. Existing flow
inpainting also changes the same-noise object prediction; this is an approximate
conditional consistency diagnostic, not calibrated uncertainty or exact
conditional sampling. Both ordinary and optimized plans, all draws, and movies
remain visible. This motivates the actuator-only ablation without asserting
that it will fix the problem.

An exploratory contact-force diagnostic fits one gravity wrench using FCL
closest points and friction cones. Analytic cases pass, but the one-point-per-link
approximation rejects some recorded grasps even at friction1. It is therefore
not used as an acceptance screen or guidance penalty. Receipts:
`gravity_wrench_analytic_v1.json`, `gravity_support_recorded_v1/summary.json`.
It omits contact patches, torsional friction and dynamic inertial forces; fitting
gravity alone is also weaker than arbitrary-wrench force closure.

The placing optimizer's legacy `target_distance_mm` progress field contained
its mean held-goal cost, while final `measure()` diagnostics already measured
the endpoint correctly. The field now consistently measures the endpoint;
`goal_cost_mm` retains the objective's temporal reduction. A regression case
with a correct final pose and earlier12mm errors distinguishes the two.
Thirteen physical-guidance tests pass. `audit_noise_prefix_endpoints.py`
remeasures saved prefixes directly from their pose tracks and targets into
`prefix_mesh_summary_v3.json`, retaining the original diagnostic value and
unchanged v2 mesh screens. For example, bowl mesh24's objective mean3.618mm
differs from its actual endpoint5.005mm. Main final evaluations were correct;
no trajectory, selection, task threshold or objective was changed by this fix.

The matched H32 full-gradient versus detached-context comparison completed on
the first development can and bowl, seed8311, six updates. Initial trajectories,
noise, targets, sampler settings and checkpoint identities match exactly.
Full gradients reduce endpoint error93.56→32.16mm for the can and31.53→8.21mm
for the bowl; detachment reaches53.97mm and29.76mm. Detachment gives exactly zero
first-window gradient and noise change; full gradients adjust both windows.
Both use1120 actual model calls including recomputation. This is a gradient
ablation, not closed-loop replanning. Geometry failures remain. Movies, traces
and pairing checks: `context_gradient_report_v1/index.html`.

`snapshot_source_audit_v1.json` verifies all archived source files against Git
blob identities: nine snapshots,440–455 tracked files each, no mismatches or
unexpected Python modules. The legacy `code_revision()` helper can discover the
ambient parent checkout when invoked from an archive. Consequently50/56 audited
run protocols contain an ambient `git_commit` different from the executed
archive. Use the verified archive revision, immutable command/source-root and
manifest receipts for actual source provenance; original protocols remain
preserved. This metadata ambiguity did not change the executed source.

The live provenance helper now recognizes a full-hash `SOURCE_REVISION` marker
before consulting Git. It records that declared source commit with unknown
dirty state and explicitly unverified archive integrity. It never labels an
archive clean merely because a marker exists. Malformed markers fail instead
of falling back to an ambient parent checkout. Seven focused provenance/CLI
tests pass; existing frozen experiment snapshots and original receipts remain
unchanged, with the independent blob audit providing their content verification.


## Primary comparison complete at 9:23 a.m. New York

The seed8311 matrix has all80 cells:16 scenes x one shared initial draw and four
objective/update combinations. All64 unique optimized runs (including retained
24-update prefixes of the four100-update runs) have independent replay audits.
`primary_protocol_qualification_v1.json` verifies equal effective settings,
checkpoint/EMA/precision/normalization and per-scene inputs, geometry, calibration
and target orientation. The main overview also checks initial trajectories and
targets bitwise. The new schedule options stayed disabled in this comparison.

| Eight scenes per task, seed8311 | Initial | Goal6 | Physical6 | Goal24 | Physical24 |
|---|---:|---:|---:|---:|---:|
| Pick median predicted final point error, mm |128.91|50.84|66.93|2.65|17.48|
| Pick median maximum mesh penetration, mm |9.86|8.51|3.93|11.59|2.58|
| Place median predicted final point error, mm |40.08|11.69|19.05|2.44|5.77|
| Place median maximum mesh penetration, mm |5.97|7.10|6.06|8.18|5.79|

Every cell still fails the complete task screen and qualified physical execution.
These medians describe predictions, not manipulation success. The report also
separates the four development and four guidance-reserved scenes per task.
The source-task versus requested held-goal/released-place mismatch is documented
in `TASK_SEMANTICS_2026-09-15.md`; source recordings are not automatically positive
controls for the newly requested task.

## Final budget priorities, 9:28 a.m. New York

The first three broad-region predictions pass their mesh task screen but fail
independent execution. Late feedback on the first can and asset8026 likewise has
no complete physical pass. The can's fresh-sample lift is accompanied by roughly
32.6kN hand/table crushing in the floating-hand engine; lower final lift alone
therefore does not imply worse overall physical quality. See the force plots and
prediction-versus-execution GIF in `grasp_execution_report_v1`.

Preserve all eight region cases, three early repeated-feedback6 cases, and both
can best-of49 banks. Then run the bowl's late observed/predicted feedback24 pair,
and deepen repeated feedback to24 updates at each of four replans on one can and
one bowl. These take precedence over further AL weight sweeps. This is a larger
compute budget, not a matched-compute comparison against six updates per replan.

Safe-boundary handoff manifests are `packet_gpu0_v10.json` (SHA256
`5c567e1b78024715f26dafb4ef915e75baddda97acf538bde86fc8a5f5a7249f`)
and `packet_gpu1_v13.json`
(`116f1abe80cfedf3926842917dec4604950830f0fc5cb12bb83a9e277bfc28e8`).
They start new runs only through14:05UTC and stop active work by14:30UTC, leaving
30minutes before the15:00UTC review. The extra five-minute launch window recovers
part of the earlier18-minute GPU0 import failure. Every active sampler finishes
before its supervisor is replaced; running manifests/source are unchanged.
The earlier failed launch is preserved, and absolute script entrypoints passed
import qualification from the worker's actual source directory. The shared
packet launcher now honors per-entry source overrides for both cwd and PYTHONPATH.


## Early-state sampling control added at 9:53 a.m. New York

The late can banks inherit a prefix that already fails the full stability test.
The first0.2s of the can and bowl instead pass the existing hard prefix checks;
timestep differences are0.0014mm/0.0010degrees for can and0.274mm/0.372degrees for
bowl. After each deeper MPC run, add one best-of49 bank from that earlier state.
This reuses the existing frozen sampler and avoids attributing a doomed-prefix
failure to candidate generation. Selection still uses the declared model cost;
replaying every candidate is a separate diagnostic with added simulator access.

Additional manifests: GPU0 `packet_gpu0_v11.json`, SHA256
`dbfe6d54030cbb6fda1907513431f03041f89b1b676bbf26652d9680ff981b62`;
GPU1 `packet_gpu1_v14.json`, SHA256
`c644d116a6fca98885950d3dd0f0e96d5f5fa77a5a414ef55abb848b16b18906`.
Each contains one early best49 bank, with a360s estimate and600s timeout. Launch
only if it fits before the unchanged14:30UTC cutoff (latest launch14:24UTC).
No current sampler is interrupted. If deeper MPC consumes that budget, the bank
is explicitly skipped. The11a.m. review deadline remains unchanged.
