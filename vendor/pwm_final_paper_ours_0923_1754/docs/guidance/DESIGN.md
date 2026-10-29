# Guidance on the unified model: design (2026-09-10)

**Current implementation note (2026-09-15):** the dated design below predates the
unconditional articulated pickup checkpoint. `guidance/unconditional.py` now
optimizes complete generated rollouts through the existing differentiable sampler,
with hidden future actions, generated gripper motion retained across windows,
replayed FPS randomness, and all-body/actuator-only pose-noise masks. The entrypoint
is `python -m guidance.run_unconditional`; the supplied-action CLI rejects fully
unconditional checkpoints. Picking costs and signed collision-component diagnostics
are in `guidance/pick_objective.py`. See [current progress](UNCONDITIONAL_PICK_PROGRESS.md)
for historical validation. The [September15 campaign](NOISE_OPT_OVERNIGHT_2026-09-15.md)
uses the clean-context checkpoint across16 pick/place scenes. It adds proposal-free
mesh costs, optional constraint/schedule experiments, exact persistent physics
state, and [observed-state feedback](FEEDBACK_REPLANNING_2026-09-15.md) through
`python -m guidance.run_feedback`, including repeated replanning and sample selection.
The review entry page is `validation/guidance_noise_2026-09-15/review_11am/index.html`.
The optimizer is integrated and tested; reliable physical task completion remains
unresolved. Neither FK steering nor goal inpainting is implemented by these additions.

Status (2026-09-10): Layer 0 is implemented under `guidance/` (sampler, plan,
objectives, planners, horizon, run_guidance CLI; tests in
`tests/test_guidance_core.py` and `tests/test_guidance_planners.py`). First smoke
on the UNet 40K checkpoint, pushing, one 16-frame window: the batch's own plan
lands the block 4.3 mm from its ground-truth pose; on a 4 cm offset target,
best-of-4 under the GT plan 46.8 mm, CEM 36.2 mm (1,440 NFE), joint optimization
32.6 mm (320 NFE), tiny budgets. Layer 1 is not started. The historical
`guided_inference*.py` / `noise_opt_guidance*.py` entrypoints were retired with the
specialist model families (`docs/cleanup/ROUND4_RETIREMENTS.md`, in git history, last present at `79a3c84`); their validated
lessons are carried here, their code is not.

## 1. What the clean model gives a planner

- One family, two backbones (Transformer, UNet), pose-native: rigid bodies are
  pose tokens whose 6-D deltas are jointly diffused with the particles and
  supervised by the same v-loss for every rigid body, actuated or not
  (`utils/trajectory_objective.py`).
- The actuator's future is an INPUT. `batch.pose_actions` `[B, F, K, 6]` holds the
  achieved deltas of actuated bodies; the backbones add them to the actuated pose
  tokens' future rows, passive bodies read a learned no-action token
  (`models/unet_std.py:_pose_tokens`, `models/transformer.py`). There is no action
  dropout: the model has never generated an actuator trajectory on its own.
- Sampling is a stateless Euler loop over a diffusion-forcing scheduling matrix
  (`inference.py:run_diffusion_inference_df`, `diffusion/scheduling.py`). It is
  `@torch.no_grad`; the multi-window drivers (`autoregressive_rollout`,
  `streaming_rollout`) are too.
- Temporal attention direction is a config choice on both backbones
  (`docs/SUPPORTED.md`, "Temporal attention direction"). Causal is the trained
  default of every existing checkpoint.

So on this branch "guidance" means: choose the actuator trajectory (and/or the
sampler's noise) such that the PASSIVE outcome meets an objective. It is inverse
dynamics through the world model.

## 2. Layer 0: what to integrate now

Package `guidance/` (no new model code, no new training code):

| file | contents |
| --- | --- |
| `sampler.py` | `sample_window(model, diffusion, batch, noise, *, steps, schedule, checkpoint=True)`: the DF Euler loop without `no_grad`, model calls under `torch.utils.checkpoint`, initial noise and `batch.pose_actions` are the leaves. Returns particles and poses in physical units through the checkpoint's normalization, identical to `run_diffusion_inference_df` when nothing requires grad (test). |
| `objectives.py` | callables `(prediction, batch) -> (scalar, diagnostics)`: passive-body target pose (translation, geodesic rotation), particle-set target (per-particle L2, Chamfer), progress terms for receding horizon (object height, object-to-target distance), penalties: phantom motion (passive motion without actuator proximity), penetration, table plane. All in physical units. |
| `planners.py` | `best_of_k` (sample K windows, pick by objective; no gradients); `cem_waypoints` (actuator trajectory parameterized by W waypoints in SE(3) plus open flags, CEM/MPPI over waypoints, candidates evaluated by the sampler; the matched-compute baseline); `noise_opt` (Adam over the initial noise); `actuator_opt` (Adam over `pose_actions`); `joint_opt` (both). Carried hyperparameters: lr 0.05 to 0.20, K=4, phantom weight 100, actuator-only gradients, no terminal inpainting. |
| `horizon.py` | receding horizon: optimize one window, commit `stride` frames through the same contract as `autoregressive_rollout` (predicted particle history plus full predicted rigid context), re-plan. Terminal objectives become progress objectives until the last window. |
| `run_guidance.py` | CLI: checkpoint, domain, episode set, objective, planner, budget (NFE, candidates, horizon), seed; writes one JSON per episode with the objective trace, diagnostics per step, wall time and NFE. |

Evaluation protocol for Tier-1 #5 (planning at matched compute): same checkpoint,
same episodes, three NFE budgets, report success rate against a target tolerance,
final target distance, phantom and penetration rates, seconds per action, and NFE
times candidates times horizon. Gradient-free planners at the same NFE are the
control; a gradient planner that wins only at higher NFE has not won.

Tests: sampler equals the production sampler under no-grad; objective gradients
finite and nonzero; each planner lowers its objective on a fixed batch from a
fixed seed; receding-horizon commit matches `autoregressive_rollout` frame for
frame when the planner is the identity.

Cost: two to three days. Everything in Layer 1 plugs into `planners.py` and
`objectives.py` without touching Layer 0.

## 3. Layer 1: how diffusion-forcing guidance can be better

The historical methods (DPS in x_t, MPGD in x0, noise optimization, action and
joint optimization) share one failure: they ascend a gradient on the model's own
prediction, and the optimizer finds inputs the model mis-predicts rather than
plans the world executes. The record shows it: terminal inpainting had to be
banned, phantom penalties of 100 to 1000 were needed, and per-frame translation
was clipped to the stats half-range. Improvements, in the order to try:

1. **Stay on the model's manifold: sampling-based guidance.** Resampling among
   the model's own samples cannot produce a trajectory the model would not have
   generated. Best-of-K is the trivial case. Feynman-Kac steering / twisted SMC
   runs K particles through the denoising chain and resamples by
   `exp(objective on the x0 estimate)` at chosen steps. Under diffusion forcing
   the natural resampling points are the steps where a frame commits to noise
   zero. Cost is K forward chains, no backward. The one-pass mean predictor
   (`docs/iclr` GenCeption result: one pass matches the 50-step sampler on MPM L2)
   makes candidate scoring one NFE per window, so CEM over actuator waypoints
   with the mean predictor inside and the full sampler only for the winner is
   the cheapest planner available.
2. **Noise-level-aware guidance weight under DF.** A gradient on a frame at high
   noise is uninformative; on a frame near zero noise it is too late. Per-frame
   gain as a function of that frame's current level (zero above and below a
   band). The pyramid schedule denoises EARLY frames first, so a terminal
   objective spends most steps on frames that are still noise. For terminal
   goals use `full_sequence`, or an inverted pyramid, which needs bidirectional
   attention to be well posed.
3. **Goal inpainting with bidirectional models (Diffuser-style planning).** With
   bidirectional attention the terminal frame can be clamped to the goal (passive
   body at the target, noise level zero from the start) and the intermediate
   frames plus the actuator trajectory denoised between start and goal. This is
   the opposite of the retired terminal-inpainting cheat: there the model
   predicted the block and the cheat overwrote it; here the goal is an
   observation and the actuator is what gets generated. It requires the actuator
   to be a generated variable, which needs action dropout (item 4).
4. **Action dropout fine-tune.** Replace `pose_actions` on actuated bodies by the
   no-action token with probability p (0.1 to 0.3) during a fine-tune from the
   200K checkpoints. The model then generates actuator trajectories, supports
   classifier-free guidance on actions, and item 3 becomes possible. One or two
   GPU-days; a config flag plus one line in the pose-token assembly.
5. **Amortized goals.** Add a goal token (target pose of one passive body) with
   dropout; sample plans conditioned on the goal with classifier-free guidance,
   no test-time optimization. Same fine-tune as item 4 with one more input.
6. **Verify plans outside the model.** Replay the planned actuator trajectory in
   the simulator that produced the training data (Genesis, `collection/`) and
   score the executed outcome. A planner is judged by executed success, not by
   the world model's own belief. This is the guard that makes items 1 to 5
   comparable and is the closed-loop number the ICLR plan asks for.

What "better" means, measured: executed success at matched NFE, phantom and
penetration rates without penalties, and the gap between predicted and executed
outcome (the model's self-consistency under its own plans).

## 4. Order and dates

| when | item |
| --- | --- |
| this week | Layer 0 on the pushing and pickup checkpoints; matched-compute table on pushing |
| next | best-of-K and FK steering (Layer 1.1) as planner entries; noise-band gains (1.2) |
| after the 200K pair | action-dropout fine-tune (1.4), then goal inpainting on a bidirectional checkpoint (1.3), simulator replay (1.6) |
| later | amortized goals (1.5) |
