# Supported unified scope

Run commands from the repository root. Fresh configs reject unknown, retired or
incompatible options before model allocation and corpus access.

| Choice | Recipe | Trainer |
|---|---|---|
| TrajectoryTransformer, approximately 333M parameters | `configs/transformer.yaml` | `train.py` |
| Hierarchical UNet, approximately 332M parameters | `configs/unet.yaml` | `train.py` |

Both use C4/F16, delta targets, world-frame six-channel native rigid poses and
flow matching with per-frame diffusion forcing. Velocity and sample
parameterizations are explicit alternatives. The historical **sample-mode
pose loss is velocity-weighted**, unlike the particle sample loss; changing
parameterization changes that objective and is not a cleanup equivalence claim.

Shared defaults are readable in `utils/config.py`; YAML contains backbone
differences. Model defaults preserve historical checkpoint behavior rather than
silently inheriting today's recipe.

## Domain and normalization choices

```bash
python train.py --config configs/transformer.yaml 'data.domains=[mpm,pickup]'
python train.py --config configs/unet.yaml 'data.domains=[cloth]' \
  normalization.profile=xy_p99_global run_name=cloth_unet_global
```

A specialist is a domain-selected unified model, not a separate architecture.
Default validation and train-eval follow selected positive-weight domains.
An explicit `data.eval_domains` permits cross-domain validation.
Domain weights describe batch draws, not equal particle counts or objective mass.

`best.pt` is crowned by `log.best_metric`. The default `sample_mm` is the
full-sampler error in millimetres (`validate_sampler`: EMA weights, generation
from pure noise, fixed per-batch seeds), because the one-pass loss does not track
generation quality and is not comparable across recipes that see different
information. `loss` restores the historical gate, with `log.best_gate` (ema/raw)
selecting which weights' loss. `sample_mm` requires `log.sampler_batches > 0` and
`log.best_gate: ema`; both are configuration errors, not silent fallbacks. The
gate metric's name is stored beside its value, and changing it restarts the best
search instead of comparing millimetres against a loss.

`train.action_dropout` (0 to 1, default 0) hides the actuated bodies' given future
motion per batch element: their pose tokens read the learned no-action token, so the
model generates the actuator. It hides only the pose-token action stream and therefore
requires the UNet with `unet_actions: false`. At 1 (unconditional) the validation loss,
the sampler gate and `eval_unified` sample with the actions hidden, and evaluation
receipts record `"actions": "hidden"`; a partial recipe is validated and evaluated with
its actions given. `train.init_from` starts a new run from another checkpoint's EMA
weights (fresh optimizer, schedule and step 0). Its network contract and normalization
must match the new configuration, and a resumed run ignores it.

### Opt-in water domain (Water-3D)

`water` (DeepMind Learning-to-Simulate Water-3D, SPH water in a closed box) is supported
but never selected by default: name it in `data.domains`, set `model.num_mat_types=7`
(material id 6) and select a normalization artifact with a water entry
(`normalization.artifact=stats/normalization_water3d.json`); the UNet also needs a
material-6 entry in `model.unet_l1_by_mat`. The four reference entries, the global profile
and every existing checkpoint snapshot are unchanged. Data, constants, commands and the
overfit record: [WATER3D.md](WATER3D.md).

### Opt-in rl_push domain (RL pushing, low-dim release)

`rl_push` (a Panda gripper pushing one of 1,202 reconstructed objects toward a goal, 20 Hz)
is supported but never selected by default: name it in `data.domains` and select a
normalization artifact whose `rl_push` entry carries particle and pose scales
(`normalization.artifact=stats/normalization_rl_push.json`). The release stores poses, not
points; `data/rl_push.py` builds points in the pick-and-place layout (pickup's own gripper link
points, 512 object points), starts windows at frame 3, holds out the objects listed in
`data/manifests/rl_push_val_assets.json` and drops the episodes in
`data/manifests/rl_push_exclusions.json`. The frozen artifact, the global profile and every
existing checkpoint snapshot are unchanged. Layout, decisions and commands: [RL_PUSH.md](RL_PUSH.md).

### Opt-in indexed collections

`mpm_fresh`, `mpm_drag`, `folding`, `yikai`, and `dex` use frozen episode
indices through the same unified trainer. Each requires explicit
`data.roots.<domain>.train` and `.val` directories containing `INDEX.json`, and
its measured entries in a collections scales artifact: `stats/normalization_collections_20260919_v7.json`
for runs on pwm-data-v1 (v6 predates the one-scale pushing entry, v5 predates the filtered `dex_v2_filtered2` folder, v4 the regenerated `dex_v2`, v3 the
thinned folding folder;
`..._20260916_v2.json` is the pilots' artifact).
The validation directory holds a `val` index or a release's `test` index: in
`pwm-data-v1` (see [DATASETS.md](DATASETS.md)) the test folders are the validation
folders by decision of 2026-09-17, so validation curves and `best.pt` read the
test set and every number reported from it says so. The recipe builder takes
`<release>/<domain>/test` when there is no `val` folder. They are never selected
by default.

Training windows of an indexed domain are drawn uniformly unless the recipe says
otherwise. `data.window_motion.<domain> = {alpha, floor_mm}` draws a window start with
probability proportional to (mean future step of the predicted particles + floor)^alpha
(cloth vertices only for folding; MPM: every particle); `data.batch_across_episodes`
lists the equal-point-count MPM domains whose training batches take one window from
each of B different scenes with the same object count instead of B windows of one
scene. Both apply to training only; validation windows and per-episode validation
batches are unchanged, and alpha 0 reproduces the uniform draw exactly. The collection
recipe carries the values decided on 2026-09-17: alpha 1 with floors 2 / 0.25 / 0.02 mm
for mpm_fresh / mpm_drag / folding, and cross-episode batches for mpm_fresh and mpm_drag. The collection recipe builder supplies the
required action settings, unclipped targets and balanced validation counts.

Clocks, action semantics, split limitations, preparation commands and verification
receipts are documented in [DATASETS.md](DATASETS.md).
Dragging supplies commanded particle motion with hold/release gates; Dex supplies
achieved hand motion, without claiming a recovered torque-action interface.
Checkpoints and evaluation receipts pin the episode-index hashes.

### Explicit dataset roots

`data.roots` points a domain at a dataset release instead of the registered
`unified://v1` corpus; domains it does not name keep their registered roots:

```yaml
data:
  domains: [mpm, cloth]
  roots:
    mpm:   {train: /abs/release/mpm/train, val: /abs/release/mpm/val}
    cloth: {train: /abs/release/cloth/train, val: /abs/release/cloth/val,
            exclusions: {train: [/abs/release/cloth/exclusions_train.json]}}
```

Paths are absolute, canonicalized, and must exist when the config resolves, before
any model is allocated; `val` must not overlap `train`. Without `val`, validation
is the deterministic last-10%-of-sorted-shards split of the training roots (per
root for pickup, pooled for pushing); MPM requires both roots, and indexed
collections require both frozen index directories. An overridden
domain applies only the exclusion files it names: the registered v1 lists belong
to the v1 corpus and are not carried over. The override is part of the saved run
configuration, and the calibration fingerprint records the canonical roots, so a
checkpoint states which release it trained on. Evaluation reads the same release:
`eval_unified`, guidance and `scripts/dev/check_checkpoint.py` score a checkpoint on its
saved roots and exclusions; the command-line root flags apply only to registered domains.
Resume refuses a change of these roots or exclusion lists. Each checkpoint also records
the sha256 of every exclusion file the run applied, registered lists included, so a list
edited in place is refused too. Checkpoints saved before 2026-09-14 lack those hashes and
resume with a warning.

Both profiles live in `stats/normalization.json`:

- `xy_p99_per_domain`: 20 symmetric XY/Z motion scales.
- `xy_p99_global`: 4 scales shared across all domains, fixed even for specialists.

The global profile is max-over-the-fixed-domain reference, not a pooled p99.
Its original values are restored, not refitted. See
[normalization](../stats/README.md) for physical units, provenance and the
single collect/fit/audit CLI. New candidates are explicit, separately versioned
artifacts. Neither profile is declared best.

## Evaluation

Start with [EVALUATION.md](EVALUATION.md): which command answers which question, the
protocol, the metric vocabulary and what a receipt records.

| Entrypoint | Protocol |
|---|---|
| `scripts/unified/eval_unified.py` | Domain-selected fixed windows and overlapping autoregression; explicit cross-domain evaluation |
| `scripts/eval_object_rollout.py` | Native passive MPM WDS, named populations/counts, window/full (the shared rollout)/streaming |
| `scripts/inference_mpm/inference_mpm_multi.py` | Passive MPM H5 decoding with the shared rollout and scores, and visualization |
| `scripts/unified/eval_paper_mpm_unified.py` | Paper native-frame crops and ID population contract |
| `scripts/unified/eval_paper_cloth_table3.py` | Paper cloth native-frame boundary and materialized cache contract |

See each command's `--help` and the
[complete MPM population inventory](CODE_WALKTHROUGH.md#mpm-evaluation-populations).
Unseen, unseen-clean, selection, count-OOD and pinned subsets are not
interchangeable. Windowed and streaming autoregression are different protocols.
Paper evaluators preserve their first scored frame independently of context C.

Checkpoint-owned normalization, representation and context take precedence.
Explicit evaluation ablations are labeled; empty/nonfinite metrics fail closed.
Evaluators count coverage in clips (`--clips all` or a positive count; `--max_batches`
is retired), read the split in stratified order (task roots and object-count
directories interleaved), roll out with `inference.windowed_rollout` and score with
`utils.evaluation.TrajectoryScores`. `eval_unified` reseeds the sampler per batch from
`--seed` and the batch index, so a domain's number does not depend on which domains
were evaluated before it in the same command, and its per-domain results record
`shards_read` and `clips_per_shard_read`, the shards the scored clips came from.
New checkpoints contain numeric normalization snapshots with checksums.
Only compatible historical unified checkpoints migrate; unsupported specialists
fail with recovery instructions instead of partially loading weights.

Native evaluation (`scripts/dev/native_eval_unified.py` and
`scripts/dev/native_zero_shot.py`) and guidance also use the checkpoint's positive
`data.future_len`. Missing or nonpositive lengths retain the per-domain fallback,
matching the main evaluator. Longer requested horizons use multiple model windows.
Both also follow the checkpoint's action policy: a checkpoint trained with
`train.action_dropout=1` is sampled with actions hidden, and their receipts record `actions`.
Guidance receipts report `nfe_per_window` for a full configured model window;
`nfe_total` accounts for actual planner calls, including shorter horizons.

Rotation calculations in `utils/pose_ops.py` use a quaternion-based principal
SO(3) logarithm and geodesic error in [0, pi]. This intentionally corrects the
historical trace/antisymmetric formula, which returned zero for exact half-turns
and could reverse rotation-guidance gradients near pi. The shared logarithm also
constructs pose targets, context deltas and revolute-joint increments, so these
calculations no longer promise bitwise agreement with the historical formula.
At exactly pi the logarithm's axis sign is non-unique; gradients are tested on
either side of that branch cut. Tests retain exact historical `so3_exp` parity
and check both backbones with the corrected logarithm supplied to both sides.
Recompute historical rotation scores with the same corrected metric before
comparing them with new scores.

Rigid geometry in `utils/pose_native.py` runs with autocast disabled: the shared
input rewrite, clean context pose deltas and output materialization use FP32 pose
arithmetic, including articulated forward kinematics. Pose predictions are promoted
before denormalization; learned layers retain the caller's mixed precision, and
rewritten particle buffers retain their input dtype. No learned weights or
checkpoint keys change. This intentionally changes historical AMP arithmetic;
FP32 backbone parity and CPU/GPU AMP geometry and gradient tests cover the repair.
`scripts/dev/benchmark_pose_precision.py` measures its time and allocated GPU memory
against a pinned pre-fix revision on one full-size eager model with synthetic inputs
and unchanged weights. It does not measure compiled or distributed training throughput.

Fresh Transformer blocks explicitly zero the four AdaLN modulation projections
(shift, scale and residual gate). Attention, feedforward and embedding weights keep
their existing random initialization. This implements the documented AdaLN-Zero
identity start; diffusers' `AdaLayerNormZeroSingle` does not initialize its own
projection to zero. The change affects fresh training, including runs with the
same historical seed. Checkpoint loading restores all learned modulation weights;
no state-dict keys change and no initialization runs after loading.
Tests check identity at startup, gate and branch learning, and exact saved-weight
outputs on CPU and CUDA in FP32/BF16. Frozen-backbone parity uses identical loaded
weights; fresh initialization has a separate test showing only modulation changes.
An improvement in training quality has not been established by these contract tests.

## Config caveats

No numerical retuning is hidden in consolidation:

- Transformer still has a 200K schedule; UNet stops at 40K on a 200K schedule.
  That historical budget difference is a comparison confound, not a recommendation.
  Precisely, `stop_after_step=40000` is an inclusive zero-based label: a fresh
  run attempts labels 0 through 40000 (40,001 attempts). `train.steps=200000`
  permits 200,000 attempts. Accepted optimizer updates can be fewer after rejection.
- Transformer FPS was deterministic despite its old `fps_random_start` option.
  The no-op setting is removed. UNet's separately implemented randomness remains.
- Auxiliary geometry/context supervision and augmentation have narrower valid
  domains than ordinary training. Strict preflight rejects unsupported combinations.
  Strain auxiliaries require MPM-only training and evaluation and full-valid,
  contiguous equal-sized object blocks. Particle auxiliaries have no effect on
  pure-rigid batches, whose particle decoder is skipped. Floor loss excludes
  invalid/unpredicted rigid rows; fresh canonical recipes explicitly use `floor_z=0`.
- Regional compilation is a UNet policy; ragged operations and PyTorch3D may
  remain eager. This is not a full-graph or B4-memory guarantee.
- Temporal attention direction is a config choice, default causal in both
  canonical recipes. Transformer: `model.causal_temporal` (true = lower-triangular
  over future frames; false = bidirectional). UNet: block-causal by default
  (context frames see each other, future frames see context and earlier future);
  `model.unet_bidirectional_temporal: true` drops the mask entirely so every frame
  attends to every frame and the per-frame noise levels carry the ordering (DFoT
  convention); `model.unet_causal_context: true` is the fully causal flash path
  and is exclusive with it. Diffusion-forcing training and both samplers do not
  depend on the choice. One-step validation loss is not comparable across it
  (bidirectional sees more); compare rollouts. Checkpoints without the field
  load with the default. `tests/test_temporal_attention_direction.py` checks
  that a last-frame noisy-state perturbation reaches earlier frames only when
  bidirectional.
- Known-action access differs between backbones with action conditioning enabled.
  The Transformer's causal mask applies to future-state self-attention; its
  separate cross-attention can read the entire supplied action plan. The causal
  UNet adds actions to their corresponding future pose tokens, so prediction
  step i can read actions through step i, including the current action, but not
  later actions. Both receive actions. Enabling bidirectional UNet attention
  opens access to both later actions and later noisy states. This is a
  conditioning-policy difference, not evidence of broken UNet conditioning.
  Record action access when reporting comparisons; match it when an experiment
  is intended to isolate backbone effects. No prediction-quality benefit from
  changing this policy has been established by the access diagnostics.

- Training-time noise levels are a config choice, default unchanged. Diffusion forcing
  draws every future frame's level independently (`train.timestep_schedule: independent`),
  which covers every sampler and puts almost no mass on the joint patterns any one sampler
  visits. `train.timestep_schedule` may instead name a shaped schedule or a mixture, a list
  or a `{name: weight}` map with one schedule drawn per sample (on the command line
  `train.timestep_schedule.<name>=<weight>`): `flat` is one level for all future frames,
  the states of `df_schedule: full_sequence`; `ramp` is the pyramid staircase drawn as a
  family, `clip(phase + i * slope + jitter)`, whose slope defaults to the run's own sampler
  (`df_uncertainty_scale / flow_matching.inference_steps`; `train.ramp_slope`, a number or
  `[lo, hi]`, and `train.ramp_jitter`, default half a sampler step, override it).
  `tests/test_timestep_schedules.py` records the production sampler and checks that every
  state it visits is a ramp and that ramp draws stay within one sampler step of a visited
  state (independent draws: under 0.1%). Under `ramp` at 16 frames and slope 0.04 frame 0
  is already finished in 36.6% of the draws and frame 15 in none, and a finished frame
  teaches nothing, so a share of `independent` keeps the early frames' signal (85% for
  frame 0 at a ramp weight of 0.4) and the patterns no staircase contains (goal
  inpainting). Context frames stay clean; the retired `noise_context` is refused. One-step
  validation loss is not comparable across schedules; compare rollouts.
- A sample head (`flow_matching.prediction_type: sample`, the network outputs the clean
  per-frame delta) scores particles clean target against clean target, but by default still
  scores rigid poses in velocity space: the clean pose error divided by the frame's noise
  level `t`, so its square carries `1/t^2`, up to 1,000,000 at the smallest drawn level
  (kept for checkpoints trained that way). On pose-only pushing that default gives pre-clip
  gradient norms up to 6,521 from scratch and most steps are rejected by the absolute
  gradient-norm gate. `train.pose_loss_space: sample` (default `velocity`) scores rigid
  poses like particles, clean target against clean target; the same check then shows norms
  up to 1.6 on pushing and 2.5 on folding_8mm with no rejected step, the rigid term being
  37 to 46% of folding's loss. New sample-head recipes with rigid bodies should set it.
- `train.velocity_weight_cap` (sample heads only, default off) multiplies every clean-target
  error, particles and rigid poses alike, by `min(1/t^2, cap)` per frame: the velocity loss
  without its singularity at the clean end. It is a weight on clean-target terms, so it
  requires `train.pose_loss_space: sample`; a velocity head already carries the uncapped
  weight and the option is refused there. The weighted loss and its gradients are steeper than
  the flat ones, and `train.grad_norm_spike_threshold` is an ABSOLUTE pre-clip bound
  (default 50) that applies once `train.spike_grace_steps` (500) have passed, still inside
  warm-up. Functional checks from scratch on 2026-09-19 (103 M UNet, sample head, 700
  steps, norms logged every 10 steps): mpm_fresh flat 0.1 to 1.2, cap 400 up to 63 (median
  8.5 after the grace steps, none of 20 logged steps rejected), cap 400 with a
  0.5 / 0.4 / 0.1 independent / ramp / flat mixture up to 161 (3 of 20 rejected); pushing,
  whose loss is the pose term alone, default sample mode up to 6,521 (13 of 20 rejected),
  cap 400 up to 121 (9 of 20), cap 400 with the mixture up to 357 (7 of 20). A rejected
  step costs budget, not correctness, but a capped recipe should set the threshold (or the
  grace steps) for its own gradient scale. Resume refuses a changed schedule or cap;
  `train.init_from` starts a continuation with different ones.

All synchronized attempts count toward the step budget; optimizer, scheduler and
EMA advance only on accepted updates. Atomic checkpoints retain the next-attempt
boundary and rank-local runtime state. Exact fixed-input continuation does not
imply exact resumption of WebDataset/worker input order.

## Guidance (Layer 0)

`guidance/` plans the actuator trajectory of pose-native batches through the
world model: a differentiable copy of the production single-window sampler
(bitwise equal under no-grad), actuator plans as world pose deltas of the
actuated rigid bodies applied to every batch field the backbones read,
objectives in physical units, gradient-free (best-of-K, waypoint CEM) and
gradient (noise / actuator / joint Adam) planners with NFE accounting, and a
receding-horizon driver that commits the model's own output as context.
Entry point `guidance/run_guidance.py`; design and the improvement path in
`docs/guidance/DESIGN.md`. Supplied-action plans require non-articulated bodies;
`apply_plan` rejects articulated fingers. Particle targets are available, but no
specialized cloth planning objective is provided.

`python -m guidance.run_unconditional` is the experimental articulated pickup
path for fully unconditional (`action_dropout=1`), pose-native UNet checkpoints
with `unet_actions=false`. It supports one entirely rigid scene at a time,
preserves generated actuator motion, and optimizes complete rollouts through
carried context. Picking diagnostics include two-finger proximity and sampled
collision-component intersections; these do not certify physical execution.
See [the controlled diagnostic](guidance/UNCONDITIONAL_PICK_GUIDANCE_2026-09-14.md)
for tests, paired samples, geometric limitations and compute receipts. The
supplied-action CLI rejects fully unconditional checkpoints.

## Deferred and historical

Active MPM drag is an **explicitly selected experimental training path** with a
bounded gentle-v4 train-only collection release; this does not establish model
quality or a held-out evaluation release. See the
[dataset format and training guide](../collection/mpm/docs/DRAG_DATASET_FORMAT.md)
for approved paths, field semantics, smoke checks and required split preparation,
and [drag readiness](../collection/mpm/docs/DRAG_READINESS.md) for evidence and
limitations. `data.roots.mpm.task: drag` selects
versioned commanded-velocity exports; default MPM readers remain passive and reject
active inputs. Both models accept temporal hold/release flags without new parameters.
UNet additionally requires `model.unet_actions=true`. Historical unversioned drag
shards are not automatically compatible.

Guidance beyond Layer 0, DDPM, standalone specialist pose heads,
frame-zero/anchor-offset targets and planning sweeps are outside this checkout's
supported scope. Recover their source at `robert/pre-unified-only-2026-09-06`
(`c5fac33`); earlier latent/multi-mixer experiments remain at the preceding
pre-cleanup reference.

Keep SO(3), Kabsch and native rigid-body geometry: these are active unified
dependencies, not legacy specialist implementations. Live manifests are
exclusion policies, not generated stats clutter.

The round-four plan (`docs/cleanup/ROUND4_PLAN.md`, in git history, last present at `79a3c84`) lists verification gates.
Recorded regression coverage does not establish full-corpus generalization,
multi-node correctness, production throughput or an architecture ranking.
