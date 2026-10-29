# Clean-branch code walkthrough

Scope: the five-layer review and the round-four unified-only cleanup.
This describes the supported implementation, not the larger historical experiment
tree in `PROJECT_MASTER_REFERENCE.md`. Verification and changes are recorded in
the round reports of the 2026-09 cleanup (`docs/cleanup/ROUND2.md` to `ROUND4.md`, in git history, last present at `79a3c84`). Supported
recipes remain in [SUPPORTED.md](SUPPORTED.md).

## 1. Data sources, exclusions, windows, batches

Start in [data/unified_dataset.py](../data/unified_dataset.py):
`build_domain_dataset` chooses a corpus adapter, fixed coordinate shift,
normalization snapshot, split, and exclusion lists. `data.domains` selects any
supported subset; a specialist is the same unified architecture trained on one
domain. Primary recipes use C4/F16; explicit overrides remain part of the saved
experiment contract. Shared raw shard reading lives in
[data/dataset.py](../data/dataset.py), not separate specialist trainers.

All roots below are relative to `/data/group_data/katefgroup/VLA/3DPCWM/`.

| Domain | Training / validation population | Default window policy |
|---|---|---|
| MPM | Train `robert_mpm_v2/webdataset_full49`; val `robert_mpm_v2_test_seen`. | Random training subwindow; validation subwindow seeded by sample-key CRC32. `first_window=True` explicitly selects the prefix. |
| Pickup | Four `robert_pickup_data_{place,convex,pick,upright}_success` roots; last 10% of sorted shards held out **per root**. | Random training subwindow; validation prefix. |
| Cloth | Materialized `unified_v1/cloth/{train,val}`; garment split and stride 2 already applied upstream. | Training windows weighted by future cloth motion plus a 0.02 mm/frame floor; validation prefix. |
| Pushing | `pushing_data_actions`; last 10% of sorted shards held out. | Random training subwindow; validation prefix. |

`data.roots` replaces a domain's registered roots with an explicit dataset
release (`resolve_data_roots` validates the mapping before any shard is opened;
`build_domain_dataset` builds the same adapters on the given roots, and only the
exclusion files the release names are applied). See
[explicit dataset roots](SUPPORTED.md#explicit-dataset-roots).

The opt-in indexed collections (`mpm_fresh`, `mpm_drag`, `folding`, `yikai`,
`dex`) use `data/indexed_episodes.py` and `data/recording_adapters.py` through
that same factory. They require explicit frozen indices and measured normalization;
the four registered defaults above retain their original behavior. See
[the dataset document](DATASETS.md) for units, conditioning, splits,
terminal boundaries and real-data verification.

These are not interchangeable test sets. Pickup validation is asset-matched to
training, not the reserved 142-asset holdout. Paper MPM and long-horizon cloth
protocols have separate entrypoints and information boundaries.

### MPM evaluation populations

Do not collapse these into a single "MPM eval" label. The following tar counts
were re-read on 2026-09-06 UTC; they count materialized shards, not generator scenes
or benchmark episodes. Roots remain relative to the group directory above.

| Root | Tars | Role and limitation |
|---|---:|---|
| `robert_mpm_v2/webdataset_full49` | 208 | Training, counts 1–4. Not evaluation. |
| `robert_mpm_v2_test_seen` | 10 | Counts 1–4, seen asset types; **selection/validation for unified models**, so not their untouched final test. |
| `robert_mpm_v2_val_unseen_fulltar` | 4 | First tar per count 1–4; specialist checkpoint selection, recorded 128 episodes. |
| `robert_mpm_v2_test_unseen` | 11 | Full unseen-object pool, including those four validation tars. Not a clean specialist final test. |
| `robert_mpm_v2_test_unseen_00001plus` | 7 | Remaining unseen tars, recorded 158 episodes; specialist final-test view. |
| `robert_mpm_v2_test_unseen_clean` | 7 | Alternate view of the 00001+ unseen tars; not another independent test population. |
| `robert_mpm_v2_ood` | 17 | Counts 5–7, held-out assets; count and object generalization. |
| `robert_mpm_v2_ood_first_tar` | 3 | One tar each at counts 5–7; diagnostic subset, not full OOD coverage. |
| `robert_mpm_v2_viz_pinned` | 3 | Pinned visualization shards at counts 1, 2 and 4; not a representative benchmark. |
| `robert_mpm_test/webdataset_full49` | 10 | Paper-era v1 ID root, counts 1–4 only. An OOD paper request needs an explicit suitable root. |
| `robert_mpm/webdataset_test` | 54 | Legacy v1 test directory; not approved here as a fresh training-validation split. |
| `robert_mpm_drag_setA/webdataset_test` | 23 | Legacy active-drag test directory; likewise not reassigned to validation. |
| `robert_mpm_v2_drag/webdataset_full49` | 1,318 | Separate active-drag training corpus, unused by retained recipes. |
| `robert_mpm_v2_drag_{test_seen,test_unseen,ood}/webdataset_full49` | 16 / 14 / 6 | Separate active-drag eval corpora; the six OOD tars cover counts 5–6, not 7. No retained model was newly evaluated on these. |

The historical specialist validation views stay documented even though their
model/config stacks are retired from the active branch. All ten passive evaluation
populations are explicitly named by `scripts/eval_object_rollout.py --population`;
`--wds` selects a custom root/count directory. Active-drag adapters and unified
guidance are deferred, not silently treated as passive MPM. Read-only realpath
checks alone cannot establish asset/content disjointness. The historical
v1 OOD cloud-overlap finding in the master reference is not erased by cleanup.
No test corpus was relabelled, regenerated, deleted or scored in this inventory.
Cloth's exact 38 held-out garment identities now live in
[`data/manifests/cloth_holdout.json`](../data/manifests/cloth_holdout.json), not
a specialist model configuration. Materialization retains the original split.

### Sampling and exclusions

Exclusions are dataset policy, not cleanup debris:

- MPM: the 64 collapsed training episodes are removed by `mpm_blocklist_train.json`.
  The recorded collapsed evaluation episodes remain for comparability.
- Pickup: training applies the combined phantom list and the external asset
  holdout list. Validation applies the asset holdout list, not the training
  phantom list.
- Cloth: `cloth_blocklist_train.json` and `cloth_blocklist_val.json` apply to their
  respective splits (418 and 57 whole-episode entries, with path aliases).
- Pushing: no additional unified blocklist.

[data/blocklists.py](../data/blocklists.py) resolves full shard identity, historical
path aliases, sample keys, and optional episode indices. Missing required lists
fail closed. No exclusions or corpus files changed in this round.

The train stream draws **one domain per step**, then a domain-pure batch. The
domain RNG seed is shared across ranks; WebDataset splits the actual inputs by
rank/worker. Domain weights are batch-draw probabilities, not a guarantee of equal
particle counts or equal loss contribution. Zero-weight domains are not opened.
Validation and train-evaluation default to the selected active domains, so an
MPM-only run never opens the other corpora. `data.eval_domains` is an explicit
cross-domain evaluation override. Validation round-robin traverses finite splits and deliberately avoids rank
shard splitting so collective per-domain reductions align.

`BaseGraspDataset.__iter__` buckets by particle count, drains exact requested
sample counts from pre-batched records, and carries remainders. A configured
bucket-memory cap can repeat samples when flushing a rare partial bucket. That
affects effective sampling and must accompany an experiment description.
The flush now counts individual samples: repeating a three-episode record four
times previously produced B=12 despite a requested B=4.
Processing/contract exceptions now identify the shard and sample and propagate.
Cloth's existing unusable-gripper-link metadata exclusion uses an explicit
`SkipSample`, not a catch-all error handler. That exclusion, the separate
NaN/Inf-position filter and WebDataset tar warning handler still skip bad inputs.
This is not a guarantee that every upstream corruption is fatal.

[TrajectoryBatch](../data/batch.py) owns:

- positions `[B,C+F,N,3]`, actions `[B,F,N,3]`, masks/IDs/physics `[B,N]`;
- valid prefix lengths `[B]`, target type, normalizers and source-domain metadata;
- optional per-link rigid templates, poses, particle/body mapping, and future
  six-channel pose targets/actions.

`.to()`, `.slice()`, `.float()`, and `.half()` now share one field-wise transform.
Metadata survives every transform; integer IDs and boolean masks stay typed.
Collation rejects incompatible shapes, dtypes, devices, targets, stats, domains,
or missing optional fields. It does **not** make arbitrary particle layouts legal
for every model: the UNet additionally requires matching contiguous object
blocks across elements and no padded rows.

Unsupported unified options now fail before I/O instead of disappearing. The
`first_window`, shuffle, and shard-splitting requests are forwarded where the
loader supports them. The unsupported historical specialist format switches are
not part of the active training API.

Calibration has a separate **finite raw** traversal in
[`data/calibration.py`](../data/calibration.py): deterministic shard permutation,
per-(seed, canonical shard, sample key) window seeds, no caller RNG mutation,
actual temporal validity masks and a selected episode/window ledger. It uses
the same physical adapters without opening normalization assets. Its source
fingerprint records roots, tar size/mtime, byte-hashed exclusions and sidecars,
window policy/C/F/stride, canonical shifts and adapter code. This is not a claim
that all tar contents have been cryptographically hashed.

## 2. Physical units, normalization, noise, targets

[data/normalization.py](../data/normalization.py) now contains the linear normalizer
without importing corpus loaders. `data.dataset.NormalizationStats` remains an
alias for old imports and pickled batch fixtures.

Positions are metres; particle deltas are **displacements per stored frame**,
not metres/second. Native-pose deltas are world-frame translation in metres and
rotation vectors in radians. Physics inputs are Young's modulus (Pa), Poisson's
ratio, and density (kg/m³). A material label distinguishes measured properties
from fixed rigid/cloth fills. `DOMAIN_DT` is metadata: the target pipeline does
not divide by dt. Different corpora therefore do not share a physical forecast
duration merely because they share F=16.

Canonical shifts (metres) are fixed per domain: MPM `(-.75,-.75,-.10)`, pickup
`(-.45,0,-.10)`, pushing `(0,0,0)`, cloth `(-.65,0,-.80)`. The cloth materializer
already applied its shift. There is no per-episode recentring. Unified position
bounds are ±0.8 m, separately from the motion bounds.

The exact preserved arithmetic is:

```text
normalize(x)   = 2 * (x - min) / (max - min + 1e-8) - 1
denormalize(y) = (y + 1) / 2 * (max - min) + min
```

The epsilon means even a symmetric normalizer is only approximately `x/half_range`.
Removing it or changing operation order would change checkpoint numerics. The
normalizer does not clip by default; the main objective clips future targets
when `train.clip_deltas=True`. Context conditioning and model action paths have
their own clipping policies; they must not be replaced with one global clamp.

There is **one versioned normalization artifact and two selectable profiles**:
`xy_p99_per_domain` stores 20 motion scales; `xy_p99_global` stores four. Current xy-p99
uses `max(p99|dx|, p99|dy|)` for the shared x/y half-range and `p99|dz|` for z.
This is **not** p99 of the horizontal displacement magnitude. Pose scales group
horizontal translation and rotation similarly and cover both training-policy
and prefix-policy windows from the **training split**. Global4 is the componentwise
maximum over the fixed four-domain reference, sharing particle and pose
translation. It is not a pooled-data percentile and never changes merely because
a specialist selects one domain. Neither profile is declared the winner.

`resolve_normalization` returns values, units, profile, schema/artifact version,
provenance and SHA-256 in one self-contained JSON snapshot; `domain_normalizers`
constructs the same float32 bound tensors from either profile. Checkpoints embed
the snapshot. Resume/evaluation reject incompatible values instead of following
historical absolute NPZ paths. The old URI `unified://xy_p99` is an exact alias;
the old literal `per-domain` is a **different retired baseline**, not an alias.

The frozen bundle reconstructs all seven per-domain and seven shared historical
motion-bound pairs exactly. It is a migration, not a refit or a retrospective
claim of holdout hygiene. Historical standalone native-C2 raw-RoPE behavior and
other specialist conventions remain at the frozen revision; current native/paper
MPM evaluation is explicitly unified and always applies its canonical shift.

[`scripts/compute_stats.py`](../scripts/compute_stats.py) has `collect`, `fit`
and `audit` stages, implemented in [`data/stats_fitting.py`](../data/stats_fitting.py).
Collection uses bounded deterministic per-domain/policy particle-frame and
rigid-body-frame reservoirs; it is **not episode-uniform sampling**. Cache identity
includes sources, exclusions, window contract, collector/geometry code, runtime
versions and capacity. Fitting requires all four training domains, derives both
profiles from the same samples, and takes maximum per-policy p99 bounds. Zero
scales fail rather than receiving an invented floor. New fits require separate
candidate versions and cannot overwrite frozen or existing artifacts.

Audits report physical distributions, normalized magnitudes, clipping and
single-transition clipped-target reconstruction. They do not integrate trajectories
or establish a paper ranking. Validation caches are audit-only; final-test data
is never used to fit or select a candidate. Full recipe details are in
[stats/README.md](../stats/README.md).

The active unified batch/target contract is **delta only**. Absolute-position and
offset experiments belong to historical reproduction, not alternative active
normalization semantics.

[TrajectoryFlowMatching](../diffusion/flow_matching.py) forms
`x_t=(1-t)x0+t*epsilon` with unit Gaussian noise in **normalized coordinates**.
Training DF draws separate future-frame noise levels. A full-sequence *sampling*
schedule remains available; it is distinct from changing the training objective. Future levels exclude t=0 because sample-to-velocity
conversion divides by t. Context frames stay clean in new runs: `train.noise_context`
(DF noise levels on the history) is refused because it noised every particle
independently, putting rigid bodies' particles away from their pose. Checkpoints
trained with it still load and sample. `train.light_context_noise_std` adds independent
Gaussian noise to normalized deformable context deltas. The UNet integrates forward
from the recorded oldest frame, so its internal anchor moves; future delta targets
remain unchanged. This option retains its existing semantics and clipping defaults.
`train.light_pose_noise_std` is its rigid-body analog: every
free rigid body's history transitions get light noise in normalized pose-delta units,
integrated back from the clean anchor, jointed bodies follow their parents, and the
rigid particle rows of the history move with their bodies
([jitter_rigid_history](../utils/pose_native.py)). The anchor, the future and every
target stay clean, and sampling never uses it.

`train.light_pose_noise_anchor=true` also perturbs the passive bodies' anchor, with
two target policies selected by `train.light_pose_noise_target`:

| Target policy | Future supervision |
|---|---|
| `restore` (default, including older anchor checkpoints) | Recorded future poses/particles stay fixed. Only the first moved free-body pose transition is patched to return to the recording. |
| `carry` | Recorded world pose increments and joint increments stay fixed. Future poses/particles are rebuilt from the perturbed anchor. |

`train.light_pose_noise_type` picks how the history transitions are drawn. `iid`
(default) draws each independently. `random_walk` draws increments at
`std/sqrt(C-1)` and accumulates them, so the transition noise is correlated across
the window and `std` still names its magnitude at the most recent transition, which
matches the final-transition scale, not the anchor-displacement scale. A moved anchor
is built by integrating forward from the exact oldest pose, composing rotations at
every step. GNS
compared `only_last`, `correlated`, `uncorrelated` and `random_walk` and reports the
random walk best; our `iid` is their `uncorrelated`. With
`anchor=true, type=random_walk, scope=passive, target=carry` this adapts the GNS recipe to
whole-body rigid poses. Two differences remain: per-particle noise is not
available on a rigid body (ISSUE-001), and the target is a multi-frame delta rather
than a single acceleration.

The loader, history encoder and patcher share [pose_sequence_deltas](../utils/pose_native.py).
A revolute child's parent-relative transform is invariant under the common parent
correction, so its scalar joint targets stay unchanged. Carry preserves a free body's
world increments with `R'_f=R_f (R_anchor^T R'_anchor)` and a constant translation
offset. Applying a constant left/world rotation instead would conjugate its rotation
increments. Particle deltas always follow the resulting positions.

Both moved-anchor policies require positive noise std and `train.clip_pose_deltas=false`;
clipping would change the advertised target. `clip_deltas` remains independent because
the pose-native particle loss excludes rigid rows. There is no global clipping-off
policy in the defaults or the saved MPM recipe.

Body scope is explicit: `train.light_pose_noise_scope=auto` preserves the historical
all-body history-only / passive moved-anchor behavior. Set `passive` for history-only
experiments to hold scope fixed. Moved anchors currently require passive scope. This
is a restricted augmentation policy: default rollouts can carry actuator error too.
Given actions stay recorded, and actuated children of passive parents are conservatively
refused until all action representations have an explicit policy. Passive joint-coordinate
error itself is not independently injected.

Carry and restore use identical context corruption and RNG draws at equal seeds/settings.
With `iid`, a moved anchor consumes an extra draw; `random_walk` integrates the history
draws instead. Equal seeds alone do not match downstream diffusion draws between recipes
that consume different draws. Train/validate draw parity
holds within each recipe; sampling does not use training jitter. The large-t diagnostic
receives the same effective batch as the objective, including augmentation and action dropout.

These are experiments, not established effectiveness results. Default GNS carries the
position offset while correcting next velocity, and uses random-walk velocity noise
followed by integration to positions; this repo's independent delta noise differs.
GNS also tested position-restoring targets and reported worse performance on GOOP;
that supports a carry baseline but does not prove restoration unlearnable or establish
the best choice for this rigid model. Random restoration can learn geometry-dependent
corrections; ambiguity and the forced first-step velocity correction are risks to measure.
See the [GNS target code](https://github.com/google-deepmind/deepmind-research/blob/master/learning_to_simulate/learned_simulator.py#L204-L223),
[noise code](https://github.com/google-deepmind/deepmind-research/blob/master/learning_to_simulate/noise_utils.py#L21-L47)
and [Supplement C.2](https://proceedings.mlr.press/v119/sanchez-gonzalez20a/sanchez-gonzalez20a-supp.pdf).

A normalization change therefore changes the raw-metre corruption and the
integrated noisy positions used by geometry, as well as loss scaling. The bounded
near-static MPM failures are **not proven to be a wrong-stats bug**: the inspected
fixture has no target clipping, and an exact-target sampler reconstructs it
accurately. MPM particle/position bounds are bitwise identical between the two
frozen profiles; joint profile comparisons also change other-domain updates.
Scale, parameterization and optimization interactions remain open questions;
the completed diagnostic evidence and its limits are in the round-four report (`docs/cleanup/ROUND4.md`, in git history, last present at `79a3c84`).

## 3. Features, geometry, attention, decoding

Read [models/factory.py](../models/factory.py) for strict construction, then
[models/transformer.py](../models/transformer.py),
[models/unet_std.py](../models/unet_std.py), and the shared
[utils/pose_native.py](../utils/pose_native.py).

The unified Transformer embeds motion, material/physics, timestep and
actuation information; separately conditions on history/actions; compresses
objects to spatial latents; mixes temporal and spatial information; and decodes
to particle rows. Native rigid-pose tokens provide the rigid output branch.
Standalone joint-MPM and specialist pose heads are retired, recoverable at
`c5fac33`. Stateless temporal embedding, physics preparation, action masking and
gather operations live directly in [TrajectoryTransformer](../models/transformer.py).
Learned modules and state-dict keys of retained backbones remain unchanged.
Six nonzero model cases check forward, input/parameter gradients, AdamW updates
and four Euler particle/pose integration steps against frozen source at identical
loaded weights. Fresh Transformer modulation projections now explicitly start at
zero, making each block identity; the remaining random initialization and RNG
consumption are unchanged. `test_transformer_initialization.py` checks that the
gates and branches learn and that loading raw/EMA weights preserves saved outputs.
Both sides use the corrected shared SO(3) logarithm; its intentional half-turn
repair is covered by endpoint, reconstruction and gradient tests in `test_pose_ops.py`.
The shared input rewrite, context pose deltas and output materialization disable
autocast around FP32 geometry, including joint composition. Casting inputs to FP32
alone does not protect matrix operations from AMP. Neural-network layers keep the
caller's precision; `test_pose_precision.py` checks reconstruction, input gradients
and the return to mixed precision in both backbones on CPU and CUDA.
Sixty additional objective cases cover both parameterizations, scene types,
context-noise choices and optional losses. This is not a model redesign.

The default stratified Transformer decoder retains each particle's full-resolution
noisy-future embedding as its query residual alongside the compressed latent
stream. Fine particle information is therefore not available only through pooled
latent values.

The UNet processes `(C-1)+F` transition tokens. Its base features are normalized
delta (3), offset from frame 0 (3), height z (1), and physics (3). The optional
particle action channels add displacement (3) and an actuation flag (1).
The current unified UNet recipe leaves those channels off: all actuators are
rigid bodies with action conditioning on their pose tokens. This is not suitable
as-is for a deformably actuated MPM-drag model.

The UNet hierarchy uses within-object L0 attention, a material-keyed L1 point
budget, coarser spatial/temporal blocks, skip connections and PointNet pooling/
unpooling in the supported recipe. Registers begin at the configured coarse
level. Decoding returns deformable particle channels and native rigid-pose
channels; pure-rigid batches can legitimately return `None` for particles.

The optional `configs/unet_adaptive.yaml` replaces the material table with a
count-based allocation, protecting small objects and capping both per-object and
scene L1 particle counts. See [allocation policy and limits](UNET_ADAPTIVE_ALLOCATION.md).
The original recipe and historical checkpoint allocation remain unchanged.

The UNet stores L0 encoder-block outputs before pooling and merges them into the
mirrored full-resolution decoder blocks through learned long skips. This route
is separate from the PointNet unpool skip gate. Neither backbone's fine-resolution
route guarantees an exact identity map: small CPU Jacobian witnesses demonstrate
local noisy-input access, not convergence or a causal explanation of rollout errors.

Two geometric conventions are deliberately preserved:

| Path | Coordinates supplied to geometric operations |
|---|---|
| Unified Transformer | Integrated noisy positions at the **end** of transitions, with its configured spatial RoPE calibration. |
| Unified UNet | Integrated noisy positions at the **start** of transitions (exclusive cumulative sum), divided by its fixed 0.133155 m geometry radius; detached from gradients. Offset/height features use position normalization separately. |

The UNet's own transition noise therefore does not appear in that transition's
geometry, although preceding noisy increments do. Equal parameter counts do not
make these recipes a backbone-only ablation: geometry conventions, feature
normalization, spatial bottlenecks and conditioning differ. None of those trained
semantics was changed in this cleanup.

For native poses, noisy six-channel deltas are integrated from the known context
anchor. Rigid particle rows are reconstructed from those noisy poses and fixed
templates **before** geometry/attention. After sampling, the same integration
and rendering reconstruct rigid predictions. Particle and pose channels cannot
independently predict contradictory rigid geometry.

The batch contains future GT for losses/metrics, but that does not make it a
permitted feature. New tests hold context, actions, templates, noise and timesteps
fixed, perturb passive future positions/poses/targets, and require bit-exact
outputs for Transformer and UNet in deformable, mixed and rigid scenes. Nonzero
heads and a changed-noise control make these non-vacuous checks, not a proof for
every possible experimental flag combination.

## 4. Objectives, updates, EMA, continuation

[utils/trajectory_objective.py](../utils/trajectory_objective.py) now owns the loss
math and options. [train.py](../train.py) owns orchestration, diagnostics and
stateful updates. Train/validation use the same objective options. Historical
helper imports from `train` remain aliases.

The important distinction is between **network output parameterization** and
**loss coordinates**:

| Output mode | Particle loss | Native-pose loss |
|---|---|---|
| Velocity | MSE against `epsilon-x0` | MSE against `epsilon_pose-x0_pose` |
| Sample | MSE against clean `x0` | Convert to velocity with `(x_t-pred_x0)/t`, then velocity MSE |

Thus a pose clean-target error e contributes `e²/t²` in sample mode, while a
particle error e contributes `e²`. A controlled test checks e=0.1 at t=.05 and
t=.5. This asymmetry is preserved, explicitly documented, and remains a future
scientific ablation. **MPM has no native-pose term**, so it does not directly
explain MPM's near-static failure.

Native-pose particle loss counts deformable rows only; rigid targets have their
own term weighted by `lambda_pose_native`. Other auxiliaries have different
coordinates/units and are opt-in. The `spread_axes`-only initialization bug is
fixed. Retired offset/standalone-pose auxiliaries and per-object noise settings are
rejected. Requested mesh loss no longer silently becomes zero when its dependency
is missing; fresh mesh experiments are cloth-only because its object-ID convention
is not a domain-generic semantic selector.

The UNet `supervise_context` recipe is retired with `train.noise_context` (new
configs refuse both). It supervised **particle** context predictions, not pose
context predictions. Its combined loss is
`(future_loss * n_future + particle_context_loss * n_context) / (n_future + n_context)`,
where the counts include valid deformable coordinates only. On mixed cloth
batches this also multiplies the future pose term by
`n_future / (n_future + n_context)`; `lambda_pose_native` is therefore not its
effective final weight. This historical arithmetic is preserved for direct
objective calls; no training config can enable it.

[utils/optimization.py](../utils/optimization.py) accepts/rejects an entire
gradient accumulation group. Non-finite losses/gradients or spikes trigger a
collective rejection; rejected groups do not update weights or Adam moments.
Gradients are cleared after the decision. Attempt counters and accepted-update
counters are distinct: LR scheduling and EMA advance with accepted updates.
`train.steps`, logging/checkpoint labels, grace periods and `next_step` count
**attempted synchronized accumulation groups**, including rejected groups; they
do not count individual microbatches. Consequently, rejected groups reduce the
number of optimizer updates within a fixed run budget, and a nonconstant learning
rate schedule can stop before its nominal final value. This is the preserved
budget convention, not an automatic retry policy. Resume tests include a rejected
group and check that model/Adam, accepted-update scheduling, EMA and attempt
counters continue exactly across its checkpoint boundary.

[utils/ema.py](../utils/ema.py), [utils/lr_schedule.py](../utils/lr_schedule.py),
and [utils/checkpoint.py](../utils/checkpoint.py) retain strict state restoration
and atomic checkpoint replacement. Loading validates learned keys/shapes; only
verified deterministic buffers can be regenerated. Checkpoints include model,
optimizer, scheduler, EMA and rank-local RNG/scaler/loop state at a synchronized
boundary. This reproduces **fixed-input numerical continuation**, not WebDataset
iterator/worker queue positions or exact streaming corpus replay. The exact
continuation checks use small CPU models and fixed inputs; they do not establish
exact replay of full-size stochastic GPU training.

## 5. Sampling, autoregression, evaluation, visualization

[inference.py](../inference.py) contains shared flow-matching, per-frame DF and
passive-MPM rolling-buffer samplers. [utils/conditioning.py](../utils/conditioning.py)
preserves the trained context timestep representation: absent embedding, clean
t=0, and the explicit clean-t=1 recipe are not interchangeable. Sample outputs
are converted to velocity for FM Euler updates; pose and particle channels use
the same time schedule.

`inference_steps` is not generally a model-call count. With the campaign's F16
pyramid schedule and `inference_steps=25`, each frame receives 25 nonzero Euler
updates, while the rollout makes 40 full-window model calls. The corresponding
full-sequence schedule uses 25 calls. Report the schedule and future length along
with the configured step count.

[scripts/unified/eval_unified.py](../scripts/unified/eval_unified.py) selects the
checkpoint's embedded normalization snapshot and C/F lengths. Its default domain
list follows the checkpoint's training subset; `--domains` explicitly enables
cross-domain evaluation. Root overrides construct local datasets rather than
mutating global training/evaluation roots. Fixed-window evaluation and overlapping
autoregression are separate modes. Autoregression is `inference.windowed_rollout`,
shared by every evaluator and by guidance's receding horizon: it carries predicted
particle history and the full predicted rigid-pose context, and the final overlapping
window commits only its previously uncovered tail. It does not reset passive state to
GT between windows. Rolling-buffer streaming is a different protocol again.
[EVALUATION.md](EVALUATION.md) maps questions to commands and defines every metric key.
[EVAL_PROTOCOLS.md](EVAL_PROTOCOLS.md) fixes the scored clips of each pwm-data-v1 test folder
(`scripts/unified/freeze_eval_cases.py`, `utils/eval_cases.py`, `data/manifests/eval_cases/`).

Future **achieved actuator motion** is supplied as conditioning. Pushing never
uses passive-block `absolute_targets` as actions. These experiments test
trajectory prediction given an actuator path, not the accuracy of a low-level
controller following raw commands.

Unified particle errors are Euclidean distances in mm, averaged within an episode
and then across episodes (`utils.evaluation.TrajectoryScores`, used by every evaluator
except the paper protocols). Rigid errors are translation mm / rotation geodesic
degrees of each body's template fitted onto its predicted particles, for every
checkpoint type, split passive/actuated. Cloth Chamfer is the sum of directional mean
nearest-neighbour distances in cm at explicit frame marks, not squared distance.
Paper/native-MPM harnesses retain their own grouping and temporal conventions;
their scores must not be relabelled as this unified protocol. In particular:

- Paper MPM fixes its first prediction at native frame 8 and scores frames 8–23,
  8–48 and frame 48. It crops the context according to checkpoint C, and only
  calls an aggregate `ID_*` when all four ID groups are present.
- Paper cloth fixes its first prediction at post-stride frame 30/native frame 60.
  Its two-episode cache records context, source/output hashes and exact span;
  unverified or stale caches require a new output root. A different data vintage
  does not by itself prove that episodes were held out from a checkpoint.
- Native WDS MPM has window, full-clip (the shared windowed rollout) and rolling-buffer protocols. Its
  ten named passive populations include seen/unseen, alternate views, OOD,
  paper-era inputs and diagnostic subsets without claiming those are independent
  test sets. All use the unified canonical adapter and checkpoint snapshot.
- H5 MPM decoding preserves its separate raw input format, applies the identical
  canonical shift, and uses checkpoint C/F, noise schedule and normalization, the
  shared rollout and the shared scores.
- Historical cloth block-commit, standalone pickup/pushing-pose and guidance
  protocols are retired to the frozen revision, not relabelled as unified scores.

Audited evaluation paths reject empty budgets/runs, invalid particle sets and non-finite
predictions/metrics. Subset/body masks are resolved per element, not borrowed from
element 0. Unified evaluation JSON records resolved C/F, seed, sampling settings, checkpoint hash,
code revision/dirty state, corpus roots, normalization snapshot and active exclusion hashes.
These identify declared inputs, not a full inventory of all corpus bytes; per domain
the JSON also records the shards the scored clips came from. Evaluators take
`--clips all` for a complete finite pass or a positive clip count; zero, negative and
the retired `--max_batches` are errors. Evaluation reads interleave task roots and
object-count directories, so a partial pass covers every one of them. The per-frame
curve requires a common clip horizon.

Rollout dumps retain float32 positions: float16 rounding near metre-scale
coordinates can erase sub-mm motion. Dumps select the first episode by default;
`max_motion` is an explicit visualization selection, not an unbiased benchmark
sample. Motion scoring includes the anchor-to-first-future transition, including
one-frame rollouts. Point decimation is deterministic and for viewing only.
Existing GIF/plot tests cover frame completeness, explicit bounds and comparison
contracts. No model-quality claim follows from a visually pleasing rollout.

## What this review does not settle

The cleanup does not itself establish an architecture ranking, normalization-only
causal result or near-static memorization success. Completed joint-profile,
static/moving MPM parameterization and moving-cloth clipping diagnostics are in
the round-four report, alongside fixed-checkpoint precision comparisons and
their explicit trained-on-window scope. Near-static MPM still fails its no-motion
baseline at the tested budgets; other window/task outcomes differ. Fixed-weight
FP32 inference is not FP32 retraining, and eight sampling seeds are not eight
training seeds or a held-out generalization interval.

Next questions concern controlled optimizer/LR, EMA and context-noise interventions,
matched training budgets, broader train-only calibration coverage and untouched
final evaluation populations. The cleaned stack makes those tests easier to audit;
it does not turn completed engineering checks into paper-readiness evidence.
