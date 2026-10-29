# Evaluating a checkpoint

Every evaluator on this branch uses one rollout (`inference.windowed_rollout`), one
scorer (`utils/evaluation.py`, `TrajectoryScores`) and one metric vocabulary. This page
says which command answers which question, what the numbers mean, and what a receipt
must show before a number is quoted. The datasets of release pwm-data-v1 additionally have
frozen case lists with trivial references per clip: [EVAL_PROTOCOLS.md](EVAL_PROTOCOLS.md).

## Which command

| Question | Command | Set |
|---|---|---|
| How good is this checkpoint on pwm-data-v1? (numbers you report) | `eval_protocol.py --checkpoint <run>/checkpoints/final.pt` ([EVAL_PROTOCOLS.md](EVAL_PROTOCOLS.md#the-one-command)) | every frozen case set of every dataset the checkpoint trained on, whole-clip rollouts as the headline, with per-frame curves, trivial references and saved predictions |
| How good is this checkpoint on the four older datasets? (numbers you report) | `eval_unified.py --rollout --clips all` | the validation split of every domain the checkpoint trained on |
| A quick check while developing | `eval_unified.py --rollout --clips 200` | the first 200 clips of that split's stratified order; label it as a quick check |
| One model window from ground-truth context | `eval_unified.py --clips all` (no `--rollout`) | the same split |
| A frozen case set of a pwm-data-v1 dataset ([EVAL_PROTOCOLS.md](EVAL_PROTOCOLS.md)) | `eval_unified.py --cases <manifest> --case_set <name> --clips all` (`--rollout --rollout_frames 128` for rollout sets) | exactly the manifest's clips, with `rows.predicted` / `rows.moving` / `rows.still` |
| The rollout with every window sampled at one noise level | `eval_unified.py --rollout --clips all --schedule full_sequence` | the same split |
| MPM generalization (seen, unseen, OOD object counts) | `eval_object_rollout.py --population <name> --rollout full --clips all` | a named MPM population ([table](CODE_WALKTHROUGH.md#mpm-evaluation-populations)) |
| Native episodes (SkillWeaver / Yikai) | `scripts/dev/native_eval_unified.py --rollout` | every selected window of the episodes you pass |
| Paper Table 1 (MPM) or Table 3 (cloth) | `eval_paper_mpm_unified.py`, `eval_paper_cloth_table3.py` | the published episodes and frames |
| Which checkpoint training keeps as `best.pt` | automatic | `validate_sampler` on the first `log.sampler_batches` validation batches |

```bash
# the standard: rollout over the whole validation split
PYTHONPATH=. python scripts/unified/eval_unified.py --checkpoint runs/<run>/checkpoints/final.pt \
    --domains pickup --rollout --rollout_frames 36 --clips all --seed 0 --out eval.json

# a named MPM population
PYTHONPATH=. python scripts/eval_object_rollout.py --checkpoint runs/<run>/checkpoints/final.pt \
    --population v2_unseen_00001plus --rollout full --clips all --max_objects 4 --out unseen.json
```

## The protocol

- **Set.** The checkpoint's validation split. A run trained with `data.roots` is scored
  on that saved release with its own exclusion lists (the saved paths must exist), and
  `--mpm_val_root` and `--pickup_roots` re-point only registered domains;
  `--test_folder DOMAIN=FOLDER` scores another test folder of an indexed dataset of that release
  (a report folder such as `pwm-data-v1/mpm_fresh/test_seen`, see
  [EVAL_PROTOCOLS.md](EVAL_PROTOCOLS.md#report-folders)). Otherwise the
  split is the last 10% of sorted shards (per task root for pickup, pooled for pushing),
  after the registered exclusions. `--clips all` reads all of it; `--clips N` reads the
  first N clips in the order below. A clip is one
  scored trajectory. Shards pack several clips per record, so a count of batches or
  records meant a different number of clips in every domain; `--max_batches` is retired.
- **Order.** Evaluation reads alternate records across dataset roots, then across the
  directories below each root (`stratified_shard_groups` in `data/dataset.py`), so any
  prefix covers every pickup task and every MPM object count. Training reads are never
  reordered. Before 2026-09-14 evaluation read shards one after another: 40 batches
  scored one place shard for pickup and only 1- and 2-object scenes for MPM. Setting a
  dataset's `stratify_shards` attribute to `False` restores that order to reproduce an
  old receipt.
- **Checkpoint settings.** Context and future lengths, normalization snapshot, sampler
  schedule and EMA weights come from the checkpoint. Overriding a window or the schedule
  prints a warning and is recorded.
- **Schedule.** Each window is sampled with the checkpoint's `df_schedule`, `pyramid` for
  every current checkpoint: later frames in the window stay noisier for longer, and every
  frame is clean before the window is committed. `--schedule full_sequence` samples the
  whole window at one shared noise level instead (25 model calls per window rather than
  40 at 16 frames and 25 steps), still committing one window at a time. Training always
  uses diffusion forcing; the flag changes sampling only.
- **Rollout.** `inference.windowed_rollout`. Model windows use the trained future
  length up to the remaining horizon; the last window can be shorter. Every window
  starts at the latest committed state. After the first window the context is the
  model's own output, particles and carried rigid-body poses. Ground truth enters
  only as the actuator's actions, and
  checkpoints trained with `train.action_dropout=1` get those hidden too
  (`"actions": "hidden"`). The horizon is `min(--rollout_frames, clip frames - context)`:
  36 frames for pickup clips and 45 for MPM clips at context 4. Guidance plans on the
  same windows (`inference.window_bounds`, `inference.rollout_window`).
  Before the 2026-09-14 continuity fix, a partial final window slid backward and
  discarded regenerated overlap. That could introduce a discontinuity where its
  suffix joined the committed trajectory. Receipts from that protocol must be
  distinguished from the corrected forward-window rollout; full windows are unchanged.
- **Seeds.** Each batch is seeded from `--seed` and its position, so a domain's numbers
  do not depend on which domains the same command evaluated before it.

## The vocabulary

Every evaluator above except the two paper protocols reports these keys.

| Key | Meaning |
|---|---|
| `n_episodes` | clips scored |
| `particle_mean_mm`, `particle_final_mm` | Euclidean error over a clip's valid particles, averaged over its scored frames (or taken at the last one), then averaged over clips |
| `particle_at16_mm`, `particle_at32_mm` | the same at the 16th or 32nd scored frame, when the horizon reaches it |
| `particle_per_frame_mm`, `particle_per_frame_n` | the per-frame curve: at frame h the mean over the clips that reach h, and how many do (MPM populations, the h5 tool, and every `--cases` receipt, which adds the same pair for deformable rows and the passive bodies' pose) |
| `particle_mse_mm2`, `particle_mse_per_frame_mm2` | the squared Euclidean distance in mm^2, averaged like the error in mm (`--cases` receipts) |
| `chamfer_mean_mm`, `chamfer_final_mm`, `chamfer_per_frame_mm` | symmetric Chamfer distance of the predicted rows at every frame in mm; it needs no point correspondence (`--cases` receipts, under `rows.predicted`) |
| `reference.zero`, `reference.hold` | the keys above for predicting no motion and for holding the last context velocity, scored on the same clips (`--cases` receipts) |
| `deform_mean_mm`, `deform_final_mm` | the particle error on deformable rows only (cloth) |
| `chamfer_cm_at{m}`, `chamfer_n_at{m}` | symmetric Chamfer on deformable rows at frame m in cm (the sum of both directional mean nearest-neighbour distances), and the clips it covers |
| `pose_passive_mm`, `pose_passive_deg` | bodies without actions (the object): each body's template fitted onto its predicted particles (Kabsch) and compared with the ground-truth pose, translation mm and geodesic degrees, averaged over frames, then over clips and bodies |
| `pose_actuated_mm`, `pose_actuated_deg` | the same for actuated bodies (gripper links, rod) |
| `horizon_frames_min`, `horizon_frames_max` | frames scored per clip |

Pose errors come from the predicted particles for every checkpoint type, so particle-only
and pose-native checkpoints are measured the same way.

`sample_mm` in the training log, which selects `best.pt`, is the particle error of one
16-frame window from ground-truth context on the first validation batches of the same
stratified stream: an early read of the table's first window, not the table.

## The receipt

Quote a number only with its `eval_unified` receipt, which records

- `checkpoint`, `checkpoint_sha256`, `git_commit`, `git_dirty`, `torch_version`
- `clips_requested` (`"all"` or N) and, per domain, `n_episodes`
- `rollout`, `rollout_frames_requested` and, per domain, `horizon_frames_min` and `horizon_frames_max`
- `context_len`, `future_len`, `inference_steps`, `df_schedule` (the schedule used),
  `checkpoint_df_schedule`, `df_uncertainty_scale`, `normalization`, `seed`, `actions`,
  `clamp_actuated`
- `cases` when a frozen case set was scored: the manifest, its sha256, the set and its size
- `inputs` per domain, read from the dataset that was scored: `roots`, `roots_source`
  (`registered`, `checkpoint data.roots` or `command line`), normalization and exclusion hashes
- per domain, what was actually scored: `shards_declared`, `shards_read`, `clips_per_shard_read`

## Not comparable

- Receipts made before 2026-09-14: sequential reads, one process-level seed,
  `max_batches` counted in records, and `eval_object_rollout --rollout full` used strided
  chunks with a shorter last window. Re-run under this protocol before comparing.
- A quick check (`--clips N`) and the whole split.
- One window and a rollout, windowed rollout and streaming, pyramid and full-sequence
  sampling, different horizons, and different sets (seen, unseen, OOD, pinned).
- The paper protocols, which keep their published episodes, crops, units (cm) and rollout.
