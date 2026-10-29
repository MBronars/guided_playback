# Evaluation protocols for pwm-data-v1

One protocol per dataset of the release
`/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1`. It fixes which
clips of a test folder are scored, how, and which trivial numbers every model number is read against.
[EVALUATION.md](EVALUATION.md) stays the guide to the evaluator, its metric vocabulary and its receipts;
this page adds the frozen case lists on top of it.

Status 2026-09-19: frozen for all eight training datasets of the release: `mpm_fresh` and `mpm_drag` with their report folders,
`folding_8mm`, `pushing`, `rl_push`, `yikai`, `dex_v2_filtered2` (the dex folder to train and evaluate on; its parents
`dex_v2_filtered` and `dex_v2` stay for provenance) and `pickup`. Pushing, rl_push and yikai have small test folders
(100 / 200 / 200 episodes). The long recordings (folding, yikai, dex) are also rolled out to the last frame of every test episode.
The current case list of each dataset is named in one table, `CURRENT` in `scripts/unified/eval_protocol.py`
(see "Freezing and versions"). No 200K checkpoint has been scored under these protocols yet.

## The one command

```bash
PYTHONPATH=. python scripts/unified/eval_protocol.py --checkpoint <run>/checkpoints/final.pt
```

scores a checkpoint, a specialist or a unified model alike, on every dataset it was trained on: the dataset's current case list,
every case set of it, one `eval_unified.py --cases` command per set, with everything below fixed. It takes no option that changes a
number (no seed, window, clamping or case filter), so everyone who runs it on a checkpoint gets the same receipts. Use it for every
reported number; the single commands further down are what it runs.

- **Sampling schedule.** `--schedule pyramid | full_sequence`, default the checkpoint's own `df_schedule`. It is the one protocol
  setting, because both are in use. Each schedule has its own output folder and is named in every receipt and summary.
- **Whole-clip rollouts are the headline.** Every comparison reads the rollout that reaches the end of the clip: `rollouts_full`
  for folding, yikai and dex, `rollouts_first` for the other datasets, whose clips that set already covers whole.
- **Output**, default `<run>/eval_protocol/<checkpoint name>_<schedule>/`: per case set a receipt `<dataset>/<set>.json`, its
  per-case sidecar `<set>.cases.json`, the evaluator's log, and for the headline set `<set>.predictions/` (`--predictions all | none`
  changes which sets save predictions); `SUMMARY.json` and `SUMMARY.md` put every set next to its trivial references and fail loudly
  when the evaluator's no-motion number differs from the frozen list's, which would mean other clips were read.
- `--report_folders` adds the MPM report folders. `--datasets` and `--sets` restrict a run and mark the summary partial. `--resume`
  keeps receipts that match the checkpoint, case list, set and schedule. `--dry_run` prints the commands.

What a receipt of a case set holds, on the predicted rows and next to the usual keys:

| key | meaning |
|---|---|
| `particle_mean_mm`, `particle_final_mm`, `particle_per_frame_mm` with `particle_per_frame_n` | Euclidean distance in mm, averaged over a clip's rows and frames, then over clips; per frame over the clips that reach the frame, so clips of different lengths share one curve |
| `particle_mse_mm2`, `particle_mse_per_frame_mm2` | the squared distance in mm^2, averaged the same way |
| `chamfer_mean_mm`, `chamfer_final_mm`, `chamfer_per_frame_mm` | symmetric Chamfer distance of the predicted rows at every frame: the sum of the two directional mean nearest-neighbour distances. It needs no point correspondence, so it also compares with methods that predict unordered points |
| `pose_passive_mm`, `pose_passive_deg`, `pose_passive_per_frame_*` | the passive rigid bodies' pose error, fitted from the predicted points |
| `reference.zero`, `reference.hold` | the same keys for predicting no motion and for holding the last context velocity, scored by the same scorer on the same clips |

The sidecar `<set>.cases.json` holds one record per case: its id, start, horizon, stratum and tag, and its per-frame curves
(`predicted_mm`, `predicted_mse_mm2`, `predicted_chamfer_mm`, `moving_mm`, `all_mm`, `pose_passive_mm`, `pose_passive_deg`, and the
trivial `zero_predicted_mm`, `zero_predicted_chamfer_mm`, `hold_predicted_mm`), so any horizon, stratum or tag is read without
scoring again. `<set>.predictions/<position in the set>.npz` holds a case's predicted positions at every scored frame (`pred`
[frames, rows, 3], metres, float32, on the clip's valid rows; `rows` their indices in the clip's particle order; `predicted` which
of them the model predicts; `id`, `start`, `horizon`), for later visualization and as the layout in which another method's
predictions will be scored on the same cases.

## What holds for every dataset

- **Set.** The dataset's `test` folder, whole, with the exclusion lists the recipe names. There is no
  development split (decision 2026-09-17): the test folder is also the validation folder of training, so the
  curves and `best.pt` read it. Reported numbers therefore come from `final.pt`, the fixed end of the
  schedule, and every table says that the test set also served as validation.
- **Reading.** `scripts/unified/eval_unified.py` through the checkpoint's saved `data.roots`: context 4,
  future 16, the checkpoint's normalization snapshot, sampler settings (25 steps, pyramid schedule) and EMA
  weights. Yikai clips start at frame 10 or later, rl_push clips at frame 3 or later, every other dataset at
  frame 0 (`data/recording_adapters.FIRST_START`, `data/rl_push.RL_PUSH_FIRST_START`).
- **Rollouts.** `inference.windowed_rollout`: after the first window the context is the model's own output;
  ground truth enters only as the actuator's actions.
- **Seed.** `--seed 0`, one seed for every reported number (user decision 2026-09-19: no multi-seed averages). The sampler is
  stochastic, so a table says so.
- **Schedule.** The checkpoint's own `df_schedule` (pyramid for every pwm-data-v1 recipe) unless the command names
  `full_sequence`; a receipt records both.
- **Metrics.** The vocabulary of [EVALUATION.md](EVALUATION.md#the-vocabulary), averaged within a clip and
  then across clips. The headline per dataset is named below.
- **Receipt.** A number is quoted with its `eval_unified` receipt, which records the case manifest, its
  sha256 and the case set (`cases`).

## Case sets

`scripts/unified/freeze_eval_cases.py` reads a test folder through a training recipe exactly as the evaluator
does (`evaluation_dataset`, rigid-body sidecar on) and writes `data/manifests/eval_cases/<dataset>_v1.json`
(`kind: pwm_eval_cases`). A case is `(id, start, horizon)`: the clip `frames start .. start + 4 + horizon - 1`
of one episode, the last `horizon` frames scored. `utils/eval_cases.load_cases` reads a manifest.

| case set | what it is | datasets |
|---|---|---|
| `windows` | one 16-frame window per test episode, exactly the window the loader's evaluation mode yields (indexed datasets: stratified order, start drawn from the episode id; rl_push: start drawn from subset seed, object and episode; pushing: the prefix window). It is what training-time validation reads. | all |
| `windows_motion` | four 16-frame windows per episode, at the within-episode quantiles 0.125 / 0.375 / 0.625 / 0.875 of the window's mean predicted step (tags `q1` to `q4`; ties go to the earliest start) | `folding_8mm`, `yikai`, `dex_v2_filtered2` (long recordings) |
| `rollouts_first` | rollouts from the first allowed frame. Indexed datasets: the first 256 episodes of the stratified order (all 174 dex episodes), 128 frames; folding since `folding_8mm_v3`: the first 50 episodes of that order, which goes round the fold actions. rl_push: 256 episodes taken round-robin over the 117 objects in the loader's order, each object's episodes in `sha256(object:episode)` order, to the episode's end. pushing: whole 40-frame clips, the first 8 episodes of the 32 shards first in `sha256('pushing-rollouts:' + name)` order. | all |
| `rollouts_active` | the same episodes, 128 frames from the start whose scored frames carry the most predicted motion (ties: earliest) | `folding_8mm`, `yikai`, `dex_v2_filtered2` |
| `rollouts_full` | the same episodes from the first allowed frame to the episode's last frame: 898 scored frames for each of folding's 50 rollout clips, 233 to 921 for yikai, 35 to 396 for dex. **The headline of these three datasets** (user decision 2026-09-19: every comparison uses whole-episode rollouts). The other datasets need no such set: their `rollouts_first` already ends at the clip's last frame (45 to 109 frames for MPM, 36 for pushing and pickup, the episode's end for rl_push) | `folding_8mm`, `yikai`, `dex_v2_filtered2` |

Scoring a set (each manifest repeats its commands under `evaluator`):

```bash
# windows and windows_motion: one model window per case
PYTHONPATH=. python scripts/unified/eval_unified.py --checkpoint <run>/checkpoints/final.pt --domains yikai \
    --cases data/manifests/eval_cases/yikai_v2.json --case_set windows_motion --clips all --seed 0 --out <receipt>.json
# rollout sets: --rollout with --rollout_frames at least the set's horizon (128; 36 for pushing)
PYTHONPATH=. python scripts/unified/eval_unified.py --checkpoint <run>/checkpoints/final.pt --domains folding \
    --cases data/manifests/eval_cases/folding_8mm_v1.json --case_set rollouts_active --rollout --rollout_frames 128 \
    --chamfer_marks 16,64,128 --clips all --seed 0 --out <receipt>.json
```

`--case_tag q4` and `--case_stratum <name>` keep one group of a set (a motion quantile, `beyond`, a fold
action, a yikai population). `--cases` reads the clips through the checkpoint's own validation dataset (`utils/eval_cases.CaseDataset`),
refuses a manifest frozen on another test index or folder, and adds the particle keys on the `predicted`,
`moving` and `still` rows under `rows`. Without `--cases`, `--clips all` scores the `windows` clips in the
loader's order, and `--rollout --rollout_frames 128 --clips 256` the `rollouts_first` clips of an indexed
dataset, both without row sets. A `--clips` prefix cannot express the rl_push and pushing rollout sets
(it would cover five objects or one shard).

Why the motion-selected sets exist: the first 128 frames of a long recording are mostly the approach. On
the frozen lists the zero-motion error of a rollout from the first frame is 1.8 mm on folding and 7.5 mm on
yikai; from the most active span it is 14.7 mm and 55.7 mm. A model can look good on `rollouts_first` by
predicting little.

The freezer checks itself on the real data: every 25th episode's window from the adapter must equal the slice
of the whole recording the cases are cut from, position for position and number for number, and the first
three episodes are also read through the loader's iterator. The reader is checked the other way round: a
model that predicts no motion, run through `eval_domain` and `rollout_domain` on a case set, scores the
manifest's trivial references (`tests/test_eval_cases.py`), and on the real release the reader's clips
reproduce the frozen rows of all twelve sets.

## Row sets and trivial references

Rows of a clip (`utils/eval_cases.scored_rows`; the batch carries the rigid sidecar, as in the evaluator):

| rows | definition | why |
|---|---|---|
| `all` | every valid particle | the evaluator's `particle_mean_mm`; it includes actuators, which follow given actions |
| `predicted` | `all` minus rigid actuators and fixtures; commanded deformable particles stay | what the model is asked to predict (the overfit harness's `primary_mask`) |
| `moving` | predicted rows that leave their last context position by 5 mm or more at some scored frame | error where something happens |
| `still` | the other predicted rows | motion the model invented |

Every case stores, in mm, the error of predicting no motion and of holding the last context velocity on the
predicted rows (mean over scored frames and rows, and at the last frame), the zero-motion error on `all` and on
`moving` rows, the mean step of the predicted rows, and the zero-motion pose error of the clip's passive rigid
bodies (mm and degrees; exactly the scorer's `pose_passive_*` for that prediction). A set's `summary` holds the
episode-weighted means, per stratum and per tag; pose summaries named `_pairs_` use the scorer's (clip, body)
weighting. A model number is always read next to the trivial number of the same set.

The evaluator's usual keys are `particle_mean_mm` (all rows), `deform_mean_mm` (cloth rows, which are
folding's predicted rows) and `pose_passive_*` (the predicted bodies of the rigid datasets). With `--cases`
it also reports `rows.predicted`, `rows.moving` and `rows.still` (particle keys; `n_episodes` there counts the
clips that have such rows).

## Frozen protocols

Reference numbers below are rendered from the manifests by
`/data/user_data/robertre/scratchpad/eval_protocols_0918/render_tables.py`; they describe the data, not a model.

Whole sets (episode-weighted means; pose column: the scorer's clip-and-body weighting):

| dataset | case set | cases | horizon (frames) | cases with moving rows | zero motion, predicted rows (mm; mean / final) | zero motion, moving rows (mm) | hold velocity, predicted rows (mm; mean / final) | zero motion, passive pose (mm / deg) |
|---|---|---|---|---|---|---|---|---|
| `dex_v2_filtered2` | `windows` | 174 | 16 | 155 | 13.26 / 19.34 | 14.90 | 28.91 / 52.75 | 8.44 / 32.77 |
| `dex_v2_filtered2` | `windows_motion` | 680 | 16 | 599 | 12.04 / 17.99 | 13.67 | 26.38 / 48.81 | 7.30 / 30.31 |
| `dex_v2_filtered2` | `rollouts_first` | 174 | 35 to 128 | 174 | 30.97 / 37.51 | 30.97 | 123.19 / 231.36 | 21.85 / 78.86 |
| `dex_v2_filtered2` | `rollouts_active` | 170 | 77 to 128 | 170 | 29.18 / 34.22 | 29.18 | 137.31 / 262.62 | 18.63 / 81.59 |
| `dex_v2_filtered` | `windows` | 190 | 16 | 168 | 13.28 / 19.19 | 15.02 | 28.53 / 51.91 | 8.64 / 32.04 |
| `dex_v2_filtered` | `windows_motion` | 744 | 16 | 657 | 12.19 / 18.03 | 13.81 | 27.18 / 50.24 | 7.63 / 29.88 |
| `dex_v2_filtered` | `rollouts_first` | 190 | 35 to 128 | 190 | 32.06 / 38.59 | 32.06 | 117.75 / 218.95 | 23.07 / 80.68 |
| `dex_v2_filtered` | `rollouts_active` | 186 | 77 to 128 | 186 | 29.63 / 34.80 | 29.63 | 140.18 / 268.78 | 19.30 / 81.74 |
| `dex_v2` | `windows` | 194 | 16 | 171 | 13.23 / 19.10 | 15.00 | 28.49 / 51.85 | 8.65 / 31.79 |
| `dex_v2` | `windows_motion` | 760 | 16 | 672 | 12.19 / 18.01 | 13.80 | 27.22 / 50.31 | 7.67 / 29.72 |
| `dex_v2` | `rollouts_first` | 194 | 35 to 128 | 194 | 32.26 / 38.84 | 32.26 | 116.68 / 216.67 | 23.24 / 81.13 |
| `dex_v2` | `rollouts_active` | 190 | 77 to 128 | 190 | 29.90 / 35.12 | 29.90 | 139.84 / 268.04 | 19.62 / 81.68 |
| `folding_8mm` | `windows` | 236 | 16 | 191 | 0.54 / 1.01 | 4.36 | 0.37 / 0.75 | n/a / n/a |
| `folding_8mm` | `windows_motion` | 944 | 16 | 746 | 0.53 / 0.97 | 4.40 | 0.39 / 0.78 | n/a / n/a |
| `folding_8mm` | `rollouts_first` | 236 | 128 | 236 | 1.81 / 2.98 | 4.53 | 2.80 / 5.90 | n/a / n/a |
| `folding_8mm` | `rollouts_active` | 236 | 128 | 236 | 14.66 / 26.61 | 29.89 | 11.57 / 24.68 | n/a / n/a |
| `mpm_drag_test_seen` | `windows` | 100 | 16 | 75 | 10.49 / 17.23 | 24.09 | 9.72 / 21.89 | n/a / n/a |
| `mpm_drag_test_seen` | `rollouts_first` | 100 | 45 to 109 | 100 | 32.53 / 40.91 | 48.60 | 32.38 / 40.69 | n/a / n/a |
| `mpm_drag` | `windows` | 100 | 16 | 82 | 9.06 / 15.21 | 19.10 | 8.37 / 18.53 | n/a / n/a |
| `mpm_drag` | `rollouts_first` | 100 | 45 to 109 | 100 | 27.73 / 34.61 | 42.27 | 27.59 / 34.43 | n/a / n/a |
| `mpm_fresh_test_count_seen` | `windows` | 100 | 16 | 99 | 37.33 / 46.08 | 47.76 | 80.55 / 160.15 | n/a / n/a |
| `mpm_fresh_test_count_seen` | `rollouts_first` | 100 | 45 | 100 | 201.55 / 220.23 | 203.32 | 681.64 / 1437.56 | n/a / n/a |
| `mpm_fresh_test_count_unseen` | `windows` | 100 | 16 | 95 | 27.72 / 34.54 | 39.50 | 66.60 / 131.27 | n/a / n/a |
| `mpm_fresh_test_count_unseen` | `rollouts_first` | 100 | 45 | 100 | 196.40 / 215.19 | 200.35 | 688.36 / 1447.22 | n/a / n/a |
| `mpm_fresh_test_seen` | `windows` | 100 | 16 | 80 | 40.14 / 49.66 | 56.34 | 91.22 / 179.51 | n/a / n/a |
| `mpm_fresh_test_seen` | `rollouts_first` | 100 | 45 | 100 | 210.07 / 230.21 | 214.35 | 702.45 / 1492.68 | n/a / n/a |
| `mpm_fresh` | `windows` | 100 | 16 | 77 | 36.43 / 48.07 | 52.14 | 104.78 / 205.15 | n/a / n/a |
| `mpm_fresh` | `rollouts_first` | 100 | 45 | 100 | 209.08 / 230.76 | 211.17 | 708.14 / 1503.07 | n/a / n/a |
| `pickup` | `windows` | 200 | 16 | 149 | 28.89 / 43.01 | 39.03 | 32.36 / 65.65 | 27.80 / 12.23 |
| `pickup` | `rollouts_first` | 200 | 36 | 200 | 74.05 / 216.00 | 74.05 | 109.44 / 313.17 | 71.54 / 29.28 |
| `pushing` | `windows` | 100 | 16 | 64 | 11.96 / 22.43 | 18.80 | 4.06 / 10.10 | 11.52 / 7.67 |
| `pushing` | `rollouts_first` | 100 | 36 | 84 | 25.91 / 49.79 | 30.92 | 15.61 / 41.12 | 25.08 / 16.19 |
| `rl_push` | `windows` | 200 | 16 | 196 | 104.48 / 173.68 | 106.61 | 99.31 / 184.33 | 102.00 / 21.09 |
| `rl_push` | `rollouts_first` | 200 | 16 to 93 | 200 | 127.97 / 238.12 | 127.97 | 131.33 / 245.27 | 124.73 / 28.67 |
| `yikai` | `windows` | 200 | 16 | 120 | 4.48 / 7.62 | 33.55 | 4.62 / 9.84 | 4.08 / 1.66 |
| `yikai` | `windows_motion` | 800 | 16 | 483 | 4.71 / 8.05 | 35.47 | 4.80 / 10.16 | 4.25 / 1.40 |
| `yikai` | `rollouts_first` | 200 | 128 | 171 | 6.59 / 28.14 | 31.93 | 16.58 / 47.53 | 5.68 / 3.24 |
| `yikai` | `rollouts_active` | 200 | 128 | 200 | 55.04 / 94.26 | 223.34 | 69.90 / 133.41 | 51.02 / 13.41 |

Groups (motion quantile tags, yikai populations and extent tag, folding actions and dex teachers on the active rollouts; the manifests hold every group, those of the superseded dex folders are left out here; an MPM folder is one category, so it has no groups):

| dataset | case set | group | cases | cases with moving rows | zero motion, predicted rows (mm) | zero motion, moving rows (mm) | zero motion, passive pose (mm / deg) |
|---|---|---|---|---|---|---|---|
| `dex_v2_filtered2` | `windows_motion` | tag `q1` | 170 | 128 | 7.78 | 10.32 | 4.13 / 20.41 |
| `dex_v2_filtered2` | `windows_motion` | tag `q2` | 170 | 146 | 10.31 | 11.98 | 6.08 / 25.95 |
| `dex_v2_filtered2` | `windows_motion` | tag `q3` | 170 | 159 | 13.16 | 14.09 | 8.24 / 32.63 |
| `dex_v2_filtered2` | `windows_motion` | tag `q4` | 170 | 166 | 16.91 | 17.35 | 10.76 / 42.24 |
| `dex_v2_filtered2` | `rollouts_active` | stratum `in_hand_repose-sharpa` | 42 | 42 | 34.16 | 34.16 | 27.14 / 86.04 |
| `dex_v2_filtered2` | `rollouts_active` | stratum `in_hand_repose-xhand1` | 34 | 34 | 32.08 | 32.08 | 26.34 / 74.23 |
| `dex_v2_filtered2` | `rollouts_active` | stratum `in_palm_rotate-xhand1` | 47 | 47 | 28.12 | 28.12 | 19.41 / 70.28 |
| `dex_v2_filtered2` | `rollouts_active` | stratum `screwdriver-sharpa` | 23 | 23 | 22.49 | 22.49 | 4.07 / 88.96 |
| `dex_v2_filtered2` | `rollouts_active` | stratum `screwdriver-xhand1` | 24 | 24 | 24.89 | 24.89 | 5.24 / 99.31 |
| `folding_8mm` | `windows_motion` | tag `q1` | 236 | 108 | 0.12 | 3.32 | n/a / n/a |
| `folding_8mm` | `windows_motion` | tag `q2` | 236 | 175 | 0.17 | 3.44 | n/a / n/a |
| `folding_8mm` | `windows_motion` | tag `q3` | 236 | 227 | 0.35 | 3.52 | n/a / n/a |
| `folding_8mm` | `windows_motion` | tag `q4` | 236 | 236 | 1.48 | 6.46 | n/a / n/a |
| `folding_8mm` | `rollouts_active` | stratum `0` | 46 | 46 | 13.19 | 24.79 | n/a / n/a |
| `folding_8mm` | `rollouts_active` | stratum `1` | 48 | 48 | 11.47 | 28.81 | n/a / n/a |
| `folding_8mm` | `rollouts_active` | stratum `2` | 45 | 45 | 18.65 | 32.46 | n/a / n/a |
| `folding_8mm` | `rollouts_active` | stratum `3` | 48 | 48 | 15.84 | 29.37 | n/a / n/a |
| `folding_8mm` | `rollouts_active` | stratum `4` | 49 | 49 | 14.36 | 33.86 | n/a / n/a |
| `pickup` | `windows` | stratum `convex` | 50 | 36 | 3.54 | 4.90 | 3.04 / 2.91 |
| `pickup` | `windows` | stratum `pick` | 50 | 35 | 3.73 | 5.59 | 3.27 / 6.36 |
| `pickup` | `windows` | stratum `place` | 50 | 50 | 106.34 | 106.34 | 103.09 / 38.15 |
| `pickup` | `windows` | stratum `upright` | 50 | 28 | 1.93 | 4.53 | 1.80 / 1.50 |
| `pickup` | `rollouts_first` | stratum `convex` | 50 | 50 | 72.84 | 72.84 | 70.89 / 21.61 |
| `pickup` | `rollouts_first` | stratum `pick` | 50 | 50 | 51.10 | 51.10 | 49.38 / 26.27 |
| `pickup` | `rollouts_first` | stratum `place` | 50 | 50 | 129.32 | 129.32 | 123.91 / 55.40 |
| `pickup` | `rollouts_first` | stratum `upright` | 50 | 50 | 42.92 | 42.92 | 41.99 / 13.81 |
| `yikai` | `windows` | stratum `fail` | 26 | 13 | 3.77 | 26.71 | 3.40 / 2.11 |
| `yikai` | `windows` | stratum `gen` | 174 | 107 | 4.59 | 34.38 | 4.17 / 1.60 |
| `yikai` | `windows` | tag `beyond` | 2 | 2 | 7.04 | 41.86 | 6.50 / 0.51 |
| `yikai` | `windows_motion` | stratum `fail` | 104 | 51 | 3.05 | 23.91 | 2.52 / 1.30 |
| `yikai` | `windows_motion` | stratum `gen` | 696 | 432 | 4.96 | 36.84 | 4.47 / 1.42 |
| `yikai` | `windows_motion` | tag `beyond` | 9 | 5 | 2.91 | 31.25 | 2.91 / 0.87 |
| `yikai` | `windows_motion` | tag `q1` | 200 | 12 | 0.07 | 4.11 | 0.06 / 0.04 |
| `yikai` | `windows_motion` | tag `q2` | 200 | 95 | 0.93 | 8.62 | 0.77 / 0.54 |
| `yikai` | `windows_motion` | tag `q3` | 200 | 178 | 3.91 | 19.93 | 3.46 / 1.46 |
| `yikai` | `windows_motion` | tag `q4` | 200 | 198 | 13.93 | 64.22 | 12.72 / 3.56 |
| `yikai` | `rollouts_first` | stratum `fail` | 26 | 23 | 4.51 | 20.31 | 4.15 / 3.19 |
| `yikai` | `rollouts_first` | stratum `gen` | 174 | 148 | 6.90 | 33.73 | 5.87 / 3.25 |
| `yikai` | `rollouts_first` | tag `beyond` | 3 | 3 | 2.45 | 10.68 | 2.51 / 2.05 |
| `yikai` | `rollouts_active` | stratum `fail` | 26 | 26 | 24.45 | 99.45 | 22.45 / 10.31 |
| `yikai` | `rollouts_active` | stratum `gen` | 174 | 174 | 59.62 | 241.85 | 54.63 / 13.80 |
| `yikai` | `rollouts_active` | tag `beyond` | 4 | 4 | 41.93 | 136.89 | 48.14 / 9.47 |

Manifests:

| manifest | sha256 | code | test index / folder index sha256 |
|---|---|---|---|
| `data/manifests/eval_cases/dex_v2_filtered2_v1.json` | `c0851e88f03b` | `7a64e57` | `da1603c49384` |
| `data/manifests/eval_cases/dex_v2_filtered_v1.json` | `6afcc9b8c502` | `ef0eb7d` | `7c78f356bdcb` |
| `data/manifests/eval_cases/dex_v2_v1.json` | `246a16bc438c` | `0754324` | `9f31b1ae82ed` |
| `data/manifests/eval_cases/folding_8mm_v1.json` | `8d6539522293` | `af01c51` | `01577416c646` |
| `data/manifests/eval_cases/mpm_drag_test_seen_v1.json` | `7c480a007297` | `71a23cf` | `859302603e65` |
| `data/manifests/eval_cases/mpm_drag_v1.json` | `772542710356` | `71a23cf` | `175a5b99349c` |
| `data/manifests/eval_cases/mpm_fresh_test_count_seen_v1.json` | `8f3332e330b9` | `71a23cf` | `26959e2af50e` |
| `data/manifests/eval_cases/mpm_fresh_test_count_unseen_v1.json` | `4d1b135396db` | `71a23cf` | `d1e2e8eaea94` |
| `data/manifests/eval_cases/mpm_fresh_test_seen_v1.json` | `f70c8af1a187` | `71a23cf` | `4d5a9f8224c5` |
| `data/manifests/eval_cases/mpm_fresh_v1.json` | `ac5becd2110b` | `71a23cf` | `11e5b064d6e4` |
| `data/manifests/eval_cases/pickup_v1.json` | `90e6b4e5eb4a` | `eb93719` | `600b39462d34` |
| `data/manifests/eval_cases/pushing_v2.json` | `0b39060482b1` | `0edca8a` | `7781aef2f1de` |
| `data/manifests/eval_cases/pushing_v1.json` (the complete folder, now `test_full`; no longer matches `test`) | `34b9833fc836` | `af01c51` | `0c1800eb92c6` |
| `data/manifests/eval_cases/rl_push_v2.json` | `472315b83aee` | `0edca8a` | `2173a5ecd684` |
| `data/manifests/eval_cases/rl_push_v1.json` (the complete folder, now `test_full`; no longer matches `test`) | `3674b6a9e970` | `af01c51` | `82df904d4bbf` |
| `data/manifests/eval_cases/yikai_v2.json` | `3193b64f18d4` | `0edca8a` | `6e332c53c281` |
| `data/manifests/eval_cases/yikai_v1.json` (the complete folder, now `test_full`; no longer matches `test`) | `d0a44905ad63` | `af01c51` | `0e5a1581ea29` |

### folding_8mm

- Set: `folding_8mm/test`, 236 clips of 49 garments, 902 frames each at 30 Hz, five fold actions (45 to 49
  clips each), cloth thinned to 8 mm (at most 10,000 vertices) plus 2,000 gripper points. Stratum: action.
- Headline: cloth error, `deform_mean_mm` and `deform_final_mm` (equal to the predicted rows), on
  `windows_motion` tag `q4` and on `rollouts_active`; Chamfer at frames 16, 64 and 128 of the rollouts
  (`--chamfer_marks 16,64,128`). Report `still` rows next to `moving` rows: most cloth does not move in a
  window (45 of the 236 validation windows have no moving vertex at all).
- The loader's window is a weak signal here: predicting no motion already scores 0.54 mm on `windows`.
  Validation curves on folding move in fractions of a millimetre; judge folding by the motion-selected sets.

### yikai

- Set: `yikai/test`, 200 episodes of 100 scenes (174 `gen`, 26 `fail`): since 2026-09-19 a seeded subset, proportional over population and
  difficulty level, of the official `wm_v2_eval`, which stays complete in `yikai/test_full` (548 episodes of 104 scenes);
  247 to 935 frames at 20 Hz, whole scenes without the table, clips from frame 10. Stratum: population.
- Headline: `pose_passive_mm` / `pose_passive_deg` and the predicted-row particle error on `windows_motion`
  tag `q4` and on `rollouts_active`; `still` rows separately (distractors and the resting object: phantom
  motion). `fail` episodes are reported separately from `gen`.
- Cases that overlap frames where a point lies beyond 0.8 m in x or y carry the tag `beyond`
  (`data/manifests/yikai_extent_tags_20260918.json`; decision 2026-09-18: keep every episode, report tagged
  cases separately as well as inside the totals).

### rl_push

- Set: `rl_push/test`, 200 episodes of the 117 test objects (new pushes on shapes seen in training; 83 objects give two
  episodes, 34 one): since 2026-09-19 a seeded subset of the complete folder `rl_push/test_full` (5,598 episodes, 15 excluded by
  `rl_push_exclusions_20260918.json`), drawn among the episodes with the 23 frames a window needs and off that list; 20 Hz; 512 object points and
  671 gripper points. Stratum: object category.
- Headline: `pose_passive_mm` / `pose_passive_deg` of the pushed object and its particle error, on `windows`
  and on `rollouts_first` (to the episode's end, 16 to 93 frames).
- Pushes are abrupt: holding the last velocity is no better than predicting no motion (99 against 104 mm
  over a window), so both references are weak and the model number carries the information.

### pushing

- Set: `pushing/test`, 100 episodes of 40 frames, 256 points (block and rod): since 2026-09-19 one stored sample of 4 episodes
  from each of 25 seeded shards of the complete folder `pushing/test_full` (the last 40 of 404 sorted shards, 10,148 episodes),
  among the samples whose episodes all pass the rule that cleaned train (tilt, z range, in-plane jump). Episode ids are the
  same in both folders. The frame interval of this corpus is not verified.
- Headline: `pose_passive_mm` / `pose_passive_deg` of the block and its particle error, on `windows` and on
  `rollouts_first` (36 frames).
- Holding the last velocity is a strong reference here (4.1 mm over a window against 12.0 mm for no motion):
  a pushing number is read against hold-velocity first. In a third of the prefix windows the block has not
  been touched yet (36 of 100 cases have no moving row).

### pickup

- Set: `pickup/test`, 200 episodes of 40 frames, 1,183 points (the shared 671-point Franka gripper and a 512-point object): 50
  per collection (place, convex, pick, upright) on 141 of the 142 held-out assets, so this set measures unseen objects. The
  registered `pickup` domain keeps those assets out of both of its splits, so no earlier model has trained or validated on them.
  The release reads the corpus with the translation (-0.45, 0, -0.11), table top at z = 0 (`data.roots.pickup.shift_m`).
- Headline: `pose_passive_mm` / `pose_passive_deg` of the object and its particle error, on `windows` and on `rollouts_first`
  (every episode rolled out whole, 36 frames), with the collection as stratum: placing and the three pick variants differ.
- In 51 of the 200 prefix windows the object has not moved yet. No motion scores 28.9 mm on the
  predicted rows of a window and 74.0 mm over the whole clip, holding the last velocity 32.4 and 109.4 mm.

### dex_v2_filtered2

- Set: `dex_v2_filtered2/test`, 174 episodes of the five tactile-genesis teachers on 24 held-out objects (3 boxes, 20
  objaverse meshes, screwdriver model 0 with a corrected grasp file): in_hand_repose-sharpa 44, in_hand_repose-xhand1 34, in_palm_rotate-xhand1 49, screwdriver-sharpa 23, screwdriver-xhand1 24;
  2,158 or 2,194 points (the object and the hand links). Stratum: teacher. It is `dex_v2/test` minus the 20 episodes the
  user's two filter decisions of 2026-09-18 excluded (`dex_v2_filtered2/manifests/dex_v2_exclusions2.json`: the four audit
  exclusions of `dex_v2_filtered`, the repose-XHand1 episodes on boxes, and milder single-finger rides).
- Headline: the object, which is the only predicted body (the hand is the actuator): `pose_passive_mm` /
  `pose_passive_deg` and its particle error, per teacher (`--case_stratum`), on `windows` and on the rollout sets.
- The object turns in almost every window (155 of 174 validation windows have moving rows), so
  the motion-selected sets add little here: a rollout from the first frame already carries 31.0 mm of
  zero-motion error against 29.2 mm from the most active span. Rotation dominates: predicting no motion
  is 33 degrees off over a window and 79 degrees over 128 frames.
  170 of the 174 episodes are long enough for motion-selected windows.
- Dex points are computed from link poses, so a window differs from the slice of its episode in the last float32
  bit; the frozen numbers are those of the adapter's own window of each case, the clip the reader yields.
- Every case of `dex_v2_filtered2_v1.json` is a case of `dex_v2_v1.json` with the same start, horizon and numbers, to the last digit; the 140 cases of the 20 excluded test episodes are gone and the stratified order shifts. The evaluator's case reader reproduces all 1,198 cases. Scales artifact v6 serves this folder; the cases do not depend on the scales.

### dex_v2_filtered and dex_v2 (superseded by dex_v2_filtered2)

- `dex_v2/test` (194 episodes, the unfiltered regeneration) and `dex_v2_filtered/test` (190 episodes, the first filter, the dex
  folder for a few hours of 2026-09-18). `dex_v2_v1.json` and `dex_v2_filtered_v1.json` stay frozen and valid for a checkpoint
  that validates on one of those folders; nothing new should be trained or reported on them. Their whole-set rows remain in the
  table above for comparison.

### mpm_fresh

- Sets: the four passive categories of the MPM evaluation suite, 100 episodes each, 49 states at 1/24 s, 1,024 points per
  object, every particle predicted (no actuator). `mpm_fresh/test` is the unseen-shape category with 1 to 4 objects: it is the
  validation folder of training and the headline test (user decision 2026-09-18). The other three are report folders that no
  recipe reads; they are scored with `--test_folder mpm_fresh=<release>/mpm_fresh/<folder>` and their own manifest.
- Case sets: `windows` (the loader's evaluation window, what validation reads) and `rollouts_first` (the whole clip from frame
  0, 45 scored frames). Clips are too short for motion-selected sets. Stratum: the suite's `evaluation_category`, one per folder.

| folder | what it is | manifest | windows with moving rows | zero motion, window (mm) | zero motion, whole clip (mm; mean / final) |
|---|---|---|---|---|---|
| `test` | unseen shapes, 1 to 4 objects: validation and headline | `mpm_fresh_v1.json` | 77 of 100 | 36.4 | 209.1 / 230.8 |
| `test_seen` | training shapes, 1 to 4 objects | `mpm_fresh_test_seen_v1.json` | 80 of 100 | 40.1 | 210.1 / 230.2 |
| `test_count_unseen` | unseen shapes, 5 to 7 objects | `mpm_fresh_test_count_unseen_v1.json` | 95 of 100 | 27.7 | 196.4 / 215.2 |
| `test_count_seen` | training shapes, 5 to 7 objects | `mpm_fresh_test_count_seen_v1.json` | 99 of 100 | 37.3 | 201.5 / 220.2 |

- Headline: `rollouts_first` on `test`, particle error on the predicted rows, read against the zero-motion column: a drop moves
  about 0.2 m over a clip, so holding the last velocity is useless here (708 mm). About a fifth of the
  evaluation windows of the 1 to 4 object folders start after the scene has settled and have no moving row; the whole-clip
  rollouts always move.
- Seen against unseen shapes and 1 to 4 against 5 to 7 objects are comparisons between folders of 100 episodes each: report
  them with their spread, as the suite's design note asks, not as a full factorial.

### mpm_drag

- Sets: the two drag categories of the same suite, 100 episodes each, 1 to 4 objects, 49 / 81 / 113 states. `mpm_drag/test`
  is `drag_unseen` (validation and headline); `test_seen` is the report folder. The recorded drag commands are given to the
  model as actions; every particle is predicted, the driven patch included.
- Case sets: `windows` and `rollouts_first` (the whole clip: 45, 77 or 109 scored frames). Stratum: `evaluation_category`.

| folder | what it is | manifest | windows with moving rows | zero motion, window (mm) | zero motion, whole clip (mm; mean / final) |
|---|---|---|---|---|---|
| `test` | unseen shapes: validation and headline | `mpm_drag_v1.json` | 82 of 100 | 9.1 | 27.7 / 34.6 |
| `test_seen` | training shapes | `mpm_drag_test_seen_v1.json` | 75 of 100 | 10.5 | 32.5 / 40.9 |

- Headline: `rollouts_first` on `test`, on the predicted rows and on the moving rows (most of a drag scene stands still:
  27.7 mm on all rows against 42.3 mm on the rows that move). Holding the last velocity equals zero motion here.
- Not built: a separate score for the driven patch, the rest of the dragged object and the other objects.

## Report folders

`pwm-data-v1` gives `mpm_fresh` and `mpm_drag` test folders that no recipe reads (`test_seen`, `test_count_unseen`,
`test_count_seen`). The evaluator scores one with `--test_folder DOMAIN=FOLDER`: the folder replaces the checkpoint's validation
folder for that command, must hold a test index of the same dataset that is disjoint from the train folder, and is named in the
receipt (`test_folders`, `inputs.<domain>.roots`). The case reader still refuses a manifest frozen on another index, so a report
folder is always scored with its own manifest:

```bash
PYTHONPATH=. python scripts/unified/eval_unified.py --checkpoint <run>/checkpoints/final.pt --domains mpm_fresh --clips all \
    --test_folder mpm_fresh=<release>/mpm_fresh/test_seen --cases data/manifests/eval_cases/mpm_fresh_test_seen_v1.json \
    --case_set rollouts_first --rollout --rollout_frames 128 --out <receipt>.json
```

## Freezing and versions

```bash
# one dataset (CPU, minutes); recipes: /data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1/configs/
PYTHONPATH=. python scripts/unified/freeze_eval_cases.py --recipe <configs>/unet_yikai_100m.yaml --domain yikai \
    --rollouts 0 --rollout_frames 128 --extent_tags data/manifests/yikai_extent_tags_20260918.json --full_rollouts \
    --out data/manifests/eval_cases/yikai_v3.json
PYTHONPATH=. python scripts/unified/freeze_eval_cases.py --recipe <configs>/unet_folding_8mm_100m.yaml --domain folding \
    --name folding_8mm --rollouts 50 --rollout_frames 128 --full_rollouts --out data/manifests/eval_cases/folding_8mm_v3.json
PYTHONPATH=. python scripts/unified/freeze_eval_cases.py --recipe <configs>/unet_rl_push.yaml --domain rl_push \
    --rollouts 0 --out data/manifests/eval_cases/rl_push_v2.json
PYTHONPATH=. python scripts/unified/freeze_eval_cases.py --recipe <configs>/unet_pushing.yaml --domain pushing \
    --out data/manifests/eval_cases/pushing_v2.json
PYTHONPATH=. python scripts/unified/freeze_eval_cases.py --recipe <configs>/unet_pickup_100m.yaml --domain pickup \
    --out data/manifests/eval_cases/pickup_v1.json
# MPM: the dataset's own test folder, then each report folder through --test_folder (names mpm_fresh_test_seen, ...)
PYTHONPATH=. python scripts/unified/freeze_eval_cases.py --recipe <configs>/unet_mpm_fresh.yaml --domain mpm_fresh \
    --rollouts 0 --rollout_frames 128 --out data/manifests/eval_cases/mpm_fresh_v1.json
PYTHONPATH=. python scripts/unified/freeze_eval_cases.py --recipe <configs>/unet_mpm_fresh.yaml --domain mpm_fresh \
    --name mpm_fresh_test_seen --test_folder <release>/mpm_fresh/test_seen --rollouts 0 --rollout_frames 128 \
    --out data/manifests/eval_cases/mpm_fresh_test_seen_v1.json
# dex: a recipe written with --folder dex=dex_v2_filtered2 (the manifests of dex_v2 and dex_v2_filtered came from the same command)
PYTHONPATH=. python scripts/unified/freeze_eval_cases.py --recipe <configs>/unet_dex_v2_filtered2_100m.yaml --domain dex \
    --name dex_v2_filtered2 --rollouts 256 --rollout_frames 128 --full_rollouts \
    --out data/manifests/eval_cases/dex_v2_filtered2_v2.json
```

A frozen list is never edited. A changed rule or test folder gets a new file (`_v2`), and tables name the
version they used. The `_v1` files are the third freeze of 2026-09-18 (revisions `1defe9d`, `df2e4de`, `af01c51`):
all three pinned the same cases and numbers; the later ones added the pose references and the evaluator
commands, before any checkpoint was scored. `dex_v2_v1.json` was frozen once, at `0754324`, the six MPM manifests at `71a23cf` (a first freeze at `124a33c` pinned the same cases; the second adds `--test_folder` to the report folders' commands; both equal, case by case, the dry runs made on the staged suite before it entered the release), `dex_v2_filtered_v1.json` once, at `ef0eb7d`, and `dex_v2_filtered2_v1.json` once, at `7a64e57`. A manifest records the code revision, the recipe and its sha256, the test index sha256 and
the exclusion lists' sha256; freezing the same folder twice gives the same cases.

**`_v2` of pushing, rl_push and yikai (2026-09-19).** The user asked for small test folders: 100 episodes for pushing, 200 for
rl_push, 200 for yikai. In the release the complete folder of each was renamed `test_full` (a report folder, never trained on)
and `test` became a seeded subset of it (`<release>/build/test_small_0919/`). `pushing_v2.json`, `rl_push_v2.json` and
`yikai_v2.json` were frozen on the new `test` folders at `0edca8a`, with every test episode as a rollout case; they equal, case by
case, a rehearsal freeze made on the staged folders before the install. The `_v1` files of these three datasets describe the
complete folders and no longer match `test`: the reader refuses them there, as it should. Yikai's `_v2` numbers on all rows also
reflect the shared Franka gripper points of `f480125` (671 instead of 667 actuated rows); predicted rows are unaffected.

**Whole-episode rollouts (2026-09-19).** `folding_8mm_v2.json`, `yikai_v3.json` and `dex_v2_filtered2_v2.json` were frozen at `e366c0c`
with `--full_rollouts`, which adds the `rollouts_full` set. Their other four sets pin the clips of the previous version, case by case
(same episode, start, horizon, stratum and tag); of the stored trivial numbers a few moved by 0.0001 mm, float32 rounding on the
machine of this freeze. `CURRENT` in `scripts/unified/eval_protocol.py` names the current list of every dataset, and a new version
changes that table in the same commit. Without `--full_rollouts` the freezer writes what it wrote before.

**Folding rolls out 50 episodes (decided 2026-09-19, frozen 2026-09-20 at `71e93a2`).** A whole folding clip is 898 scored frames,
57 windows of 25 sampler steps on up to 10,000 points, so 236 whole-clip rollouts cost most of a day per checkpoint on one GPU.
`folding_8mm_v3.json` (user decision: 50 cases) was frozen with `--rollouts 50`: its three rollout sets hold the first 50 episodes of
the stratified order, ten clips of each of the five fold actions and the same clips as the first 50 cases of each rollout set of
`folding_8mm_v2.json`, and its two window sets keep all 236 test clips. `folding_8mm_v2.json` stays in the tree for the complete
rollout sets; no checkpoint was scored on it. The folding rollout rows of the reference tables above describe the 236-clip sets of
`_v1`; the 50-clip sets of `_v3` read, on the predicted rows (mean / final):

| case set of `folding_8mm_v3` | cases | horizon (frames) | zero motion (mm) | zero motion, moving rows (mm) | hold velocity (mm) |
|---|---|---|---|---|---|
| `rollouts_first` | 50 | 128 | 1.68 / 2.79 | 4.33 | 2.62 / 5.52 |
| `rollouts_active` | 50 | 128 | 15.27 / 28.52 | 32.31 | 12.23 / 25.65 |
| `rollouts_full` (headline) | 50 | 898 | 19.54 / 33.29 | 22.11 | 34.20 / 66.20 |

## Open items

- **Checkpoint monitor.** Nothing scores the numbered checkpoints of a run automatically yet; every receipt
  is one `eval_unified.py` command.
- **Per-group model scores.** A receipt holds one set of numbers; the per-case sidecar holds every case's curves with its stratum
  and tag, so a group's score is an average over that file. `--case_tag q4` or `--case_stratum gen` still score one group per command.
- **Other methods on the same cases.** The saved-prediction layout is fixed (see "The one command"). Still to build: the scorer
  mode that reads such files in place of a model, and an export of the cases (context, actions, ground truth as plain arrays) for
  methods outside this code base.
- **Cost of whole-episode rollouts.** A folding clip is 898 scored frames, 57 windows of 25 sampler steps, which is why folding
  rolls out 50 clips. Yikai (200 clips of up to 921 frames) and dex (174 clips of up to 396 frames) still take hours per checkpoint
  on one GPU; splitting one case set across GPUs is not built.
- **MPM validation category.** Resolved 2026-09-18 by the release layout: `mpm_fresh/test` and `mpm_drag/test` are the
  unseen 1 to 4 object categories, the other categories are report folders. No loader filter exists or is needed.
- **Drag row groups.** A separate score for the driven patch, the rest of the dragged object and the other objects.
