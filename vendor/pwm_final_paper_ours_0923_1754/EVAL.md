# Evaluating on pwm-data-v1

The exact evaluation protocol of every dataset of the release `pwm-data-v1`. How the checkpoints are trained is in [TRAIN.md](TRAIN.md).
The reasoning behind the case sets, their selection rules and the freezer are in [docs/EVAL_PROTOCOLS.md](docs/EVAL_PROTOCOLS.md), and the
evaluator's vocabulary in [docs/EVALUATION.md](docs/EVALUATION.md).

## The one command

```bash
PYTHONPATH=. python scripts/unified/eval_protocol.py --checkpoint <run>/checkpoints/final.pt                         # the checkpoint's own schedule
PYTHONPATH=. python scripts/unified/eval_protocol.py --checkpoint <run>/checkpoints/final.pt --schedule full_sequence
```

It scores a specialist or a unified model on every dataset the checkpoint was trained on, which it reads from the checkpoint. Per dataset it
takes the current frozen case list (table below) and scores every case set of it. Use it for every reported number, so that everyone gets the
same receipts from the same checkpoint.

## What is fixed

- **Checkpoint.** `final.pt`, EMA weights. The test folder is also the validation folder of training, so `best.pt` has seen it; a table says so.
- **Seed 0**, one seed. The sampler is stochastic.
- **Reading.** Through the checkpoint's own saved data paths, scales and window: 4 context frames, 16 predicted frames per window, 25 sampler
  steps. Given actions stay given. Nothing is clamped.
- **Rollouts.** After the first window the context is the model's own output; ground truth enters only as the actuator's actions.
- **Schedule.** `pyramid` or `full_sequence`, the one setting of the command, default the checkpoint's own. Each has its own output folder.
- **Headline.** The rollout that reaches the end of the clip: `rollouts_full` for folding, yikai and dex, `rollouts_first` for the others.
- **Cases.** Frozen lists in `data/manifests/eval_cases/`, never edited; the tests pin their sha256. The evaluator refuses a list frozen on
  another test folder, and the summary fails when the no-motion error it computes differs from the list's, so other clips cannot slip in.

## Metrics

All on the **predicted rows**: a clip's valid particles minus rigid actuators and fixtures, whose motion is an input. Averaged over a clip's rows
and frames first, then over clips.

| metric | keys in a receipt | definition |
|---|---|---|
| Euclidean error, mm | `particle_mean_mm`, `particle_final_mm`, `particle_per_frame_mm` | distance between predicted and true position; mean over the scored frames, at the last frame, and the curve per frame (with `particle_per_frame_n`, the clips that reach the frame) |
| MSE, mm^2 | `particle_mse_mm2`, `particle_mse_per_frame_mm2` | the squared distance, averaged the same way |
| Chamfer distance, mm | `chamfer_mean_mm`, `chamfer_final_mm`, `chamfer_per_frame_mm` | sum of the two directional mean nearest-neighbour distances between the predicted and the true point set, at every frame. Needs no point correspondence |
| Moving and still rows, mm | `rows.moving`, `rows.still` | the Euclidean keys on the predicted rows that leave their last context position by 5 mm or more, and on the others (error there is motion the model invented) |
| Rigid pose error | `pose_passive_mm`, `pose_passive_deg`, per frame too | each passive body's pose fitted from its predicted points, translation in mm and rotation in degrees |
| Trivial references | `reference.zero`, `reference.hold` | every key above for predicting no motion and for holding the last context velocity, scored by the same scorer on the same clips. A model number is read next to them |

## What is saved

Default `<run>/eval_protocol/<checkpoint name>_<schedule>/`:

- `<dataset>/<case set>.json`: the receipt. Checkpoint and case-list sha256, code revision, seed, schedule, window, the metrics above with their
  per-frame curves, the trivial references;
- `<dataset>/<case set>.cases.json`: one record per case (id, start, horizon, stratum, tag) with its per-frame curves for the model and the two
  trivial predictions, so any horizon, stratum or tag is read without scoring again;
- `<dataset>/<case set>.predictions/<position in the set>.npz` for the headline set (`--predictions all | none` changes that): `pred`
  [frames, rows, 3] in metres, float32, on the clip's valid rows; `rows`, their indices in the clip's particle order; `predicted`, which of them the
  model predicts; `id`, `start`, `horizon`. For later visualization, and the layout in which another method's predictions will be scored;
- `<dataset>/<case set>.log`, the evaluator's output; `SUMMARY.json` and `SUMMARY.md`, every set next to its trivial references.

## Datasets

Errors in mm on the predicted rows, episode-weighted means over the set; `pose` is the passive bodies' no-motion pose error. **Bold** marks the headline set.

### mpm_fresh

Test folder `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_fresh/test`, 100 episodes. Case list `data/manifests/eval_cases/mpm_fresh_v1.json` (sha256 `ac5becd2110b`), clips start at frame 0 or later.

| case set | cases | scored frames | no motion, mean / final | held velocity, mean / final | no motion, moving rows | no motion, pose |
|---|---|---|---|---|---|---|
| `windows` | 100 | 16 | 36.43 / 48.07 | 104.78 / 205.15 | 52.14 |  |
| **`rollouts_first`** | 100 | 45 | 209.08 / 230.76 | 708.14 / 1503.07 | 211.17 |  |

Report folders, scored with `--report_folders`, never used for validation: `test_seen` (`mpm_fresh_test_seen_v1.json`, 100 episodes), `test_count_seen` (`mpm_fresh_test_count_seen_v1.json`, 100 episodes), `test_count_unseen` (`mpm_fresh_test_count_unseen_v1.json`, 100 episodes).

### mpm_drag

Test folder `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_drag/test`, 100 episodes. Case list `data/manifests/eval_cases/mpm_drag_v1.json` (sha256 `772542710356`), clips start at frame 0 or later.

| case set | cases | scored frames | no motion, mean / final | held velocity, mean / final | no motion, moving rows | no motion, pose |
|---|---|---|---|---|---|---|
| `windows` | 100 | 16 | 9.06 / 15.21 | 8.37 / 18.53 | 19.10 |  |
| **`rollouts_first`** | 100 | 45 to 109 | 27.73 / 34.61 | 27.59 / 34.43 | 42.27 |  |

Report folders, scored with `--report_folders`, never used for validation: `test_seen` (`mpm_drag_test_seen_v1.json`, 100 episodes).

### folding_8mm

Test folder `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/folding_8mm/test`, 236 episodes. Case list `data/manifests/eval_cases/folding_8mm_v3.json` (sha256 `827cb6b455c1`), clips start at frame 0 or later.

| case set | cases | scored frames | no motion, mean / final | held velocity, mean / final | no motion, moving rows | no motion, pose |
|---|---|---|---|---|---|---|
| `windows` | 236 | 16 | 0.54 / 1.01 | 0.37 / 0.75 | 4.36 |  |
| `windows_motion` | 944 | 16 | 0.53 / 0.97 | 0.39 / 0.78 | 4.40 |  |
| `rollouts_first` | 50 | 128 | 1.68 / 2.79 | 2.62 / 5.52 | 4.33 |  |
| `rollouts_active` | 50 | 128 | 15.27 / 28.52 | 12.23 / 25.65 | 32.31 |  |
| **`rollouts_full`** | 50 | 898 | 19.54 / 33.29 | 34.20 / 66.20 | 22.11 |  |

The three rollout sets hold 50 of the 236 test clips, 10 of each of the 5 fold actions: a whole clip is 898 scored frames, 57 sampler windows. The two window sets hold every test clip.

### pushing

Test folder `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/pushing/test`, 100 episodes. Case list `data/manifests/eval_cases/pushing_v2.json` (sha256 `0b39060482b1`), clips start at frame 0 or later.

| case set | cases | scored frames | no motion, mean / final | held velocity, mean / final | no motion, moving rows | no motion, pose |
|---|---|---|---|---|---|---|
| `windows` | 100 | 16 | 11.96 / 22.43 | 4.06 / 10.10 | 18.80 | 11.52 mm / 7.67 deg |
| **`rollouts_first`** | 100 | 36 | 25.91 / 49.79 | 15.61 / 41.12 | 30.92 | 25.08 mm / 16.19 deg |

### rl_push

Test folder `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/rl_push/test`, 200 episodes. Case list `data/manifests/eval_cases/rl_push_v2.json` (sha256 `472315b83aee`), clips start at frame 3 or later.

| case set | cases | scored frames | no motion, mean / final | held velocity, mean / final | no motion, moving rows | no motion, pose |
|---|---|---|---|---|---|---|
| `windows` | 200 | 16 | 104.48 / 173.68 | 99.31 / 184.33 | 106.61 | 102.00 mm / 21.09 deg |
| **`rollouts_first`** | 200 | 16 to 93 | 127.97 / 238.12 | 131.33 / 245.27 | 127.97 | 124.73 mm / 28.67 deg |

### yikai

Test folder `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/yikai/test`, 200 episodes. Case list `data/manifests/eval_cases/yikai_v3.json` (sha256 `c0d0223d0419`), clips start at frame 10 or later.

| case set | cases | scored frames | no motion, mean / final | held velocity, mean / final | no motion, moving rows | no motion, pose |
|---|---|---|---|---|---|---|
| `windows` | 200 | 16 | 4.48 / 7.62 | 4.62 / 9.84 | 33.55 | 4.08 mm / 1.66 deg |
| `windows_motion` | 800 | 16 | 4.71 / 8.05 | 4.80 / 10.16 | 35.47 | 4.25 mm / 1.40 deg |
| `rollouts_first` | 200 | 128 | 6.59 / 28.14 | 16.58 / 47.53 | 31.93 | 5.68 mm / 3.24 deg |
| `rollouts_active` | 200 | 128 | 55.04 / 94.26 | 69.90 / 133.41 | 223.34 | 51.02 mm / 13.41 deg |
| **`rollouts_full`** | 200 | 233 to 921 | 61.51 / 98.89 | 102.22 / 181.12 | 212.25 | 56.24 mm / 16.02 deg |

### dex_v2_filtered2

Test folder `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/dex_v2_filtered2/test`, 174 episodes. Case list `data/manifests/eval_cases/dex_v2_filtered2_v2.json` (sha256 `4a62f090da50`), clips start at frame 0 or later.

| case set | cases | scored frames | no motion, mean / final | held velocity, mean / final | no motion, moving rows | no motion, pose |
|---|---|---|---|---|---|---|
| `windows` | 174 | 16 | 13.26 / 19.34 | 28.91 / 52.75 | 14.90 | 8.44 mm / 32.77 deg |
| `windows_motion` | 680 | 16 | 12.04 / 17.99 | 26.38 / 48.81 | 13.67 | 7.30 mm / 30.31 deg |
| `rollouts_first` | 174 | 35 to 128 | 30.97 / 37.51 | 123.19 / 231.36 | 30.97 | 21.85 mm / 78.86 deg |
| `rollouts_active` | 170 | 77 to 128 | 29.18 / 34.22 | 137.31 / 262.62 | 29.18 | 18.63 mm / 81.59 deg |
| **`rollouts_full`** | 174 | 35 to 396 | 34.32 / 43.39 | 331.68 / 647.68 | 34.32 | 24.72 mm / 88.38 deg |

### pickup

Test folder `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/pickup/test`, 200 episodes. Case list `data/manifests/eval_cases/pickup_v1.json` (sha256 `90e6b4e5eb4a`), clips start at frame 0 or later.

| case set | cases | scored frames | no motion, mean / final | held velocity, mean / final | no motion, moving rows | no motion, pose |
|---|---|---|---|---|---|---|
| `windows` | 200 | 16 | 28.89 / 43.01 | 32.36 / 65.65 | 39.03 | 27.80 mm / 12.23 deg |
| **`rollouts_first`** | 200 | 36 | 74.05 / 216.00 | 109.44 / 313.17 | 74.05 | 71.54 mm / 29.28 deg |

## Comparing models

A specialist and a unified model are compared per dataset, on the same case sets, from their `SUMMARY.json`. There is no pooled number over
datasets. State the schedule, that one seed was used, and that the test folder also served as validation. The two pushing runs are compared
on `pushing_v2` and by sampler error, never by validation loss.

## Not built yet

- Scoring another method's saved predictions with this scorer (the file layout above is fixed), and an export of the cases as plain arrays
  for methods outside this code base.
- Splitting one case set across GPUs. The whole-episode sets are long (yikai 200 clips of up to 921 scored frames, dex 174 of up to 396,
  folding 50 of 898): hours per checkpoint on one GPU.
