# Training on pwm-data-v1

The exact training protocol of every job on the release `pwm-data-v1`, as confirmed on 2026-09-19: nine specialists, one dataset each,
and two unified runs on all eight datasets. How a checkpoint is scored is in [EVAL.md](EVAL.md). Per-dataset background (collection,
units, splits, exclusions) is in [docs/DATASETS.md](docs/DATASETS.md).

- Release: `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/<dataset>/{train,test}`. There is no development split: the `test` folder is also the validation
  folder of training, so the curves and `best.pt` read it. Reported numbers come from `final.pt`.
- Scales: `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/stats/normalization_collections_20260919_v7.json`, profile `xy_p99_per_domain`, one entry per dataset.
- Recipes: `/data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1/configs/unet_<job>.yaml`, written by `scripts/unified/collection_recipe.py`.

## Start a job

```bash
bash launch/submit_pushing_100m.sh --print    # checks, prints the sbatch command, submits nothing
bash launch/submit_pushing_100m.sh            # submits
```

One script per job in [launch/](launch/README.md). A script submits `launch/train_pwm_data_v1.sbatch` from the checkout it lives in. It
refuses a checkout with local changes and a recipe whose sha256 differs from the one it records. A run reads the code again whenever it
starts or resumes, so submit from a checkout that stays on one commit until the run ends (`git worktree add --detach <path> <commit>`).
Running a script again resumes its run from `latest.pt`.

Any account of the group `kfragki2_group` can run a script as it is: the release and the recipes are group-readable, and every other
location defaults to the calling account's own, so two people who run the same job never share a run folder. Runs and SLURM logs go to
`LOG_DIR` (default `/data/group_data/katefgroup/runs_pwm_<account>`), temporary files to `SCRATCH` (default
`/data/user_data/<account>/scratchpad/launch_pwm_data_v1`), and the job activates the environment `PWM_CONDA_ENV` (default `pwm_blackwell`,
built as in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)) from the conda installation `PWM_CONDA_SH` (default
`$HOME/miniconda3/etc/profile.d/conda.sh`). Set any of the four in the calling shell; nothing in the repo needs editing.

## The jobs

| script | dataset | what | train episodes | distinct 20-frame windows | test episodes | batch per GPU | estimated hours | recipe sha256 |
|---|---|---|---|---|---|---|---|---|
| `submit_mpm_fresh_100m.sh` | `mpm_fresh` | passive MPM drops, 1 to 4 deformable objects | 47,288 | 1.42 M | 100 | 16 | 58 | `b3bd7695e8cd` |
| `submit_mpm_drag_100m.sh` | `mpm_drag` | MPM objects dragged by a commanded patch | 2,333 | 0.14 M | 100 | 8 | 30 | `ae5c1889b77a` |
| `submit_folding_8mm_100m.sh` | `folding_8mm` | cloth folded by two grippers, thinned to 8 mm spacing | 4,510 clips | 3.98 M | 236 | 4 | 44 | `14140808625d` |
| `submit_pushing_100m.sh` | `pushing` | a rod pushes a block | 92,651 of 93,184 | 1.95 M | 100 | 16 | 34 | `33028dc63e6e` |
| `submit_rl_push_100m.sh` | `rl_push` | a Franka gripper pushes objects | 50,815 usable | 1.13 M | 200 | 16 | 32 | `fa15b3a04a15` |
| `submit_yikai_100m.sh` | `yikai` | whole pick-and-place scenes | 11,015 | 5.25 M | 200 | 8 | 56 | `3c46c4eb3318` |
| `submit_dex_v2_filtered2_100m.sh` | `dex_v2_filtered2` | dexterous hands turning and moving objects | 16,999 | 5.36 M | 174 | 8 | 34 | `6d17c8b0d29a` |
| `submit_pickup_100m.sh` | `pickup` | pick and place, test on unseen objects | 42,253 of 50,092 | 0.89 M | 200 | 16 | about 30 | `63b7434aac90` |

Every specialist is a UNet with 103.10 M parameters on two RTX PRO 6000. `submit_pushing_100m_light_pose_noise.sh` is the pushing
specialist with rigid history noise (recipe `c1fe96d70d75`), and `submit_joint8_500m.sh` / `submit_joint8_100m.sh` are the unified runs
(recipes `dd82310918f6` / `c6473a593491`), both described below.

## What a run saves

Under `LOG_DIR/pwm_v1_<job>_unet_bidir/`, which is `/data/group_data/katefgroup/runs_pwm_robertre/pwm_v1_<job>_unet_bidir/` for the account `robertre`:

- `config.yaml`, the resolved configuration, which every checkpoint also carries together with its normalization snapshot;
- `checkpoints/step_<N>.pt` every 20,000 steps, `latest.pt` at every validation, `best.pt` (lowest sampler error in mm on the validation
  batches, EMA weights) and `final.pt`, the reported checkpoint. 16 bytes per parameter: 1.65 GB at 103 M, 8.04 GB at 502 M, 13 files per run;
- the SLURM log `LOG_DIR/slurm/pwm_v1_<job>_<job id>.log`: code revision, recipe sha256, normalization digest and corpus identity before
  the first step, then one training line per 100 steps and one validation block per validation (loss and sampler error in mm per dataset);
- the WandB run in project `particle_world_model`.

## Settings shared by every specialist

| item | value |
|---|---|
| Clipping | none. Targets, poses, positions and the sampler are all unclipped |
| Objective | flow matching with x0 prediction: the network outputs the clean normalized motion. The loss is a plain squared error, the same at every noise level. Deformable particles are scored per particle. Rigid bodies are scored through their pose with the same plain squared error, which is the pose-loss fix (`train.pose_loss_space=sample`). 1,000 training timesteps, 25 sampling steps |
| Noise levels in training | per sample, 60% of the time every frame draws its own noise level and 40% of the time the frames follow the staircase that our sampler visits at test time (`train.timestep_schedule: {independent: 0.6, ramp: 0.4}`). No loss weight cap |
| Optimizer | AdamW, lr 1e-4, betas 0.9 / 0.999, weight decay 1e-3, gradient clip 1.0, bf16, seed 42 |
| EMA | 0.9999 with warm-up |
| Schedule | 200,000 steps, 5,000 warm-up, cosine to 1e-6 |
| Validation | every 1,000 steps on the test folder: 40 loss batches and 8 sampler batches, except rl_push with 117 and 60, see its table. `best.pt` by sampler error in mm on EMA weights. `final.pt` is the reported checkpoint |
| Checkpoints | 1.65 GB each: every 20K steps plus latest, best and final, about 21 GB per run |
| Step rejection | the loss spike rule is OFF in these recipes (`train.loss_spike_threshold: .inf`, decision of 2026-09-19): under the plain x0 loss it skipped motion-rich batches. A step is still skipped when its loss or gradient is not finite, or when its gradient norm exceeds 50 after the first 500 steps. Every applied update is clipped to gradient norm 1.0. A skipped step does not advance the learning-rate schedule |
| Other constants in the code | the particle loss covers deformable rows only. Rigid bodies enter through the pose term, averaged over all rigid bodies including the given ones. The sampler error behind `best.pt` averages over all valid particles, given ones included |
| GPUs | two RTX PRO 6000 in one job, NCCL timeout 60 min, job limit 71.5 h. Resubmitting the same command resumes from `latest.pt` |

## mpm_fresh

Job `mpm_fresh_100m`, recipe `configs/unet_mpm_fresh_100m.yaml` (`b3bd7695e8cd`).

| item | value |
|---|---|
| Data paths, vintage, size | train `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_fresh/train`, test `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_fresh/test`. Campaign `fresh-10432785-v4`. Test = category `passive_unseen` of the MPM evaluation suite `test-600-20260917-v3`, adopted 2026-09-18. 150.3 GB of hard links plus 0.3 GB of test shards |
| Train data | `pwm-data-v1/mpm_fresh/train`: 47,288 passive-drop episodes, 26.3 h, 1 to 4 objects, 1,024 points per object, 49 frames at 24 Hz. 1.42 M distinct 20-frame windows |
| Validation = test | `mpm_fresh/test`: 100 episodes on unseen shapes, 25 each with 1 to 4 objects. No shape is shared with train |
| Window | 4 context + 16 future frames. Training starts are weighted by motion with a 2 mm floor. A batch mixes scenes of equal point count. Evaluation windows are fixed |
| Stats, how | per-frame particle displacement, all valid particles, all 19 frame pairs of a window. 1,024 train episodes x 4 uniform windows, 191 M rows into a uniform 262,144-row reservoir. 99th percentile of the absolute value, xy = larger of x and y. Train folder only, seed 916 |
| Stats, values | xy 22.37 mm per frame, z 76.66 mm per frame. Target = displacement / scale. Positions / 0.8 m. Largest normalized target in a 16,000-window scan: 7.1 |
| Model | UNet, 103,096,585 parameters. Widths 64/128/256/512, encoder depths 1,2,4,9, decoder 1,2,4, head dim 64, bidirectional attention, compiled |
| Context noise | 0.05 of the scale on every context particle, training only: 1.1 mm xy and 3.8 mm z per frame |
| Batch | 16 per GPU x 2 GPUs = 32 windows per step, 6.4 M windows in 200K steps, 4.5 passes over the distinct windows |
| Wall clock | measured on 2026-09-19: 1.07 steps/s at this batch on two RTX PRO 6000 Blackwell Server Edition, so 200K steps take 52 h of training. A validation with its checkpoint write took 115 s in that run (the first one 129 s), which adds 6.4 h over 200 validations, so about 58 h in total against the 71.5 h job limit |

## mpm_drag

Job `mpm_drag_100m`, recipe `configs/unet_mpm_drag_100m.yaml` (`ae5c1889b77a`).

| item | value |
|---|---|
| Data paths, vintage, size | train `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_drag/train`, test `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/mpm_drag/test`. Gentle-v4 dragging, 31 published blocks. Test = category `drag_unseen` of the same evaluation suite. 16.7 GB of hard links plus 0.8 GB of test shards |
| Train data | `pwm-data-v1/mpm_drag/train`: 2,333 gentle-v4 drag episodes, 2.1 h, 1 to 4 objects, 1,024 points per object, 4,724 of the 10,000 training shapes. Episodes of 49, 81 or 113 frames at 24 Hz. 139,622 distinct 20-frame windows |
| Validation = test | `mpm_drag/test`: 100 episodes on 191 unseen shapes, 25 each with 1 to 4 objects. No shape, episode or file is shared with train |
| Window | 4 context + 16 future frames. Training starts are weighted by motion with a 0.25 mm floor. A batch mixes scenes of equal point count. Evaluation windows are fixed |
| Stats, how | per-frame particle displacement, all valid particles with the driven patch included, all 19 frame pairs of a window. All 2,333 train episodes x 4 uniform windows, 432 M rows into the 262,144-row reservoir. 99th percentile of the absolute value, xy = larger of x and y. Train only, seed 916. Four seeds agree within 2% |
| Stats, values | xy 7.31 mm per frame, z 6.15 mm per frame. The driven patch raises the scale by 2.5% in xy and 7% in z. Largest normalized target in the scan: 11.7 in z |
| Model | UNet, 103,096,841 parameters, 256 more than mpm_fresh for the four action channels. Same widths and depths, bidirectional attention, compiled |
| Context noise | 0.05 of the scale on every context particle including the driven patch, training only: 0.37 mm xy and 0.31 mm z per frame |
| Batch | 8 per GPU x 2 GPUs = 16 windows per step, 3.2 M windows in 200K steps, 22.9 passes over the distinct windows. Held at 16 windows per step because the dataset is small: at 32 per step it would make 46 passes |
| Wall clock | measured on 2026-09-19: 2.11 steps/s at this batch on two RTX PRO 6000 Blackwell Server Edition, so 200K steps take 26 h of training. A validation with its checkpoint write took 59 s in that run (the first one 75 s), which adds 3.3 h over 200 validations, so about 30 h in total against the 71.5 h job limit |

## folding_8mm

Job `folding_8mm_100m`, recipe `configs/unet_folding_8mm_100m.yaml` (`14140808625d`).

| item | value |
|---|---|
| Data paths, vintage, size | train `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/folding_8mm/train`, test `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/folding_8mm/test`. SIM1 folding, collector indices `folding-30hz-noon-20260916-v4`, thinned on 2026-09-18. Hard links of the 611.4 GB `folding/` folder plus 1 MB of subset files |
| Train data | `pwm-data-v1/folding_8mm/train`: 4,510 clips of 939 garments, 37.6 h, SIM1 garment folding at 30 Hz, 902 frames per clip. Cloth thinned to 8 mm spacing with at most 10,000 cloth points, plus 2,000 gripper points, so 3,627 to 12,000 points per scene. 3.98 M distinct 20-frame windows |
| Validation = test | `folding_8mm/test`: 236 clips of 49 held-out garments, 2.0 h. The split is by garment |
| Window | 4 context + 16 future frames. Training starts are weighted by motion with a 0.02 mm floor. One GPU's batch is several windows of ONE clip, so a step sees two clips at any batch size. Evaluation windows are fixed |
| Stats, how | same procedure as MPM: 512 of 4,510 clips x 4 uniform windows, 322 M particle rows into the 262,144-row reservoir, 99th percentile, train only, seed 916. Every valid particle counts, the gripper points included |
| Stats, values | particle xy 1.80 mm, z 1.72 mm per frame. Gripper pose translation 2.74 / 2.81 mm, rotation 0.0135 / 0.0003 rad per frame. The grippers set the particle scale, so cloth targets reach 0.53 xy and 0.33 z at their 99th percentile. The 2026-09-19 test of a cloth-only scale changed the moving-cloth error by 1.6%, so this scale stays. Largest normalized cloth target in the scan: 17.5, and 5.5% of windows exceed 10 |
| Model | UNet, 103,096,585 parameters, same widths and depths, bidirectional attention. Eager, because folding has hundreds of point counts |
| Context noise | 0.05 of the scale on cloth context rows only, 0.09 mm per frame per axis. Gripper rows stay clean |
| Batch | 4 per GPU x 2 GPUs = 8 windows per step, 1.6 M windows in 200K steps, 0.4 passes over the distinct windows. One GPU's batch is 4 windows of ONE clip, so a step sees two clips. A larger batch adds windows of the same two clips, not more clips |
| Wall clock | measured on 2026-09-19: 1.53 steps/s at this batch on two RTX PRO 6000 Blackwell Server Edition, so 200K steps take 36 h of training. A validation with its checkpoint write took 139 s in that run (the first one 147 s), which adds 7.7 h over 200 validations, so about 44 h in total against the 71.5 h job limit |

## pushing

Job `pushing_100m`, recipe `configs/unet_pushing_100m.yaml` (`33028dc63e6e`).

| item | value |
|---|---|
| Data paths, vintage, size | train `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/pushing/train`, test `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/pushing/test`. The historical `pushing_data_actions` corpus, copied on 2026-09-17. 27.3 GB |
| Train data | `pwm-data-v1/pushing/train`: the legacy rod-pushing corpus, 93,184 episodes of 40 frames in 364 shards, 256 points per scene. Read without the 533 episodes in which the simulation failed, so 92,651 episodes and 1.95 M distinct 20-frame windows |
| Excluded episodes | 533 train episodes (0.57%): block tilt above 5 degrees (442), centroid more than 5 mm out of plane (57), or an in-plane jump above 10 mm in one frame (122). Decision of 2026-09-19. The list ships with the release and the recipe builder writes it into the recipe. The loader was verified to yield 92,651 episodes |
| Validation = test | `pushing/test`: 100 episodes since 2026-09-19. 25 of the 40 test shards, one stored sample of 4 episodes from each, drawn with a fixed seed among the samples whose 4 episodes all pass the train cleaning rule, so none of the 23 flagged test episodes is in it. The complete 10,148 episodes stay in the release as `pushing/test_full` and are not trained on. Validation reads all 100 windows at every validation |
| Window | 4 context + 16 future frames, uniform starts, batches mix episodes. Evaluation windows are fixed |
| Bodies and actions | two rigid bodies of 128 points each. The rod is given as a pose action on every future frame. The block is predicted through its pose. No deformable particles |
| Stats, how | the frozen four-domain entry of 2026-08-28 (99th percentile of per-frame motion), changed on 2026-09-19 to ONE scale for all axes by decision. No new fit |
| Stats, values | translation 3.88 mm per frame on every axis, rotation 0.0543 rad per frame on every axis. Before, the vertical and tilt scales were 0.17 mm and 0.0026 rad, about 20 times finer, because pushing is planar. Largest normalized target fell from 111 to 5.7 |
| Model | UNet, 103,096,585 parameters, same widths and depths, bidirectional attention, compiled |
| Context noise | none in effect. The 0.05 setting jitters deformable particles only, and this dataset has none. The rigid-body analogue (`train.light_pose_noise_std`) exists but has no supporting result (it lost on pickup at 200K), so it stays off |
| Batch | 16 per GPU x 2 GPUs = 32 windows per step, 6.4 M windows in 200K steps, 3.3 passes over the distinct windows |
| Wall clock | measured on 2026-09-19: 1.87 steps/s at this batch on two RTX PRO 6000 Blackwell Server Edition, so 200K steps take 30 h of training. A validation with its checkpoint write took 72 s in that run (the first one 75 s), which adds 4.0 h over 200 validations, so about 34 h in total against the 71.5 h job limit |

## rl_push

Job `rl_push_100m`, recipe `configs/unet_rl_push_100m.yaml` (`fa15b3a04a15`).

| item | value |
|---|---|
| Data paths, vintage, size | train `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/rl_push/train`, test `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/rl_push/test`. `rl_push_data_09_14`, re-frozen on 2026-09-18 as object folders. 0.3 GB |
| Train data | `pwm-data-v1/rl_push/train`: 1,085 objects, 51,555 successful pushes by an RL policy, 50,815 of them usable after the pinned exclusions and the minimum length, 31.6 h, 13 to 100 frames at 20 Hz. 1,183 points per scene. 1.13 M distinct 20-frame windows, starting at frame 3 or later |
| Excluded episodes | 204 train and 15 test episodes on the pinned list of 2026-09-18: sink or off-table motion, unexplained kicks, contact launches, gripper penetration |
| Validation = test | `rl_push/test`: 200 episodes since 2026-09-19, from all 117 test objects: 83 objects give two episodes and 34 one, drawn with a fixed seed among the usable episodes off the exclusion list. The complete 5,598 episodes stay in the release as `rl_push/test_full` and are not trained on. These are new pushes on shapes whose categories are in train. 116 of the 117 test objects have a near-twin in train, so this is an in-distribution test |
| Window | 4 context + 16 future frames, every window of a batch comes from a random episode, uniform starts. Evaluation windows are fixed |
| Bodies and actions | four rigid bodies: the object with 512 points is predicted through its pose. The Panda hand and its two fingers, 671 points, are given as pose actions. The fingers are prismatic children of the hand. No deformable particles |
| Stats, how | 99th percentile of per-frame motion on the train folder with the exclusion list, fit of 2026-09-18 by `scripts/dev/rl_push_stats.py`. Within 1.5% of the 2026-09-15 entry |
| Stats, values | particle xy 46.0 mm, z 20.3 mm per frame. Pose translation 44.8 / 20.4 mm, rotation 0.169 / 0.220 rad per frame. Largest normalized target in the scan: 4.6 |
| Model | UNet, 103,096,585 parameters, same widths and depths, bidirectional attention, compiled |
| Context noise | none in effect. The 0.05 setting jitters deformable particles only, and this dataset has none. The rigid-body analogue (`train.light_pose_noise_std`) exists but has no supporting result (it lost on pickup at 200K), so it stays off |
| Batch | 16 per GPU x 2 GPUs = 32 windows per step, 6.4 M windows in 200K steps, 5.6 passes over the distinct windows |
| Wall clock | measured on 2026-09-19: 2.13 steps/s at this batch on two RTX PRO 6000 Blackwell Server Edition, so 200K steps take 26 h of training. A validation with its checkpoint write took 49 s in that run (the first one 64 s), which adds 2.7 h over 200 validations, so about 29 h in total against the 71.5 h job limit |

## yikai

Job `yikai_100m`, recipe `configs/unet_yikai_100m.yaml` (`3c46c4eb3318`).

| item | value |
|---|---|
| Data paths, vintage, size | train `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/yikai/train`, test `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/yikai/test`. Yikai `wm_v2` and the official `wm_v2_eval`, table-free geometry copies built on 2026-09-18. 273.3 GB |
| Train data | `pwm-data-v1/yikai/train`: 11,015 pick-and-place episodes of 2,134 scenes, 77.2 h, 240 to 1,287 frames at 20 Hz. The whole scene is modelled: the grasped object and every distractor. 2,203 to 4,763 points per scene, the table dropped. 5.25 M distinct 20-frame windows |
| Validation = test | `yikai/test`: 200 episodes since 2026-09-19: 100 of the 104 scenes, 174 successful and 26 failed attempts, drawn with a fixed seed in proportion over population and difficulty level from the owner's official evaluation release. That complete release, 548 episodes, stays as `yikai/test_full` and is not trained on. Asset- and scene-disjoint from train. Validation still reads 40 and 8 episodes of 8 windows |
| Window | 4 context + 16 future frames, uniform starts from frame 10 because distractors settle in the first ten frames. One GPU's batch is several windows of ONE episode, so a step sees two episodes at any batch size. Evaluation windows are fixed |
| Bodies and actions | every object is a rigid body of 512 points and is predicted through its pose. The gripper, 667 points in a hand and two prismatic fingers, is given as pose actions. No deformable particles |
| Stats, how | same procedure as MPM: 1,024 train episodes x 4 uniform windows, 248 M particle rows and 519 K pose rows into the reservoirs, 99th percentile, train only, seed 916, table excluded |
| Stats, values | particle xy 12.3 mm, z 12.8 mm per frame. Pose translation 13.3 / 13.9 mm, rotation 0.071 / 0.099 rad per frame. The predicted objects alone reach 9.0 / 8.2 mm, within a factor of two of the scale. 85% of object rows are still. Largest normalized target in the scan: 13.1, and 0.25% of windows exceed 10 |
| Model | UNet, 103,096,585 parameters, same widths and depths, bidirectional attention. Eager, because scenes have many point counts |
| Context noise | none in effect. The 0.05 setting jitters deformable particles only, and this dataset has none. The rigid-body analogue (`train.light_pose_noise_std`) exists but has no supporting result (it lost on pickup at 200K), so it stays off |
| Batch | 8 per GPU x 2 GPUs = 16 windows per step, 3.2 M windows in 200K steps, 0.6 passes over the distinct windows. One GPU's batch is 8 windows of ONE episode, so a step sees two episodes. A larger batch adds windows of the same two episodes, not more episodes |
| Wall clock | measured on 2026-09-19: 1.09 steps/s at this batch on two RTX PRO 6000 Blackwell Server Edition, so 200K steps take 51 h of training. A validation with its checkpoint write took 93 s in that run (the first one 93 s), which adds 5.2 h over 200 validations, so about 56 h in total against the 71.5 h job limit. At batch 16 the same recipe ran at 0.50 steps/s on two PRO 6000 with 21.8 GiB (job 10500470), which is 111 h for 200K steps, so batch 8 by the agreed rule |

## dex_v2_filtered2

Job `dex_v2_filtered2_100m`, recipe `configs/unet_dex_v2_filtered2_100m.yaml` (`6d17c8b0d29a`).

| item | value |
|---|---|
| Data paths, vintage, size | train `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/dex_v2_filtered2/train`, test `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/dex_v2_filtered2/test`. `tactile_genesis_teachers_v2`, regenerated on 2026-09-18, minus 2,934 episodes. Hard links of `dex_v2` (9.4 GB) |
| Train data | `pwm-data-v1/dex_v2_filtered2/train`: 16,999 episodes of five dexterous-hand teachers on 117 objects, 78.6 h, 20 to 400 frames at 20 Hz. 2,158 or 2,194 points per scene. 5.36 M distinct 20-frame windows |
| Excluded episodes | 2,914 train and 20 test episodes are already out of this folder: simulator artifacts, single-finger rides, and every repose-XHand1 episode on boxes. a decisions of 2026-09-18 |
| Validation = test | `dex_v2_filtered2/test`: 174 episodes on 24 held-out objects: 3 boxes, 20 objaverse meshes and screwdriver model 0 |
| Window | 4 context + 16 future frames, uniform starts. One GPU's batch is several windows of ONE episode, so a step sees two episodes at any batch size. Evaluation windows are fixed |
| Bodies and actions | 14 rigid bodies with the XHand1 and 24 with the Sharpa hand. The object with 512 points is the only predicted body. The 13 or 23 hand links, 1,646 or 1,682 points, are given as pose actions, with revolute joints. The pose loss averages over all bodies, so the object is one term of 14 or 24. No deformable particles |
| Stats, how | same procedure as MPM: 1,024 train episodes x 4 uniform windows, 169 M particle rows and 1.2 M pose rows, 99th percentile, train only, seed 916. Fit on this exact folder on 2026-09-18 |
| Stats, values | particle xy 12.8 mm, z 10.6 mm per frame. Pose translation 13.1 / 11.5 mm, rotation 0.302 / 0.178 rad per frame. The object alone reaches 8.0 / 5.8 mm and 0.37 / 0.28 rad, within a factor of two of the scale. Largest normalized target in the scan: 3.7 |
| Model | UNet, 103,096,585 parameters, same widths and depths, bidirectional attention. Eager |
| Context noise | none in effect. The 0.05 setting jitters deformable particles only, and this dataset has none. The rigid-body analogue (`train.light_pose_noise_std`) exists but has no supporting result (it lost on pickup at 200K), so it stays off |
| Batch | 8 per GPU x 2 GPUs = 16 windows per step, 3.2 M windows in 200K steps, 0.6 passes over the distinct windows. One GPU's batch is 8 windows of ONE episode, so a step sees two episodes. A larger batch adds windows of the same two episodes, not more episodes |
| Wall clock | measured on 2026-09-19: 1.91 steps/s at this batch on two RTX PRO 6000 Blackwell Server Edition, so 200K steps take 29 h of training. A validation with its checkpoint write took 88 s in that run (the first one 88 s), which adds 4.9 h over 200 validations, so about 34 h in total against the 71.5 h job limit. At batch 16 the same recipe ran at 0.97 steps/s on two PRO 6000 with 11.1 GiB (probe inside dev-blackwellb), which is 57 h of training plus about 20 h of validation, above the job limit, so batch 8 by the agreed rule |

## pickup

Job `pickup_100m`, recipe `configs/unet_pickup_100m.yaml` (`63b7434aac90`).

| item | value |
|---|---|
| Data paths, vintage, size | train `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/pickup/train`, test `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/pickup/test`. The earlier pick-and-place corpus `VLA/3DPCWM/robert_pickup_data_{place,convex,pick,upright}_success`, IsaacLab replays of four RL policies with a Franka Panda, collected before 2026-08. 62.2 GB. installed in the release |
| Train data | 196 shards, byte copies of the source shards: every shard of convex, pick and upright (49 each) and 49 of the 63 place shards by seeded permutation, so placing is one collection in four. Read with the exclusion list it yields 42,253 episodes of 40 frames on 800 objects: place 10,458 (24.8%), convex 10,210, pick 10,715, upright 10,870. 0.89 M distinct 20-frame windows. No development split: the old last-10%-of-shards validation split is part of train |
| Excluded episodes | 7,839 of the 50,092 stored episodes: every episode of a held-out object (7,607) and the audited episodes (285: phantom object motion, far placements, teleports, rotation glitches, drops; 53 are on both lists). These are the registered domain's two lists, restated relative to the release folder. Verified through the production loader on 12 shards: the episodes it yields equal the index and none is of a held-out object |
| Validation = test | `pickup/test`: 200 episodes on 141 of the 142 held-out objects, 50 per collection, audited episodes never eligible, spread over the objects by a seeded round-robin. So this test set measures UNSEEN OBJECTS. The registered domain kept these objects out of both of its splits, so no earlier model has trained or validated on them. Rewritten from the source samples row by row: all 2,600 arrays are bit-identical to their source rows. Validation reads all 200 windows for the loss and 128 for the sampler |
| Window | 4 context + 16 future frames, uniform starts, batches mix episodes. Evaluation windows are fixed. The registered domain's default was 2 + 38 frames |
| Bodies and actions | four rigid bodies, 1,183 points: the object (512 points, predicted through its pose) and the shared Franka gripper (hand 241, left finger 212, right finger 218 points). The gripper is given as pose actions on every future frame, and the fingers are prismatic children of the hand. No deformable particles |
| Table height | the collection ran with a table setting of 0.1 m and the registered loader shifts by -0.10 m, but the table top is at 0.110 m: through the registered loader the gripper's lowest point over 2,400 windows is 9.9 mm and objects rest at 10 to 11 mm. The release folder states a shift of (-0.45, 0, -0.11) m and the recipe carries it as `data.roots.pickup.shift_m`. With it the gripper's lowest point on the test windows is 0.0 mm and objects rest with their lowest sampled point at about 2 mm. The registered domain is unchanged |
| Stats, how | the v7 artifact's existing `pickup` entry, the frozen four-domain fit of 2026-08-28 (99th percentile of per-frame motion). No new fit: on 3,200 windows each, the 99th percentiles of the new train folder are within 3% of the registered train split for particles and within 6% for the pose channels, sampling noise included |
| Stats, values | particles 17.2 mm per frame in the plane and 28.8 mm vertical. Poses: translation 15.4 mm in the plane and 28.2 mm vertical, rotation 0.130 rad about the horizontal axes and 0.189 rad about the vertical axis. Largest normalized target on the new train windows: 3.1 for particles, 5.0 for poses |
| Model | UNet, 103,096,585 parameters, same widths and depths as the other specialists, bidirectional attention, compiled (always 1,183 points) |
| Context noise | none in effect. The 0.05 setting jitters deformable particles only, and this dataset has none. Rigid history noise is off here, as in the other plain runs. This is the dataset the 2026-09-16/17 noise comparison ran on. A second pickup run with the noise would be the like-for-like repeat of it |
| Batch | 16 per GPU x 2 GPUs = 32 windows per step, 6.4 M windows in 200K steps, 7.2 passes over the distinct windows. Same as rl_push, which has the same point count |
| Wall clock | from the smoke only: 1.18 steps/s over its logged windows, which include validations. rl_push, with the same point count, batch and model, measured 2.11 steps/s, so expect about 26 h of training plus validation, about 30 h against the 71.5 h limit. No short training was run for pickup |

## pushing with rigid history noise

Job `pushing_100m_light_pose_noise` (`c1fe96d70d75`): the pushing specialist above with one change, a comparison run next to it.
The two runs are compared by sampler error and on the frozen `pushing_v2` cases, never by validation loss, because the noisy run's
validation loss is computed on noisy context.

| config key | plain pushing run | this run | what it does |
|---|---|---|---|
| `train.light_context_noise_std` | 0.05 | 0.05 | Gaussian jitter on the context motion of DEFORMABLE particles. Rigid rows stay exact. Pushing has no deformable particles, so this does nothing in either run |
| `train.light_pose_noise_std` | 0.0 | 0.05 | Gaussian noise on the history of RIGID bodies, applied to the whole body's pose so the body stays rigid. In normalized pose units: 0.05 of the pushing scales, 3.88 mm and 0.0543 rad per frame |
| `train.light_pose_noise_anchor` | not set, off | true | also moves the body's pose at the last context frame, the frame the prediction starts from |
| `train.light_pose_noise_target` | not set | restore | the first predicted transition is rewritten to start at the moved pose and still end on the recorded frame, so the model learns to return to the recorded trajectory. The alternative `carry` keeps the recorded increments from the moved pose |
| `train.light_pose_noise_type` | not set | iid | every history transition draws its own noise. The alternative `random_walk` accumulates the draws, as GNS does |
| which bodies, resolved by the code | none | the block only | with the moved last frame the code restricts the noise to predicted bodies. The rod, whose motion is given, keeps its exact history |

## Unified runs: one model on all eight datasets

`joint8_500m` (502 M parameters) and `joint8_100m` (103 M, the specialists' size), each on four RTX PRO 6000.

| item | 502 M run `joint8_500m` | 103 M run `joint8_100m` |
|---|---|---|
| Recipe | `configs/unet_joint8_500m.yaml`, sha256 `dd82310918f6` | `configs/unet_joint8_100m.yaml`, sha256 `c6473a593491` |
| Model | UNet, 502,218,277 parameters, widths 128/320/640/1280, encoder depths 1,2,4,8, decoder 1,2,4, bidirectional attention | UNet, 103,096,841 parameters, the specialists' widths 64/128/256/512 and depths 1,2,4,9 / 1,2,4, bidirectional attention |
| Same in both | the specialists' protocol: x0 prediction with the pose loss fix, no clipping, context noise 0.05 on deformable particles, EMA 0.9999 with warm-up, noise-level mixture 0.6 / 0.4, loss spike rule off, AdamW lr 1e-4, 5,000 warm-up steps, cosine to 1e-6, seed 42, scales v7 with each dataset's own entry |  |
| Differs from every specialist | nothing is compiled, because point counts differ between datasets. The per-particle action channel is on for all datasets, because mpm_drag needs it, so the network has 4 more input channels than six of the specialists. Batch 4 per GPU instead of 4 to 16 | the same |

| item | value |
|---|---|
| Data paths, vintage, size | the train and test folders of the eight datasets under `/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/pwm-data-v1/`: mpm_fresh, mpm_drag, folding_8mm, pushing, rl_push, yikai, dex_v2_filtered2, pickup. The same folders, exclusion lists and scales as the eight specialists, see their tables. Pickup carries its table shift `data.roots.pickup.shift_m` |
| Dataset draw | every step draws ONE dataset with probability 1/8, the same on all GPUs, and the whole batch comes from it. 200,000 steps give each dataset about 25,000 steps |
| Batch | 4 per GPU x 4 GPUs = 16 windows per step. For folding, yikai and dex one GPU's batch is 4 windows of ONE clip or episode, so a step sees 4 of them |
| Windows | 4 context + 16 future frames. Training starts are weighted by motion for mpm_fresh, mpm_drag and folding, uniform elsewhere, as in the specialists |
| Validation | on the eight test folders in strict rotation: 40 loss batches and 40 sampler batches, so 5 and 5 per dataset, 20 windows each. For rl_push a batch is one object, 1 or 2 windows. It selects `best.pt` by the mean sampler error in mm over the eight datasets, which the large-motion datasets dominate: mpm_fresh and rl_push. `final.pt` is the reported checkpoint, so this affects nothing reported |
| Loss balance | unweighted. After 3,000 steps the specialists' loss levels differ 7 times between datasets and their gradient norms 5 times, all below the clip. Watch the per-dataset validation lines, which the log prints at every validation |
| Action channel clip | the code clips the per-particle action channel for every dataset except mpm_drag. 1.7% to 6.5% of gripper, pusher or hand commands exceed the range. It loses nothing, because every rigid body's command also enters unclipped through the pose path |
| Checkpoints | 8.04 GB each for 502 M, 1.65 GB for 103 M: every 20K steps plus latest, best and final, 13 files, about 105 GB and 21 GB |
| Wall clock, 502 M | measured on two RTX PRO 6000: 1.10 steps/s at batch 4, so 200K steps take 51 h of training. One validation took 313 s, and the sampler runs on one GPU, so more GPUs do not shorten it. Validating every 1,000 steps adds 19 h: 69 h in total, 2 h under the 71.5 h job limit. That is no margin: the step rate on 4 GPUs is not measured, the gradient exchange cost 5% on two GPUs and grows with four, and a run 4% slower than this needs a resubmission. Validating every 2,000 steps adds 9 h: 60 h in total. The run validates every 2,000 steps: `submit_joint8_500m.sh` carries it as `log.val_every=2000` |
| Wall clock, 103 M | 2.61 steps/s on two GPUs, so 21 h of training. One validation took 182 s, which adds 12 h at every 1,000 steps: 33 h in total |

How much data each dataset sees in a unified run, against its specialist:

| dataset | train episodes | specialist sees, windows | unified sees, windows | times fewer | passes over its distinct windows |
|---|---|---|---|---|---|
| mpm_fresh | 47,288 | 6.4 M | 0.40 M | 16 | 0.28 |
| mpm_drag | 2,333 | 3.2 M | 0.40 M | 8 | 2.86 |
| folding_8mm | 4,510 clips | 1.6 M | 0.40 M | 4 | 0.10 |
| pushing | 92,651 after 533 exclusions | 6.4 M | 0.40 M | 16 | 0.21 |
| rl_push | 50,815 usable | 6.4 M | 0.40 M | 16 | 0.35 |
| yikai | 11,015 | 3.2 M | 0.40 M | 8 | 0.08 |
| dex_v2_filtered2 | 16,999 | 3.2 M | 0.40 M | 8 | 0.07 |
| pickup | 42,253 after 7,839 exclusions | 6.4 M | 0.40 M | 16 | 0.45 |
