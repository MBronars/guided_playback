# Launch scripts

The training jobs on the pwm-data-v1 release, one script per job: nine specialists (one dataset each) and two unified
runs (one model on all eight datasets). Dataset collection launchers live with their domain, not here. See
[MPM collection](../collection/mpm/README.md) and the
[collection layout convention](../collection/README.md#domain-layout-convention).

## The jobs

Every job is a UNet with bidirectional attention, 200K steps, on RTX PRO 6000 GPUs, under the specialist protocol of
2026-09-19: x0 prediction with the rigid pose loss in sample space, light context noise 0.05, EMA 0.9999 with warm-up,
noise levels 0.6 independent + 0.4 ramp, the loss spike rule off, scales artifact v7.

| script | job | GPUs | estimated hours |
|---|---|---|---|
| `submit_mpm_fresh_100m.sh` | mpm_fresh specialist, 103.10 M, batch 16 per GPU | 2 | 58 |
| `submit_mpm_drag_100m.sh` | mpm_drag specialist, 103.10 M, batch 8 per GPU | 2 | 30 |
| `submit_folding_8mm_100m.sh` | folding_8mm specialist, 103.10 M, batch 4 per GPU | 2 | 44 |
| `submit_pushing_100m.sh` | pushing specialist, 103.10 M, batch 16 per GPU | 2 | 34 |
| `submit_pushing_100m_light_pose_noise.sh` | pushing specialist with rigid history noise 0.05 (`train.light_pose_noise_std`), the comparison run next to `pushing_100m` | 2 | 34 |
| `submit_rl_push_100m.sh` | rl_push specialist, 103.10 M, batch 16 per GPU | 2 | 32 |
| `submit_yikai_100m.sh` | yikai specialist, 103.10 M, batch 8 per GPU | 2 | 56 |
| `submit_dex_v2_filtered2_100m.sh` | dex_v2_filtered2 specialist, 103.10 M, batch 8 per GPU | 2 | 34 |
| `submit_pickup_100m.sh` | pickup specialist, 103.10 M, batch 16 per GPU | 2 | about 30 |
| `submit_joint8_500m.sh` | unified run on all eight datasets, 502.22 M, batch 4 per GPU, validation every 2,000 steps | 4 | about 60 |
| `submit_joint8_100m.sh` | unified run on all eight datasets, 103.10 M, batch 4 per GPU | 4 | about 33 |

```bash
bash launch/submit_pushing_100m.sh --print    # run the checks, print the sbatch command, submit nothing
bash launch/submit_pushing_100m.sh            # submit
```

A script submits `train_pwm_data_v1.sbatch` from the checkout it lives in, with the job's recipe name, resources and
overrides. Before that it refuses a checkout with local changes and a recipe whose sha256 differs from the one it
records. Submit from a checkout that stays at one revision for the whole run, because a resubmission reads the code
again: `git worktree add --detach <path> <revision>`. Running a script again resumes its run from `latest.pt`.

The recipes are written by `scripts/unified/collection_recipe.py --indices <release>` and live outside the repo, under
`ROOT` (default `/data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1`): `configs/unet_<job>.yaml`, next to
`PREFLIGHT_<dataset>.md`, the job's full setup table. Nothing is submitted before the user confirms that table: it is
the pre-flight of [docs/SUPPORTED.md](../docs/SUPPORTED.md) and the project rules (data paths and vintage, eval
protocol, baseline consistency, code-baked constants). The nine specialist tables were confirmed on 2026-09-19. The two
unified scripts submit only once `PREFLIGHT_joint8.md` carries the line `Status: CONFIRMED by the user`.

The release's test folder is the validation folder, so `best.pt` and the curves read the test set. `final.pt` is the
reported checkpoint, scored on the frozen cases of [docs/EVAL_PROTOCOLS.md](../docs/EVAL_PROTOCOLS.md).

## What the jobs run through

| file | role |
|---|---|
| `train_pwm_data_v1.sbatch` | the SLURM template behind every job: `<unet\|transformer> <recipe name>`, 2 GPUs unless the submit script asks for more, 71.5 h, save every 20K, NCCL timeout 60 min (rank 0 writes checkpoints while the other ranks wait), resubmission resumes from `latest.pt` |
| `train_collections.sh` | the launcher the template calls: validates the recipe (`--check`) and prints the code revision, the normalization digest and the corpus identity before the first step (`--run`) |
| `smoke_pwm_data_v1.sbatch` | two-leg smoke of one recipe through `train_collections.sh`: fresh start, then explicit resume, short validation on the release's test folder. Proves loader, validation, checkpoint files and the resume contract; WandB off, no number is reported |
| `overfit_pwm_data_v1.sbatch`, `overfit_collections.sh` | fixed-batch overfit of one dataset (`scripts/unified/overfit_prod.py`), pinned fixtures drawn with the recipe's window policy, 1 GPU. A trainability check in mm against the zero-motion and hold-velocity baselines, never a held-out number |

## From another account

Nothing in the repo needs editing (a submit script refuses a checkout with local changes). Every location defaults to
the calling account's own, so two people who run the same job never share a run folder; a shared one would make the
second job resume the first one's `latest.pt`.

| variable | default | what |
|---|---|---|
| `LOG_DIR` | `/data/group_data/katefgroup/runs_pwm_<your account>` | run folders `pwm_v1_<job>_unet_bidir/` and the SLURM logs `slurm/` |
| `SCRATCH` | `/data/user_data/<your account>/scratchpad/launch_pwm_data_v1` | temporary files |
| `PWM_CONDA_SH` | `$HOME/miniconda3/etc/profile.d/conda.sh` | the conda installation the job activates from |
| `PWM_CONDA_ENV` | `pwm_blackwell` | the environment of [docs/ENVIRONMENT.md](../docs/ENVIRONMENT.md) (its rebuild recipe names it `pwm`) |
| `ROOT` | `/data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1` | the shared recipes and setup tables: leave it |

```bash
PWM_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh PWM_CONDA_ENV=pwm bash launch/submit_pushing_100m.sh --print
```

The release and the recipes are readable by the group `kfragki2_group`. Run directories belong on a volume with room
for 16 bytes per parameter per checkpoint: 1.65 GB at 103 M, 8.0 GB at 502 M, 13 files per run. WandB logs to the
project `particle_world_model` of whoever is logged in.

## Earlier runs

[`legacy/`](legacy/README.md) keeps the templates of finished or superseded runs as records: the primary transformer
and UNet recipes on A100s, the MPM v2 and fresh-campaign specialists, the jointed and unconditional pickup runs and the
2026-09-15 rl_push specialist.
