# Templates of earlier runs

Kept as records of how finished or superseded runs were launched. None of them is part of the pwm-data-v1 launch: the
current jobs are the `submit_<job>.sh` scripts one folder up. Submit from the repo root, for example
`sbatch launch/legacy/train_rl_push_unet.sbatch bidir`. The pre-flight in [docs/SUPPORTED.md](../../docs/SUPPORTED.md)
applies to every submission.

| file | recipe | hardware | notes |
|---|---|---|---|
| `launch_tf_a100x4.sbatch` | `configs/transformer.yaml` | 4x A100-40GB, batch 2 per GPU | ~3.5 days; resubmit to resume from `latest.pt` |
| `launch_unet_a100x2.sbatch` | `configs/unet.yaml` | 2x A100-40GB, batch 4 per GPU | overrides the recipe's 40K stop and 10K snapshot cadence |
| `train_mpm_fresh_unet.sbatch` | `configs/unet.yaml` with MPM from a fresh campaign release: `mpm` specialist or `all` four domains, `causal` or `bidir` | 2 GPUs at batch 4, or one pinned GPU at batch 8 (`mpm` only) | `MPM_RELEASE` comes from `scripts/unified/build_mpm_release.py`; a draft that never ran at length: `pwm-data-v1/mpm_fresh` and `submit_mpm_fresh_100m.sh` replace it |
| `train_mpm_v2_unet.sbatch` | `configs/unet.yaml` MPM specialist on the registered `robert_mpm_v2` corpus, validated on `robert_mpm_v2_val_unseen_fulltar`, `train.light_context_noise_std=0.05`; `causal` or `bidir` | 2 GPUs at batch 4, or one pinned GPU at batch 8 | recipe confirmed 2026-09-15; `best.pt` is chosen on the four unseen validation tars, so score unseen objects on the 00001+ tars |
| `train_pickup_jointed_unet.sbatch` | `configs/unet.yaml` pickup specialist, 331M UNet with the jointed gripper (`model.articulated_bodies`); `causal` or `bidir` | 2 GPUs at batch 4, or one pinned GPU at batch 8 | pre-flight confirmed 2026-09-12, full 200K; both runs trained with `train.noise_context=true`, which new configs refuse, so a relaunch has clean context |
| `train_pickup_uncond_unet.sbatch` | unconditional fine-tune of the 200K bidirectional pickup checkpoint (`train.action_dropout=1`, `train.init_from`), for guidance testing | one process at batch 8, optionally on a pinned GPU | 40K schedule at peak LR 5e-5; see [ISSUES.md](../../ISSUES.md) |
| `train_pickup_uncond_cleanctx.sbatch` | continues the 40K unconditional pickup UNet with clean context, peak LR 1e-5, batch 8 | 1 RTX PRO 6000, 12 h | needs the prepared local `validation/cleanctx_training_preflight_2026-09-14` artifacts; see [docs/guidance/CLEAN_CONTEXT_FINETUNE_2026-09-14.md](../../docs/guidance/CLEAN_CONTEXT_FINETUNE_2026-09-14.md) |
| `train_rl_push_unet.sbatch` | rl_push specialist on the 2026-09-14 rl_push release with `stats/normalization_rl_push.json`; `causal` or `bidir` | 2 GPUs | the 200K run of 2026-09-15, see [docs/RL_PUSH.md](../../docs/RL_PUSH.md); `submit_rl_push_100m.sh` replaces it for pwm-data-v1 |
