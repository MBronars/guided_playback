#!/bin/bash
# =============================================================================
#  pwm-data-v1 training job: rl_push_ctxgen_100m_L40s
#  rl_push specialist with the PICK-AND-PLACE ladder and a GENERATED context, UNet 104.22 M,
#  200,000 steps, batch 4 per GPU x grad_accum 4 on 2 L40S.
#
#  Two deliberate differences from launch/submit_rl_push_100m.sh, both stated in the recipe's
#  header, plus the L40S resource shape of launch/submit_yikai_100m_L40s.sh:
#    MODEL    the 3-level pick-and-place ladder at head dim 64, depths 1,2,4 / 1,2, but widths
#             192/384/768 at 3/6/12 heads. 104,221,917 parameters. A 576 L2 (3/6/9 heads) was
#             measured and rejected on 2026-09-20: 6% of the step for 30 M parameters.
#             Per-level registers at L1 and L2 (model.unet_register_mode=per_level): one learned
#             set per level, carried through that level's blocks, dropped at the seam, sized as
#             ONE OBJECT'S WORTH of that level's tokens -- 128 rows at L1, 32 at L2 (user). That
#             sizing reads the material table, which is why the table lists only rl_push's own two
#             materials: with cloth's 512 still in it, one object's worth at L1 would be 512 rows.
#    CONTEXT  generated, not read: train.noise_context + train.pose_noise_context noise the 3
#             context pose transitions at their own levels, rebuild each rigid body's context
#             particle rows from them, and predict them with the same x0 loss as the future rows.
#             train.timestep_schedule therefore shapes all 19 rows, and validation rollouts
#             denoise the context alongside the future (models.factory.context_predicted).
#    DRAW     {independent: 0.6, stream: 0.4} (user): 40% of windows are the ones a STREAMING
#             rollout shows the model, keeping c = 3 rows clean with nothing given and S sampled
#             over 4..50, jitter 0.048. log.stream_rollout_keep: 3 puts the logged rollouts and the
#             sampler validation on that staircase instead of the pyramid.
#             docs/streaming_rollout_plan.md. NOT the rl_push protocol's independent/ramp.
#    L40S     batch 16 -> 4 with grad_accum 4 (4 x 2 GPUs x 4 = the same 32 windows per optimizer
#             step), activation checkpointing off, compiled, log.save_every 0.
#             log.val_batches / sampler_batches 40 / 8, not rl_push's 117 / 60: a validation
#             rollout now walks the whole 19-row window.
#
#  Measured on one L40S of an interactive job, 2026-09-20 (the shared node also ran other work,
#  so the per-step numbers are an upper bound):
#    104,221,917 parameters.
#    compiled, batch 4, the three ladders on the same node:
#        192/384/768, registers 128/32   310 ms per 4 windows   16.1 GiB  104.22 M   (this recipe)
#        192/384/576, registers 128/32   290 ms                 15.0 GiB   73.60 M   (rejected)
#        192/384/768, registers 64/64    318 ms                 16.1 GiB  104.22 M   (measured
#          while the node was busy, so its 8 ms over the row above is not a real difference)
#      So the narrower L2 was worth about 6% of the step and 1.1 GiB -- not worth 30 M parameters
#      (user, 2026-09-20) -- and the register re-sizing is free on both counts -- it is even parameter-neutral, 128x384 + 32x768 = 64x384 + 64x768 =
#      73,728 either way. L2 carries only ~100 of the model's ~1,290 rows per frame, which is why
#      cutting its width 768 -> 576 buys single digits.
#      EAGER at the 768 ladder was 20.6 GiB, so compile buys ~26% of memory here but is not
#      load-bearing: unlike the yikai recipe this one fits either way.
#    compiled, batch 8: 1.62 it/s = 617 ms per 8 windows, peak 30.3 GiB. The SAME 78 ms per
#      window, so batch 8 buys nothing but fewer accumulation boundaries; batch 4 keeps 28 GiB
#      of the card free for the validation passes, and is what the smoke ran.
#    ONE L40S (user), so the windows per optimizer step are grad_accum x 4 and the wall clock is:
#        accum 8 = 32 windows/step (STAGED, parity)   2.48 s   200k = 138 h   6.4 M samples
#        accum 4 = 16 windows/step                    1.24 s   200k =  69 h   3.2 M samples
#        accum 2 =  8 windows/step                    0.62 s   200k =  34 h   1.6 M samples
#      32 windows per step is what the pwm-data-v1 protocol documents, and this run keeps it so it
#      stays comparable with the other specialists and with a later relaunch on more GPUs (user).
#      The invariant across GPU counts is grad_accum = 32 / (data.batch_size x GPUs).
#      before the 200 validations. general MaxTime is 48 h, so the staged row needs THREE jobs:
#      run this file again after each wall and the template resumes from latest.pt. The push50 study measured that
#      total SAMPLES, not batch size, is the lever (batch 8 x 312.5k matched batch 50 x 50k).
#      For reference, the rl_push specialist does the same 32 windows per step in 0.47 s on two
#      RTX PRO 6000 (26 h). That is a different card AND a different ladder; the ratio is not a
#      model-level statement.
#    Smoke (eager, one L40S, this recipe): SMOKE_OK. Fresh start, validation with the 19-row
#      rollout, best/latest/final.pt, resume from latest.pt, second leg. Sampler error on an
#      untrained model 152.9 mm (zero motion scores 104.5 mm on this folder).
#
#  Known limitation: inference.windowed_rollout hands each window a committed history, but a
#  context-predicting checkpoint REGENERATES those rows from noise, so a multi-window offline
#  rollout re-draws its history every window. The single-window sampler (what validation uses) is
#  correct; the streaming protocol this recipe would really want is not in this tree.
#    katef has no L40S (RTX_PRO_6000 / 6000Ada / A100_40GB), so this submits to general with qos
#    normal and the standing --exclude. general MaxTime is 48 h and 200k steps project past it:
#    run this file again and the template resumes from latest.pt.
#
#  Submits launch/train_pwm_data_v1.sbatch from the checkout this file lives in. Use a checkout that stays at one revision
#  for the whole run, because a resubmission reads the code again.
#  Refuses a checkout with modified TRACKED files and a recipe whose sha256 is not the one recorded here.
#
#  Recipe   $ROOT/configs/unet_rl_push_ctxgen_100m_L40s.yaml, sha256 81ba6c58d31c
#  ROOT     default /data/user_data/mbronars/launch_pwm_data_v1_l40s (this job's own; the recipe is
#           kept out of the checkout because this worktree is shared with other agents)
#  LOG_DIR  default /data/group_data/katefgroup/runs_pwm_<your account>: the run folder
#           pwm_v1_rl_push_ctxgen_100m_L40s_unet_bidir/ and slurm/
#  Usage    bash launch/submit_rl_push_ctxgen_100m_L40s.sh            submit
#           bash launch/submit_rl_push_ctxgen_100m_L40s.sh --print    run the checks, print the sbatch command, submit nothing
#  Running it again resumes the run from its latest.pt, first written at the first validation. A run that died before
#  that has no checkpoint: rename its run folder first (see the template's header).
# =============================================================================
set -euo pipefail
JOB=rl_push_ctxgen_100m_L40s
RECIPE_SHA256=81ba6c58d31ced18d8f8661552568dc4511a6f1f1678ba2d0be5ad5176b3c96e
ROOT=${ROOT:-/data/user_data/mbronars/launch_pwm_data_v1_l40s}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_$(id -un)}
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
# Tracked modifications are refused, as in submit_rl_push_100m.sh. Untracked files only warn:
# this worktree is shared with other agents, and their new modules are not imported by an
# architecture=unet recipe.
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "$PWD has modified tracked files: submit from a clean checkout of one revision" >&2; git status --porcelain --untracked-files=no >&2; exit 2; }
UNTRACKED=$(git ls-files --others --exclude-standard)
[[ -z "$UNTRACKED" ]] || { echo "note: untracked files present (not imported by this recipe):" >&2; echo "$UNTRACKED" | sed 's/^/  /' >&2; }
echo "$RECIPE_SHA256  $ROOT/configs/unet_$JOB.yaml" | sha256sum --check --quiet
echo "code revision: $(git rev-parse HEAD)"
# the job is exactly what this file says: no training setting leaks in from the calling shell
unset STEPS BATCH RUN_NAME REPO RESUME
# ONE GPU, so ONE rank. The template defaults to NPROC=2 and accelerate would launch a second
# process that binds device 1: "CUDA error: invalid device ordinal", 16 s into job 10515005.
NPROC=1
# log.save_every: the template hard-codes 20000 on its command line; OVERRIDES is appended last and wins.
OVERRIDES="log.save_every=0"
PWM_CONDA_ENV=pwm211                      # the template defaults to pwm_blackwell, which is for the Blackwell cards
export ROOT LOG_DIR OVERRIDES PWM_CONDA_ENV NPROC
CMD=(sbatch --job-name=pwm_v1_$JOB
     --partition=general --qos=normal --gres=gpu:L40S:1
     --cpus-per-task=8 --mem=96G --time=2-00:00:00
     --exclude=babel-o9-20,babel-o9-24,babel-p9-32
     --output="$LOG_DIR/slurm/%x_%j.log" launch/train_pwm_data_v1.sbatch unet $JOB)
if [[ "${1:-}" == "--print" ]]; then echo "cd $PWD && ROOT=\"$ROOT\" LOG_DIR=\"$LOG_DIR\" OVERRIDES=\"$OVERRIDES\" PWM_CONDA_ENV=$PWM_CONDA_ENV NPROC=$NPROC ${CMD[*]}"; exit 0; fi
mkdir -p "$LOG_DIR/slurm"
"${CMD[@]}"
