#!/bin/bash
# =============================================================================
#  Pickup, 331M UNet, jointed gripper, bidirectional: UNCONDITIONAL fine-tune.
#  The gripper's future motion is hidden from the model (train.action_dropout=1:
#  its pose tokens read the learned no-action token), so the model generates the
#  gripper trajectory together with the object. For guidance testing.
#
#  Starts from the 200K bidirectional pickup checkpoint (train.init_from loads
#  its EMA weights; fresh optimizer, schedule and step 0). Otherwise the 200K
#  recipe: pickup 4x *_success, xy_p99_per_domain, context 4 + future 16,
#  unclipped pose targets, batch 8, regional compile. The context is clean: the
#  200K runs and the first fine-tune trained with train.noise_context=true,
#  which new configs refuse (it put rigid bodies' particles away from their
#  pose, ISSUES.md ISSUE-001).
#
#  Usage (SLURM)    sbatch launch/legacy/train_pickup_uncond_unet.sbatch
#  Usage (dev job)  srun --jobid=<id> --overlap --ntasks=1 --gres=gpu:1 \
#                       bash launch/legacy/train_pickup_uncond_unet.sbatch 0
#  The optional argument pins one GPU index. One process at batch 8 either way.
#
#  Environment      STEPS schedule length (40000), LR peak (5e-5; the 200K run
#                   used 1e-4 from scratch), WARMUP (1000), ACTION_DROPOUT (1.0;
#                   between 0 and 1 also keeps the conditional model and is
#                   validated with its actions given), STOP_AFTER (inclusive
#                   zero-based stop label, 0 = the full schedule), INIT_FROM,
#                   LOG_DIR, RUN_NAME, OVERRIDES (extra key=value, space-separated).
#  best.pt          log.best_metric=sample_mm, scored on unconditional samples.
#  Eval             eval_unified hides the actions itself for this checkpoint;
#                   its receipt records "actions": "hidden".
#  Status           The first fine-tune, run pickup_jointed_unet_bidir_uncond,
#                   trained 40K steps with noisy context (finished 2026-09-14).
#                   Its pre-flight (2026-09-13) resolved to the 200K bidirectional
#                   run's config except run_name, steps, lr, warmup, save_every,
#                   action_dropout, init_from and best_metric; this version also
#                   differs in train.noise_context (false) and its default run
#                   name. Smoke on the dev job (noisy-context version): loads the
#                   200K EMA weights, trains, validates unconditionally, resumes
#                   (the resumed state supersedes init_from), evaluates.
# =============================================================================

#SBATCH --job-name=pickup_uncond_unet
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_robertre/slurm/%x_%j.log

set -euo pipefail

GPU=${1:-}

# ---- locations --------------------------------------------------------------
REPO=${REPO:-/data/user_data/robertre/particle_world_model}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_robertre}
INIT_FROM=${INIT_FROM:-/data/group_data/katefgroup/runs_pwm_robertre/handoff_pickup_jointed_0912/bidir/final.pt}
RUN_NAME=${RUN_NAME:-pickup_jointed_unet_bidir_uncond_cleanctx}
CONFIG=configs/unet.yaml

# ---- fine-tune schedule -----------------------------------------------------
STEPS=${STEPS:-40000}
LR=${LR:-5e-5}
WARMUP=${WARMUP:-1000}
ACTION_DROPOUT=${ACTION_DROPOUT:-1.0}
STOP_AFTER=${STOP_AFTER:-0}
read -r -a OVERRIDE_ARGS <<< "${OVERRIDES:-}"

# ---- environment ------------------------------------------------------------
source /home/robertre/miniconda3/etc/profile.d/conda.sh
conda activate pwm_blackwell
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES=$GPU
fi

cd "$REPO"
mkdir -p "$LOG_DIR/slurm"

echo "[$RUN_NAME] start $(date -u +%FT%TZ) host=$(hostname) rev=$(git rev-parse --short HEAD)" \
     "dirty=$(git status --porcelain | wc -l) steps=$STEPS lr=$LR warmup=$WARMUP" \
     "action_dropout=$ACTION_DROPOUT stop_after=$STOP_AFTER gpu=${GPU:-allocated}" \
     "init_from=$INIT_FROM"

# ---- resume on resubmission (the resumed state supersedes train.init_from) --
RESUME=()
CKPT="$LOG_DIR/$RUN_NAME/checkpoints/latest.pt"
if [[ -f "$CKPT" ]]; then
    echo "=== resuming from $CKPT"
    RESUME=(--resume "$CKPT")
fi

# ---- launch -----------------------------------------------------------------
PYTHONPATH=. accelerate launch --num_processes=1 --mixed_precision=bf16 train.py \
    --config "$CONFIG" \
    run_name="$RUN_NAME" \
    log.dir="$LOG_DIR" \
    'data.domains=[pickup]' \
    data.batch_size=8 \
    model.articulated_bodies=true \
    model.unet_bidirectional_temporal=true \
    train.clip_pose_deltas=false \
    train.action_dropout="$ACTION_DROPOUT" \
    train.init_from="$INIT_FROM" \
    train.steps="$STEPS" \
    train.lr="$LR" \
    train.warmup_steps="$WARMUP" \
    train.stop_after_step="$STOP_AFTER" \
    log.save_every=10000 \
    log.project=particle_world_model \
    "${OVERRIDE_ARGS[@]}" \
    "${RESUME[@]}"
