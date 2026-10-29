#!/bin/bash
# =============================================================================
#  Pickup specialist, 331M UNet, jointed gripper: each finger is a prismatic
#  child of the hand, so fingers cannot drift off it (model.articulated_bodies).
#
#  Two arms that differ ONLY in temporal attention direction:
#      causal  configs/unet.yaml default (block-causal)
#      bidir   model.unet_bidirectional_temporal=true
#
#  Usage (SLURM)    sbatch launch/legacy/train_pickup_jointed_unet.sbatch causal
#  Usage (dev job)  srun --jobid=<id> --overlap --ntasks=1 --gres=gpu:1 \
#                       bash launch/legacy/train_pickup_jointed_unet.sbatch causal 0
#  A second argument pins one GPU and runs one process at batch 8; without it
#  the script takes two GPUs at batch 4 each. Effective batch is 8 either way,
#  which is the 40K reference recipe (2x A100, batch 4 per GPU).
#
#  Status  Pre-flight confirmed 2026-09-12. pickup 4x *_success (vintage
#          2026-05-24, 189 train / 21 val shards), xy_p99_per_domain sha
#          2bc31504, context 4 + future 16, unclipped pose targets (0.55%
#          of elements clip, worst 2.4x), effective batch 8, full 200K.
#          Both runs trained with train.noise_context=true (the default until
#          2026-09-12); new configs refuse it, so a relaunch has clean context.
#          Eval: eval_unified --rollout --rollout_frames 36 --seed 0
#          --max_batches 40 on pickup val (H = min(horizon, T-C) caps at 36).
#          Those 40 batches were 136 clips from one place shard; the current
#          protocol is docs/EVALUATION.md (--clips all, stratified reads).
# =============================================================================

#SBATCH --job-name=pickup_jointed_unet
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-23:30:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_robertre/slurm/%x_%j.log

set -euo pipefail

ARM=${1:?arm: causal | bidir}
GPU=${2:-}

case "$ARM" in
    causal) EXTRA=() ;;
    bidir)  EXTRA=(model.unet_bidirectional_temporal=true) ;;
    *)      echo "arm must be causal or bidir" >&2; exit 2 ;;
esac

# ---- locations --------------------------------------------------------------
REPO=${REPO:-/data/user_data/robertre/particle_world_model}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_robertre}
RUN_NAME=${RUN_NAME:-pickup_jointed_unet_$ARM}
CONFIG=configs/unet.yaml
STEPS=${STEPS:-0}            # inclusive, zero-based; 0 = the full 200K schedule
OVERRIDES=${OVERRIDES:-}

# ---- environment ------------------------------------------------------------
source /home/robertre/miniconda3/etc/profile.d/conda.sh
conda activate pwm_blackwell
export PYTHONUNBUFFERED=1             # the log shows steps as they happen
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOG_DIR/slurm"

echo "[$RUN_NAME] start $(date -u +%FT%TZ) host=$(hostname) rev=$(git rev-parse --short HEAD)" \
     "dirty=$(git status --porcelain | wc -l) steps=$STEPS gpu=${GPU:-both} overrides=[$OVERRIDES]"

# ---- one GPU at batch 8, or two at batch 4 ----------------------------------
if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES=$GPU
    LAUNCH=(accelerate launch --num_processes=1 --mixed_precision=bf16)
    BATCH=8
else
    LAUNCH=(accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16
            --main_process_port=$((29800 + ${SLURM_JOB_ID:-0} % 100)))
    BATCH=4
fi

# ---- resume on resubmission -------------------------------------------------
RESUME=()
CKPT="$LOG_DIR/$RUN_NAME/checkpoints/latest.pt"
if [[ -f "$CKPT" ]]; then
    echo "=== resuming from $CKPT"
    RESUME=(--resume "$CKPT")
fi

# ---- launch -----------------------------------------------------------------
# OVERRIDES is intentionally unquoted: a space-separated list of key=value words.
PYTHONPATH=. "${LAUNCH[@]}" train.py \
    --config "$CONFIG" \
    run_name="$RUN_NAME" \
    log.dir="$LOG_DIR" \
    'data.domains=[pickup]' \
    data.batch_size=$BATCH \
    model.articulated_bodies=true \
    train.clip_pose_deltas=false \
    train.stop_after_step=$STEPS \
    log.save_every=20000 \
    log.project=particle_world_model \
    "${EXTRA[@]}" \
    $OVERRIDES \
    "${RESUME[@]}"
