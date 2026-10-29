#!/bin/bash
# =============================================================================
#  UNet 331M, xy_p99 per-domain, fresh 200K schedule
#
#  Hardware   2x A100-40GB, batch 4 per GPU (effective batch 8), regional compile
#  Recipe     configs/unet.yaml keeps the 40K arm's early stop and 10K snapshot
#             cadence on purpose; the two overrides below make it the full run.
#  Resume     latest.pt is picked up automatically on resubmission.
#  Status     DRAFT, NOT SUBMITTED. Requires the pre-flight OK (PREFLIGHT_200K.md).
# =============================================================================

#SBATCH --job-name=clean_unet_xy_p99_200k
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --gres=gpu:A100_40GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-23:30:00
#SBATCH --output=/data/group_data/katefgroup/VLA/3DPCWM/runs_clean/slurm/%x_%j.log

set -euo pipefail

# ---- locations --------------------------------------------------------------
REPO=/data/user_data/robertre/particle_world_model/.worktrees/iclr-clean
LOG_DIR=/data/group_data/katefgroup/VLA/3DPCWM/runs_clean
RUN_NAME=clean_unet_xy_p99_200k
CONFIG=configs/unet.yaml

# ---- environment ------------------------------------------------------------
source /home/robertre/miniconda3/etc/profile.d/conda.sh
conda activate pwm_blackwell
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOG_DIR/slurm"

# ---- resume on resubmission -------------------------------------------------
RESUME=()
CKPT="$LOG_DIR/$RUN_NAME/checkpoints/latest.pt"
if [[ -f "$CKPT" ]]; then
    echo "=== resuming from $CKPT"
    RESUME=(--resume "$CKPT")
fi

# ---- launch -----------------------------------------------------------------
PYTHONPATH=. accelerate launch \
    --multi_gpu \
    --num_processes=2 \
    --mixed_precision=bf16 \
    --main_process_port=29792 \
    train.py \
        --config "$CONFIG" \
        run_name="$RUN_NAME" \
        log.dir="$LOG_DIR" \
        train.stop_after_step=0 \
        log.save_every=20000 \
        "${RESUME[@]}"
