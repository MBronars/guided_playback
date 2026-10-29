#!/bin/bash
# =============================================================================
#  Transformer 333M, xy_p99 per-domain, full 200K schedule
#
#  Hardware   4x A100-40GB, batch 2 per GPU (effective batch 8)
#  Runtime    ~3.5 days > the partition's 3-day limit, so the job resumes from
#             latest.pt on every resubmission: just sbatch this file again.
#  Status     DRAFT, NOT SUBMITTED. Requires the pre-flight OK (PREFLIGHT_200K.md).
# =============================================================================

#SBATCH --job-name=clean_tf333m_xy_p99_200k
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --gres=gpu:A100_40GB:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-23:30:00
#SBATCH --output=/data/group_data/katefgroup/VLA/3DPCWM/runs_clean/slurm/%x_%j.log

set -euo pipefail

# ---- locations --------------------------------------------------------------
REPO=/data/user_data/robertre/particle_world_model/.worktrees/iclr-clean
LOG_DIR=/data/group_data/katefgroup/VLA/3DPCWM/runs_clean
RUN_NAME=clean_tf333m_xy_p99_200k
CONFIG=configs/transformer.yaml            # already the 200K recipe

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
    --num_processes=4 \
    --mixed_precision=bf16 \
    --main_process_port=29791 \
    train.py \
        --config "$CONFIG" \
        run_name="$RUN_NAME" \
        log.dir="$LOG_DIR" \
        data.batch_size=2 \
        "${RESUME[@]}"
