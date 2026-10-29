#!/bin/bash
# Continue the 40K unconditional pickup UNet with clean context: 10K attempts,
# peak LR 1e-5, batch 8, BF16. Uses the frozen 2026-09-14 source and preflight;
# requires the prepared local validation/cleanctx_training_preflight_2026-09-14
# artifacts and validation/overnight_2026-09-14/vendor dependencies.
#
# Check only: bash launch/legacy/train_pickup_uncond_cleanctx.sbatch --check
# Submit:     sbatch launch/legacy/train_pickup_uncond_cleanctx.sbatch
# Resume:     sbatch launch/legacy/train_pickup_uncond_cleanctx.sbatch \
#               --resume <this-run>/checkpoints/latest.pt
# Details: docs/guidance/CLEAN_CONTEXT_FINETUNE_2026-09-14.md

#SBATCH --job-name=pickup_uncond_cleanctx
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX_PRO_6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_robertre/slurm/%x_%j.log

set -euo pipefail

# Fixed paths also work when Slurm runs a spooled copy of this script.
REPO=/data/user_data/robertre/particle_world_model
PREFLIGHT="$REPO/validation/cleanctx_training_preflight_2026-09-14"
TRAIN_PYTHON=/data/user_data/robertre/conda_envs/pwm_blackwell/bin/python

cd "$REPO"
export PYTHONPATH="$PREFLIGHT/source:$REPO/validation/overnight_2026-09-14/vendor"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export WANDB_MODE=disabled

# Keep Slurm's GPU assignment. start.py verifies the prepared run before launch
# and refuses an existing output directory without an explicit --resume.
exec "$TRAIN_PYTHON" "$PREFLIGHT/start.py" "$@"
