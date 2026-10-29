#!/bin/bash
# =============================================================================
#  Smoke training of ONE pwm-data-v1 dataset with train.py (through
#  launch/train_collections.sh), one GPU, a few hundred steps in two legs:
#      leg 1  fresh start, stops at LEG1 (inclusive, zero-based)
#      leg 2  explicit --resume from leg 1's latest.pt, stops at LEG2
#  It proves the loader, the window policy, validation on the release's test
#  folder (objective pass and sampler pass), best.pt / latest.pt / final.pt and the
#  resume contract (config, normalization, roots and exclusion hashes). It is not a
#  research run: WandB is off and no number it prints is reported.
#
#  Usage (SLURM)    sbatch launch/smoke_pwm_data_v1.sbatch <unet|transformer> <dataset>
#  Usage (dev job)  srun --jobid=<id> --overlap --ntasks=1 --gres=gpu:1 \
#                       bash launch/smoke_pwm_data_v1.sbatch unet yikai
#
#  ROOT (default /data/group_data/katefgroup/runs_pwm_robertre/smoke_pwm_data_v1):
#      configs/<arch>_<dataset>.yaml   recipe from scripts/unified/collection_recipe.py
#                                      --indices <release> (written by the prep step)
#      runs/<run name>/                train.py output; three checkpoints of ~5.3 GB each
#  Attention is bidirectional (decision 2026-09-18): UNet unet_bidirectional_temporal,
#  transformer causal_temporal=false.
#  Optional  LEG1 (149)  LEG2 (299)  VAL_EVERY (50)  VAL_BATCHES (10)  SAMPLER_BATCHES (5)
#            RUN_NAME  REPO (the submit directory: submit from a pinned worktree)  SCRATCH
#            OVERRIDES (extra key=value words, e.g. train.compile=false)
# =============================================================================

#SBATCH --job-name=smoke_pwm_data_v1
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_robertre/slurm/%x_%j.log

set -euo pipefail

ARCH=${1:?arch: unet | transformer}
DATASET=${2:?dataset: a name with a recipe under ROOT/configs}
case "$ARCH" in
    unet)        ATTENTION=(model.unet_bidirectional_temporal=true) ;;
    transformer) ATTENTION=(model.causal_temporal=false) ;;
    *)           echo "arch must be unet or transformer" >&2; exit 2 ;;
esac

# ---- locations --------------------------------------------------------------
ROOT=${ROOT:-/data/group_data/katefgroup/runs_pwm_robertre/smoke_pwm_data_v1}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
SCRATCH=${SCRATCH:-/data/user_data/robertre/scratchpad/smoke_pwm_data_v1}
CONFIG=$ROOT/configs/${ARCH}_${DATASET}.yaml
RUN_NAME=${RUN_NAME:-smoke_${ARCH}_${DATASET}_${SLURM_JOB_ID:-local}}
LEG1=${LEG1:-149}
LEG2=${LEG2:-299}
OVERRIDES=${OVERRIDES:-}

[[ -f "$REPO/train.py" && -f "$REPO/launch/train_collections.sh" ]] || { echo "REPO=$REPO is not a checkout" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "missing recipe $CONFIG (run the prep step first)" >&2; exit 2; }
(( LEG1 < LEG2 )) || { echo "LEG1 must be below LEG2" >&2; exit 2; }

# ---- environment ------------------------------------------------------------
# another account: ROOT and SCRATCH are its own folders, PWM_CONDA_SH / PWM_CONDA_ENV its conda (as in train_pwm_data_v1.sbatch)
PWM_CONDA_SH=${PWM_CONDA_SH:-${HOME:-/home/$(id -un)}/miniconda3/etc/profile.d/conda.sh}
[[ -r "$PWM_CONDA_SH" ]] || { echo "cannot read $PWM_CONDA_SH: set PWM_CONDA_SH=<your conda>/etc/profile.d/conda.sh" >&2; exit 2; }
source "$PWM_CONDA_SH"
conda activate "${PWM_CONDA_ENV:-pwm_blackwell}"
export WANDB_MODE=disabled
mkdir -p "$ROOT/runs" "$SCRATCH/tmp"
cd "$REPO"

echo "[$RUN_NAME] start $(date -u +%FT%TZ) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
     "rev=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) recipe_sha=$(sha256sum "$CONFIG" | cut -c1-12)" \
     "legs=$LEG1/$LEG2 overrides=[$OVERRIDES]"

# OVERRIDES is intentionally unquoted: a space-separated list of key=value words.
COMMON=("${ATTENTION[@]}" log.val_every="${VAL_EVERY:-50}" log.val_batches="${VAL_BATCHES:-10}"
        log.sampler_batches="${SAMPLER_BATCHES:-5}" log.save_every=0 $OVERRIDES)
run_leg() {   # $1 = stop label, RESUME exported by the caller for leg 2
    REPO="$REPO" PYTHON="$(command -v python)" CONFIG="$CONFIG" LOG_DIR="$ROOT/runs" RUN_NAME="$RUN_NAME" \
    SCRATCH="$SCRATCH" PORT=$((29800 + ${SLURM_JOB_ID:-0} % 100)) \
        bash launch/train_collections.sh --run "${COMMON[@]}" train.stop_after_step="$1"
}

echo "=== leg 1: fresh start to step $LEG1"
unset RESUME
run_leg "$LEG1"
CKPT="$ROOT/runs/$RUN_NAME/checkpoints"
for name in latest.pt final.pt; do
    [[ -f "$CKPT/$name" ]] || { echo "SMOKE FAILED: leg 1 wrote no $name" >&2; exit 3; }
done
ls -la "$CKPT"

echo "=== leg 2: resume from latest.pt to step $LEG2"
export RESUME="$CKPT/latest.pt"
run_leg "$LEG2"
ls -la "$CKPT"
echo "[$RUN_NAME] SMOKE_OK $(date -u +%FT%TZ)"
