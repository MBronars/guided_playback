#!/bin/bash
# =============================================================================
#  Fixed-batch overfit of ONE pwm-data-v1 dataset with the production model
#  (scripts/unified/overfit_prod.py through launch/overfit_collections.sh), one GPU.
#  The recipe comes from scripts/unified/collection_recipe.py --indices <release>; the
#  fixture (one B=2 training batch and one held-out batch, drawn by the prep step with the
#  recipe's window policy) is pinned and never redrawn here.
#
#  Usage (SLURM)    sbatch launch/overfit_pwm_data_v1.sbatch <unet|transformer> <dataset>
#  Usage (dev job)  srun --jobid=<id> --overlap --ntasks=1 --gres=gpu:1 \
#                       bash launch/overfit_pwm_data_v1.sbatch unet yikai
#
#  ROOT (default /data/group_data/katefgroup/runs_pwm_robertre/overfit_pwm_data_v1):
#      configs/<arch>_<dataset>.yaml        the recipe written by the prep step
#      batches/<dataset>.pt|.json           the pinned training fixture
#      batches/<dataset>_hold.pt|.json      the held-out fixture (sampled at every eval, never trained on)
#      runs/<tag>/                          this job's output (curves, final 8-seed scores, receipts)
#  Optional  STEPS (5000)  EVAL_EVERY (500)  WARMUP (200)  TAG (<arch>_<dataset>_<jobid>)  DOMAIN (the dataset's domain when
#            <dataset> is a variant folder such as folding_8mm)
#            REPO (the submit directory: submit from the pinned worktree)  SCRATCH  SAVE_FINAL (0)
#
#  What it measures: fixed-window reconstruction of the fixture in mm, sampled from pure
#  noise with the recipe's sampler, next to the zero-motion and hold-velocity baselines.
#  Trainability and implementation consistency only; never a held-out or rollout number.
# =============================================================================

#SBATCH --job-name=overfit_pwm_data_v1
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_robertre/slurm/%x_%j.log

set -euo pipefail

ARCH=${1:?arch: unet | transformer}
DATASET=${2:?dataset: folding | pushing | rl_push | yikai (any dataset with a recipe and fixtures under ROOT)}
case "$ARCH" in unet|transformer) ;; *) echo "arch must be unet or transformer" >&2; exit 2 ;; esac

# ---- locations --------------------------------------------------------------
ROOT=${ROOT:-/data/group_data/katefgroup/runs_pwm_robertre/overfit_pwm_data_v1}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
SCRATCH=${SCRATCH:-/data/user_data/robertre/scratchpad/overfit_pwm_data_v1}
CONFIG=$ROOT/configs/${ARCH}_${DATASET}.yaml
TAG=${TAG:-${ARCH}_${DATASET}_${SLURM_JOB_ID:-local}}
DOMAIN=${DOMAIN:-$DATASET}          # a variant folder (folding_8mm) keeps its domain name (folding)

[[ -f "$REPO/train.py" && -f "$REPO/scripts/unified/overfit_prod.py" ]] || { echo "REPO=$REPO is not a checkout" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "missing recipe $CONFIG (run the prep step first)" >&2; exit 2; }
for f in "$ROOT/batches/$DATASET.pt" "$ROOT/batches/$DATASET.json" "$ROOT/batches/${DATASET}_hold.pt" "$ROOT/batches/${DATASET}_hold.json"; do
    [[ -f "$f" ]] || { echo "missing pinned fixture $f (run the prep step first)" >&2; exit 2; }
done
[[ -e "$ROOT/runs/$TAG" ]] && { echo "output $ROOT/runs/$TAG exists; choose another TAG" >&2; exit 2; }

# ---- environment ------------------------------------------------------------
# another account: ROOT and SCRATCH are its own folders, PWM_CONDA_SH / PWM_CONDA_ENV its conda (as in train_pwm_data_v1.sbatch)
PWM_CONDA_SH=${PWM_CONDA_SH:-${HOME:-/home/$(id -un)}/miniconda3/etc/profile.d/conda.sh}
[[ -r "$PWM_CONDA_SH" ]] || { echo "cannot read $PWM_CONDA_SH: set PWM_CONDA_SH=<your conda>/etc/profile.d/conda.sh" >&2; exit 2; }
source "$PWM_CONDA_SH"
conda activate "${PWM_CONDA_ENV:-pwm_blackwell}"
mkdir -p "$ROOT/runs" "$SCRATCH/tmp" "$(dirname "$ROOT")/slurm"
cd "$REPO"
ARTIFACT=$(python -c "import sys; from omegaconf import OmegaConf; print(OmegaConf.load(sys.argv[1]).normalization.artifact)" "$CONFIG")

echo "[$TAG] start $(date -u +%FT%TZ) host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" \
     "rev=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) recipe_sha=$(sha256sum "$CONFIG" | cut -c1-12)" \
     "fixture_sha=$(sha256sum "$ROOT/batches/$DATASET.pt" | cut -c1-12) holdout_sha=$(sha256sum "$ROOT/batches/${DATASET}_hold.pt" | cut -c1-12)" \
     "artifact=$ARTIFACT steps=${STEPS:-5000} eval_every=${EVAL_EVERY:-500} warmup=${WARMUP:-200}"

# ---- run --------------------------------------------------------------------
REPO="$REPO" PYTHON="$(command -v python)" BATCH_DIR="$ROOT/batches" OUT_ROOT="$ROOT/runs" SCRATCH="$SCRATCH" \
CONFIG="$CONFIG" DOMAIN="$DOMAIN" BATCH_NAME="$DATASET" HOLDOUT_NAME="${DATASET}_hold" TAG="$TAG" ARTIFACT="$ARTIFACT" \
STEPS="${STEPS:-5000}" EVAL_EVERY="${EVAL_EVERY:-500}" WARMUP="${WARMUP:-200}" SAVE_FINAL="${SAVE_FINAL:-0}" \
    bash launch/overfit_collections.sh
echo "[$TAG] done $(date -u +%FT%TZ)"
