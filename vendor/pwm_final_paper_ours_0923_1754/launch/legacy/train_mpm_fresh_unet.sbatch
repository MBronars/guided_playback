#!/bin/bash
# =============================================================================
#  331M UNet on a fresh MPM campaign release (data.roots.mpm), full 200K schedule.
#
#  Scope (first argument), the only difference between the two runs:
#      mpm   MPM specialist, data.domains=[mpm]
#      all   four-domain model: MPM from the release; pickup, cloth and pushing
#            keep their registered unified://v1 data
#  Arm (second argument), temporal attention as in the pickup template:
#      causal  configs/unet.yaml default (block-causal)
#      bidir   model.unet_bidirectional_temporal=true
#
#  Usage (SLURM)    MPM_RELEASE=<release> sbatch launch/legacy/train_mpm_fresh_unet.sbatch mpm bidir
#  Usage (dev job)  MPM_RELEASE=<release> srun --jobid=<id> --overlap --ntasks=1 --gres=gpu:1 \
#                       bash launch/legacy/train_mpm_fresh_unet.sbatch mpm bidir 0
#  A third argument pins one GPU and runs one process at batch 8 (scope mpm only);
#  without it the script takes two GPUs at batch 4 each. Effective batch is 8 either
#  way. Scope all refuses one GPU: see the SDPA sequence limit below.
#
#  MPM_RELEASE  a directory built by scripts/unified/build_mpm_release.py: train/ and
#               val/ hold hard links to verified campaign shards; RELEASE.json lists
#               every shard with its sha256.
#  MPM_VAL      validation folder (default MPM_RELEASE/val). A release built before
#               validation data exists has no val/ and refuses to start without it.
#  Optional     STEPS (inclusive zero-based stop label, 0 = the full 200K schedule),
#               LOG_DIR, RUN_NAME, OVERRIDES (extra key=value, space-separated).
#
#  Status  DRAFT, NOT SUBMITTED (2026-09-14). The pre-flight is not confirmed:
#          - data: passive campaign fresh-10432785-v4 (1-4 objects, 7 motion types,
#            fresh training shapes, 49 states at 1/24 s), collecting until 2026-09-15;
#            no validation data has been generated yet
#          - normalization: xy_p99_per_domain as frozen. With one estimator on 160
#            shards each, the fresh data's per-frame |delta| p99 is ~15% below the
#            registered v2 corpus in xy and z (measured 2026-09-14)
#          - context 4 + future 16 (DOMAIN_FUTURE_LEN mpm); no exclusion lists, as
#            the collector drops rejected clips before export
#          - all: configs/unet.yaml domain weights and model.articulated_bodies=false;
#            the pickup specialists train with articulated_bodies=true
#          - evaluation protocol and baselines: pending the validation data
# =============================================================================

#SBATCH --job-name=mpm_fresh_unet
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-23:30:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_robertre/slurm/%x_%j.log

set -euo pipefail

SCOPE=${1:?scope: mpm | all}
ARM=${2:?arm: causal | bidir}
GPU=${3:-}

case "$SCOPE" in
    mpm) DOMAINS=('data.domains=[mpm]') ;;
    all) DOMAINS=() ;;
    *)   echo "scope must be mpm or all" >&2; exit 2 ;;
esac
case "$ARM" in
    causal) EXTRA=() ;;
    bidir)  EXTRA=(model.unet_bidirectional_temporal=true) ;;
    *)      echo "arm must be causal or bidir" >&2; exit 2 ;;
esac
# The UNet's temporal attention runs one sequence per particle. At batch 8 a cloth batch
# (~9.4K particles) passes over 65,535 sequences to one SDPA call, and the flash kernel then
# fails with "CUDA error: invalid configuration argument" (PyTorch 2.11). Batch 4 stays under.
if [[ "$SCOPE" == all && -n "$GPU" ]]; then
    echo "scope all runs on two GPUs at batch 4; one GPU at batch 8 exceeds SDPA's 65,535-sequence limit on cloth" >&2
    exit 2
fi

# ---- locations --------------------------------------------------------------
REPO=${REPO:-/data/user_data/robertre/particle_world_model}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_robertre}
MPM_RELEASE=${MPM_RELEASE:?set MPM_RELEASE to a release from scripts/unified/build_mpm_release.py}
MPM_VAL=${MPM_VAL:-$MPM_RELEASE/val}
RUN_NAME=${RUN_NAME:-unet_${SCOPE}_${ARM}_$(basename "$MPM_RELEASE")}
CONFIG=configs/unet.yaml
STEPS=${STEPS:-0}            # inclusive, zero-based; 0 = the full 200K schedule
OVERRIDES=${OVERRIDES:-}

if [[ ! -f "$MPM_RELEASE/RELEASE.json" || ! -d "$MPM_RELEASE/train" ]]; then
    echo "MPM_RELEASE=$MPM_RELEASE is not a built release (needs RELEASE.json and train/)" >&2; exit 2
fi
if [[ ! -d "$MPM_VAL" ]]; then
    echo "no validation folder at $MPM_VAL: this release has no val split, set MPM_VAL" >&2; exit 2
fi

# ---- environment ------------------------------------------------------------
source /home/robertre/miniconda3/etc/profile.d/conda.sh
conda activate pwm_blackwell
export PYTHONUNBUFFERED=1             # the Slurm log shows steps as they happen
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
mkdir -p "$LOG_DIR/slurm"

echo "[$RUN_NAME] start $(date -u +%FT%TZ) host=$(hostname) rev=$(git rev-parse --short HEAD)" \
     "dirty=$(git status --porcelain | wc -l) steps=$STEPS gpu=${GPU:-both}" \
     "release=$MPM_RELEASE val=$MPM_VAL overrides=[$OVERRIDES]"

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
    "${DOMAINS[@]}" \
    data.roots.mpm.train="$MPM_RELEASE/train" \
    data.roots.mpm.val="$MPM_VAL" \
    data.batch_size=$BATCH \
    train.stop_after_step=$STEPS \
    log.save_every=20000 \
    log.project=particle_world_model \
    "${EXTRA[@]}" \
    $OVERRIDES \
    "${RESUME[@]}"
