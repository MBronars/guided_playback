#!/bin/bash
# =============================================================================
#  rl_push specialist, 331M UNet, jointed gripper (model.articulated_bodies), on a
#  frozen copy of the low-dim RL push release, full 200K schedule.
#
#  Arm (first argument), temporal attention as in the pickup template:
#      causal  configs/unet.yaml default (block-causal)
#      bidir   model.unet_bidirectional_temporal=true
#
#  Usage (SLURM)    sbatch launch/legacy/train_rl_push_unet.sbatch bidir        (from the repo root)
#  Usage (dev job)  srun --jobid=<id> --overlap --ntasks=1 --gres=gpu:2 \
#                       bash launch/legacy/train_rl_push_unet.sbatch bidir
#  Two GPUs at BATCH per GPU (default 8, effective batch 16, as confirmed in the
#  2026-09-15 pre-flight). The pickup and MPM templates run effective batch 8.
#
#  RL_PUSH_RELEASE  frozen release folder with RELEASE.json (every file's sha256);
#                   default: releases/rl-push-lowdim-20260914 on the shared SSD
#  RL_PUSH_RELEASE_SHA256 expected RELEASE.json digest; override for a different release
#  Optional         STEPS (inclusive zero-based stop label, 0 = the full 200K schedule),
#                   LOG_DIR, RUN_NAME, BATCH, OVERRIDES (extra key=value, space-separated)
#
#  Status  Pre-flight confirmed 2026-09-15: the release as of 2026-09-14 22:51 (the
#          audited snapshot), validation = the held-out objects in
#          data/manifests/rl_push_val_assets.json, outliers = data/manifests/rl_push_exclusions.json
#          (both applied to both splits), scales stats/normalization_rl_push.json, context 4 +
#          future 16 from frame 3, unclipped particle and pose targets, bidirectional arm.
#          Eval: eval_unified --domains rl_push --clips all --seed 0 (fixed windows, and
#          --rollout to each episode's end). No baseline comparison. docs/RL_PUSH.md.
# =============================================================================

#SBATCH --job-name=rl_push_unet
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
case "$ARM" in
    causal) EXTRA=() ;;
    bidir)  EXTRA=(model.unet_bidirectional_temporal=true) ;;
    *)      echo "arm must be causal or bidir" >&2; exit 2 ;;
esac

# ---- locations --------------------------------------------------------------
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_robertre}
RL_PUSH_RELEASE=${RL_PUSH_RELEASE:-/data/group_data/katefgroup-ssd/world_model/particle_world_model/releases/rl-push-lowdim-20260914}
RL_PUSH_RELEASE_SHA256=${RL_PUSH_RELEASE_SHA256:-afd2fe3a1df508477a6121392088f939fcce39074d59ea76769fa7acefa4d734}
RUN_NAME=${RUN_NAME:-rl_push_jointed_unet_$ARM}
CONFIG=configs/unet.yaml
STEPS=${STEPS:-0}            # inclusive, zero-based; 0 = the full 200K schedule
BATCH=${BATCH:-8}
OVERRIDES=${OVERRIDES:-}
MANIFESTS="[./data/manifests/rl_push_exclusions.json,./data/manifests/rl_push_val_assets.json]"

if [[ ! -f "$REPO/train.py" || ! -f "$REPO/data/rl_push.py" ]]; then
    echo "REPO=$REPO is not a checkout with the rl_push domain" >&2; exit 2
fi
if [[ ! -f "$RL_PUSH_RELEASE/RELEASE.json" || ! -d "$RL_PUSH_RELEASE/objects" ]]; then
    echo "RL_PUSH_RELEASE=$RL_PUSH_RELEASE is not a frozen release (needs RELEASE.json and objects/)" >&2; exit 2
fi

# ---- environment ------------------------------------------------------------
source /home/robertre/miniconda3/etc/profile.d/conda.sh
conda activate pwm_blackwell
export PYTHONUNBUFFERED=1             # the log shows steps as they happen
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
python -m utils.rl_push_release "$RL_PUSH_RELEASE" --sha256 "$RL_PUSH_RELEASE_SHA256"
mkdir -p "$LOG_DIR/slurm"

echo "[$RUN_NAME] start $(date -u +%FT%TZ) host=$(hostname) rev=$(git rev-parse --short HEAD)" \
     "dirty=$(git status --porcelain | wc -l) steps=$STEPS batch_per_gpu=$BATCH release=$RL_PUSH_RELEASE" \
     "exclusions_sha=$(sha256sum data/manifests/rl_push_exclusions.json | cut -c1-12)" \
     "val_sha=$(sha256sum data/manifests/rl_push_val_assets.json | cut -c1-12) overrides=[$OVERRIDES]"

LAUNCH=(accelerate launch --multi_gpu --num_processes=2 --mixed_precision=bf16
        --main_process_port=$((29800 + ${SLURM_JOB_ID:-0} % 100)))

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
    'data.domains=[rl_push]' \
    data.roots.rl_push.train="$RL_PUSH_RELEASE" \
    data.roots.rl_push.exclusions.train="$MANIFESTS" \
    data.roots.rl_push.exclusions.val="$MANIFESTS" \
    data.batch_size=$BATCH \
    normalization.artifact=stats/normalization_rl_push.json \
    model.articulated_bodies=true \
    train.clip_deltas=false \
    train.clip_pose_deltas=false \
    train.stop_after_step=$STEPS \
    log.save_every=20000 \
    log.project=particle_world_model \
    "${EXTRA[@]}" \
    $OVERRIDES \
    "${RESUME[@]}"
