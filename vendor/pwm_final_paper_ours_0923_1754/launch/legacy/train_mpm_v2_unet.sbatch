#!/bin/bash
# =============================================================================
#  MPM specialist, 331M UNet, on the registered robert_mpm_v2 dropping corpus,
#  validated on held-out unseen objects, full 200K schedule.
#
#  Arm (first argument), temporal attention as in the pickup template:
#      causal  configs/unet.yaml default (block-causal)
#      bidir   model.unet_bidirectional_temporal=true
#
#  Usage (SLURM)    sbatch launch/legacy/train_mpm_v2_unet.sbatch bidir
#  Usage (dev job)  srun --jobid=<id> --overlap --ntasks=1 --gres=gpu:2 \
#                       bash launch/legacy/train_mpm_v2_unet.sbatch bidir
#  A second argument pins one GPU and runs one process at batch 8; without it
#  the script takes two GPUs at batch 4 each. Effective batch is 8 either way.
#  A step started with srun --overlap dies with the allocation that holds its srun
#  client, not only with the one it trains in. To tie a long run to its own dev
#  allocation only, start it from a process inside it: an ssh to the node from one
#  of that allocation's steps is adopted into the allocation.
#
#  Recipe (user decisions 2026-09-15)
#    train    robert_mpm_v2/webdataset_full49: 208 shards, 1-4 objects, 6,553 episodes
#             after data/manifests/mpm_blocklist_train.json (64 collapsed episodes)
#    val      robert_mpm_v2_val_unseen_fulltar: 128 unseen-object episodes, the 00000.tar
#             of each object count of robert_mpm_v2_test_unseen. best.pt is chosen here,
#             so robert_mpm_v2_test_seen, the 00001+ unseen tars and robert_mpm_v2_ood
#             stay untouched for evaluation
#    context  clean history plus train.light_context_noise_std=0.05 (deformable history
#             jitter, the August MPM UNet's value)
#    rest     configs/unet.yaml and utils/config.py defaults: context 4 + future 16,
#             xy_p99_per_domain, velocity flow matching, pyramid diffusion forcing,
#             lr 1e-4 cosine with 5K warmup, EMA 0.99, validation every 1K steps
# =============================================================================

#SBATCH --job-name=mpm_v2_unet
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
RUN_NAME=${RUN_NAME:-mpm_v2_unet_$ARM}
CONFIG=configs/unet.yaml
STEPS=${STEPS:-0}            # inclusive, zero-based; 0 = the full 200K schedule
OVERRIDES=${OVERRIDES:-}
CORPUS=/data/group_data/katefgroup/VLA/3DPCWM
# Resume compares the exclusion paths and their sha256: keep this path stable for a run.
EXCLUSIONS=${EXCLUSIONS:-/data/user_data/robertre/particle_world_model/data/manifests/mpm_blocklist_train.json}

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
    'data.domains=[mpm]' \
    data.roots.mpm.train="$CORPUS/robert_mpm_v2/webdataset_full49" \
    data.roots.mpm.val="$CORPUS/robert_mpm_v2_val_unseen_fulltar" \
    "data.roots.mpm.exclusions.train=[$EXCLUSIONS]" \
    data.batch_size=$BATCH \
    train.light_context_noise_std=0.05 \
    train.stop_after_step=$STEPS \
    log.save_every=20000 \
    log.project=particle_world_model \
    "${EXTRA[@]}" \
    $OVERRIDES \
    "${RESUME[@]}"
