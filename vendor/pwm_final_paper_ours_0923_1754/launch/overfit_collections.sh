#!/usr/bin/env bash
# Bounded engineering fit on an already allocated GPU. Does not allocate GPUs,
# resume production runs, or enable WandB. Exact inputs/outputs are required.
set -euo pipefail
REPO=${REPO:?set isolated checkout}
PYTHON=${PYTHON:?set model-environment python}
BATCH_DIR=${BATCH_DIR:?set pinned fixture directory}
OUT_ROOT=${OUT_ROOT:?set new output root}
SCRATCH=${SCRATCH:?set scratch directory}
CONFIG=${CONFIG:?set resolved recipe or production config}
DOMAIN=${DOMAIN:?set dataset domain}
TAG=${TAG:?set unique run tag}
STEPS=${STEPS:-1000}
WARMUP=${WARMUP:-100}
EVAL_EVERY=${EVAL_EVERY:-500}
ARTIFACT=${ARTIFACT:?set normalization artifact}
BATCH_NAME=${BATCH_NAME:-$DOMAIN}
HOLDOUT_NAME=${HOLDOUT_NAME:-${BATCH_NAME}_hold}
SAVE_FINAL=${SAVE_FINAL:-0}
SAVE_ARGS=()
if [[ "$SAVE_FINAL" == 1 ]]; then SAVE_ARGS+=(--save_final); fi
if [[ "${REBIND_NORMALIZATION:-0}" == 1 ]]; then SAVE_ARGS+=(--rebind_normalization); fi
if [[ -n "${GPU_UUID:-}" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_UUID"; fi
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/torchinductor" TRITON_CACHE_DIR="$SCRATCH/triton"
export WANDB_MODE=disabled OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True
exec "$PYTHON" scripts/unified/overfit_prod.py \
  --config "$CONFIG" --domain "$DOMAIN" --batch_name "$BATCH_NAME" \
  --holdout_name "$HOLDOUT_NAME" --batch_dir "$BATCH_DIR" --require_saved_batches \
  --out_root "$OUT_ROOT" --tag "$TAG" --steps "$STEPS" --warmup "$WARMUP" \
  --eval_every "$EVAL_EVERY" --eval_rng isolated "${SAVE_ARGS[@]}" \
  --override "data.domains=[$DOMAIN]" data.context_len=4 data.future_len=16 data.batch_size=2 \
    "normalization.artifact=$ARTIFACT" normalization.profile=xy_p99_per_domain \
    model.articulated_bodies=true train.compile=false train.noise_context=false \
    train.light_context_noise_std=0 train.light_pose_noise_std=0 \
    train.clip_deltas=false train.clip_pose_deltas=false
