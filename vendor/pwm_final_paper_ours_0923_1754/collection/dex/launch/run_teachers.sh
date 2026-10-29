#!/bin/bash
# Generate the five teacher shards with the tactile_genesis_teachers_v1 settings (2,048 envs each, seeds 0-4).
# The xhand1 teachers run one after another on the first GPU, the sharpa teachers on the second. On a node with both:
#   bash collection/dex/launch/run_teachers.sh <tactile-genesis checkout> <new out_dir> [gpu_a] [gpu_b]
# then label outcomes:  <checkout>/dexterous-hands/.venv/bin/python -m collection.dex.outcomes <out_dir>
set -uo pipefail
TG=$(realpath "${1:?tactile-genesis checkout}")
OUT=$(realpath -m "${2:?output directory}")
GPU_A=${3:-0}
GPU_B=${4:-1}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PY=$TG/dexterous-hands/.venv/bin/python
CACHE=${XDG_CACHE_HOME:-$(dirname "$TG")/cache}    # Genesis kernel caches; keep them off small home quotas
mkdir -p "$OUT"
cd "$REPO"
one() {   # gpu teacher seed
    echo "[$2] start $(date -u +%FT%TZ) gpu=$1"
    CUDA_VISIBLE_DEVICES=$1 XDG_CACHE_HOME=$CACHE "$PY" -m collection.dex.generate --tactile_genesis "$TG" \
        --teacher "$2" --out_dir "$OUT" --num_envs 2048 --perturb_frac 0.2 --noise_std 0.3 --seed "$3" > "$OUT/$2.log" 2>&1
    echo "[$2] exit=$? end=$(date -u +%FT%TZ)"
}
( one "$GPU_A" in_palm_rotate-xhand1 0; one "$GPU_A" in_hand_repose-xhand1 1; one "$GPU_A" screwdriver-xhand1 2 ) &
( one "$GPU_B" in_hand_repose-sharpa 3; one "$GPU_B" screwdriver-sharpa 4 ) &
wait
echo "[all] done $(date -u +%FT%TZ)"
