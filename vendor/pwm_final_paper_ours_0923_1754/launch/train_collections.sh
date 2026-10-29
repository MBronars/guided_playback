#!/usr/bin/env bash
# Run an explicitly generated collection recipe on an existing allocation.
# This wrapper does not allocate GPUs. See docs/DATASETS.md.
# Required: REPO PYTHON CONFIG LOG_DIR RUN_NAME SCRATCH
# Optional: NPROC (1), PORT (29716), RESUME (explicit checkpoint)
# Usage: bash launch/train_collections.sh --check [key=value ...]
#        bash launch/train_collections.sh --run   [key=value ...]
set -euo pipefail
MODE=${1:?usage: --check or --run, then optional key=value overrides}
shift
case "$MODE" in --check|--run) ;; *) echo 'Use --check or --run' >&2; exit 2;; esac
REPO=${REPO:?set isolated checkout}
PYTHON=${PYTHON:?set model-environment python}
CONFIG=${CONFIG:?set generated collection recipe}
LOG_DIR=${LOG_DIR:?set output parent directory}
RUN_NAME=${RUN_NAME:?set a new explicit run name}
SCRATCH=${SCRATCH:?set writable scratch directory}
NPROC=${NPROC:-1}
PORT=${PORT:-29716}
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then echo 'NPROC must be positive' >&2; exit 2; fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then echo 'Use a simple run name without path separators' >&2; exit 2; fi
cd "$REPO"
export PYTHONPATH="$REPO" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export TMPDIR="$SCRATCH/tmp" MPLCONFIGDIR="$SCRATCH/matplotlib"
export TORCHINDUCTOR_CACHE_DIR="$SCRATCH/torchinductor" TRITON_CACHE_DIR="$SCRATCH/triton"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
OVERRIDES=("$@" "run_name=$RUN_NAME" "log.dir=$LOG_DIR")
"$PYTHON" - "$CONFIG" "${OVERRIDES[@]}" <<'PY'
import json, sys
from utils.config import load_config
from utils.checkpoint import corpus_identity
from utils.reproducibility import code_revision
cfg = load_config(sys.argv[1], sys.argv[2:])
print(json.dumps(dict(code=code_revision(), domains=list(cfg.data.domains),
    normalization=cfg.normalization.sha256, corpus=corpus_identity(cfg)), indent=2))
PY
if [[ "$MODE" == --check ]]; then exit 0; fi
RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    [[ -f "$RESUME" ]] || { echo "Missing explicit RESUME checkpoint: $RESUME" >&2; exit 2; }
    RESUME_ARGS=(--resume "$RESUME")
elif [[ -e "$LOG_DIR/$RUN_NAME" ]]; then
    echo "Output already exists: $LOG_DIR/$RUN_NAME; use a new RUN_NAME or explicit RESUME" >&2
    exit 2
fi
LAUNCH=("$PYTHON" -m accelerate.commands.launch --num_processes="$NPROC" --main_process_port="$PORT")
if (( NPROC > 1 )); then LAUNCH+=(--multi_gpu); fi
exec "${LAUNCH[@]}" train.py --config "$CONFIG" "${OVERRIDES[@]}" "${RESUME_ARGS[@]}"
