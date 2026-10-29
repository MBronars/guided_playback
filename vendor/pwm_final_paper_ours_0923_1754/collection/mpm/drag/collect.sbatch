#!/bin/bash
#SBATCH --job-name=mpm-drag
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --no-requeue
set -euo pipefail
umask 0007
: "${1:?Use the verified drag launch bundle}"
: "${2:?Pass the qualified shared interpreter}"
export PYTHONPATH="$1/source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$1"
exec "$2" -m collection.mpm.drag.launch run --bundle "$1"
