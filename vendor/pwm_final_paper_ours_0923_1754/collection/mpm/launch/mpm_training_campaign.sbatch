#!/bin/bash
#SBATCH --job-name=mpm-train-4gpu-48h
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX_PRO_6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@18000
#SBATCH --no-requeue
set -euo pipefail
: "${MPM_TRAINING_BUNDLE:?Use mpm_collect.py to set the verified run bundle}"
: "${MPM_TRAINING_OUTPUT:?Use mpm_collect.py to set the new run output}"
: "${MPM_TRAINING_PYTHON:?Use mpm_collect.py to select the pinned shared interpreter}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export MPLCONFIGDIR="/tmp/mpm-training-mpl-${SLURM_JOB_ID}"
exec "$MPM_TRAINING_PYTHON" "$MPM_TRAINING_BUNDLE/mpm_training_campaign.py" run \
  --bundle "$MPM_TRAINING_BUNDLE" --root "$MPM_TRAINING_OUTPUT"
