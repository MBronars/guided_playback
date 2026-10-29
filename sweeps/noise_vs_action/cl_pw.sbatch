#!/bin/bash
#SBATCH --job-name=nva_cl_pw
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action_dev/slurm/cl_pw_%j.out
# Closed-loop PointWorld + CEM (128 x 3, replan the remaining horizon, commit 5): planning on this L40S, Isaac in the claude-host GPU (flock).
source /data/user_data/mbronars/packages/guided_playback/sweeps/noise_vs_action/env.sh
P=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action/pool_cl
SRUN="unset \$(env | grep -o '^SLURM_[A-Z_]*'); flock $P/isaac.lock srun --jobid=\$(squeue -u mbronars -h -n claude-host -t R -o %i | head -1) --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=0"
$PY -u bin/pick_pw_cem_closed_loop.py --tag nva_pwcem128 --ckpt /data/group_data/katefgroup/runs_pw_mbronars/pointworld_pickup_open/latest.pt \
  --stats /data/group_data/katefgroup/runs_pw_mbronars/pickup_open_stream/norm_stats --commit 5 --plan_gpus 0 --population 128 --gpu_srun "$SRUN"
