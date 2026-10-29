#!/bin/bash
#SBATCH --job-name=nva_cl_drivers
#SBATCH --partition=katef-cpu
#SBATCH --qos=katef_cpu_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action_dev/slurm/cl_drivers_%j.out
# Closed-loop table: one driver per planner (100-s open-loop settings), all through the pool_cl workers; Isaac launches go
# into the running claude-host GPU job, one at a time (flock).
source /data/user_data/mbronars/packages/guided_playback/sweeps/noise_vs_action/env.sh
P=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action/pool_cl
SRUN="unset \$(env | grep -o '^SLURM_[A-Z_]*'); flock $P/isaac.lock srun --jobid=\$(squeue -u mbronars -h -n claude-host -t R -o %i | head -1) --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=0"
for tag in dno_pool32_I5_P4 cem_noise_K32_R3 cem_act_K32_R3 bok_K128; do
  $PY -u bin/closed_loop_pick_pool.py --pool $P --tag $tag --gpu_srun "$SRUN" > $P/driver_$tag.log 2>&1 &
done
wait
