#!/bin/bash
#SBATCH --job-name=nva_pw_worker
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action_dev/slurm/pw_worker_%j.out
# PointWorld + CEM reference curve: one L40S worker pulling units of the PointWorld pool (gp.pool layout).
#   sbatch --export=ALL,POOL=<pool root> sweeps/noise_vs_action/worker_pw.sbatch
cd /data/user_data/mbronars/packages/pwm_final_paper
/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python -u /data/user_data/mbronars/reports/pw_mpc/pool_worker_pw.py "$POOL" --max_minutes ${MAXMIN:-235}
