#!/bin/bash
#SBATCH --job-name=pile_pw_plan_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action/slurm/pile_pw_plan_gen_%A_%a.out
# PointWorld + CEM pile planning worker (timed, L40S only). Submit as an array:
#   sbatch --array=0-11 --export=ALL,POOL=<pool> plan_worker.sbatch
cd /data/user_data/mbronars/packages/pwm_final_paper
/data/user_data/mbronars/.miniconda3/envs/pointworld-env/bin/python -u /data/user_data/mbronars/reports/pw_mpc/pool_worker_pw_pile.py plan "$POOL" --max_minutes ${MAXMIN:-2870}
