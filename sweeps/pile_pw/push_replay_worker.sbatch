#!/bin/bash
#SBATCH --job-name=push_pw_replay
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action/slurm/push_pw_replay_%A_%a.out
# SAPIEN replay worker for the PointWorld push pool (untimed; any GPU). Waits while plans are still coming.
#   sbatch --array=0-7 --export=ALL,POOL=<pool> replay_worker.sbatch
cd /data/user_data/mbronars/packages/pwm_final_paper
/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python -u /data/user_data/mbronars/reports/pw_mpc/pool_worker_pw_push.py replay "$POOL" --max_minutes ${MAXMIN:-235}
