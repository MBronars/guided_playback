#!/bin/bash
#SBATCH --job-name=nva_replayer
#SBATCH --partition=katef-cpu
#SBATCH --qos=katef_cpu_qos
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action_dev/slurm/replayer_%j.out
# Central Isaac replayer: the loop runs here on CPU; each Isaac launch goes into whichever claude-host GPU job is running
# (looked up at launch time, so the loop survives the host job being resubmitted). Cross-job srun needs this job's SLURM_* unset.
#   sbatch --export=ALL,POOL=<pool root> sweeps/noise_vs_action/replayer.sbatch
source /data/user_data/mbronars/packages/guided_playback/sweeps/noise_vs_action/env.sh
SRUN='unset $(env | grep -o "^SLURM_[A-Z_]*"); srun --jobid=$(squeue -u mbronars -h -n claude-host -t R -o %i | head -1) --overlap --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=0'
$PY -u -m gp.pool replay "$POOL" --gpu_srun "$SRUN" --every 120
