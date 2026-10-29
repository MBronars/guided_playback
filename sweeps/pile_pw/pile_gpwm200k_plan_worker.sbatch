#!/bin/bash
#SBATCH --job-name=pile_gpwm200k_plan
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --exclude=babel-s5-28   # 27 Oct: whole pool-32 DNO batches came out NaN there
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action/slurm/pile_gpwm200k_plan_%A_%a.out
# gpwm pile (mould drop50 final.pt) planning worker (one L40S, preempt + requeue, user 27 Oct): gp.pool work on the pile pool, no
# compile cache (the pick cache is for pickup); prewarm.json gives one untimed plan first.
#   sbatch --array=0-7 --export=ALL,POOL=<pool> pile_gpwm200k_plan_worker.sbatch
source /data/user_data/mbronars/packages/guided_playback/sweeps/noise_vs_action/env.sh
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/nva_ind_XXXX)
$PY -u -m gp.pool work "$POOL" --shapes "${SHAPES:-all}" --max_minutes ${MAXMIN:-230}
rm -rf $TORCHINDUCTOR_CACHE_DIR
