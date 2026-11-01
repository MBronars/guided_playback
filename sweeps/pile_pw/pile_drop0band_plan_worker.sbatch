#!/bin/bash
#SBATCH --job-name=pile_drop0band_plan
#SBATCH --partition=general
#SBATCH --exclude=babel-s5-28   # 27 Oct: whole pool-32 DNO batches came out NaN there
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action/slurm/pile_gpwm200k_gen_%A_%a.out
# planning worker for the always-conditioned drop0-band pile checkpoint (act_cmd + ctx_level, 31 Oct)
# compile cache (the pick cache is for pickup); prewarm.json gives one untimed plan first.
#   sbatch --array=0-7 --export=ALL,POOL=<pool> pile_gpwm200k_plan_worker.sbatch
source /data/user_data/mbronars/packages/guided_playback/sweeps/noise_vs_action/env.sh
export GP_PAPER_REPO=/data/user_data/mbronars/packages/pwm_final_wt/pile-drop0-band   # the band key in the drop0-band recipe validates only on this branch
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/nva_ind_XXXX)
$PY -u -m gp.pool work "$POOL" --shapes "${SHAPES:-all}" --max_minutes ${MAXMIN:-1400}
rm -rf $TORCHINDUCTOR_CACHE_DIR
