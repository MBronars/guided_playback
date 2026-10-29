#!/bin/bash
#SBATCH --job-name=pile_gpwm_plan_gen
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action_dev/slurm/pile_gpwm_plan_gen_%A_%a.out
# One planning worker (one L40S): loads the model + compile cache once, then claims pool units until none are left for its
# shapes or its time is nearly up. Preemption = the process dies; its claim's lease expires and another worker redoes it.
#   sbatch --export=ALL,POOL=<pool root>,SHAPES=<s1,s2> sweeps/noise_vs_action/worker.sbatch
source /data/user_data/mbronars/packages/guided_playback/sweeps/noise_vs_action/env.sh
export TORCHINDUCTOR_CACHE_DIR=$(mktemp -d /tmp/nva_ind_XXXX)
$PY -u -m gp.pool work "$POOL" --shapes "${SHAPES:-all}" --cache $NVA_CACHE --max_minutes ${MAXMIN:-235}
rm -rf $TORCHINDUCTOR_CACHE_DIR
