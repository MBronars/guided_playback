#!/bin/bash
#SBATCH --job-name=push_diag
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/user_data/mbronars/packages/guided_playback/logs/push_diag_node_%j.out
# One warm block-pushing server per GPU (3 dedicated cores each, port 8840+g) + one queue worker per GPU (sweeps/queue/qworker.py).
# Jobs, workers and replays share the remaining cores. The agent fills /data/group_data/.../guided_playback_budget/queue/pending.
cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; H=$(hostname); ROOT=/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget
Q=$ROOT/queue; mkdir -p $Q/pending
CPUS=($($P -c "import os; print(' '.join(map(str, sorted(os.sched_getaffinity(0)))))")); NG=$(nvidia-smi -L | wc -l); CPS=3
REST=$(IFS=,; echo "${CPUS[*]:$((CPS * NG))}"); echo "cpus ${CPUS[*]} gpus $NG rest $REST"
for g in $(seq 0 $((NG - 1))); do
  MINE=$(IFS=,; echo "${CPUS[*]:$((CPS * g)):$CPS}")
  CUDA_VISIBLE_DEVICES=$g taskset -c $MINE $P -u bin/serve.py --adapter clean_push --runs $ROOT/runs \
    --ckpt /data/user_data/mbronars/packages/pwm_final_clean/runs/push0418_rigid_x0_rope_regpass_mix_cont200k_stream/checkpoints/final.pt \
    --port $((8840 + g)) > logs/push_diag_serve_${SLURM_JOB_ID}_g$g.log 2>&1 &
done
for g in $(seq 0 $((NG - 1))); do until curl -s http://$H:$((8840 + g))/status > /dev/null; do sleep 5; done; done
for g in $(seq 0 $((NG - 1))); do
  taskset -c $REST $P -u sweeps/queue/qworker.py $Q $g http://$H:$((8840 + g))/guide $REST > logs/push_diag_qworker_${SLURM_JOB_ID}_g$g.log 2>&1 &
done
wait
