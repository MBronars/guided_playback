#!/bin/bash
#SBATCH --job-name=budget_push
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --cpus-per-task=32
# 2-GPU job: sbatch --gres=gpu:L40S:2 --cpus-per-task=16 --mem=96G sweeps/budget_push/node.sbatch
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=/data/user_data/mbronars/packages/guided_playback/logs/budget_node_%j.out
# One warm push server per GPU + one cell worker per server (docs/BUDGET_PUSH_PLAN.md).
# Planning is kernel-launch bound at these batch sizes, so each server gets 3 DEDICATED cores; workers and every SAPIEN
# replay are confined to the remaining cores so replays never steal CPU from a timed plan.
cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; H=$(hostname); ROOT=/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget
CPUS=($($P -c "import os; print(' '.join(map(str, sorted(os.sched_getaffinity(0)))))")); echo "cpus ${CPUS[*]}"
NG=$(nvidia-smi -L | wc -l); CPS=${CPS:-3}; REST=$(IFS=,; echo "${CPUS[*]:$((CPS * NG))}"); echo "gpus $NG, $CPS cores per server"
for g in $(seq 0 $((NG - 1))); do
  MINE=$(IFS=,; echo "${CPUS[*]:$((CPS * g)):$CPS}")
  CUDA_VISIBLE_DEVICES=$g taskset -c $MINE $P -u bin/serve.py --adapter clean_push --runs $ROOT/runs \
    --ckpt /data/user_data/mbronars/packages/pwm_final_clean/runs/push0418_rigid_x0_rope_regpass_mix_cont200k_stream/checkpoints/final.pt \
    --port $((8830 + g)) > logs/budget_serve_${SLURM_JOB_ID}_g$g.log 2>&1 &
  SERVERS="$SERVERS $!"
  echo "server g$g cores $MINE"
done
for g in $(seq 0 $((NG - 1))); do until curl -s http://$H:$((8830 + g))/status > /dev/null; do sleep 5; done; done
echo "workers + replays on cores $REST"
for g in $(seq 0 $((NG - 1))); do
  taskset -c $REST $P -u sweeps/budget_push/worker.py --cells ${CELLS:-sweeps/budget_push/cells.json} --url http://$H:$((8830 + g))/guide --replay_gpu $g --replays ${REPLAYS:-4} > logs/budget_worker_${SLURM_JOB_ID}_g$g.log 2>&1 &
  WORKERS="$WORKERS $!"
  sleep 20
done
wait $WORKERS
echo ALL_WORKERS_DONE; kill $SERVERS
