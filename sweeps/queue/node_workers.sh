#!/bin/bash
# Queue workers only (no resident servers: every job starts its own fresh server via with_server*.sh), one per GPU of this job.
# Writes this node's per-GPU dedicated-core map (3 cores per GPU for the job's server) and confines workers + replays to the rest.
cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; H=$(hostname); Q=/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget/queue
CPUS=($($P -c "import os; print(' '.join(map(str, sorted(os.sched_getaffinity(0)))))")); NG=$(nvidia-smi -L | wc -l); CPS=3
$P -c "
import json; c='${CPUS[*]}'.split(); json.dump({str(g): ','.join(c[3*g:3*g+3]) for g in range($NG)}, open('$Q/cores_${H}_${SLURM_JOB_ID}.json','w'))"
REST=$(IFS=,; echo "${CPUS[*]:$((CPS * NG))}"); echo "$H gpus $NG rest $REST $(date)"
for g in $(seq 0 $((NG - 1))); do
  taskset -c $REST $P -u sweeps/queue/qworker.py $Q $g none $REST > logs/push_diag_qworker_${SLURM_JOB_ID}_g$g.log 2>&1 &
done
wait
