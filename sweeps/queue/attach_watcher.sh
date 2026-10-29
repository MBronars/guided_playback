#!/bin/bash
# Every 2 min: (1) start queue workers inside every RUNNING push_diag job not yet attached (srun --overlap);
# (2) re-queue jobs whose node is no longer held by a running push_diag job (the runners resume from finished scenarios).
Q=/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget/queue; M=$Q/attached; mkdir -p $M
while true; do
  for J in $(squeue -u mbronars -h -n push_diag -t R -o %i); do
    [ -f $M/$J ] && continue
    C=$(squeue -j $J -h -o %C); G=$(( C >= 32 ? 4 : 2 ))
    setsid nohup srun --jobid $J --overlap -c $C --mem=60G --cpu-bind=none --gres=gpu:L40S:$G bash /data/user_data/mbronars/packages/guided_playback/sweeps/queue/node_workers.sh > /data/user_data/mbronars/packages/guided_playback/logs/attach_$J.log 2>&1 < /dev/null &
    touch $M/$J; echo "$(date) attached $J ($G GPUs, $C cpus)"
  done
  LIVE=" $(squeue -u mbronars -h -n push_diag -t R -o %N | tr '\n' ' ') "
  for f in $Q/running/*.json; do
    [ -e "$f" ] || continue
    b=$(basename $f .json); host=${b##*__}; host=${host%_g*}
    case "$LIVE" in *" $host "*) ;; *)
      mv "$f" "$Q/pending/${b%__*}.json" && echo "$(date) requeued ${b%__*} (node $host gone)";;
    esac
  done
  sleep 120
done
