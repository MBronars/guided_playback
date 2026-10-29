#!/bin/bash
# old-config PointWorld (paper May settings) open + closed on the given scenarios, on babel-q9-28 GPU $1 (server port 519$1); SAPIEN on the same GPU.
cd /data/user_data/mbronars/packages/guided_playback; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; L=runs/pushing_baselines/logs; G=$1; shift
for e in "$@"; do E=$(printf %03d $e); for mode in ol cl; do
  N=pw_${mode}_hard42_ep${E}; [ -f runs/pushing_baselines/$N/summary.json ] && continue; [ -f runs/pushing_baselines/$N.claimed ] && continue; touch runs/pushing_baselines/$N.claimed
  C=$([ $mode = ol ] && echo 60 || echo 10)
  $P -u bin/closed_loop_baseline_push.py --method pw --ep $e --commit $C --name $N --gpu $G --pw_url http://127.0.0.1:519$G > $L/$N.log 2>&1 || echo "FAIL $N"
  echo "DONE $N $(grep -o '"final_mm": [0-9.]*' runs/pushing_baselines/$N/summary.json) $(grep -o '"plan_seconds_total": [0-9.]*' runs/pushing_baselines/$N/summary.json)"
done; done; echo QUEUE_DONE
