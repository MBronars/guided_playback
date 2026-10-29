#!/bin/bash
# block-pushing baselines, 5 hard seed-42 scenarios: GNS, PointWorld old config (pw, dynamic horizon cap 4 = paper May),
# PointWorld frame-60 cost (pw60), each open (commit 60) and closed (commit 10). Order: gns and pw60 first, pw (old) last.
cd /data/user_data/mbronars/packages/guided_playback; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; L=runs/pushing_baselines/logs
for m in gns pw60 pw; do for e in 5 11 14 29 37; do E=$(printf %03d $e); for mode in ol cl; do
  N=${m}_${mode}_hard42_ep${E}; [ -f runs/pushing_baselines/$N/summary.json ] && continue
  C=$([ $mode = ol ] && echo 60 || echo 10)
  $P -u bin/closed_loop_baseline_push.py --method $m --ep $e --commit $C --name $N > $L/$N.log 2>&1 || echo "FAIL $N"
  echo "DONE $N $(grep -o '"final_mm": [0-9.]*' runs/pushing_baselines/$N/summary.json) $(grep -o '"plan_seconds_total": [0-9.]*' runs/pushing_baselines/$N/summary.json)"
done; done; done; echo QUEUE_DONE
