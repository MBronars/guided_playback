#!/bin/bash
# After all four p9-28 queues report done: any gallery case with planned runs but no pick.json (e.g. a dropped client connection) gets its chain re-run
# (gallery_chain2 skips existing result.json / analysis.json / arm_all.done), using GPU 0's server for any missing plan.
cd /data/user_data/mbronars/packages/guided_playback
until [ $(grep -l "QUEUE[0-3]_DONE" logs/resume_p928_q0.log logs/resume_p928_q1.log logs/resume_p928_q2.log logs/resume_p928_q3.log 2>/dev/null | wc -l) -ge 4 ]; do sleep 60; done
export GP_SRV=http://babel-p9-28:8768 GP_REPLAY_GPU=0
for base in $(python3 -c "import json; print(' '.join(c['base'] for c in json.load(open('sweeps/gallery_cases.json')) if not c.get('failed')))"); do
  [ -f runs/pickup/${base}_pick.json ] && continue
  [ -f runs/pickup/${base}_s0/result.json ] || { echo "SKIP $base: never planned"; continue; }
  REQ=$(python3 -c "import json; d=json.load(open('runs/pickup/${base}_s0/result.json')); print(json.dumps({k:d[k] for k in ('domain','task','select','episodes','transitions','steps','offset','optimizer','lr','iterations','shell','freeze_frames','still_weight','still_frames','box_weight','clearance_mm','wall_height','wall_thickness','wall_width','wall_at','upright_weight','upright_frames','obstacle_asset','obstacle_rpy','obstacle_layout','object_mesh','hand_start','hand_above_mm','desc') if k in d and d[k] is not None}))")
  echo "FINISH $base"; bash sweeps/gallery_chain2.sh $base "$REQ" "0 1" > logs/${base}_chain_finish.log 2>&1; tail -1 logs/${base}_chain_finish.log
done
echo FINISH_MISSING_DONE
