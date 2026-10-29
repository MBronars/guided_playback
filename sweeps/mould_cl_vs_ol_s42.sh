#!/bin/bash
# pile manipulation, cloud-centroid seed 42 scenarios 1..4: $1 = closed|open. Server babel-l9-20:8810 (mould step_60000), Genesis on this node's GPU 0.
cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; S=/data/user_data/mbronars/packages/pwm_efficient/scenarios/cloud_centroid/seed_42
until grep -q CASE0_DONE logs/mould_case0.log; do sleep 20; done
for e in 001 002 003 004; do
  if [ "$1" = closed ]; then N=clmould_s42_ep${e}_k25; C=25; else N=olmould_s42_ep${e}; C=150; fi
  [ -f runs/mould/$N/summary.json ] && continue
  $P -u bin/closed_loop_mould.py --url http://babel-l9-20:8810/guide --scenario $S/ep_$e.npz --name $N --commit $C --replay_gpu 0 > logs/$N.log 2>&1
  echo "DONE $N $(grep -o '"sim_final_mm": [0-9.]*' runs/mould/$N/summary.json)"
done; echo QUEUE_DONE $1
