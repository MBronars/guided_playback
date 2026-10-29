#!/bin/bash
# 5 hard seed-42 scenarios (the paper's pushing video episodes), push0418 cont200k_stream final.pt, tail-padded sampler.
# $1 = mode (closed|open), $2 = server url. closed: replan every 10 actions; open: one 60-frame plan, all 60 executed.
cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; S=/data/user_data/mbronars/packages/PointWorld/scenarios/hard/seed_42
for e in 005 011 014 029 037; do
  if [ "$1" = closed ]; then N=clpad_hard42_ep${e}_k10; C=10; else N=olpad_hard42_ep${e}; C=60; fi
  [ -f runs/pushing/$N/summary.json ] && continue
  $P -u bin/closed_loop_push.py --url $2 --scenario $S/ep_$e.npz --name $N --commit $C --context 3 --min_plan 26 --pad_tail --replay_gpu 0 > logs/$N.log 2>&1
  echo "DONE $N $(grep -o '"sim_final_mm": [0-9.]*' runs/pushing/$N/summary.json)"
done
echo "QUEUE_DONE $1"
