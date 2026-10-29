#!/bin/bash
# closed loop with ZERO executed context: each replan anchors on the current SAPIEN frame only (history_context 0),
# tail padded, reward on task frame 60, commit 10. $1 = server url, $2.. = scenario ids.
cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; S=/data/user_data/mbronars/packages/PointWorld/scenarios/hard/seed_42; U=$1; shift
for e in "$@"; do N=clctx0_hard42_ep${e}_k10
  [ -f runs/pushing/$N/summary.json ] && continue
  $P -u bin/closed_loop_push.py --url $U --scenario $S/ep_$e.npz --name $N --commit 10 --context 0 --min_plan 26 --pad_tail --replay_gpu 0 > logs/$N.log 2>&1
  echo "DONE $N $(grep -o '"sim_final_mm": [0-9.]*' runs/pushing/$N/summary.json)"
done; echo QUEUE_DONE
