#!/bin/bash
# Full-arm Isaac replay (table extended under the base) for every runs/rl_push/<prefix>* run that has its floating replay
# and no arm replay yet; loops until the driver log says "summary:" and everything is done.  usage: arm_loop.sh <prefix> <driver log>
cd /data/user_data/mbronars/packages/guided_playback; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
PREFIX=$1; DLOG=$2
while true; do
  did=0
  for d in runs/rl_push/${PREFIX}*/; do
    r=${d%/}; [ -f $r/replay_isaac/analysis.json ] || continue; [ -f $r/replay_isaac_arm/analysis.json ] && continue
    GP_ISAAC_EXTRA="--table_x0 -0.5" $P -u bin/replay.py --run $r --sim isaac_arm --variants guided --gpu 1 > logs/replay_$(basename $r)_arm.log 2>&1 || echo "ARM FAILED $r"
    echo "== $(basename $r) arm"; GP_TILT_SUFFIX=_arm $P sweeps/rl_push/tilt.py $r | grep "^\*"; did=1
  done
  if [ $did = 0 ]; then grep -q "^summary:" $DLOG 2>/dev/null && { echo ARM_LOOP_DONE; break; }; sleep 30; fi
done
