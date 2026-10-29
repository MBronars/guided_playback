#!/bin/bash
# Replay each assetsweep run in Isaac as soon as its rollout.npz exists (sequential, GPU 0).
ROOT=/data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
cd $ROOT
LABELS=$($P -c "import json; print(' '.join(e['label'] for e in json.load(open('sweeps/assetsweep_select.json'))))")
GPID=$(cat sweeps/assetsweep_tools/guidance.pid)
for L in $LABELS; do
  R=runs/pickup/assetsweep_$L
  until [ -f $R/rollout.npz ] || ! kill -0 $GPID 2>/dev/null; do sleep 15; done
  [ -f $R/rollout.npz ] || { echo "== $L: no rollout"; continue; }
  if [ -f $R/replay_isaac/analysis.json ]; then echo "== $L: replay exists, skip"; continue; fi
  sleep 5
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=0 $P bin/replay.py --run $R --gpu 0 > sweeps/assetsweep_tools/replay_$L.log 2>&1
  echo "== $L: replay rc $? in $(( $(date +%s) - t0 )) s"
done
echo "replays done"
