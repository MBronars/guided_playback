#!/bin/bash
cd /data/user_data/mbronars/packages/guided_playback; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
until grep -q BATCH3_DONE logs/replay_batch3.log 2>/dev/null; do sleep 5; done
for r in rlp_bowl5_e0_own_adam rlp_bowl5_e0_own_adam_up1 rlp_bowl5_e0_own_adam_up0.5_ct5; do
  $P -u bin/replay.py --run runs/rl_push/$r --variants guided --gpu 1 --suffix _coacd > logs/replay_${r}_coacd.log 2>&1 || { echo "FAILED $r"; tail -30 logs/replay_${r}_coacd.log; tail -20 runs/rl_push/$r/replay_isaac_coacd/replay.log; }
  echo "== $r coacd"; GP_TILT_SUFFIX=_coacd $P sweeps/rl_push/tilt.py runs/rl_push/$r
done
echo BATCH4_DONE
