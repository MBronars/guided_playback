#!/bin/bash
cd /data/user_data/mbronars/packages/guided_playback; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
$P -u bin/replay.py --run runs/rl_push/rlp_bowl5_e0_own_adam_up1 --variants guided,unguided --gpu 1 > logs/replay_rlp_bowl5_e0_own_adam_up1.log 2>&1
echo "== up1"; $P sweeps/rl_push/tilt.py runs/rl_push/rlp_bowl5_e0_own_adam_up1
GP_ISAAC_EXTRA="--kp_pos 100 --kp_rot 100" $P -u bin/replay.py --run runs/rl_push/rlp_bowl5_e0_own_adam --variants guided --gpu 1 --suffix _kp100 > logs/replay_rlp_bowl5_e0_own_adam_kp100.log 2>&1
echo "== adam kp100"; GP_TILT_SUFFIX=_kp100 $P sweeps/rl_push/tilt.py runs/rl_push/rlp_bowl5_e0_own_adam
echo BATCH1_DONE
