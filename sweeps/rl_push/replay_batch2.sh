#!/bin/bash
cd /data/user_data/mbronars/packages/guided_playback; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
GP_ISAAC_MU=0.5 $P -u bin/replay.py --run runs/rl_push/rlp_bowl5_e0_own_adam --variants guided --gpu 1 --suffix _mu0.5 > logs/replay_rlp_bowl5_e0_own_adam_mu0.5.log 2>&1
echo "== adam mu0.5"; GP_TILT_SUFFIX=_mu0.5 $P sweeps/rl_push/tilt.py runs/rl_push/rlp_bowl5_e0_own_adam
GP_ISAAC_MU=0.5 $P -u bin/replay.py --run runs/rl_push/rlp_bowl5_e0_own_adam_up1 --variants guided --gpu 1 --suffix _mu0.5 > logs/replay_rlp_bowl5_e0_own_adam_up1_mu0.5.log 2>&1
echo "== up1 mu0.5"; GP_TILT_SUFFIX=_mu0.5 $P sweeps/rl_push/tilt.py runs/rl_push/rlp_bowl5_e0_own_adam_up1
echo BATCH2_DONE
