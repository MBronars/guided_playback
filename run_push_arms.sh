#!/bin/bash
# Two guidance schedules on the paper's random/seed_42 PushBoundary scenarios, same warm server, same scenarios.
set -x
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
U=http://babel-o9-24:8795/guide
cd /data/user_data/mbronars/packages/guided_playback
# Arm A: the validated push config (decaying lr)
$P bin/paper_eval_push.py --url $U --difficulty random --seed 42 --n 50 --members 3 \
   --steps 8 --iterations 10 --lr 0.2 --lr_final 0.02 --replay_gpu 0 --tag pushrand42_decay
# Arm B: the mould-settled config (constant lr, fewer steps and iterations)
$P bin/paper_eval_push.py --url $U --difficulty random --seed 42 --n 50 --members 3 \
   --steps 6 --iterations 6 --lr 0.1 --lr_final -1 --replay_gpu 0 --tag pushrand42_flat
