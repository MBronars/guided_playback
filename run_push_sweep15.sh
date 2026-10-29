#!/bin/bash
# Guidance-schedule sweep on 15 diverse random/seed_42 scenarios (evenly spaced in target angle, 58-123 mm).
# All S=6, shell, full churn, population 3, corrected scenes. Runs alongside the B / C / A arms on the same server.
set -x
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
U=http://babel-o9-24:8795/guide
SC=6,27,5,32,17,2,26,18,24,42,3,43,34,33,49
cd /data/user_data/mbronars/packages/guided_playback
run() { $P bin/paper_eval_push.py --url $U --difficulty random --seed 42 --scenarios $SC --members 3 --steps 6 --replay_gpu 0 --no_warmup "$@"; }
run --iterations 6  --lr 0.2  --lr_final -1   --tag sw15_lr0.2_it6
run --iterations 6  --lr 0.05 --lr_final -1   --tag sw15_lr0.05_it6
run --iterations 6  --lr 0.3  --lr_final -1   --tag sw15_lr0.3_it6
run --iterations 12 --lr 0.2  --lr_final 0.02 --tag sw15_lr0.2to0.02_it12
run --iterations 6  --lr 0.1  --lr_final -1   --t_freeze 0.5 --tag sw15_lr0.1_it6_tf0.5
run --iterations 12 --lr 0.05 --lr_final -1   --tag sw15_lr0.05_it12
