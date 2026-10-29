#!/bin/bash
# S=10 arms on the same 15 diverse random/seed_42 scenarios (shell, full churn, population 3, corrected scenes).
set -x
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
U=http://babel-o9-24:8795/guide
SC=6,27,5,32,17,2,26,18,24,42,3,43,34,33,49
cd /data/user_data/mbronars/packages/guided_playback
run() { $P bin/paper_eval_push.py --url $U --difficulty random --seed 42 --scenarios $SC --members 3 --steps 10 --replay_gpu 0 "$@"; }
run --iterations 6  --lr 0.1  --lr_final -1   --tag sw15_S10_lr0.1_it6
run --iterations 6  --lr 0.2  --lr_final -1   --tag sw15_S10_lr0.2_it6
run --iterations 12 --lr 0.1  --lr_final -1   --tag sw15_S10_lr0.1_it12
run --iterations 10 --lr 0.2  --lr_final 0.02 --tag sw15_S10_lr0.2to0.02_it10
