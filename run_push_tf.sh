#!/bin/bash
# t_freeze exploration (S=6, 6 it, shell, full churn, population 3). t_freeze 0.0 is banned (never optimize the clean rows).
set -x
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
U=http://babel-o9-24:8795/guide
SC=6,27,5,32,17,2,26,18,24,42,3,43,34,33,49
cd /data/user_data/mbronars/packages/guided_playback
# confirmation of the best arm on all 50
$P bin/paper_eval_push.py --url $U --difficulty random --seed 42 --n 50 --members 3 --steps 6 --replay_gpu 0 --no_warmup \
   --iterations 6 --lr 0.1 --lr_final -1 --t_freeze 0.5 --tag pushrand42_tf0.5_fix
run() { $P bin/paper_eval_push.py --url $U --difficulty random --seed 42 --scenarios $SC --members 3 --steps 6 --replay_gpu 0 --no_warmup "$@"; }
run --iterations 6 --lr 0.1 --lr_final -1 --t_freeze 0.3  --tag sw15_lr0.1_it6_tf0.3
run --iterations 6 --lr 0.1 --lr_final -1 --t_freeze 0.65 --tag sw15_lr0.1_it6_tf0.65
run --iterations 6 --lr 0.2 --lr_final -1 --t_freeze 0.5  --tag sw15_lr0.2_it6_tf0.5
run --iterations 6 --lr 0.05 --lr_final -1 --t_freeze 0.5 --tag sw15_lr0.05_it6_tf0.5
