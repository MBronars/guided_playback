#!/bin/bash
# Shell-off and Adam arms on the same 15 scenarios (S=6, full churn -- churn-off is not implemented for clean_push).
set -x
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
U=http://babel-o9-24:8795/guide
SC=6,27,5,32,17,2,26,18,24,42,3,43,34,33,49
cd /data/user_data/mbronars/packages/guided_playback
run() { $P bin/paper_eval_push.py --url $U --difficulty random --seed 42 --scenarios $SC --members 3 --steps 6 --replay_gpu 0 --no_warmup "$@"; }
run --iterations 6 --lr 0.1  --lr_final -1 --no_shell                    --tag sw15_lr0.1_it6_noshell
run --iterations 6 --lr 0.05 --lr_final -1 --no_shell                    --tag sw15_lr0.05_it6_noshell
run --iterations 6 --lr 0.1  --lr_final -1 --optimizer adam              --tag sw15_adam0.1_it6
run --iterations 6 --lr 0.1  --lr_final -1 --optimizer adam --no_shell   --tag sw15_adam0.1_it6_noshell
