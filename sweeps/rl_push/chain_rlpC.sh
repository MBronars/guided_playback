#!/bin/bash
# after the rlpB driver finishes: recipe C = same as B but 40 transitions (2 s horizon) on the same 18 cases, then its arm loop
cd /data/user_data/mbronars/packages/guided_playback; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
until grep -q "^summary:" logs/rlpB_driver.log 2>/dev/null; do sleep 20; done
$P -u sweeps/rl_push/run_cases.py --tag rlpC --cases 13,4,42,66,27,63,81,146,104,151,8,70,53,161,113,58,28,93 --members 12 --transitions 40 --steps 10 --iterations 10 --lr 0.2 --lr_final 0.02 --t_freeze 0.0 --optimizer adam --goal own --upright_weight 1.0 --upright_frames 40 --extra '{"contact_weight": 2.0}' --replay_gpu 1 --replay_variants guided > logs/rlpC_driver.log 2>&1
