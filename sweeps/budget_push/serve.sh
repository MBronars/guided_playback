#!/bin/bash
# Warm block-pushing server for the time-budget study. $1 = GPU, $2 = port.
cd /data/user_data/mbronars/packages/guided_playback
export CUDA_VISIBLE_DEVICES=$1
exec /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python -u bin/serve.py --adapter clean_push \
  --ckpt /data/user_data/mbronars/packages/pwm_final_clean/runs/push0418_rigid_x0_rope_regpass_mix_cont200k_stream/checkpoints/final.pt \
  --port $2
