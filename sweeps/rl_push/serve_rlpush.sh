#!/bin/bash
# rl_push server: dedicated rl_push checkpoint step_180000, compiled, GPU 1 of babel-l9-16 (job 10560734), port 8797.
# Warmup compiles the grad + no-grad graphs at population B=12 (static shapes: keep every request at 12 members).
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
cd /data/user_data/mbronars/packages/guided_playback
GP_RELOAD=1 CUDA_VISIBLE_DEVICES=1 exec $P -u bin/serve.py --adapter paper \
  --ckpt /data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_rl_push_ctxgen_100m_uncond_unet_bidir/checkpoints/step_180000.pt \
  --port 8797 --warmup --warmup_domain rl_push --warmup_transitions 60 --warmup_episodes 0,0,0,0,0,0,0,0,0,0,0,0 --warmup_iterations 1
