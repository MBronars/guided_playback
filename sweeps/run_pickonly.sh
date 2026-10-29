#!/bin/bash
# pickup-only checkpoint (cont200k final.pt): pick to a diagonal target, 3 val episodes x population 4, the earlier recipe (Adam lr 0.1, 16 it, streaming 25 steps/frame)
cd /data/user_data/mbronars/packages/pwm_final_clean
export PATH=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin:$PATH PYTHONPATH=.
CK=/data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt
R=/data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_pick_success
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/pickonly_pop4x3.json
CUDA_VISIBLE_DEVICES=0 python guidance/streaming_guidance.py --checkpoint $CK --roots $R --select $SEL --batch_size 4 --offset=0.20,0,0.10 --iterations 16 --lr 0.1 --seed 0 --out /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/guidance_streaming/pick_x20z10_pop4 > /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/guidance_streaming/pick_x20z10_pop4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python guidance/streaming_guidance.py --checkpoint $CK --roots $R --select $SEL --batch_size 4 --offset=0,0.20,0.10 --iterations 16 --lr 0.1 --seed 0 --out /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/guidance_streaming/pick_y20z10_pop4 > /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/guidance_streaming/pick_y20z10_pop4.log 2>&1 &
wait
