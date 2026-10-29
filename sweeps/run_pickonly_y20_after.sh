#!/bin/bash
# wait for the +x chain (pid 3188845: guidance, import, Isaac replays) to end, then run +y on GPU 0 only
while kill -0 3188845 2>/dev/null; do sleep 20; done
O=/data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/guidance_streaming
[ -f $O/pick_y20z10_pop4.log ] && mv $O/pick_y20z10_pop4.log $O/pick_y20z10_pop4_OOM_gpu1.log
cd /data/user_data/mbronars/packages/pwm_final_clean
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python guidance/streaming_guidance.py --checkpoint /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt --roots /data/group_data/katefgroup/VLA/3DPCWM/robert_pickup_data_pick_success --select /data/user_data/mbronars/packages/guided_playback/sweeps/pickonly_pop4x3.json --batch_size 4 --offset=0,0.20,0.10 --iterations 16 --lr 0.1 --seed 0 --out $O/pick_y20z10_pop4 > $O/pick_y20z10_pop4.log 2>&1
echo "y20 exit $?"
cd /data/user_data/mbronars/packages/guided_playback; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
$P bin/import_clean_pickup.py $O/pick_y20z10_pop4 pickonly_y20z10
for r in runs/pickup/pickonly_y20z10_*; do CUDA_VISIBLE_DEVICES=0 $P -u bin/replay.py --run $r > logs/$(basename $r)_replay.log 2>&1; echo "$r replay $?"; done
