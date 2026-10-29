#!/bin/bash
# All 24 unconditional draws (seed 0 = ground30_nowall run; seeds 1-5 = uncond_milk_carton_seed*) -> floating specs (if missing) -> arm replay + hull render, no X.
cd /data/user_data/mbronars/packages/guided_playback; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; CAM="--eye 0.70,-1.10,0.52 --look 0.42,0.20,0.21"
for s in 1 3 4 5; do R=runs/pickup/uncond_milk_carton_seed$s; [ -f $R/replay_isaac/analysis.json ] || CUDA_VISIBLE_DEVICES=1 $P -u bin/replay.py --run $R > logs/uncond_milk_carton_seed${s}_replay.log 2>&1; echo "float seed $s"; done
for R in runs/pickup/ground30_nowall_milk_carton_asset_0037 runs/pickup/uncond_milk_carton_seed1 runs/pickup/uncond_milk_carton_seed2 runs/pickup/uncond_milk_carton_seed3 runs/pickup/uncond_milk_carton_seed4 runs/pickup/uncond_milk_carton_seed5; do
  $P bin/replay_arm.py --run $R --members 0,1,2,3 --variants unguided --gpu 1 --asset none --no_target --no_obstacle $CAM 2>&1 | grep -E "^\[arm\] member|Error|Traceback"; echo "ARM $R"; done
echo GALLERY_REPLAYS_DONE
