#!/bin/bash
cd /data/user_data/mbronars/packages/guided_playback
until grep -q PASTA_ARM_ALL_DONE logs/gal_pasta_arm_all.log 2>/dev/null; do sleep 20; done
for s in 0 1; do touch runs/pickup/gal_pasta_air_fr24_s$s/replay_isaac_arm/isaac/arm_all.done; done
bash sweeps/gallery_repick.sh gal_pasta_air_fr24 "0 1" > logs/gal_pasta_air_fr24_repick.log 2>&1; echo REPICK_PASTA_DONE
until grep -q STAGE2_DONE logs/stage2_objects.log 2>/dev/null; do sleep 30; done
for b in gal_bottle_air_fr24 gal_can_air_fr24; do bash sweeps/gallery_repick.sh $b "0 1" > logs/${b}_repick.log 2>&1; echo "REPICK_DONE $b"; done
echo REPICK_STAGE2_DONE
