#!/bin/bash
# Stage 2: air lifts (+15 cm z) for four more objects, directions chosen so the target stays <= 0.62 m from the robot base.
cd /data/user_data/mbronars/packages/guided_playback
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
for spec in "carton:11:milk carton:0.30,0,0.15:30 cm forward" "pasta:9:pasta box:0.30,0,0.15:30 cm forward" "bottle:16:water bottle:0,-0.30,0.15:30 cm to -y" "can:13:soda can:0,0.30,0.15:30 cm to +y"; do
  IFS=: read tag ep label off dirs <<< "$spec"
  REQ='{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"'$ep,$ep,$ep,$ep'","transitions":60,"steps":10,"offset":"'$off'","optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"desc":"stage 2: '"$label"', air target '"$dirs"' and 15 cm up, no obstacle; Adam lr 0.1, 40 it, shell, freeze_frames 24, still_weight 5 (frames <24), pop 4 per seed"}'
  bash sweeps/gallery_chain.sh gal_${tag}_air_fr24 "$REQ" "0 1" > logs/gal_${tag}_air_fr24_chain.log 2>&1
done
echo STAGE2_DONE
