#!/bin/bash
# Stage 2b (second pickup server, GPU 1, port 8769): air lifts (+15 cm) for five more objects whose start radius is 0.35-0.51 m; directions keep the target <= 0.66 m.
cd /data/user_data/mbronars/packages/guided_playback
export GP_SRV=http://babel-o9-24:8769
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
for spec in "teabox:6:tea box:0.30,0,0.15:30 cm forward" "juice:10:juice carton:0.30,0,0.15:30 cm forward" "tomato:19:tomato:0,0.30,0.15:30 cm to +y" "tuna:14:tuna can:0,-0.30,0.15:30 cm to -y" "mustard:15:mustard bottle:0.30,0,0.15:30 cm forward"; do
  IFS=: read tag ep label off dirs <<< "$spec"
  REQ='{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"'$ep,$ep,$ep,$ep'","transitions":60,"steps":10,"offset":"'$off'","optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"desc":"stage 2b: '"$label"', air target '"$dirs"' and 15 cm up, no obstacle; Adam lr 0.1, 40 it, shell, freeze_frames 24, still_weight 5 (frames <24), pop 4 per seed"}'
  bash sweeps/gallery_chain2.sh gal_${tag}_air_fr24 "$REQ" "0 1" > logs/gal_${tag}_air_fr24_chain.log 2>&1
done
echo STAGE2B_DONE
