#!/bin/bash
# Stage 2c (GPU 1 server :8769, after stage 2b): air lifts (+15 cm) for DIVERSE objects (sweeps/diverse_select.json), then the lemon, then the jar obstacle cases.
cd /data/user_data/mbronars/packages/guided_playback
until grep -q STAGE2B_DONE logs/stage2b_objects.log 2>/dev/null; do sleep 30; done
export GP_SRV=http://babel-o9-24:8769
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/diverse_select.json
for spec in "wineglass:0:wine glass:0,-0.3,0.15:30 cm to -y" "glass:2:drinking glass:0,0.3,0.15:30 cm to +y" "winebottle:5:wine bottle:0,0.3,0.15:30 cm to +y" "oilbottle:7:oil bottle:0,-0.3,0.15:30 cm to -y" "ketchup:8:ketchup bottle:0,-0.3,0.15:30 cm to -y" "saltshaker:10:salt shaker:0,0.3,0.15:30 cm to +y" "orange:12:orange:-0.3,0,0.15:30 cm back" "pear:13:pear:0,0.3,0.15:30 cm to +y" "mokapot:14:moka pot:0,0.3,0.15:30 cm to +y"; do
  IFS=: read tag ep label off dirs <<< "$spec"
  REQ='{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"'$ep,$ep,$ep,$ep'","transitions":60,"steps":10,"offset":"'$off'","optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"desc":"stage 2c: '"$label"', air target '"$dirs"' and 15 cm up, no obstacle; Adam lr 0.1, 40 it, shell, freeze_frames 24, still_weight 5 (frames <24), pop 4 per seed"}'
  bash sweeps/gallery_chain2.sh gal_${tag}_air_fr24 "$REQ" "0 1" > logs/gal_${tag}_air_fr24_chain.log 2>&1
done
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
REQ='{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"18,18,18,18","transitions":60,"steps":10,"offset":"0,0.30,0.15","optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"desc":"stage 2c: lemon, air target 30 cm to +y and 15 cm up, no obstacle; Adam lr 0.1, 40 it, shell, freeze_frames 24, still_weight 5 (frames <24), pop 4 per seed"}'
bash sweeps/gallery_chain2.sh gal_lemon_air_fr24 "$REQ" "0 1" > logs/gal_lemon_air_fr24_chain.log 2>&1
echo STAGE2C_DONE
bash sweeps/stage3b_jar.sh
