#!/bin/bash
# Stage 1: peanut butter jar, air target +30 cm forward +15 cm up, no obstacle; grasp-phase freeze 24 + stillness 5; pop 8 (seeds 0,1); plus an 80-iteration arm (seed 0).
cd /data/user_data/mbronars/packages/guided_playback
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
BASE='{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"1,1,1,1","transitions":60,"steps":10,"offset":"0.30,0,0.15","optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"desc":"stage 1: peanut butter jar, air target +30 cm x +15 cm z, no obstacle; Adam lr 0.1, 40 it, shell, freeze_frames 24, still_weight 5 (frames <24), pop 4 per seed"}'
bash sweeps/gallery_chain.sh gal_jar_air_fr24 "$BASE" "0 1" > logs/gal_jar_air_fr24_chain.log 2>&1
IT80=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['iterations']=80; d['desc']=d['desc'].replace('40 it','80 it'); print(json.dumps(d))" "$BASE")
bash sweeps/gallery_chain.sh gal_jar_air_fr24_it80 "$IT80" "0" > logs/gal_jar_air_fr24_it80_chain.log 2>&1
echo STAGE1_DONE
