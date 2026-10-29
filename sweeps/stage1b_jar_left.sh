#!/bin/bash
# Stage 1b: same peanut butter jar, target +30 cm y (sideways) +15 cm up: hand radius stays <= 0.6 m (the +30 cm x target was at 0.90 m, past the arm's reach).
cd /data/user_data/mbronars/packages/guided_playback
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
REQ='{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"1,1,1,1","transitions":60,"steps":10,"offset":"0,0.30,0.15","optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"desc":"stage 1b: peanut butter jar, air target +30 cm y +15 cm z (within arm reach), no obstacle; Adam lr 0.1, 40 it, shell, freeze_frames 24, still_weight 5 (frames <24), pop 4 per seed"}'
bash sweeps/gallery_chain.sh gal_jar_left_fr24 "$REQ" "0 1" > logs/gal_jar_left_fr24_chain.log 2>&1
echo STAGE1B_DONE
