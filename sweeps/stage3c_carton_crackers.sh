#!/bin/bash
# After the carton cabinet cases: carton over a ROW of three cracker boxes (15 cm tall, 33 cm wide, 5 cm thick; a lower cardboard barrier than the 21 cm cereal box), +35 cm forward, ground; 80 it. Twin = gal_carton_fwd35 (exists).
cd /data/user_data/mbronars/packages/guided_playback
until grep -q STAGE3B_CARTON_DONE logs/stage3b_carton.log 2>/dev/null; do sleep 30; done
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
CR=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cracker_box/asset_0001
COMMON='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":80,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
bash sweeps/gallery_chain2.sh gal_carton_over_crackers '{"domain":"pickup","task":"wall","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.35,0,0",'$COMMON',"box_weight":10,"clearance_mm":25,"wall_height":0.150,"wall_thickness":0.0525,"wall_width":0.328,"obstacle_asset":"'$CR'","obstacle_rpy":"0,0,90","obstacle_layout":"row:3","desc":"milk carton over a row of three cracker boxes (5.3 x 32.8 x 15.0 cm) midway, target +35 cm x on the ground; buffer 25 mm, wall weight 10; freeze 24, still 5, 80 it, shell, pop 4 per seed"}' "0 1" > logs/gal_carton_over_crackers_chain.log 2>&1
echo STAGE3C_DONE
