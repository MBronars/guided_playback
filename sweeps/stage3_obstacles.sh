#!/bin/bash
# Stage 3 (queued behind stage 2): obstacle cases with the recipe (freeze 24 + still 5, 2 seeds, floating pick), each with a no-obstacle twin to the same target.
#  carton: over a cardboard (cereal) box standing on its long edge (2.2 cm thick, 29 cm long, 21 cm tall) midway, +35 cm forward, ground | twin
#  carton: onto the cabinet top (+30 cm forward, +19.6 cm), upright term 0.3 | twin (same target in the air)
#  jar:    over the cereal box sideways (+30 cm y, ground; rpy 0,90,0 puts the thin side along y, the rendered yaw is taken from obstacle_rpy) | twin
#  jar:    onto the cabinet sideways (+30 cm y, +19.6 cm), upright 0.3 | twin
cd /data/user_data/mbronars/packages/guided_playback
until grep -q STAGE2_DONE logs/stage2_objects.log 2>/dev/null; do sleep 30; done
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
CER=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cereal_box/asset_0002
CAB=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/libero/wooden_cabinet/asset_0170
COMMON='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
run(){ bash sweeps/gallery_chain2.sh $1 "$2" "0 1" > logs/$1_chain.log 2>&1; }
# carton over cereal box, +35 cm forward, ground
run gal_carton_over_cereal '{"domain":"pickup","task":"wall","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.35,0,0",'$COMMON',"box_weight":10,"clearance_mm":25,"wall_height":0.214,"wall_thickness":0.0217,"wall_width":0.291,"obstacle_asset":"'$CER'","obstacle_rpy":"0,90,90","obstacle_layout":"single","desc":"stage 3: milk carton over a cereal box standing on its long edge (2.2 x 29 x 21 cm) midway, target +35 cm x on the ground; buffer 25 mm, wall weight 10; freeze 24, still 5, 40 it, shell"}'
run gal_carton_fwd35 '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.35,0,0",'$COMMON',"desc":"stage 3 twin: milk carton to +35 cm x on the ground, no obstacle; freeze 24, still 5, 40 it, shell"}'
# carton onto the cabinet top with the upright term
run gal_carton_on_cabinet '{"domain":"pickup","task":"wall","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.30,0,0.196",'$COMMON',"box_weight":10,"clearance_mm":3,"wall_at":1.0,"wall_height":0.196,"wall_thickness":0.196,"wall_width":0.228,"upright_weight":0.3,"upright_frames":5,"obstacle_asset":"'$CAB'","obstacle_rpy":"0,0,90","obstacle_layout":"single","desc":"stage 3: milk carton placed on top of the wooden cabinet (+30 cm x, +19.6 cm), upright term 0.3; freeze 24, still 5, 40 it, shell"}'
run gal_carton_air_cab '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.30,0,0.196",'$COMMON',"upright_weight":0.3,"upright_frames":5,"desc":"stage 3 twin: milk carton to the same point in the air (+30 cm x, +19.6 cm), no cabinet, upright term 0.3; freeze 24, still 5, 40 it, shell"}'
# jar sideways over the cereal box and onto the cabinet
run gal_jar_over_cereal '{"domain":"pickup","task":"wall","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0",'$COMMON',"box_weight":10,"clearance_mm":25,"wall_height":0.214,"wall_thickness":0.0217,"wall_width":0.291,"obstacle_asset":"'$CER'","obstacle_rpy":"0,90,0","obstacle_layout":"single","desc":"stage 3: peanut butter jar over a cereal box on its long edge midway, target +30 cm y on the ground; buffer 25 mm, wall weight 10; freeze 24, still 5, 40 it, shell"}'
run gal_jar_side30 '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0",'$COMMON',"desc":"stage 3 twin: peanut butter jar to +30 cm y on the ground, no obstacle; freeze 24, still 5, 40 it, shell"}'
run gal_jar_on_cabinet '{"domain":"pickup","task":"wall","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0.196",'$COMMON',"box_weight":10,"clearance_mm":3,"wall_at":1.0,"wall_height":0.196,"wall_thickness":0.196,"wall_width":0.228,"upright_weight":0.3,"upright_frames":5,"obstacle_asset":"'$CAB'","obstacle_rpy":"0,0,90","obstacle_layout":"single","desc":"stage 3: peanut butter jar placed on top of the wooden cabinet to the side (+30 cm y, +19.6 cm), upright term 0.3; freeze 24, still 5, 40 it, shell"}'
run gal_jar_air_cab '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0.196",'$COMMON',"upright_weight":0.3,"upright_frames":5,"desc":"stage 3 twin: peanut butter jar to the same point in the air (+30 cm y, +19.6 cm), no cabinet, upright term 0.3; freeze 24, still 5, 40 it, shell"}'
echo STAGE3_DONE
