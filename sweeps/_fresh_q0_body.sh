cd /data/user_data/mbronars/packages/guided_playback
export GP_SRV=http://$(hostname):8768 GP_REPLAY_GPU=0
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
CER=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cereal_box/asset_0002
CAB=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/libero/wooden_cabinet/asset_0170
CR=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cracker_box/asset_0001
B1=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/bowl/asset_0001/asset_0001_visual.ply
DSEL=/data/user_data/mbronars/packages/guided_playback/sweeps/diverse_select.json
COMMON='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":80,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
COMMON40='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
C='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"hand_start":"above","hand_above_mm":300'
GP_RENDER_ONLY=1 bash sweeps/gallery_repick.sh gal_jar_above_air "0 1" > logs/gal_jar_above_air_rerender.log 2>&1
bash sweeps/gallery_chain2.sh gal_bottle_above_air '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"16,16,16,16","offset":"0,-0.30,0.15",'$C',"desc":"water bottle, hand start 30 cm straight above the object, air target -30 cm y +15 cm z; freeze 24, still 5, 40 it, shell, pop 4 per seed"}' "0 1" > logs/gal_bottle_above_air_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_can_above_air '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"13,13,13,13","offset":"0,0.30,0.15",'$C',"desc":"soda can, hand start 30 cm straight above the object, air target +30 cm y +15 cm z; freeze 24, still 5, 40 it, shell, pop 4 per seed"}' "0 1" > logs/gal_can_above_air_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_wineglass_above_air '{"domain":"pickup","task":"lift","select":"'$DSEL'","episodes":"0,0,0,0","offset":"0,-0.30,0.15",'$C',"desc":"wine glass, hand start 30 cm straight above the object, air target -30 cm y +15 cm z; freeze 24, still 5, 40 it, shell, pop 4 per seed"}' "0 1" > logs/gal_wineglass_above_air_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_jar_side30 '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0",'$COMMON40',"desc":"stage 3 twin: peanut butter jar to +30 cm y on the ground, no obstacle; freeze 24, still 5, 40 it, shell"}' "0 1" > logs/gal_jar_side30_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_jar_on_cabinet '{"domain":"pickup","task":"wall","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0.196",'$COMMON',"box_weight":10,"clearance_mm":3,"wall_at":1.0,"wall_height":0.196,"wall_thickness":0.196,"wall_width":0.228,"upright_weight":0.3,"upright_frames":5,"obstacle_asset":"'$CAB'","obstacle_rpy":"0,0,90","obstacle_layout":"single","desc":"stage 3: peanut butter jar placed on top of the wooden cabinet to the side (+30 cm y, +19.6 cm), upright term 0.3; freeze 24, still 5, 80 it, shell"}' "0 1" > logs/gal_jar_on_cabinet_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_jar_air_cab '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0.196",'$COMMON40',"upright_weight":0.3,"upright_frames":5,"desc":"stage 3 twin: peanut butter jar to the same point in the air (+30 cm y, +19.6 cm), no cabinet, upright term 0.3; freeze 24, still 5, 40 it, shell"}' "0 1" > logs/gal_jar_air_cab_chain.log 2>&1
echo QUEUE0_FRESH_DONE
