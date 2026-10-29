#!/bin/bash
# Resume queue 3 on babel-p9-28 GPU 3 (job 10563761): own pickup server :8771, own Isaac/EGL replays on GPU 3.
cd /data/user_data/mbronars/packages/guided_playback
export CUDA_VISIBLE_DEVICES_ALL=$CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=3 nohup /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python -u bin/serve.py --adapter clean_pickup --ckpt /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt --port 8771 --warmup --warmup_transitions 60 --warmup_episodes 1,1,1,1 --warmup_iterations 1 > logs/serve_p928_gpu3_8771.log 2>&1 &
echo "server :8771 pid $! on $(hostname) gpu 3"
export GP_SRV=http://$(hostname):8771 GP_REPLAY_GPU=3
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
CER=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cereal_box/asset_0002
CAB=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/libero/wooden_cabinet/asset_0170
CR=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cracker_box/asset_0001
B1=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/bowl/asset_0001/asset_0001_visual.ply
COMMON='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":80,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
COMMON40='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
bash sweeps/gallery_chain2.sh gal_jar_over_cereal '{"domain":"pickup","task":"wall","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0",'$COMMON',"box_weight":10,"clearance_mm":25,"wall_height":0.214,"wall_thickness":0.0217,"wall_width":0.291,"obstacle_asset":"'$CER'","obstacle_rpy":"0,90,0","obstacle_layout":"single","desc":"stage 3: peanut butter jar over a cereal box on its long edge midway, target +30 cm y on the ground; buffer 25 mm, wall weight 10; freeze 24, still 5, 80 it, shell"}' "0 1" > logs/gal_jar_over_cereal_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_jar_side30 '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0",'$COMMON40',"desc":"stage 3 twin: peanut butter jar to +30 cm y on the ground, no obstacle; freeze 24, still 5, 40 it, shell"}' "0 1" > logs/gal_jar_side30_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_jar_on_cabinet '{"domain":"pickup","task":"wall","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0.196",'$COMMON',"box_weight":10,"clearance_mm":3,"wall_at":1.0,"wall_height":0.196,"wall_thickness":0.196,"wall_width":0.228,"upright_weight":0.3,"upright_frames":5,"obstacle_asset":"'$CAB'","obstacle_rpy":"0,0,90","obstacle_layout":"single","desc":"stage 3: peanut butter jar placed on top of the wooden cabinet to the side (+30 cm y, +19.6 cm), upright term 0.3; freeze 24, still 5, 80 it, shell"}' "0 1" > logs/gal_jar_on_cabinet_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_jar_air_cab '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0.196",'$COMMON40',"upright_weight":0.3,"upright_frames":5,"desc":"stage 3 twin: peanut butter jar to the same point in the air (+30 cm y, +19.6 cm), no cabinet, upright term 0.3; freeze 24, still 5, 40 it, shell"}' "0 1" > logs/gal_jar_air_cab_chain.log 2>&1
echo QUEUE3_DONE
