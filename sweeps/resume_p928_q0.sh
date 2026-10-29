#!/bin/bash
# Resume queue 0 on babel-p9-28 GPU 0 (job 10563761): own pickup server :8768, own Isaac/EGL replays on GPU 0.
cd /data/user_data/mbronars/packages/guided_playback
export CUDA_VISIBLE_DEVICES_ALL=$CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=0 nohup /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python -u bin/serve.py --adapter clean_pickup --ckpt /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt --port 8768 --warmup --warmup_transitions 60 --warmup_episodes 1,1,1,1 --warmup_iterations 1 > logs/serve_p928_gpu0_8768.log 2>&1 &
echo "server :8768 pid $! on $(hostname) gpu 0"
export GP_SRV=http://$(hostname):8768 GP_REPLAY_GPU=0
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
CER=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cereal_box/asset_0002
CAB=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/libero/wooden_cabinet/asset_0170
CR=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cracker_box/asset_0001
B1=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/bowl/asset_0001/asset_0001_visual.ply
COMMON='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":80,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
COMMON40='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
bash sweeps/gallery_chain2.sh gal_carton_air_cab '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.30,0,0.196",'$COMMON40',"upright_weight":0.3,"upright_frames":5,"desc":"stage 3 twin: milk carton to the same point in the air (+30 cm x, +19.6 cm), no cabinet, upright term 0.3; freeze 24, still 5, 40 it, shell"}' "0 1" > logs/gal_carton_air_cab_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_carton_over_crackers '{"domain":"pickup","task":"wall","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.35,0,0",'$COMMON',"box_weight":10,"clearance_mm":25,"wall_height":0.150,"wall_thickness":0.0525,"wall_width":0.328,"obstacle_asset":"'$CR'","obstacle_rpy":"0,0,90","obstacle_layout":"row:3","desc":"milk carton over a row of three cracker boxes (5.3 x 32.8 x 15.0 cm) midway, target +35 cm x on the ground; buffer 25 mm, wall weight 10; freeze 24, still 5, 80 it, shell, pop 4 per seed"}' "0 1" > logs/gal_carton_over_crackers_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_carton_above_air '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.30,0,0.15","transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"hand_start":"above","hand_above_mm":300,"desc":"milk carton, hand start moved to 30 cm straight above the object (orientation kept), air target +30 cm x +15 cm z; freeze 24, still 5, 40 it, shell, pop 4 per seed"}' "0 1" > logs/gal_carton_above_air_chain.log 2>&1
echo QUEUE0_DONE
