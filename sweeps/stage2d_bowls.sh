#!/bin/bash
# Bowls (synthetic scenes: the object's particles replaced by bowl-mesh points; planned on, replayed as): FIRST on GPU 1 after the mustard plans,
# server :8769 restarted for the object_mesh code; then the diverse-object queue (stage2c) follows.
cd /data/user_data/mbronars/packages/guided_playback
until [ $(grep -c "^GEN " logs/gal_mustard_air_fr24_chain.log 2>/dev/null) -ge 2 ] || grep -q "FAILED" logs/gal_mustard_air_fr24_chain.log 2>/dev/null; do sleep 20; done
until curl -s -m 5 http://babel-o9-24:8769/ | grep -q '"busy": false'; do sleep 10; done
kill 1429583; sleep 5
CUDA_VISIBLE_DEVICES=1 nohup /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python -u bin/serve.py --adapter clean_pickup --ckpt /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt --port 8769 --warmup --warmup_transitions 60 --warmup_episodes 1,1,1,1 --warmup_iterations 1 > logs/serve_pickonly_o924_gpu1_8769_v2.log 2>&1 &
echo "server 8769 restarted pid $!"
export GP_SRV=http://babel-o9-24:8769
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
B5=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/bowl/asset_0005/asset_0005_visual.ply
B1=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/bowl/asset_0001/asset_0001_visual.ply
COMMON='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
bash sweeps/gallery_chain2.sh gal_bowl5_air_fr24 '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"16,16,16,16","offset":"0,-0.30,0.15",'$COMMON',"object_mesh":"'$B5'","desc":"bowl (Kitchen_Table bowl asset_0005, 14.5 x 14.6 x 6.3 cm) in the water-bottle scene (object particles replaced by the bowl mesh points), air target 30 cm to -y and 15 cm up; freeze 24, still 5, 40 it, shell, pop 4 per seed"}' "0 1" > logs/gal_bowl5_air_fr24_chain.log 2>&1
bash sweeps/gallery_chain2.sh gal_bowl1_air_fr24 '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.30,0,0.15",'$COMMON',"object_mesh":"'$B1'","desc":"bowl (Kitchen_Table bowl asset_0001, 12.2 x 12.3 x 5.2 cm) in the milk-carton scene (object particles replaced by the bowl mesh points), air target 30 cm forward and 15 cm up; freeze 24, still 5, 40 it, shell, pop 4 per seed"}' "0 1" > logs/gal_bowl1_air_fr24_chain.log 2>&1
echo BOWLS_DONE
bash sweeps/stage2c_objects.sh
