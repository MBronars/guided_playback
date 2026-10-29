#!/bin/bash
# Resume queue 1 on babel-p9-28 GPU 1 (job 10563761): own pickup server :8769, own Isaac/EGL replays on GPU 1.
cd /data/user_data/mbronars/packages/guided_playback
export CUDA_VISIBLE_DEVICES_ALL=$CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=1 nohup /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python -u bin/serve.py --adapter clean_pickup --ckpt /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt --port 8769 --warmup --warmup_transitions 60 --warmup_episodes 1,1,1,1 --warmup_iterations 1 > logs/serve_p928_gpu1_8769.log 2>&1 &
echo "server :8769 pid $! on $(hostname) gpu 1"
export GP_SRV=http://$(hostname):8769 GP_REPLAY_GPU=1
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
CER=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cereal_box/asset_0002
CAB=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/libero/wooden_cabinet/asset_0170
CR=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cracker_box/asset_0001
B1=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/bowl/asset_0001/asset_0001_visual.ply
COMMON='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":80,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
COMMON40='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
bash sweeps/gallery_chain2.sh gal_bowl1_air_fr24 '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.30,0,0.15",'$COMMON40',"object_mesh":"'$B1'","desc":"bowl (Kitchen_Table bowl asset_0001, 12.2 x 12.3 x 5.2 cm) in the milk-carton scene (object particles replaced by the bowl mesh points), air target 30 cm forward and 15 cm up; freeze 24, still 5, 40 it, shell, pop 4 per seed"}' "0 1" > logs/gal_bowl1_air_fr24_chain.log 2>&1
DSEL=/data/user_data/mbronars/packages/guided_playback/sweeps/diverse_select.json
for spec in "wineglass:0:wine glass:0,-0.3,0.15:30 cm to -y" "glass:2:drinking glass:0,0.3,0.15:30 cm to +y" "winebottle:5:wine bottle:0,0.3,0.15:30 cm to +y" "oilbottle:7:oil bottle:0,-0.3,0.15:30 cm to -y" "ketchup:8:ketchup bottle:0,-0.3,0.15:30 cm to -y"; do
  IFS=: read tag ep label off dirs <<< "$spec"
  REQ='{"domain":"pickup","task":"lift","select":"'$DSEL'","episodes":"'$ep,$ep,$ep,$ep'","transitions":60,"steps":10,"offset":"'$off'","optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"desc":"stage 2c: '"$label"', air target '"$dirs"' and 15 cm up, no obstacle; Adam lr 0.1, 40 it, shell, freeze_frames 24, still_weight 5 (frames <24), pop 4 per seed"}'
  bash sweeps/gallery_chain2.sh gal_${tag}_air_fr24 "$REQ" "0 1" > logs/gal_${tag}_air_fr24_chain.log 2>&1
done
echo QUEUE1_DONE
