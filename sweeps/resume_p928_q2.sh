#!/bin/bash
# Resume queue 2 on babel-p9-28 GPU 2 (job 10563761): own pickup server :8770, own Isaac/EGL replays on GPU 2.
cd /data/user_data/mbronars/packages/guided_playback
export CUDA_VISIBLE_DEVICES_ALL=$CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=2 nohup /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python -u bin/serve.py --adapter clean_pickup --ckpt /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt --port 8770 --warmup --warmup_transitions 60 --warmup_episodes 1,1,1,1 --warmup_iterations 1 > logs/serve_p928_gpu2_8770.log 2>&1 &
echo "server :8770 pid $! on $(hostname) gpu 2"
export GP_SRV=http://$(hostname):8770 GP_REPLAY_GPU=2
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
CER=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cereal_box/asset_0002
CAB=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/libero/wooden_cabinet/asset_0170
CR=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/cracker_box/asset_0001
B1=/data/group_data/katefgroup-ssd/sim_scene_gen/assets/Kitchen_Table/bowl/asset_0001/asset_0001_visual.ply
COMMON='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":80,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
COMMON40='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24'
DSEL=/data/user_data/mbronars/packages/guided_playback/sweeps/diverse_select.json
for spec in "saltshaker:10:salt shaker:0,0.3,0.15:30 cm to +y" "orange:12:orange:-0.3,0,0.15:30 cm back" "pear:13:pear:0,0.3,0.15:30 cm to +y" "mokapot:14:moka pot:0,0.3,0.15:30 cm to +y"; do
  IFS=: read tag ep label off dirs <<< "$spec"
  REQ='{"domain":"pickup","task":"lift","select":"'$DSEL'","episodes":"'$ep,$ep,$ep,$ep'","transitions":60,"steps":10,"offset":"'$off'","optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"desc":"stage 2c: '"$label"', air target '"$dirs"' and 15 cm up, no obstacle; Adam lr 0.1, 40 it, shell, freeze_frames 24, still_weight 5 (frames <24), pop 4 per seed"}'
  bash sweeps/gallery_chain2.sh gal_${tag}_air_fr24 "$REQ" "0 1" > logs/gal_${tag}_air_fr24_chain.log 2>&1
done
bash sweeps/gallery_chain2.sh gal_lemon_air_fr24 '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"18,18,18,18","transitions":60,"steps":10,"offset":"0,0.30,0.15","optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"desc":"stage 2c: lemon, air target 30 cm to +y and 15 cm up, no obstacle; Adam lr 0.1, 40 it, shell, freeze_frames 24, still_weight 5 (frames <24), pop 4 per seed"}' "0 1" > logs/gal_lemon_air_fr24_chain.log 2>&1
echo QUEUE2_DONE
