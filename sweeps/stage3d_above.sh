#!/bin/bash
# Straight-above hand start test (user question 2026-09-25): carton, hand moved to 30 cm directly above the object at frame 0, air target +30 cm x +15 cm z.
# Runs after the cracker-row case; restarts the GPU 0 server first (gp/session.py changed: hand_start=above, object_mesh).
cd /data/user_data/mbronars/packages/guided_playback
until grep -q STAGE3C_DONE logs/stage3c_carton_crackers.log 2>/dev/null; do sleep 30; done
until curl -s -m 5 http://babel-o9-24:8768/ | grep -q '"busy": false'; do sleep 10; done
kill 1247644; sleep 5
CUDA_VISIBLE_DEVICES=0 nohup /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python -u bin/serve.py --adapter clean_pickup --ckpt /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt --port 8768 --warmup --warmup_transitions 60 --warmup_episodes 1,1,1,1 --warmup_iterations 1 > logs/serve_pickonly_o924_v7.log 2>&1 &
echo "server 8768 restarted pid $!"
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
bash sweeps/gallery_chain2.sh gal_carton_above_air '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"11,11,11,11","offset":"0.30,0,0.15","transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"hand_start":"above","hand_above_mm":300,"desc":"milk carton, hand start moved to 30 cm straight above the object (orientation kept), air target +30 cm x +15 cm z; freeze 24, still 5, 40 it, shell, pop 4 per seed"}' "0 1" > logs/gal_carton_above_air_chain.log 2>&1
echo STAGE3D_DONE
