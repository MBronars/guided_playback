#!/bin/bash
# Pickup-only ckpt on the compiled server (:8768). Jar (pickonly_val5 idx 1), goal +15 cm x +15 cm z, 60 frames, S=10, Adam, shell OFF, pop 4.
# Main arms (no wall / 10 cm wall at +7.5 cm; lr and iteration variants), then an asset screen (20 assets, no wall, lr 0.1, 10 it). Isaac replay after each.
cd /data/user_data/mbronars/packages/guided_playback
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
until grep -q "\[serve\] http" logs/serve_pickonly.log; do sleep 10; done
req(){ # name json
  $C bin/client.py http://localhost:8768/guide "$2" > logs/$1.req.json 2>&1
  python3 -c "import json;r=json.load(open('logs/$1.req.json'));print('$1', 'unguided', [round(v) for v in r['unguided_mm']], 'best', [round(v) for v in r['best_mm']], round(r['seconds']),'s', 'it_s', r['iteration_seconds'][1:3])" 2>&1 | tail -1
  CUDA_VISIBLE_DEVICES=0 $C -u bin/replay.py --run runs/pickup/$1 > logs/$1_replay.log 2>&1; echo "DONE $1 replay $?"
}
B='"domain":"pickup","task":"%s","episodes":"1,1,1,1","transitions":60,"steps":10,"offset":"0.15,0,0.15","wall_height":0.10,"wall_thickness":0.02,"wall_width":0.30,"clearance_mm":5,"box_weight":5,"optimizer":"adam","lr":%s,"iterations":%s,"shell":false,"seed":0'
arm(){ # name task lr it desc
  req $1 "{$(printf "$B" $2 $3 $4),\"name\":\"$1\",\"desc\":\"$5\"}"
}
arm p15_jar_nowall_lr0.1_it20 lift 0.1 20 "jar, goal +15 x +15 z, NO wall, Adam lr 0.1, 20 it, shell off"
arm p15_jar_wall10_lr0.1_it20 wall 0.1 20 "jar, goal +15 x +15 z, 10 cm wall at +7.5 cm, Adam lr 0.1, 20 it, shell off"
arm p15_jar_wall10_lr0.2_it20 wall 0.2 20 "jar, 10 cm wall, Adam lr 0.2, 20 it, shell off"
arm p15_jar_wall10_lr0.4_it20 wall 0.4 20 "jar, 10 cm wall, Adam lr 0.4, 20 it, shell off"
arm p15_jar_wall10_lr0.1_it40 wall 0.1 40 "jar, 10 cm wall, Adam lr 0.1, 40 it, shell off"
echo MAIN_DONE
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
for i in $(seq 1 19); do L=$(python3 -c "import json;print(json.load(open('$SEL'))[$i]['label'])")
  req asset_$L "{\"domain\":\"pickup\",\"task\":\"lift\",\"select\":\"$SEL\",\"episodes\":\"$i,$i,$i,$i\",\"transitions\":60,\"steps\":10,\"offset\":\"0.15,0,0.15\",\"optimizer\":\"adam\",\"lr\":0.1,\"iterations\":10,\"shell\":false,\"seed\":0,\"name\":\"asset_$L\",\"desc\":\"asset screen: $L, goal +15 x +15 z, no wall, Adam lr 0.1, 10 it, shell off\"}"
done
echo ASSETS_DONE
