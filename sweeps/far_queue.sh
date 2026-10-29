#!/bin/bash
# FAR target: goal = start + 30 cm x + 15 cm z. The 5 locked assets, shell on, Adam lr 0.1, 20 it, 60 frames, S=10, pop 4; no wall, then a 10 cm wall at the midpoint (+15 cm).
# No wall first for all 5, then a 10 cm wall at +7.5 cm. Isaac replay after each. Runs on the same server as the jar tail (requests interleave).
cd /data/user_data/mbronars/packages/guided_playback
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
run(){ # name task idx desc
  $C bin/client.py http://localhost:8768/guide "{\"domain\":\"pickup\",\"task\":\"$2\",\"select\":\"$SEL\",\"episodes\":\"$3,$3,$3,$3\",\"transitions\":60,\"steps\":10,\"offset\":\"0.30,0,0.15\",\"wall_height\":0.10,\"wall_thickness\":0.02,\"wall_width\":0.30,\"clearance_mm\":5,\"box_weight\":5,\"optimizer\":\"adam\",\"lr\":0.1,\"iterations\":20,\"shell\":true,\"seed\":0,\"name\":\"$1\",\"desc\":\"$4\"}" > logs/$1.req.json 2>&1
  CUDA_VISIBLE_DEVICES=0 $C -u bin/replay.py --run runs/pickup/$1 > logs/$1_replay.log 2>&1; echo "DONE $1 replay $?"; }
for a in "milk_carton_asset_0037 11" "peanut_butter_jar_asset_0003 1" "soda_can_asset_0012 13" "pasta_box_asset_0058 9" "water_bottle_asset_0087 16"; do set -- $a
  run far30_nowall_$1 lift $2 "FAR target: $1, goal = start + 30 cm x + 15 cm z, no wall, shell on, Adam lr 0.1, 20 it"; done
for a in "milk_carton_asset_0037 11" "peanut_butter_jar_asset_0003 1" "soda_can_asset_0012 13" "pasta_box_asset_0058 9" "water_bottle_asset_0087 16"; do set -- $a
  run far30_wall10_$1 wall $2 "FAR target over a 10 cm wall at +15 cm: $1, goal = start + 30 cm x + 15 cm z, shell on, Adam lr 0.1, 20 it"; done
echo FAR_DONE
