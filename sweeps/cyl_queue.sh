#!/bin/bash
# Standard pick: milk carton, target = start + (dx, dy, +15 cm) with (dx, dy) uniform over a 15 cm-radius disc (seed 0 draws), no wall.
# Shell on, Adam lr 0.1, 10 iterations, 60 frames, S=10, pop 4. Isaac replay on GPU 1.
cd /data/user_data/mbronars/packages/guided_playback
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
until grep -qE "\[serve\] http|Traceback" logs/serve_pickonly_o924.log; do sleep 10; done
for t in "0 -0.015,0.119,0.15" "1 0.030,0.003,0.15" "2 0.115,-0.070,0.15"; do set -- $t; N=cyl_t$1_milk_carton_it10
  $C bin/client.py http://localhost:8768/guide "{\"domain\":\"pickup\",\"task\":\"lift\",\"select\":\"$SEL\",\"episodes\":\"11,11,11,11\",\"transitions\":60,\"steps\":10,\"offset\":\"$2\",\"optimizer\":\"adam\",\"lr\":0.1,\"iterations\":10,\"shell\":true,\"seed\":0,\"name\":\"$N\",\"desc\":\"pick: milk carton, random target $1 in the 15 cm cylinder, offset $2 m, shell on, Adam lr 0.1, 10 it\"}" > logs/$N.req.json 2>&1
  echo "GEN $N"; CUDA_VISIBLE_DEVICES=1 $C -u bin/replay.py --run runs/pickup/$N > logs/${N}_replay.log 2>&1; echo "DONE $N replay $?"
done; echo CYL_DONE
