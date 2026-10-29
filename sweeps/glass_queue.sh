#!/bin/bash
# more glasses (drinking_glass x4 beyond 0056, wine_glass x3), same screen recipe (shell on, +15/+15, no wall, lr 0.1, 10 it); runs after the jar tail.
cd /data/user_data/mbronars/packages/guided_playback
until grep -q TAIL_DONE /tmp/claude-2711186/-data-user-data-mbronars/11fcf8f8-5a5b-484d-8121-63bb5025d65e/tasks/bxqw391bx.output; do sleep 20; done
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/glass_select.json
for i in 0 1 2 3 4 5 6; do L=$(python3 -c "import json;print(json.load(open('$SEL'))[$i]['label'])")
  $C bin/client.py http://localhost:8768/guide "{\"domain\":\"pickup\",\"task\":\"lift\",\"select\":\"$SEL\",\"episodes\":\"$i,$i,$i,$i\",\"transitions\":60,\"steps\":10,\"offset\":\"0.15,0,0.15\",\"optimizer\":\"adam\",\"lr\":0.1,\"iterations\":10,\"shell\":true,\"seed\":0,\"name\":\"assetS_$L\",\"desc\":\"SHELL ON asset screen: $L, goal +15 x +15 z, no wall, Adam lr 0.1, 10 it\"}" > logs/assetS_$L.req.json 2>&1
  CUDA_VISIBLE_DEVICES=0 $C -u bin/replay.py --run runs/pickup/assetS_$L > logs/assetS_${L}_replay.log 2>&1; echo "DONE assetS_$L replay $?"
done; echo GLASS_DONE
