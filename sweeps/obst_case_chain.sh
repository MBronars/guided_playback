#!/bin/bash
# After a guidance run lands: floating replay (specs) -> arm replay with the obstacle ASSET + both render styles (auto camera) -> traces + mp4s.
# usage: obst_case_chain.sh <run name>
cd /data/user_data/mbronars/packages/guided_playback; N=$1; R=runs/pickup/$N; P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; F=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python
until [ -f $R/result.json ]; do sleep 20; done; echo "guidance done: $(python3 -c "import json;print(json.load(open('$R/result.json'))['best_mm'])")"
[ -f $R/replay_isaac/analysis.json ] || CUDA_VISIBLE_DEVICES=1 $P -u bin/replay.py --run $R > logs/${N}_replay.log 2>&1
M=$(python3 -c "import json,numpy as np;print(int(np.argmin(json.load(open('$R/result.json'))['best_mm'])))"); echo "planner pick m$M"
$P bin/replay_arm.py --run $R --members $M --gpu 1 --auto_camera --render_dir render_asset 2>&1 | grep -E "^\[arm\]|Error|Traceback" | tee logs/${N}_arm.log
CAM=$(grep "camera eye" logs/${N}_arm.log | sed 's/.*camera eye \([^ ]*\) look \([^ ]*\)/--eye \1 --look \2/')
$P bin/replay_arm.py --run $R --members $M --gpu 1 --skip_sim --asset none --render_dir render_x $CAM 2>&1 | grep -E "Error|Traceback"
$F bin/arm_keyframes.py $R/replay_isaac_arm 0 $CAM --render_dir render_asset; $F bin/arm_keyframes.py $R/replay_isaac_arm 0 $CAM --render_dir render_x
for d in render_asset render_x; do g=$(ls $R/replay_isaac_arm/$d/env0_ep*.gif | head -1); ffmpeg -y -loglevel error -r 20 -i $g -movflags +faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 20 ${g%.gif}.mp4; done
echo "CHAIN_DONE $N"
