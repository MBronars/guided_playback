#!/bin/bash
# Drive the remaining sweep through the WARM server (compile paid once at boot); replays on GPU 1.
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
cd /data/user_data/mbronars/packages/guided_playback
URL=http://$(hostname):8790/guide
while ! curl -s -m 2 http://$(hostname):8790/ >/dev/null 2>&1; do sleep 10; done   # wait for the server
COMMON='"domain":"mould","task":"move","episodes":"0,0,0","transitions":150,"steps":6,"churn":true,"fix_fps":true,"optimizer":"unitgrad","offset":"0.1,0,0","t_freeze":0.8'
arm () {  # name, json-overrides
  echo "=== ARM $1"
  $P bin/client.py $URL "{$COMMON,\"name\":\"$1\",$2}" | grep -E '"seconds"|best_mm|unguided_mm'
  $P bin/replay.py --run runs/mould/$1 --variants guided --gpu 1
}
arm mould60k_s6_lr01_noshell '"iterations":6,"lr":0.1,"shell":false,"desc":"S=6, 6 it, lr 0.10 flat, NO shell"'
arm mould60k_s6_lr01_it8     '"iterations":8,"lr":0.1,"shell":true,"desc":"S=6, 8 it, lr 0.10 flat, shell"'
arm mould60k_s6_lr02_flat    '"iterations":6,"lr":0.2,"shell":true,"desc":"S=6, 6 it, lr 0.20 flat, shell"'
arm mould60k_s6_lr01_tf05    '"iterations":6,"lr":0.1,"shell":true,"t_freeze":0.5,"desc":"S=6, 6 it, lr 0.10 flat, shell, t_freeze 0.5"'
arm mould60k_s8_lr01_flat    '"iterations":6,"lr":0.1,"shell":true,"steps":8,"desc":"S=8, 6 it, lr 0.10 flat, shell"'
echo ALL SERVER ARMS DONE
