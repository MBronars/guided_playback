#!/bin/bash
# After the shell-on asset screen: stop the first queue before its jar arms (they would index the asset list), then run the jar arms with the jar = assetsweep idx 0 (batch000_e1).
cd /data/user_data/mbronars/packages/guided_playback
OUT=/tmp/claude-2711186/-data-user-data-mbronars/11fcf8f8-5a5b-484d-8121-63bb5025d65e/tasks/bszmnbuvn.output
until grep -q ASSETS_DONE $OUT; do sleep 5; done
Q=$(ps -eo pid,args | awk '$2=="bash" && $3=="sweeps/pick15_shellon_queue.sh" {print $1}'); kill $Q; sleep 1
CL=$(ps -eo pid,args | awk '$3=="bin/client.py" {print $1}'); kill $CL 2>/dev/null
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
sleep 400   # the server finishes the request that was already sent (lock), ~5 min
req(){ $C bin/client.py http://localhost:8768/guide "$2" > logs/$1.req.json 2>&1
  python3 -c "import json;r=json.load(open('logs/$1.req.json'));print('$1', r['episodes'] if 'episodes' in r else '', 'unguided', [round(v) for v in r['unguided_mm']], 'best', [round(v) for v in r['best_mm']], round(r['seconds']),'s')" 2>&1 | tail -1
  CUDA_VISIBLE_DEVICES=0 $C -u bin/replay.py --run runs/pickup/$1 > logs/$1_replay.log 2>&1; echo "DONE $1 replay $?"; }
B='"domain":"pickup","task":"%s","select":"'$SEL'","episodes":"0,0,0,0","transitions":60,"steps":10,"offset":"0.15,0,0.15","wall_height":0.10,"wall_thickness":0.02,"wall_width":0.30,"clearance_mm":5,"box_weight":5,"optimizer":"adam","lr":%s,"iterations":%s,"shell":true,"seed":0'
arm(){ req $1 "{$(printf "$B" $2 $3 $4),\"name\":\"$1\",\"desc\":\"$5\"}"; }
arm p15S_jarref_nowall_lr0.1_it20 lift 0.1 20 "SHELL ON: jar (batch000_e1), goal +15 x +15 z, no wall, Adam lr 0.1, 20 it"
arm p15S_jarref_wall10_lr0.1_it20 wall 0.1 20 "SHELL ON: jar, 10 cm wall at +7.5 cm, Adam lr 0.1, 20 it"
arm p15S_jarref_wall10_lr0.2_it20 wall 0.2 20 "SHELL ON: jar, 10 cm wall, Adam lr 0.2, 20 it"
arm p15S_jarref_wall10_lr0.4_it20 wall 0.4 20 "SHELL ON: jar, 10 cm wall, Adam lr 0.4, 20 it"
arm p15S_jarref_wall10_lr0.1_it40 wall 0.1 40 "SHELL ON: jar, 10 cm wall, Adam lr 0.1, 40 it"
echo TAIL_DONE
