#!/bin/bash
# Overhead-start lifts on GPU 0 (server :8768 already running): hand moved to 30 cm straight above the object, same targets as the working lifts.
cd /data/user_data/mbronars/packages/guided_playback
export GP_SRV=http://babel-p9-28:8768 GP_REPLAY_GPU=0
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
DSEL=/data/user_data/mbronars/packages/guided_playback/sweeps/diverse_select.json
C='"transitions":60,"steps":10,"optimizer":"adam","lr":0.1,"iterations":40,"shell":true,"freeze_frames":24,"still_weight":5,"still_frames":24,"hand_start":"above","hand_above_mm":300'
run(){ bash sweeps/gallery_chain2.sh $1 "$2" "0 1" > logs/$1_chain.log 2>&1; }
run gal_jar_above_air '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"1,1,1,1","offset":"0,0.30,0.15",'$C',"desc":"peanut butter jar, hand start 30 cm straight above the object, air target +30 cm y +15 cm z; freeze 24, still 5, 40 it, shell, pop 4 per seed"}'
run gal_bottle_above_air '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"16,16,16,16","offset":"0,-0.30,0.15",'$C',"desc":"water bottle, hand start 30 cm straight above the object, air target -30 cm y +15 cm z; freeze 24, still 5, 40 it, shell, pop 4 per seed"}'
run gal_can_above_air '{"domain":"pickup","task":"lift","select":"'$SEL'","episodes":"13,13,13,13","offset":"0,0.30,0.15",'$C',"desc":"soda can, hand start 30 cm straight above the object, air target +30 cm y +15 cm z; freeze 24, still 5, 40 it, shell, pop 4 per seed"}'
run gal_wineglass_above_air '{"domain":"pickup","task":"lift","select":"'$DSEL'","episodes":"0,0,0,0","offset":"0,-0.30,0.15",'$C',"desc":"wine glass, hand start 30 cm straight above the object, air target -30 cm y +15 cm z; freeze 24, still 5, 40 it, shell, pop 4 per seed"}'
echo QUEUE0B_DONE
