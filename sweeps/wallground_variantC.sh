#!/bin/bash
# Variant C (longer horizon: 90 transitions) of the ground-target wall case, queued behind A/B on the same server.
cd /data/user_data/mbronars/packages/guided_playback; C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
P=$1; until grep -q VARIANTS_DONE_$P logs/wallground_variants_$P.log; do sleep 20; done
if [ "$P" = pickonly ]; then U=http://babel-q9-16:8768/guide; E="11,11,11,11"; X="\"select\":\"$SEL\","; D="PICKUP-ONLY ckpt"; else U=http://babel-o9-24:8791/guide; E="17,17,17,17"; X='"churn":false,"fix_fps":true,"joint_limits":"13.5,53.5",'; D="TWO-DOMAIN 120k"; fi
N=wallgroundC_${P}_T90_it20_lr0.1
[ -f runs/pickup/$N/result.json ] || $C bin/client.py $U "{\"domain\":\"pickup\",\"task\":\"wall\",\"episodes\":\"$E\",\"transitions\":90,\"steps\":10,\"offset\":\"0.30,0,0\",\"optimizer\":\"adam\",\"seed\":0,\"wall_height\":0.10,$X\"lr\":0.1,\"iterations\":20,\"shell\":true,\"name\":\"$N\",\"desc\":\"$D: variant C, ground target +30 cm x, 10 cm wall; 90 frames (4.5 s), Adam lr 0.1, 20 it, shell, wall weight 5, clearance 5 mm\"}" > logs/$N.req.json 2>&1
[ -f runs/pickup/$N/replay_isaac/analysis.json ] || CUDA_VISIBLE_DEVICES=1 $C -u bin/replay.py --run runs/pickup/$N > logs/${N}_replay.log 2>&1; echo "DONE $N"; echo VARIANTC_DONE_$P
