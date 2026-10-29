#!/bin/bash
# Two variants per checkpoint of the ground-target wall case (milk carton, +30 cm x on the ground, 10 cm wall midway), pop 4, Isaac replay after each.
cd /data/user_data/mbronars/packages/guided_playback; C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
run(){ # name url episodes extra_json desc
  N=$1; [ -f runs/pickup/$N/result.json ] || $C bin/client.py $2 "{\"domain\":\"pickup\",\"task\":\"wall\",\"episodes\":\"$3\",\"transitions\":60,\"steps\":10,\"offset\":\"0.30,0,0\",\"optimizer\":\"adam\",\"seed\":0,\"wall_height\":0.10,$4,\"name\":\"$N\",\"desc\":\"$5\"}" > logs/$N.req.json 2>&1
  [ -f runs/pickup/$N/replay_isaac/analysis.json ] || CUDA_VISIBLE_DEVICES=1 $C -u bin/replay.py --run runs/pickup/$N > logs/${N}_replay.log 2>&1; echo "DONE $N"; }
A='"lr":0.2,"iterations":40,"shell":true,"box_weight":20,"clearance_mm":20'
B='"lr":0.1,"iterations":30,"shell":false,"box_weight":10,"clearance_mm":10'
if [ "$1" = pickonly ]; then U=http://babel-q9-16:8768/guide; E="11,11,11,11"; X="\"select\":\"$SEL\","; P=pickonly
  run wallgroundA_${P}_it40_lr0.2_bw20_clr20 $U $E "$X$A" "PICKUP-ONLY ckpt: variant A, ground target +30 cm x, 10 cm wall; Adam lr 0.2, 40 it, shell, wall weight 20, clearance 20 mm"
  run wallgroundB_${P}_it30_lr0.1_noshell_bw10_clr10 $U $E "$X$B" "PICKUP-ONLY ckpt: variant B, ground target +30 cm x, 10 cm wall; Adam lr 0.1, 30 it, NO shell, wall weight 10, clearance 10 mm"
else U=http://babel-o9-24:8791/guide; E="17,17,17,17"; X='"churn":false,"fix_fps":true,"joint_limits":"13.5,53.5",'; P=two120k
  run wallgroundA_${P}_it40_lr0.2_bw20_clr20 $U $E "$X$A" "TWO-DOMAIN 120k: variant A, ground target +30 cm x, 10 cm wall; Adam lr 0.2, 40 it, shell, wall weight 20, clearance 20 mm"
  run wallgroundB_${P}_it30_lr0.1_noshell_bw10_clr10 $U $E "$X$B" "TWO-DOMAIN 120k: variant B, ground target +30 cm x, 10 cm wall; Adam lr 0.1, 30 it, NO shell, wall weight 10, clearance 10 mm"
fi; echo VARIANTS_DONE_$P
