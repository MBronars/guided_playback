#!/bin/bash
set -e
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
cd /data/user_data/mbronars/packages/guided_playback
CK=/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mould_ctxgen_100m_uncond_unet_bidir/checkpoints/step_60000.pt
for arm in "lr001 0.01 0.001" "lr0005 0.005 0.0005"; do
  set -- $arm; n=$1; lr=$2; lrf=$3
  echo "=== ARM $n (lr $lr -> $lrf)"
  CUDA_VISIBLE_DEVICES=0 $P bin/guide.py --adapter paper --ckpt $CK --domain mould --task move \
    --episodes 0,1,2,3 --transitions 150 --steps 10 --churn --t_freeze 0.8 --fix_fps --optimizer unitgrad --shell \
    --offset 0.1,0,0 --iterations 10 --lr $lr --lr_final $lrf --name mould60k_move_10cm_$n \
    --desc "dough centroid +10 cm x, 150 transitions, unit-grad + shell, t_freeze 0.8, lr $lr -> $lrf"
  CUDA_VISIBLE_DEVICES=0 $P bin/replay.py --run runs/mould/mould60k_move_10cm_$n --variants guided
done
echo ALL ARMS DONE
