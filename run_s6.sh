#!/bin/bash
set -e
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
cd /data/user_data/mbronars/packages/guided_playback
CK=/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mould_ctxgen_100m_uncond_unet_bidir/checkpoints/step_60000.pt
N=mould60k_move_10cm_s6_pop3
export CUDA_VISIBLE_DEVICES=1
S=$(date +%s)
$P bin/guide.py --adapter paper --ckpt $CK --domain mould --task move \
  --episodes 0,0,0 --transitions 150 --steps 6 --churn --t_freeze 0.8 --fix_fps --optimizer unitgrad --shell \
  --offset 0.1,0,0 --iterations 6 --lr 0.05 --lr_final 0.005 --name $N \
  --desc "POPULATION of 3 on episode 0; S=6, 6 iterations, particle churn, t_freeze 0.8, lr 0.05 -> 0.005"
echo "GUIDANCE WALL TIME $(( $(date +%s) - S )) s"
$P bin/replay.py --run runs/mould/$N --variants guided,unguided
echo ALL DONE
