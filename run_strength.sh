#!/bin/bash
set -e
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
cd /data/user_data/mbronars/packages/guided_playback
CK=/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mould_ctxgen_100m_uncond_unet_bidir/checkpoints/step_60000.pt
export CUDA_VISIBLE_DEVICES=1
run () {  # name, lr, lr_final_args, desc
  echo "=== ARM $1"
  $P bin/guide.py --adapter paper --ckpt $CK --domain mould --task move --episodes 0,0,0 \
    --transitions 150 --steps 6 --churn --t_freeze 0.8 --fix_fps --optimizer unitgrad --shell \
    --offset 0.1,0,0 --iterations 6 --lr $2 $3 --name $1 --desc "$4"
  $P bin/replay.py --run runs/mould/$1 --variants guided
}
run mould60k_s6_lr005_flat 0.05 ""                    "S=6, 6 it, pop 3, lr 0.05 CONSTANT (no decay)"
run mould60k_s6_lr01_flat  0.1  ""                    "S=6, 6 it, pop 3, lr 0.10 CONSTANT (no decay)"
run mould60k_s6_lr01_dec   0.1  "--lr_final 0.01"     "S=6, 6 it, pop 3, lr 0.10 -> 0.01"
echo ALL ARMS DONE
