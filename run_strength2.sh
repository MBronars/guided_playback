#!/bin/bash
set -e
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
cd /data/user_data/mbronars/packages/guided_playback
CK=/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mould_ctxgen_100m_uncond_unet_bidir/checkpoints/step_60000.pt
export CUDA_VISIBLE_DEVICES=1
run () {  # name, steps, iterations, extra args, desc
  echo "=== ARM $1"
  $P bin/guide.py --adapter paper --ckpt $CK --domain mould --task move --episodes 0,0,0 \
    --transitions 150 --steps $2 --churn --fix_fps --optimizer unitgrad \
    --offset 0.1,0,0 --iterations $3 --lr 0.1 $4 --name $1 --desc "$5"
  $P bin/replay.py --run runs/mould/$1 --variants guided
}
run mould60k_s6_lr01_noshell 6 6  "--t_freeze 0.8"           "S=6, 6 it, lr 0.10 flat, NO shell"
run mould60k_s6_lr01_it8     6 8  "--t_freeze 0.8 --shell"   "S=6, 8 it, lr 0.10 flat, shell (~146 s warm)"
run mould60k_s6_lr02_flat    6 6  "--t_freeze 0.8 --shell --lr 0.2"  "S=6, 6 it, lr 0.20 flat, shell"
run mould60k_s6_lr01_tf05    6 6  "--t_freeze 0.5 --shell"   "S=6, 6 it, lr 0.10 flat, shell, t_freeze 0.5"
run mould60k_s8_lr01_flat    8 6  "--t_freeze 0.8 --shell"   "S=8, 6 it, lr 0.10 flat, shell"
echo ALL ARMS2 DONE
