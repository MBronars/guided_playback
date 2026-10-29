#!/bin/bash
# two-domain checkpoint (latest.pt, step 48k) on the jar scene (pwm-data-v1 pickup train pick/00044.tar, loader index 3 via a one-shard root).
# Goal +25 cm x +15 cm z, 13 cm wall at +12.5 cm, 60 frames, S=10 streaming, no churn, Adam lr 0.1, 20 it, joint limits, pop 4; shell ON then OFF; eager; Isaac replay.
# Model code: snapshot of pwm_final_paper (ours side of an in-progress merge in the shared worktree), via GP_PAPER_REPO.
cd /data/user_data/mbronars/packages/guided_playback
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
export GP_PAPER_REPO=/data/user_data/mbronars/packages/guided_playback/vendor/pwm_final_paper_ours_0923_1754
export GP_PICKUP_VAL_ROOT=/data/user_data/mbronars/packages/guided_playback/data_roots/pickup_pick00044/test
for sh in shell noshell; do
  N=two48k_T60_wall13_x25z15_${sh}_it20_jar; EXTRA=$([ $sh = shell ] && echo --shell)
  CUDA_VISIBLE_DEVICES=0 $C -u bin/guide.py --adapter paper --ckpt /data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_two_ctxgen_100m_uncond_unet_bidir/checkpoints/latest.pt \
    --domain pickup --task wall --episodes 3,3,3,3 --transitions 60 --steps 10 --offset 0.25,0,0.15 --wall_height 0.13 --wall_thickness 0.02 --wall_width 0.30 \
    --clearance_mm 5 --box_weight 5 --optimizer adam --lr 0.1 --iterations 20 $EXTRA --fix_fps --joint_limits 13.5,53.5 --seed 0 --eager --name $N \
    --desc "TWO-DOMAIN checkpoint (step 48k), jar: lift over a 13 cm wall at +12.5 cm to goal +25 cm x +15 cm z; 60 frames, S=10, no churn, Adam lr 0.1, 20 it, $sh; population 4" > logs/$N.log 2>&1
  echo "$N guide exit $?"; grep -E "best mm|Error" logs/$N.log | cut -c1-150
  CUDA_VISIBLE_DEVICES=0 $C -u bin/replay.py --run runs/pickup/$N > logs/${N}_replay.log 2>&1; echo "replay exit $?"
  $C sweeps/wall_report.py runs/pickup/$N
done
