#!/bin/bash
# pickup-only ckpt, best asset only (val batch000_e1): goal = start + 25 cm x + 15 cm z, 10 cm wall (2 cm thick, 30 cm wide) at +12.5 cm x.
# S=10 bf16, Adam lr 0.1, 10 it, shell on/off, population 4; eager (the running server predates the wall-keeps-z change); Isaac replay, no render.
cd /data/user_data/mbronars/packages/guided_playback
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
CK=/data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt
for sh in shell noshell; do
  N=pickonlyS10_wall_x25z15_${sh}_it10_batch000_e1
  EXTRA=$([ $sh = shell ] && echo --shell)
  CUDA_VISIBLE_DEVICES=0 $C -u bin/guide.py --adapter clean_pickup --ckpt $CK --task wall --episodes 1,1,1,1 --transitions 39 --steps 10 \
    --offset 0.25,0,0.15 --wall_height 0.10 --wall_thickness 0.02 --wall_width 0.30 --clearance_mm 5 --box_weight 5 \
    --optimizer adam --lr 0.1 --iterations 10 $EXTRA --seed 0 --eager --name $N \
    --desc "pickup-only ckpt, best asset (val batch000_e1): lift over a 10 cm wall at +12.5 cm to goal +25 cm x +15 cm z; S=10 bf16 eager, Adam lr 0.1, 10 it, $sh; population 4" > logs/$N.log 2>&1
  echo "$N exit $?"; grep -E "^it |wall centre|best mm" logs/$N.log | tail -4 | cut -c1-160
  CUDA_VISIBLE_DEVICES=0 $C -u bin/replay.py --run runs/pickup/$N > logs/${N}_replay.log 2>&1; echo "replay $N exit $?"
done
