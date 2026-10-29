#!/bin/bash
# best asset (val batch000_e1), 60 transitions, walls 10 cm and 13 cm at +7.5 cm x, goal +15 cm x +15 cm z; S=10 bf16 eager, Adam lr 0.1, 20 it, shell on, pop 4; Isaac replay
cd /data/user_data/mbronars/packages/guided_playback
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
CK=/data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt
for h in 10 13; do
  N=pickonlyS10_T60_wall${h}_x15z15_shell_it20_batch000_e1
  CUDA_VISIBLE_DEVICES=0 $C -u bin/guide.py --adapter clean_pickup --ckpt $CK --task wall --episodes 1,1,1,1 --transitions 60 --steps 10 \
    --offset 0.15,0,0.15 --wall_height 0.$h --wall_thickness 0.02 --wall_width 0.30 --clearance_mm 5 --box_weight 5 \
    --optimizer adam --lr 0.1 --iterations 20 --shell --seed 0 --eager --name $N \
    --desc "best asset (val batch000_e1), 60 transitions: lift over a $h cm wall at +7.5 cm to goal +15 cm x +15 cm z; S=10 bf16 eager, Adam lr 0.1, 20 it, shell on; population 4" > logs/$N.log 2>&1
  echo "$N exit $?"; grep -E "^it 10|best mm|Error" logs/$N.log | cut -c1-170
  CUDA_VISIBLE_DEVICES=0 $C -u bin/replay.py --run runs/pickup/$N > logs/${N}_replay.log 2>&1; echo "replay exit $?"
done
