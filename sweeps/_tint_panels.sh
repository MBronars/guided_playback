cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; F=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python
CAM="--eye=0.70,-1.10,0.52 --look=0.42,0.20,0.21"; N=runs/pickup/ground30_nowall_milk_carton_asset_0037; Wd=runs/pickup/wall12w20ground_pickonly_it40_lr0.1_shell_bw10_clr10
L="flock /tmp/gp_isaac_$(hostname).lock"
for t in "tan:0.82,0.66,0.42" "slate:0.50,0.54,0.62" "blue2:0.42,0.60,0.86"; do n=${t%%:*}; c=${t#*:}
  $L $P bin/replay_arm.py --run $N --members 0,1,2,3 --variants unguided --gpu 2 --skip_sim --no_target --no_obstacle --render_envs 2 --render_dir render_asset_unguided_tint_$n --object_tint $c $CAM 2>&1 | grep -E "renders|Error|Traceback"
  $F bin/arm_keyframes.py $N/replay_isaac_arm 2 $CAM --render_dir render_asset_unguided_tint_$n --sim_dir isaac_unguided --end 39
  $L $P bin/replay_arm.py --run $Wd --members 3 --gpu 2 --skip_sim --obstacle_size 0.02,0.55,0.12 --render_dir render_asset_wall55_tint_$n --object_tint $c $CAM 2>&1 | grep -E "renders|Error|Traceback"
  $F bin/arm_keyframes.py $Wd/replay_isaac_arm 0 $CAM --render_dir render_asset_wall55_tint_$n --sim_dir isaac_wall0.02x0.55x0.12
done; echo PANELS_RENDERED
