cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; F=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python
CAM="--eye=0.70,-1.10,0.52 --look=0.42,0.20,0.21"; N=runs/pickup/ground30_nowall_milk_carton_asset_0037; Wd=runs/pickup/wall12w20ground_pickonly_it40_lr0.1_shell_bw10_clr10
L="flock /tmp/gp_isaac_$(hostname).lock"; T="--object_tint 0.82,0.66,0.42 --table_color 96,68,48 --target_color 110,215,120"; S=tan_walnut
$L $P bin/replay_arm.py --run $N --members 0 --gpu 2 --skip_sim --render_dir render_asset_tint_$S $T $CAM 2>&1 | grep -E "Error|Traceback"
$F bin/arm_keyframes.py $N/replay_isaac_arm 0 $CAM --render_dir render_asset_tint_$S
$L $P bin/replay_arm.py --run $N --members 0,1,2,3 --variants unguided --gpu 2 --skip_sim --no_target --no_obstacle --render_envs 2 --render_dir render_asset_unguided_tint_$S $T $CAM 2>&1 | grep -E "Error|Traceback"
$F bin/arm_keyframes.py $N/replay_isaac_arm 2 $CAM --render_dir render_asset_unguided_tint_$S --sim_dir isaac_unguided --end 39
$L $P bin/replay_arm.py --run $Wd --members 3 --gpu 2 --skip_sim --obstacle_size 0.02,0.55,0.12 --render_dir render_asset_wall55_tint_$S $T $CAM 2>&1 | grep -E "Error|Traceback"
$F bin/arm_keyframes.py $Wd/replay_isaac_arm 0 $CAM --render_dir render_asset_wall55_tint_$S --sim_dir isaac_wall0.02x0.55x0.12
$F bin/four_panel.py --style $S --tight; $F bin/four_panel.py --style $S
echo WALNUT_DONE
