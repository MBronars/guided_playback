cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; F=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python
CAM="--eye=0.70,-1.10,0.52 --look=0.42,0.20,0.21"; R=runs/pickup/ground30_nowall_milk_carton_asset_0037
for t in "tan:0.82,0.66,0.42" "slate:0.50,0.54,0.62" "blue2:0.42,0.60,0.86"; do n=${t%%:*}; c=${t#*:}
  flock /tmp/gp_isaac_$(hostname).lock $P bin/replay_arm.py --run $R --members 0 --gpu 2 --skip_sim --render_dir render_asset_tint_$n --object_tint $c $CAM 2>&1 | grep -E "tint|renders|Error|Traceback"
  $F bin/arm_keyframes.py $R/replay_isaac_arm 0 $CAM --render_dir render_asset_tint_$n
done; echo TINT_DONE
