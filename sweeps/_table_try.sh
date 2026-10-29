cd /data/user_data/mbronars/packages/guided_playback
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; F=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python
CAM="--eye=0.70,-1.10,0.52 --look=0.42,0.20,0.21"; N=runs/pickup/ground30_nowall_milk_carton_asset_0037; L="flock /tmp/gp_isaac_$(hostname).lock"
for t in "walnut:96,68,48" "white:238,238,236" "slatedark:70,78,90" "sage:150,168,148" "bluegrey:156,172,192"; do n=${t%%:*}; c=${t#*:}
  $L $P bin/replay_arm.py --run $N --members 0 --gpu 2 --skip_sim --render_dir render_asset_tint_tan_table_$n --object_tint 0.82,0.66,0.42 --table_color $c $CAM 2>&1 | grep -E "Error|Traceback"
  $F bin/arm_keyframes.py $N/replay_isaac_arm 0 $CAM --render_dir render_asset_tint_tan_table_$n
done; echo TABLE_TRY_DONE
