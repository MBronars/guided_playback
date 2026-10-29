# Shared environment for the pick noise-vs-action pool (sourced by worker and replayer jobs).
cd /data/user_data/mbronars/packages/guided_playback
export GP_PAPER_REPO=/data/user_data/mbronars/packages/pwm_final_wt/baselines GP_PICKUP_VAL_ROOT=/data/user_data/mbronars/scratchpad/pick_success_val GP_PICKUP_SHIFT_M=-0.45,0,-0.10 WANDB_MODE=disabled
export PWM_MODEL_OVERRIDES=unet_rope_cache=true,unet_fused_qkv=true,unet_bf16_stream=true
export GP_ISAAC_EXTRA="--physics_dt 0.0020833333333333333"               # Isaac: 1/480 s, normal (binary) gripper close
export PY=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
export NVA_CACHE=/data/group_data/katefgroup/runs_pwm_mbronars/noise_vs_action_dev/compile_cache/pick_L40S.mega
