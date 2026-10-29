#!/bin/bash
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
cd /data/user_data/mbronars/packages/guided_playback
CK=/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mould_ctxgen_100m_uncond_unet_bidir/checkpoints/step_60000.pt
for m in reduce-overhead max-autotune-no-cudagraphs; do
  echo "=== MODE $m"
  CUDA_VISIBLE_DEVICES=1 timeout 2400 $P bin/bench.py --ckpt $CK --domain mould --transitions 150 --steps 10 \
    --batches 3 --only scaling --compile_mode $m 2>&1 | grep -E "compile setup|B=3|Error|error" || echo "  FAILED (see log)"
done
echo MODES DONE
