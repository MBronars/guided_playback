#!/bin/bash
# wait for server :8768 (GPU 0) to finish its warmup, then start :8769-:8771 on GPUs 1-3 (they reuse the node-local inductor cache)
cd /data/user_data/mbronars/packages/guided_playback
until curl -s -m 5 http://babel-p9-28:8768/ | grep -q '"busy": false'; do sleep 15; done
echo "server 0 ready $(date +%H:%M)"
for k in 1 2 3; do
  port=$((8768+k))
  CUDA_VISIBLE_DEVICES=$k nohup /data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python -u bin/serve.py --adapter clean_pickup --ckpt /data/user_data/mbronars/packages/pwm_final/runs/pickup_uncond_x0_regpass_cont200k/checkpoints/final.pt --port $port --warmup --warmup_transitions 60 --warmup_episodes 1,1,1,1 --warmup_iterations 1 > logs/serve_p928_gpu${k}_${port}_v2.log 2>&1 &
  echo "server :$port restarted pid $! gpu $k"
  until curl -s -m 5 http://babel-p9-28:$port/ | grep -q '"busy": false'; do sleep 15; done; echo "server $k ready $(date +%H:%M)"
done
echo SERVERS_READY
