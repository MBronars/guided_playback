#!/bin/bash
# stop the last resident push server (babel-p9-20, port 8840) once the only job using it (f31) has finished
Q=/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget/queue
until [ -f $Q/done/f31_T_bo128_s1.json ] || [ -f $Q/failed/f31_T_bo128_s1.json ]; do sleep 60; done
srun --jobid 10590675 --overlap -n 1 -c 1 --mem=1G --cpu-bind=none bash -c 'for p in $(pgrep -f "bin/serve.py --adapter clean_push --runs"); do a=$(ps -o args= -p $p); case "$a" in *python*serve.py*--port\ 8840*) kill $p; echo stopped $p;; esac; done'
