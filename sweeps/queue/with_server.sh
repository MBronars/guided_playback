#!/bin/bash
# Run one queue job against its OWN fresh server (GP_RELOAD=1: current gp/ code), pinned to this GPU's dedicated server cores.
# usage (inside a queue cmd): bash sweeps/queue/with_server.sh {GPU} {HOST} <command using $URL>
G=$1; H=$2; shift 2
Q=/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget/queue; ROOT=/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
CF=$Q/cores_${H}_${SLURM_JOB_ID}.json; [ -f $CF ] || CF=$Q/cores_$H.json              # per-job core map (two jobs can share a node)
CORES=$(python3 -c "import json; print(json.load(open('$CF'))['$G'])")
BUS=$(nvidia-smi -i $G --query-gpu=pci.bus_id --format=csv,noheader | cut -d: -f2)   # physical GPU: unique on the node, unlike the job-local index
PORT=$((10000 + 16#$BUS))
GP_RELOAD=1 taskset -c $CORES $P -u bin/serve.py --adapter clean_push --runs $ROOT/runs --port $PORT \
  --ckpt /data/user_data/mbronars/packages/pwm_final_clean/runs/push0418_rigid_x0_rope_regpass_mix_cont200k_stream/checkpoints/final.pt > $Q/logs/server_${H}_g${G}_$$.log 2>&1 &
SP=$!; trap "kill $SP" EXIT
until curl -s http://$H:$PORT/status > /dev/null; do sleep 5; done
export URL=http://$H:$PORT/guide
eval "$@"
