#!/bin/bash
# Genesis replays of the mould MPPI plans (mppi42_K6_ep000..049) on GPU 0, started only after the pickup queue (GPU 0 planning) is done,
# so neither planner's reported time is contended. Then the summary pass (paper_eval_mppi.py without --no_replay: plans + replays exist -> rows only).
cd /data/user_data/mbronars/packages/guided_playback; C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
until grep -q MPPIFULL5x5_DONE /tmp/claude-2711186/-data-user-data-mbronars/11fcf8f8-5a5b-484d-8121-63bb5025d65e/tasks/bo2ckzkf2.output; do sleep 30; done
echo "pickup queue done $(date)"
for ep in $(seq -f %03g 0 49); do R=runs/mould/mppi42_K6_ep$ep
  until [ -f $R/result.json ]; do sleep 30; done
  [ -f $R/replay_genesis/analysis.json ] || { t0=$(date +%s); $C -u bin/replay.py --run $R --variants guided --members 0 --gpu 0 > $R/replay_stdout.log 2>&1; echo "REPLAY $ep $(( $(date +%s) - t0 )) s"; }
done
$C -u bin/paper_eval_mppi.py --url http://localhost:8790/plan --seed 42 --n 50 --K 6 --commit 5 --replay_gpu 0 --tag mppi42_K6 > logs/mppi42_K6_summary.log 2>&1
echo MOULD_MPPI_DONE
