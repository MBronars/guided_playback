#!/bin/bash
# E1b on the interactive GPUs (timing NOT comparable: 4 CPUs): per-iteration imagined curves, 20 dev scenarios, P 4, 20 iterations, no replay.
# $1 = GPU (server port 8870+GPU), then "name:t_freeze:lr" specs.
G=$1; shift; H=$(hostname); P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python
until curl -s http://$H:$((8870 + G))/status > /dev/null; do sleep 5; done
for spec in "$@"; do IFS=: read N TF LR <<< "$spec"
  $P -u sweeps/push_diag/run_cfg.py --url http://$H:$((8870 + G))/guide --set dev --n 20 --no_replay --P 4 --I 20 --t_freeze $TF --lr $LR --name iter_$N
done; echo ITER_DONE
