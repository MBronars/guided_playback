#!/bin/bash
# No-guidance planner (MPPI argmin, full 19-row windows, K 20 for ~150 s, commit 5) on the 25 benchmark picks (sweeps/pick_benchmark_5x5.json targets, seed 0).
# Plans on the server (GPU 0, /plan); each finished plan is replayed in Isaac on GPU 1 while the next one generates.
cd /data/user_data/mbronars/packages/guided_playback
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
replay(){ N=$1; [ -f runs/pickup/$N/replay_isaac/analysis.json ] && return; r0=$(date +%s.%N)
  CUDA_VISIBLE_DEVICES=1 $C -u bin/replay.py --run runs/pickup/$N > logs/${N}_replay.log 2>&1; r1=$(date +%s.%N)
  echo "{\"replay_s\": $(echo "$r1 - $r0" | bc)}" > runs/pickup/$N/timing_replay.json; echo "DONE $N"; }
RP=""
for a in "milk_carton 11" "peanut_butter_jar 1" "soda_can 13" "pasta_box 9" "water_bottle 16"; do set -- $a
  while read tid off; do N=mppiargmin_full19_t${tid}_$1_K20
    if [ ! -f runs/pickup/$N/result.json ]; then t0=$(date +%s.%N)
      $C bin/client.py http://localhost:8768/plan "{\"domain\":\"pickup\",\"task\":\"lift\",\"select\":\"$SEL\",\"episodes\":\"$2,$2,$2,$2\",\"transitions\":60,\"steps\":10,\"offset\":\"$off\",\"seed\":0,\"K\":20,\"commit\":5,\"full_window\":true,\"name\":\"$N\",\"desc\":\"MPPI argmin, full 19-row windows, K 20, commit 5, $1 t$tid\"}" > logs/$N.req.json 2>&1
      t1=$(date +%s.%N); echo "{\"request_s\": $(echo "$t1 - $t0" | bc)}" > runs/pickup/$N/timing_req.json; fi
    echo "GEN $N"; [ -n "$RP" ] && wait $RP; replay $N & RP=$!
  done < sweeps/cyl_targets.txt; done
wait; echo MPPIFULL5x5_DONE
