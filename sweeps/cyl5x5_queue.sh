#!/bin/bash
# 5 picks x 5 locked assets: target = start + (dx, dy, +15 cm), (dx, dy) uniform over a 15 cm disc, the SAME 5 seed-0 targets for every asset (sweeps/cyl_targets.txt).
# Shell on, Adam lr 0.1, 10 it, 60 frames, S=10, pop 4. Isaac replay on GPU 1. Timing per run -> runs/pickup/<run>/timing.json (request wall clock, server s/it, replay wall clock).
cd /data/user_data/mbronars/packages/guided_playback
C=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; SEL=/data/user_data/mbronars/packages/guided_playback/sweeps/assetsweep_select.json
until grep -q CYL_DONE /tmp/claude-2711186/-data-user-data-mbronars/11fcf8f8-5a5b-484d-8121-63bb5025d65e/tasks/byhjeax8z.output; do sleep 10; done
one(){ # asset idx tid offset
  N=cyl_t$3_$1_it10
  if [ ! -f runs/pickup/$N/result.json ]; then t0=$(date +%s.%N)
    $C bin/client.py http://localhost:8768/guide "{\"domain\":\"pickup\",\"task\":\"lift\",\"select\":\"$SEL\",\"episodes\":\"$2,$2,$2,$2\",\"transitions\":60,\"steps\":10,\"offset\":\"$4\",\"optimizer\":\"adam\",\"lr\":0.1,\"iterations\":10,\"shell\":true,\"seed\":0,\"name\":\"$N\",\"desc\":\"pick: $1, random target $3 in the 15 cm cylinder, offset $4 m, shell on, Adam lr 0.1, 10 it\"}" > logs/$N.req.json 2>&1
    t1=$(date +%s.%N); echo "{\"request_s\": $(echo "$t1 - $t0" | bc)}" > runs/pickup/$N/timing_req.json; else t1=$(date +%s.%N); fi
  echo "GEN $N"
  if [ ! -f runs/pickup/$N/replay_isaac/analysis.json ]; then r0=$(date +%s.%N); CUDA_VISIBLE_DEVICES=1 $C -u bin/replay.py --run runs/pickup/$N > logs/${N}_replay.log 2>&1; r1=$(date +%s.%N)
    echo "{\"replay_s\": $(echo "$r1 - $r0" | bc)}" > runs/pickup/$N/timing_replay.json; fi
  echo "DONE $N"; }
for a in "milk_carton 11" "peanut_butter_jar 1" "soda_can 13" "pasta_box 9" "water_bottle 16"; do set -- $a
  while read tid off; do one $1 $2 $tid $off; done < sweeps/cyl_targets.txt; done
echo CYL5x5_DONE
