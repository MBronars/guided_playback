#!/bin/bash
# Gallery chain: plan the same case under seeds 0 and 1 (2 x pop 4 = 8 members), floating-hand replay of all 8,
# pick the member that LIFTS (>100 mm) with the smallest generated distance to the target, then commanded-arm replay
# + both render styles + traces + mp4s for that member only.
# usage: gallery_chain.sh <base name> '<request json without "name"/"seed">' [extra seeds, default "0 1"]
cd /data/user_data/mbronars/packages/guided_playback
N=$1; REQ=$2; SEEDS=${3:-"0 1"}
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; F=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python
SRV=http://babel-o9-24:8768
until curl -s -m 5 $SRV/ | grep -q '"busy": false'; do sleep 15; done
for s in $SEEDS; do
  R=runs/pickup/${N}_s$s
  if [ ! -f $R/result.json ]; then
    J=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['name']='${N}_s$s'; d['seed']=$s; print(json.dumps(d))" "$REQ")
    $P bin/client.py $SRV/guide "$J" > logs/${N}_s$s.req.json 2>&1
  fi
  [ -f $R/result.json ] || { echo "FAILED ${N}_s$s: no result.json (see logs/${N}_s$s.req.json)"; exit 1; }
  echo "GEN ${N}_s$s: $(python3 -c "import json;print(json.load(open('$R/result.json'))['best_mm'])")"
  [ -f $R/replay_isaac/analysis.json ] || CUDA_VISIBLE_DEVICES=1 $P -u bin/replay.py --run $R > logs/${N}_s${s}_replay.log 2>&1
done
PICK=$(python3 - "$N" $SEEDS <<'EOF'
import json, sys
N = sys.argv[1]; seeds = sys.argv[2:]; rows = []
for s in seeds:
    a = json.load(open(f"runs/pickup/{N}_s{s}/replay_isaac/analysis.json"))
    for r in a["rows"]:
        if r["variant"] == "guided": rows.append((s, r))
lift = [(s, r) for s, r in rows if r["sim_lift_mm"] > 100]
print("LIFT_TABLE " + "; ".join(f"s{s}m{r['member']}: lift {r['sim_lift_mm']:.0f} gen {r.get('gen_to_target_mm', -1):.0f} sim {r.get('sim_to_target_mm', -1):.0f}" for s, r in rows), file=sys.stderr)
pool = lift if lift else rows
s, r = min(pool, key=lambda x: x[1].get("gen_to_target_mm", 1e9))
print(f"{s} {r['member']} {len(lift)} {r['sim_lift_mm']:.0f} {r.get('gen_to_target_mm', -1):.0f} {r.get('sim_to_target_mm', -1):.0f} {len(rows)}")
EOF
)
set -- $PICK; S=$1; M=$2; echo "gallery pick ${N}_s$S m$M (lifting members $3/$7; floating lift $4 mm, gen $5 mm, sim $6 mm)"
R=runs/pickup/${N}_s$S
$P bin/replay_arm.py --run $R --members $M --gpu 1 --auto_camera --render_dir render_asset 2>&1 | grep -E "^\[arm\]|Error|Traceback" | tee logs/${N}_arm.log
CAM=$(grep "camera eye" logs/${N}_arm.log | sed 's/.*camera eye \([^ ]*\) look \([^ ]*\)/--eye \1 --look \2/')
$P bin/replay_arm.py --run $R --members $M --gpu 1 --skip_sim --asset none --render_dir render_x $CAM 2>&1 | grep -E "Error|Traceback"
$F bin/arm_keyframes.py $R/replay_isaac_arm 0 $CAM --render_dir render_asset; $F bin/arm_keyframes.py $R/replay_isaac_arm 0 $CAM --render_dir render_x
for d in render_asset render_x; do g=$(ls $R/replay_isaac_arm/$d/env0_ep*.gif | head -1); ffmpeg -y -loglevel error -r 20 -i $g -movflags +faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 20 ${g%.gif}.mp4; done
echo "CHAIN_DONE $N pick s$S m$M"
