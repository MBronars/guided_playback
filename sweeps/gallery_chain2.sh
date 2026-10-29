#!/bin/bash
# Gallery chain v2: plan seeds (pop 4 each) -> floating-hand replay of all members -> gallery_repick.sh (arm replay of every lifting member, pick the one the ARM holds, render it).
# usage: gallery_chain2.sh <base> '<request json without name/seed>' ["0 1"]
cd /data/user_data/mbronars/packages/guided_playback; N=$1; REQ=$2; SEEDS=${3:-"0 1"}
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; SRV=${GP_SRV:-http://babel-o9-24:8768}
until curl -s -m 5 $SRV/ | grep -q '"busy": false'; do sleep 15; done
for s in $SEEDS; do
  R=runs/pickup/${N}_s$s
  if [ ! -f $R/result.json ]; then
    J=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d['name']='${N}_s$s'; d['seed']=$s; print(json.dumps(d))" "$REQ")
    $P bin/client.py $SRV/guide "$J" > logs/${N}_s$s.req.json 2>&1
  fi
  [ -f $R/result.json ] || { echo "FAILED ${N}_s$s: no result.json (see logs/${N}_s$s.req.json)"; exit 1; }
  echo "GEN ${N}_s$s: $(python3 -c "import json;print(json.load(open('$R/result.json'))['best_mm'])")"
  [ -f $R/replay_isaac/analysis.json ] || flock /tmp/gp_isaac_$(hostname).lock env flock /tmp/gp_isaac.lock env CUDA_VISIBLE_DEVICES=${GP_REPLAY_GPU:-1} $P -u bin/replay.py --run $R > logs/${N}_s${s}_replay.log 2>&1
done
python3 - "$N" $SEEDS <<'PY'
import json, sys; N = sys.argv[1]; out = []
for s in sys.argv[2:]:
    a = json.load(open(f"runs/pickup/{N}_s{s}/replay_isaac/analysis.json"))
    out += [f"s{s}m{r['member']}: lift {r['sim_lift_mm']:.0f} gen {r.get('gen_to_target_mm', -1):.0f} sim {r.get('sim_to_target_mm', -1):.0f}" for r in a["rows"] if r["variant"] == "guided"]
print("LIFT_TABLE " + "; ".join(out))
PY
bash sweeps/gallery_repick.sh $N "$SEEDS"
