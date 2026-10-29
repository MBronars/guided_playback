#!/bin/bash
# Arm-based pick: for each seed run, commanded-arm replay of EVERY member that lifted in the floating hand (one Isaac launch per seed,
# numbers only), then pick the member the ARM carried (peak lift > 100 mm) with the smallest final distance to the target, render that env only.
# arm_hold in pick.json = carried AND ended within 100 mm of the target.
# usage: gallery_repick.sh <base> ["0 1"]
cd /data/user_data/mbronars/packages/guided_playback; N=$1; SEEDS=${2:-"0 1"}
P=/data/user_data/mbronars/.miniconda3/envs/pwm211/bin/python; F=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python
if [ "$GP_RENDER_ONLY" = "1" ]; then                                   # re-render the existing pick (pick.json) only
  PICK=$(python3 -c "import json;p=json.load(open('runs/pickup/$N'+'_pick.json'));print(p['run'].split('_s')[-1], p['member'], p['env'], p['arm_hold'], p['lifting'], p['members'], round(p['arm_final_mm']), round(p['arm_peak_lift']))")
else
for s in $SEEDS; do
  R=runs/pickup/${N}_s$s; [ -f $R/replay_isaac/analysis.json ] || { echo "no floating replay for $R"; continue; }
  M=$(python3 -c "import json;a=json.load(open('$R/replay_isaac/analysis.json'));print(','.join(str(r['member']) for r in a['rows'] if r['variant']=='guided' and r['sim_lift_mm']>100))")
  [ -z "$M" ] && { echo "seed $s: no lifting members"; continue; }
  if [ ! -f $R/replay_isaac_arm/isaac/arm_all.done ]; then
    [ -d $R/replay_isaac_arm/isaac ] && mv $R/replay_isaac_arm/isaac $R/replay_isaac_arm/isaac_prev_$(date +%H%M%S)
    flock /tmp/gp_isaac_$(hostname).lock flock /tmp/gp_isaac.lock $P bin/replay_arm.py --run $R --members $M --gpu ${GP_REPLAY_GPU:-1} --skip_render 2>&1 | grep -E "^\[arm\] member|Error|Traceback"
    [ -f $R/replay_isaac_arm/isaac/replay.npz ] && touch $R/replay_isaac_arm/isaac/arm_all.done
  fi
done
PICK=$(python3 - "$N" $SEEDS <<'PY'
import json, sys, os, numpy as np
N = sys.argv[1]; seeds = sys.argv[2:]; SH = np.array([0.45, 0, 0.10]); rows = []
for s in seeds:
    R = f"runs/pickup/{N}_s{s}"; f = f"{R}/replay_isaac_arm/isaac/replay.npz"
    if not os.path.exists(f): continue
    Z = np.load(f, allow_pickle=True); z = np.load(f"{R}/rollout.npz"); res = {int(r["episode"]): r for r in json.load(open(f"{R}/replay_isaac_arm/isaac/replay.json"))["results"] if r["variant"] == "guided"}
    fl = {r["member"]: r for r in json.load(open(f"{R}/replay_isaac/analysis.json"))["rows"] if r["variant"] == "guided"}
    for k in range(len(Z["env_episode"])):
        m = int(Z["env_episode"][k]); op = Z["obj_pos"][k]; tgt = z["target"][m] + SH
        rows.append(dict(seed=s, member=m, env=k, end=float((op[-1, 2] - op[0, 2]) * 1e3), peak=float((op[:, 2].max() - op[0, 2]) * 1e3), final=float(np.linalg.norm(op[-1] - tgt) * 1e3),
                         gen=float(fl[m].get("gen_to_target_mm", -1)), float_lift=float(fl[m]["sim_lift_mm"]), hand_mean=res[m]["hand_pos_err_mm_mean"], hand_max=res[m]["hand_pos_err_mm_max"], rot_max=res[m]["hand_rot_err_deg_max"]))
n_lift = sum(1 for s in seeds if os.path.exists(f"runs/pickup/{N}_s{s}/replay_isaac/analysis.json") for r in json.load(open(f"runs/pickup/{N}_s{s}/replay_isaac/analysis.json"))["rows"] if r["variant"] == "guided" and r["sim_lift_mm"] > 100)
n_all = sum(1 for s in seeds if os.path.exists(f"runs/pickup/{N}_s{s}/replay_isaac/analysis.json") for r in json.load(open(f"runs/pickup/{N}_s{s}/replay_isaac/analysis.json"))["rows"] if r["variant"] == "guided")
lifted = [r for r in rows if r["peak"] > 100]                       # the arm carried it
hold = [r for r in lifted if r["final"] <= 100]                      # ...and it ended within 10 cm of the target (air or ground)
print("ARM_TABLE " + "; ".join(f"s{r['seed']}m{r['member']}: peak {r['peak']:.0f} end {r['end']:.0f} final {r['final']:.0f}" for r in rows), file=sys.stderr)
best = min(lifted, key=lambda r: r["final"]) if lifted else (max(rows, key=lambda r: r["peak"]) if rows else None)
if best is None: print("NONE"); sys.exit(0)
pick = dict(base=N, run=f"{N}_s{best['seed']}", member=best["member"], env=best["env"], arm_hold=len(hold), lifting=n_lift, members=n_all, arm_final_mm=best["final"], arm_peak_lift=best["peak"], arm_end_lift=best["end"],
            gen_mm=best["gen"], float_lift=best["float_lift"], arm_hand_mean=best["hand_mean"], arm_hand_max=best["hand_max"], arm_rot_max=best["rot_max"], arm_table=[{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
json.dump(pick, open(f"runs/pickup/{N}_pick.json", "w"), indent=1)
print(f"{best['seed']} {best['member']} {best['env']} {len(hold)} {n_lift} {n_all} {best['final']:.0f} {best['peak']:.0f}")
PY
)
fi
[ "$PICK" = "NONE" ] && { echo "gallery pick $N: no arm replay rows"; exit 1; }
set -- $PICK; S=$1; M=$2; K=$3; echo "gallery pick ${N}_s$S m$M env$K (arm holds $4/$5 lifting of $6 members; arm final $7 mm, peak $8 mm)"
R=runs/pickup/${N}_s$S
flock /tmp/gp_isaac_$(hostname).lock flock /tmp/gp_isaac.lock $P bin/replay_arm.py --run $R --members $M --gpu ${GP_REPLAY_GPU:-1} --skip_sim --auto_camera --render_dir render_asset --render_envs $K 2>&1 | grep -E "^\[arm\] (camera|asset)|Error|Traceback" | tee logs/${N}_arm.log
CAM=$(grep "camera eye" logs/${N}_arm.log | sed 's/.*camera eye \([^ ]*\) look \([^ ]*\)/--eye=\1 --look=\2/')
flock /tmp/gp_isaac_$(hostname).lock flock /tmp/gp_isaac.lock $P bin/replay_arm.py --run $R --members $M --gpu ${GP_REPLAY_GPU:-1} --skip_sim --asset none --render_dir render_x --render_envs $K $CAM 2>&1 | grep -E "Error|Traceback"
$F bin/arm_keyframes.py $R/replay_isaac_arm $K $CAM --render_dir render_asset; $F bin/arm_keyframes.py $R/replay_isaac_arm $K $CAM --render_dir render_x
for d in render_asset render_x; do g=$(ls $R/replay_isaac_arm/$d/env${K}_ep*.gif | head -1); ffmpeg -y -loglevel error -r 20 -i $g -movflags +faststart -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 20 ${g%.gif}.mp4; done
echo "CHAIN_DONE $N pick s$S m$M env$K"
