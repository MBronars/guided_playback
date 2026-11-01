#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=render_pick_figs
#SBATCH --output=/data/user_data/mbronars/packages/guided_playback/figures/planning_nov1/render_pick_%j.log
# Picking figures (Nov 2026 runs): Isaac replays of LNO / action CEM (GPWM) / action CEM (PointWorld) from the pools' replay batches,
# one shared camera per object (pass 1 stores the bounds, pass 2 renders with them), then PointWorld's own predicted cloud.
cd /data/user_data/mbronars/packages/guided_playback
export PYOPENGL_PLATFORM=egl
PY=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python
F=figures/planning_nov1
CASES=$(cat $F/pick_cases.txt)
$PY bin/render_pick_views.py --cases "$CASES" --views=-60:18 --frames start,end --size 2400x1800 --out $F/pick
$PY bin/render_pick_views.py --cases "$CASES" --views=-60:18 --frames start,end --size 2400x1800 --shared_cam --out $F/pick
$PY bin/make_pick_pdf.py --src $F/pick --config $F/pick_config.json --out $F/pdf/pick_planning
$PY bin/render_pick_views.py --pw_cloud --cases "$(cat $F/pick_pw_cases.txt)" --views=-60:18 --cloud_frames 0,15,30,45,60 --size 2400x1800 --shared_cam --cam_from $F/pick/cases.json --out $F/pick_pw
$PY bin/make_pick_pw_pdf.py --src $F/pick_pw --caption gap --rollouts "$(cat $F/pick_rollouts.txt)" --out $F/pdf/pick_pointworld_prediction
