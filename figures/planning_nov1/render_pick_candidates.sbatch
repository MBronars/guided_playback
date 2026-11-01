#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=render_pick_cands
#SBATCH --output=/data/user_data/mbronars/packages/guided_playback/figures/planning_nov1/render_pick_cands_%j.log
# Start / end renders of every pick problem LNO solves (<= 60 mm), small, for choosing natural-looking starting arm poses.
cd /data/user_data/mbronars/packages/guided_playback
export PYOPENGL_PLATFORM=egl
/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python bin/render_pick_views.py --cases "$(cat figures/planning_nov1/pick_candidates.txt)" --views=-60:18 --frames start,end --size 800x600 --out figures/planning_nov1/pick_candidates
