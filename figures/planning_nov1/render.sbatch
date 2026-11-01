#!/bin/bash
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=render_planning_figs
#SBATCH --output=/data/user_data/mbronars/packages/guided_playback/figures/planning_nov1/render_%j.log
# Planning-appendix figures (Nov 2026 runs): push + pile stills from the current pools (sources.json), then the PDFs.
cd /data/user_data/mbronars/packages/guided_playback
export PYOPENGL_PLATFORM=egl
PY=/data/user_data/mbronars/.miniconda3/envs/foundationpose/bin/python
F=figures/planning_nov1
M="Ours=Ours (LNO),ActGPWM=Action CEM\n(GPWM),ActPW=Action CEM\n(PointWorld)"
$PY bin/render_push_pile.py --task push --eps ${PUSH_EPS:-14,44} --size 2400x1800 --sources $F/sources.json --out $F/push
$PY bin/make_push_pdf.py --layout wide --src $F/push --methods "$M" --out $F/pdf/push_planning
$PY bin/render_push_pile.py --task pile --eps ${PILE_EPS:-33,25} --toward_paddle --pull_back --move_cue --clamp_table --no_overlay --arrow_w 12 --ring_on_top --paddle_on_table --size 2400x1800 --sources $F/sources.json --out $F/pile
$PY bin/make_pile_pdf.py --src $F/pile --methods "$M" --out $F/pdf/pile_planning
