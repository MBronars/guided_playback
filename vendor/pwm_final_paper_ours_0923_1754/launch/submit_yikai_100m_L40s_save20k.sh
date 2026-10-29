#!/bin/bash
# =============================================================================
#  pwm-data-v1 training job: yikai_100m_L40s  (SAVE-EVERY-20K VARIANT)
#  yikai specialist, UNet 103.10 M, 200,000 steps, batch 4 per GPU x grad_accum 2 on 2 L40S.
#
#  The L40S twin of launch/submit_yikai_100m.sh. Same template, same recipe discipline; the
#  differences are all measured (reports/yikai_compile, reports/yikai_rope):
#    batch 8 -> 4 with grad_accum 2. 4 x 2 GPUs x 2 accum = 16 windows per optimizer step, the
#      same 16 the batch-8 recipe used. train.py:455 counts a synchronized accumulation group as
#      one step, so train.steps keeps its meaning and stays at 200,000 for the same 3.2M windows.
#    activation checkpointing OFF. Compiled, the largest real yikai scene (8 objects, 4,767
#      points) peaks at 32.4 GB of the L40S's 44.4 GB. EAGER IT OOMs: compile cuts peak memory
#      ~31% here, so train.compile=true is load-bearing, not an optimization.
#    unet_rope_f_min pinned to 0.15. Unique range 2*pi*0.133155/0.15 = 5.58 m; the measured worst
#      attended-pair separation on yikai, max noise included, is 1.566 m = 28% of it. The old 0.5
#      (1.673 m) sat at 93.5%.
#    log.save_every 20000 -- step_<N>.pt snapshots. The original of this file pinned it to 0
#      (latest/best/final only); resumed at step 77,000 on 2026-09-20 with snapshots turned back
#      on at the user's request, so the last stretch of training keeps periodic checkpoints.
#      Same recipe, same RUN_NAME, so the template resumes this run rather than starting a new one.
#    katef has no L40S (RTX_PRO_6000 / 6000Ada / A100_40GB), so this submits to general with
#      qos normal and the standing --exclude. general MaxTime is 48 h and 200k steps project
#      past it: run this file again and the template resumes from latest.pt.
#
#  Submits launch/train_pwm_data_v1.sbatch from the checkout this file lives in. Use a checkout that stays at one revision
#  for the whole run, because a resubmission reads the code again.
#  Refuses a checkout with modified TRACKED files and a recipe whose sha256 is not the one recorded here.
#
#  Recipe   $ROOT/configs/unet_yikai_100m_L40s.yaml, sha256 6284347ccfe5
#  ROOT     default /data/user_data/mbronars/launch_pwm_data_v1_l40s (this job's own; the recipe is
#           kept out of the checkout because this worktree is shared with other agents)
#  LOG_DIR  default /data/group_data/katefgroup/runs_pwm_<your account>: the run folder
#           pwm_v1_yikai_100m_L40s_unet_bidir/ and slurm/
#  Usage    bash launch/submit_yikai_100m_L40s.sh            submit
#           bash launch/submit_yikai_100m_L40s.sh --print    run the checks, print the sbatch command, submit nothing
#  Running it again resumes the run from its latest.pt, first written at the first validation. A run that died before
#  that has no checkpoint: rename its run folder first (see the template's header).
# =============================================================================
set -euo pipefail
JOB=yikai_100m_L40s
RECIPE_SHA256=6284347ccfe5f902bc350a6bdc3663fe09d93269ec0b5cc7e37b9edeb43a0f7f
ROOT=${ROOT:-/data/user_data/mbronars/launch_pwm_data_v1_l40s}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_$(id -un)}
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
# Tracked modifications are refused, as in submit_yikai_100m.sh. Untracked files only warn:
# this worktree is shared with other agents, and their new modules (models/tube_*.py) plus this
# script itself are not imported by an architecture=unet recipe.
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "$PWD has modified tracked files: submit from a clean checkout of one revision" >&2; git status --porcelain --untracked-files=no >&2; exit 2; }
UNTRACKED=$(git ls-files --others --exclude-standard)
[[ -z "$UNTRACKED" ]] || { echo "note: untracked files present (not imported by this recipe):" >&2; echo "$UNTRACKED" | sed 's/^/  /' >&2; }
echo "$RECIPE_SHA256  $ROOT/configs/unet_$JOB.yaml" | sha256sum --check --quiet
echo "code revision: $(git rev-parse HEAD)"
# the job is exactly what this file says: no training setting leaks in from the calling shell
unset STEPS NPROC BATCH RUN_NAME REPO RESUME
# log.save_every: the template hard-codes 20000 on its command line; OVERRIDES is appended last and wins.
OVERRIDES="log.save_every=20000"   # CHANGED 2026-09-20: periodic step_<N>.pt for the last stretch
PWM_CONDA_ENV=pwm211                      # the template defaults to pwm_blackwell, which is for the Blackwell cards
export ROOT LOG_DIR OVERRIDES PWM_CONDA_ENV
CMD=(sbatch --job-name=pwm_v1_$JOB
     --partition=general --qos=normal --gres=gpu:L40S:2
     --cpus-per-task=16 --mem=96G --time=2-00:00:00
     --exclude=babel-o9-20,babel-o9-24,babel-p9-32
     --output="$LOG_DIR/slurm/%x_%j.log" launch/train_pwm_data_v1.sbatch unet $JOB)
if [[ "${1:-}" == "--print" ]]; then echo "cd $PWD && ROOT=\"$ROOT\" LOG_DIR=\"$LOG_DIR\" OVERRIDES=\"$OVERRIDES\" PWM_CONDA_ENV=$PWM_CONDA_ENV ${CMD[*]}"; exit 0; fi
mkdir -p "$LOG_DIR/slurm"
"${CMD[@]}"
