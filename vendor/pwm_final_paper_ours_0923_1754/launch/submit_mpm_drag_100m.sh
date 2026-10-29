#!/bin/bash
# =============================================================================
#  pwm-data-v1 training job: mpm_drag_100m
#  mpm_drag specialist, UNet 103.10 M, 200,000 steps, batch 8 per GPU on 2 RTX PRO 6000, about 30 h of the 71.5 h job limit.
#
#  Submits launch/train_pwm_data_v1.sbatch from the checkout this file lives in. Use a checkout that stays at one revision
#  for the whole run, because a resubmission reads the code again: git worktree add --detach <path> <revision>.
#  Refuses a checkout with local changes and a recipe whose sha256 is not the one recorded here.
#
#  Recipe   $ROOT/configs/unet_mpm_drag_100m.yaml, sha256 ae5c1889b77a
#  Table    $ROOT/PREFLIGHT_mpm_drag.md: confirmed by the user on 2026-09-19 (the mandatory pre-flight)
#  ROOT     default /data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1
#  LOG_DIR  default /data/group_data/katefgroup/runs_pwm_<your account>: the run folder pwm_v1_mpm_drag_100m_unet_bidir/ and slurm/
#           (your own, so that nobody resumes somebody else's run; the template's header lists SCRATCH and the conda settings)
#  Usage    bash launch/submit_mpm_drag_100m.sh            submit
#           bash launch/submit_mpm_drag_100m.sh --print    run the checks, print the sbatch command, submit nothing
#  Running it again resumes the run from its latest.pt, first written at the first validation. A run that died before
#  that has no checkpoint: rename its run folder first (see the template's header).
# =============================================================================
set -euo pipefail
JOB=mpm_drag_100m
RECIPE_SHA256=ae5c1889b77acacb8eec053be002ff5d91c7901ca5554dde46c1d28f42f8aebf
ROOT=${ROOT:-/data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_$(id -un)}
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
[[ -z "$(git status --porcelain)" ]] || { echo "$PWD has local changes: submit from a clean checkout of one revision" >&2; exit 2; }
echo "$RECIPE_SHA256  $ROOT/configs/unet_$JOB.yaml" | sha256sum --check --quiet
# the job is exactly what its table says: no training setting leaks in from the calling shell
unset STEPS NPROC BATCH RUN_NAME OVERRIDES REPO RESUME
export ROOT LOG_DIR
CMD=(sbatch --job-name=pwm_v1_$JOB --output="$LOG_DIR/slurm/%x_%j.log" launch/train_pwm_data_v1.sbatch unet $JOB)
if [[ "${1:-}" == "--print" ]]; then echo "cd $PWD && LOG_DIR=\"$LOG_DIR\" ${CMD[*]}"; exit 0; fi
mkdir -p "$LOG_DIR/slurm"
"${CMD[@]}"
