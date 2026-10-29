#!/bin/bash
# =============================================================================
#  pwm-data-v1 training job: joint8_500m
#  unified run, one model on all eight datasets, UNet 502.22 M, 200,000 steps, batch 4 per GPU on 4 RTX PRO 6000, validation every 2,000 steps so that 200,000 steps fit one 71.5 h job.
#
#  Submits launch/train_pwm_data_v1.sbatch from the checkout this file lives in. Use a checkout that stays at one revision
#  for the whole run, because a resubmission reads the code again: git worktree add --detach <path> <revision>.
#  Refuses a checkout with local changes and a recipe whose sha256 is not the one recorded here.
#
#  Recipe   $ROOT/configs/unet_joint8_500m.yaml, sha256 dd82310918f6
#  Table    $ROOT/PREFLIGHT_joint8.md: must carry the line 'Status: CONFIRMED by the user' (the mandatory pre-flight); until then only --print works
#  ROOT     default /data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1
#  LOG_DIR  default /data/group_data/katefgroup/runs_pwm_<your account>: the run folder pwm_v1_joint8_500m_unet_bidir/ and slurm/
#           (your own, so that nobody resumes somebody else's run; the template's header lists SCRATCH and the conda settings)
#  Usage    bash launch/submit_joint8_500m.sh            submit
#           bash launch/submit_joint8_500m.sh --print    run the checks, print the sbatch command, submit nothing
#  Running it again resumes the run from its latest.pt, first written at the first validation. A run that died before
#  that has no checkpoint: rename its run folder first (see the template's header).
# =============================================================================
set -euo pipefail
JOB=joint8_500m
RECIPE_SHA256=dd82310918f6e102b2d469025308c5a048ffed82729a62d614d075dde7a46de2
ROOT=${ROOT:-/data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_$(id -un)}
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
[[ -z "$(git status --porcelain)" ]] || { echo "$PWD has local changes: submit from a clean checkout of one revision" >&2; exit 2; }
echo "$RECIPE_SHA256  $ROOT/configs/unet_$JOB.yaml" | sha256sum --check --quiet
grep -q "^Status: CONFIRMED by the user" "$ROOT/PREFLIGHT_joint8.md" || [[ "${1:-}" == "--print" ]] || \
    { echo "$ROOT/PREFLIGHT_joint8.md does not carry the user's confirmation yet: nothing is submitted" >&2; exit 2; }
# the job is exactly what its table says: no training setting leaks in from the calling shell
unset STEPS NPROC BATCH RUN_NAME OVERRIDES REPO RESUME
export ROOT LOG_DIR
export NPROC="4"
export OVERRIDES="log.val_every=2000"
CMD=(sbatch --job-name=pwm_v1_$JOB --output="$LOG_DIR/slurm/%x_%j.log" --gres=gpu:RTX_PRO_6000:4 --cpus-per-task=32 --mem=256G launch/train_pwm_data_v1.sbatch unet $JOB)
if [[ "${1:-}" == "--print" ]]; then echo "cd $PWD && LOG_DIR=\"$LOG_DIR\" NPROC=\"4\" OVERRIDES=\"log.val_every=2000\" ${CMD[*]}"; exit 0; fi
mkdir -p "$LOG_DIR/slurm"
"${CMD[@]}"
