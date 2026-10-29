#!/bin/bash
# =============================================================================
#  pwm-data-v1 training job: mpm_fresh_100m_tube_joint_L40s
#  mpm_fresh specialist, JOINT-ATTENTION core 100.63 M, 200,000 steps, batch 4 per GPU x grad_accum 4 on 2 L40S.
#
#  The joint-attention twin of the mpm_fresh specialist (configs/unet_mpm_fresh_100m.yaml, UNet 102.74 M,
#  job 10500135 on 2 RTX PRO 6000). Same data, normalization, losses, EMA, LR and timestep schedules,
#  strides, depths, l1_by_mat, pool/interp and validation protocol. What differs, and why:
#    model.unet_core=tube_joint. Every attention is ONE kernel over the call's whole (rows x tokens)
#      sequence instead of the UNet's temporal-then-spatial pair. 26.7 TFLOP/step against the UNet's
#      10.9 on the same scene: 2.45x the arithmetic at 20% MFU against 12%.
#    L0 attends inside 512-token tubes that never cross an object; L1 inside 256-token random groups
#      that MAY cross objects; L2 and L3 are single whole-level calls. Tubing L1 is what keeps the step
#      linear in object count -- untubed, L1 is one 19,456-token call at 4 objects and costs 537 ms
#      against 462 tubed. Groups are redrawn every forward (unet_tube_grouping=random).
#    widths 88/264/352/616 at head dim 88 (1/3/4/7 heads) for 100.63 M, against 64/128/256/512 at 64.
#      L0 at 88 rather than 176 halves the cost of the level carrying the most tokens for 1.7 M params.
#    unet_dual_ffn is dropped: the joint block has no second FFN stage.
#    unet_token_adaln: the per-token material + physics vector reaches the BLOCKS, not only the seams.
#    registers from L2 at HALF an object's worth (64 rows at L2, 32 at L3), per_level -- each level owns
#      its set, carried across that level's blocks and dropped at the seam, so nothing crosses a seam.
#      A full object's worth costs 7.5 h over the run for 0.04 M params and is not needed here: every
#      frame is visible inside one attention call.
#    activation checkpointing OFF and compile_l0 ON. Compiled, the 4-object worst case peaks at 22.6 GB
#      of the L40S's 46 GB. It also removes the in-graph-checkpoint + compile gradient hazard that cost
#      job 10495865 (1e4x gradients on 1-object batches; 27% of mpm_fresh episodes are 1-object).
#    batch 16 -> 4 with grad_accum 4. 4 x 4 accum x 2 GPUs = 32 windows per optimizer step, the same 32
#      the batch-16 recipe used, so train.steps keeps its meaning and stays at 200,000.
#    unet_rope_f_min stays at the checkout's 0.15 default (unique range 5.58 m).
#    log.save_every 20000: a step_N.pt every 20k on top of latest/best/final, as the katef launchers keep
#      (13 x 1.65 GB = about 21 GB for the run), NOT the L40S twin's latest/best/final only.
#    katef has no L40S, so this submits to general with qos normal and the standing --exclude.
#      Measured step cost, weighted by the 1/2/3/4-object episode mix (27.4/25.9/23.2/23.4%), is 241 ms
#      per micro-step = 0.96 s per optimizer step, so 200k projects to ~53.5 h. general MaxTime is 48 h:
#      this run does NOT finish in one job. Run this file again to resume from latest.pt.
#
#  Refuses a checkout with modified TRACKED files and a recipe whose sha256 is not the one recorded here.
#
#  Recipe   $ROOT/configs/unet_mpm_fresh_100m_tube_joint_L40s.yaml, sha256 911a354391e3
#  Code     models/{tube_joint,joint_block,tube_groups}.py; tests/test_tube_joint.py
#  ROOT     default /data/user_data/mbronars/launch_pwm_data_v1_l40s (the recipe is kept out of the
#           checkout because this worktree is shared with other agents)
#  LOG_DIR  default /data/group_data/katefgroup/runs_pwm_<your account>
#  Usage    bash launch/submit_mpm_fresh_100m_tube_joint_L40s.sh            submit
#           bash launch/submit_mpm_fresh_100m_tube_joint_L40s.sh --print    run the checks, print the sbatch command, submit nothing
# =============================================================================
set -euo pipefail
JOB=mpm_fresh_100m_tube_joint_L40s
RECIPE_SHA256=911a354391e3fe26fde17468e100e3afd7d6ca6e47b479a16fdd1ad90ec23bb4
ROOT=${ROOT:-/data/user_data/mbronars/launch_pwm_data_v1_l40s}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_$(id -un)}
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
# Tracked modifications are refused. Untracked files only warn: this worktree is shared with other agents.
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "$PWD has modified tracked files: submit from a clean checkout of one revision" >&2; git status --porcelain --untracked-files=no >&2; exit 2; }
UNTRACKED=$(git ls-files --others --exclude-standard)
[[ -z "$UNTRACKED" ]] || { echo "note: untracked files present:" >&2; echo "$UNTRACKED" | sed 's/^/  /' >&2; }
echo "$RECIPE_SHA256  $ROOT/configs/unet_$JOB.yaml" | sha256sum --check --quiet
echo "code revision: $(git rev-parse HEAD)"
# the job is exactly what this file says: no training setting leaks in from the calling shell
unset STEPS NPROC BATCH RUN_NAME REPO RESUME
# log.save_every: stated in the recipe AND here, so it survives a change to either.
OVERRIDES="log.save_every=20000"
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
