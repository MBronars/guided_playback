#!/bin/bash
# =============================================================================
#  pwm-data-v1 training job: three_ctxgen_100m_L40s
#  yikai_small + pickup + rl_push, round robin, with the PICK-AND-PLACE ladder and a GENERATED
#  context. UNet 104.22 M, 200,000 steps, batch 16 per GPU on 2 RTX PRO 6000.
#
#  The MODEL and the OBJECTIVE are exactly the rl_push run's (job 10515068,
#  launch/submit_rl_push_ctxgen_100m_L40s.sh): 3-level 192/384/768 ladder at head dim 64,
#  per-level registers of one object's worth (128 at L1, 32 at L2), context generated through the
#  rigid poses (train.noise_context + train.pose_noise_context), draws {independent: 0.6,
#  stream: 0.4} with keep c = 3 rows, nothing given and S sampled over 4..50, and
#  log.stream_rollout_keep 3 so the logged rollouts walk that same staircase. Only the DATA differs.
#
#  Data, per /data/group_data/katefgroup/runs_pwm_mbronars/datasets/yikai_small/TRAINING.md:
#    three domains and no data.weights, so the interleave draws a domain per batch uniformly and
#      batches stay domain-pure -- the round robin that doc asks for;
#    the SHARED ruler normalization_shared_three.json at profile xy_p99_per_domain (yikai_small must
#      NOT inherit the release yikai ruler; xy_p99_global is DERIVED from this artifact, so the
#      profile is not free to change);
#    both exclusion manifests (they are what make pickup 42,253 of 50,092 and rl_push 50,815), and
#      pickup's own shift_m;
#    the yikai slot points at the yikai_small build, not the release.
#  Checked before staging: all three domains hand the model 4 rigid bodies, materials {3, 4}, free +
#  prismatic joints and NO deformable rows -- train.pose_noise_context refuses deformable particles,
#  since nothing would denoise their context. yikai_small also presents a 1,695-point / 3-object
#  shape in about 12% of clips, so regional compile builds a second graph for it.
#
#  Resources: general / qos normal / gpu:L40S:2 / 16 CPUs / 96G / 48 h, with the standing --exclude.
#  general MaxTime is 48 h and 200k steps at this shape will not finish in one job: run this file
#  again and the template resumes from latest.pt (a resume of THIS run, which is exact).
#  Compiled, no activation checkpointing, as in the L40S recipes: NOT separately measured on this
#  card at this batch (user, 2026-09-20 -- a sizing probe was queued and the run went ahead anyway).
#  If it OOMs, the levers in order are data.batch_size 12 with train.grad_accum 2 for the same 32
#  windows, then model.unet_ckpt_from_level 0 with unet_ckpt_seams / unet_ckpt_ingraph.
#
#  PWM_CONDA_ENV=pwm211: the template defaults to pwm_blackwell, which does not exist on this
#  account. pwm211 compiles and runs on the PRO 6000 (probe job 10515927).
#
#  Recipe   $ROOT/configs/unet_three_ctxgen_100m_L40s.yaml, sha256 114abd590534
#  ROOT     default /data/user_data/mbronars/launch_pwm_data_v1_three (this job's own; the recipe is
#           kept out of the checkout because this worktree is shared with other agents)
#  LOG_DIR  default /data/group_data/katefgroup/runs_pwm_<your account>: the run folder
#           pwm_v1_three_ctxgen_100m_L40s_unet_bidir/ and slurm/
#  Usage    bash launch/submit_three_ctxgen_100m_L40s.sh            submit
#           bash launch/submit_three_ctxgen_100m_L40s.sh --print    run the checks, print the sbatch command, submit nothing
#  Running it again resumes the run from its latest.pt, first written at the first validation. A run that died before
#  that has no checkpoint: rename its run folder first (see the template's header).
# =============================================================================
set -euo pipefail
JOB=three_ctxgen_100m_L40s
RECIPE_SHA256=114abd590534feed6865d4a084a4d86ab923dafffbb29ca89defe858595410ed
ROOT=${ROOT:-/data/user_data/mbronars/launch_pwm_data_v1_three}
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_$(id -un)}
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
# Tracked modifications are refused. Untracked files only warn: this worktree is shared with other
# agents, and their new modules are not imported by an architecture=unet recipe.
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "$PWD has modified tracked files: submit from a clean checkout of one revision" >&2; git status --porcelain --untracked-files=no >&2; exit 2; }
UNTRACKED=$(git ls-files --others --exclude-standard)
[[ -z "$UNTRACKED" ]] || { echo "note: untracked files present (not imported by this recipe):" >&2; echo "$UNTRACKED" | sed 's/^/  /' >&2; }
echo "$RECIPE_SHA256  $ROOT/configs/unet_$JOB.yaml" | sha256sum --check --quiet
echo "code revision: $(git rev-parse HEAD)"
# the job is exactly what this file says: no training setting leaks in from the calling shell
unset STEPS BATCH RUN_NAME REPO RESUME
NPROC=2                                   # two GPUs, two ranks (the template's default, pinned here)
OVERRIDES="log.save_every=20000"          # step_N every 20k as well as latest/best/final (user)
PWM_CONDA_ENV=pwm211
export ROOT LOG_DIR OVERRIDES PWM_CONDA_ENV NPROC
CMD=(sbatch --job-name=pwm_v1_$JOB
     --partition=general --qos=normal --gres=gpu:L40S:2
     --cpus-per-task=16 --mem=96G --time=2-00:00:00
     --exclude=babel-o9-20,babel-o9-24,babel-p9-32
     --output="$LOG_DIR/slurm/%x_%j.log" launch/train_pwm_data_v1.sbatch unet $JOB)
if [[ "${1:-}" == "--print" ]]; then echo "cd $PWD && ROOT=\"$ROOT\" LOG_DIR=\"$LOG_DIR\" OVERRIDES=\"$OVERRIDES\" PWM_CONDA_ENV=$PWM_CONDA_ENV NPROC=$NPROC ${CMD[*]}"; exit 0; fi
mkdir -p "$LOG_DIR/slurm"
"${CMD[@]}"
