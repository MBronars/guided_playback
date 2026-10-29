#!/bin/bash
# =============================================================================
#  Training run on the pwm-data-v1 release: a specialist (one dataset) or the joint
#  run (every dataset), either backbone, from a recipe written by
#  scripts/unified/collection_recipe.py --indices <release>. Goes through
#  launch/train_collections.sh, which validates the recipe and prints the code
#  revision, the normalization digest and the corpus identity before the first step.
#
#  Usage            bash launch/submit_<job>.sh [--print]: one script per actual job. It checks the checkout and the
#                   recipe's sha256, then submits this template with the job's recipe name, resources and overrides.
#  Usage (SLURM)    sbatch launch/train_pwm_data_v1.sbatch <unet|transformer> <recipe name>
#                   recipe name = the recipe's file stem after <arch>_. Launch plan of 2026-09-19, every run a
#                   UNet: <dataset>_100m for the specialists (folding_8mm, pushing, rl_push, yikai, mpm_fresh,
#                   mpm_drag, dex_v2_filtered2, pickup; collection_recipe.py --unet-size 100m, 103.10 M
#                   parameters), and for the unified model on all eight datasets joint8_500m (--unet-size 500m,
#                   502.22 M) and joint8_100m, each on 4 GPUs at batch 4 per GPU:
#                   NPROC=4 sbatch --gres=gpu:RTX_PRO_6000:4 --cpus-per-task=32 --mem=256G ... unet joint8_100m
#                   Specialist protocol of 2026-09-19 (user), written into the recipe by collection_recipe.py
#                   (--prediction-type sample --light-context-noise 0.05 --ema-decay 0.9999
#                   --timestep-schedule independent=0.6,ramp=0.4 --batch-size N): x0 prediction with the rigid pose
#                   loss in sample space, light context noise 0.05, EMA 0.9999 with warm-up, training noise levels
#                   0.6 independent + 0.4 ramp, no weight cap, no clipping; scales artifact v7; pushing reads its
#                   train folder without the 533 episodes of the release's exclusion list.
#                   Added 2026-09-19 (user): pickup_100m, the earlier pick-and-place corpus as pwm-data-v1/pickup (its recipe carries
#                   data.roots.pickup.shift_m, the folder's own table-height translation), and pushing_100m_light_pose_noise, the
#                   pushing specialist with rigid history noise (collection_recipe.py --light-pose-noise 0.05
#                   --light-pose-noise-anchor), a comparison run next to pushing_100m.
#  Usage (dev job)  srun --jobid=<id> --overlap --ntasks=1 --gres=gpu:2 \
#                       bash launch/train_pwm_data_v1.sbatch unet yikai
#
#  ROOT (default /data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1):
#      configs/<arch>_<name>.yaml      the recipe (prep step; data.roots = the release's train
#                                      and test folders, the release's current scales artifact, window policy)
#      PREFLIGHT_<name>.md             what must be confirmed before this is submitted
#  Accounts Every default below is the calling account's own, so two people who run the same job never share a run folder
#           (a shared one would make the second job resume the first one's latest.pt):
#           LOG_DIR  /data/group_data/katefgroup/runs_pwm_<account>      runs, checkpoints, SLURM logs
#           SCRATCH  /data/user_data/<account>/scratchpad/launch_pwm_data_v1
#           PWM_CONDA_SH   $HOME/miniconda3/etc/profile.d/conda.sh       the conda installation to activate from
#           PWM_CONDA_ENV  pwm_blackwell                                 the environment of docs/ENVIRONMENT.md
#           The #SBATCH --output line below cannot read a variable: launch/submit_<job>.sh passes --output under LOG_DIR;
#           with a bare sbatch from another account, pass --output yourself.
#  Output   LOG_DIR/<run name>/,
#           16 bytes per parameter per checkpoint (1.65 GB at 103 M, 5.3 GB at 331 M, 8.0 GB at 502 M):
#           step_<N>.pt every 20K, latest.pt, best.pt, final.pt.
#           The release's test folder is the validation folder (decision 2026-09-17), so
#           best.pt and the curves read the test set; final.pt is the reported checkpoint.
#  Attention is bidirectional (decision 2026-09-18).
#  GPUs: two RTX PRO 6000 (96 GB; decision 2026-09-19). Measured on the mpm_fresh specialist at batch 8 per GPU:
#  2.1 steps/s on two PRO 6000 (PCIe) against 1.3 on two A100 40GB (NVLink). Another type: sbatch --gres=gpu:<TYPE>:2 ...
#  Optional  STEPS (0 = the full 200K schedule; otherwise the inclusive zero-based stop label)
#            NPROC (2 GPUs)  BATCH (per GPU; default = the recipe's)  RUN_NAME  LOG_DIR
#            REPO (the submit directory: submit from a pinned worktree)  SCRATCH
#            OVERRIDES (extra key=value words)
#  Resubmitting the same run name resumes from its latest.pt (first written at step 1,000; a run that died
#  earlier has no latest.pt, and the launcher refuses its existing folder: rename the folder, then resubmit).
#  train.nccl_timeout_min=60: rank 0 alone writes the due checkpoint files while the other rank waits in the
#  next all-reduce. Slow writes to the run volume passed NCCL's default 10 minutes and killed the two-GPU 200K
#  run of 2026-09-15 at step 66,000; it finished after this setting.
#
#  Status  TEMPLATE. Nothing is submitted without the mandatory pre-flight of CLAUDE.md,
#          confirmed by the user for the specific run: PREFLIGHT_<name>.md lists it.
# =============================================================================

#SBATCH --job-name=pwm_data_v1
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --gres=gpu:RTX_PRO_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2-23:30:00
#SBATCH --output=/data/group_data/katefgroup/runs_pwm_robertre/slurm/%x_%j.log

set -euo pipefail

ARCH=${1:?arch: unet | transformer}
NAME=${2:?recipe name: a dataset folder of the release, or joint}
case "$ARCH" in
    unet)        ATTENTION=(model.unet_bidirectional_temporal=true) ;;
    transformer) ATTENTION=(model.causal_temporal=false) ;;
    *)           echo "arch must be unet or transformer" >&2; exit 2 ;;
esac

# ---- locations --------------------------------------------------------------
ROOT=${ROOT:-/data/group_data/katefgroup/runs_pwm_robertre/launch_pwm_data_v1}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ME=$(id -un)                       # the defaults are the calling account's own (see Accounts in the header)
LOG_DIR=${LOG_DIR:-/data/group_data/katefgroup/runs_pwm_$ME}
SCRATCH=${SCRATCH:-/data/user_data/$ME/scratchpad/launch_pwm_data_v1}
CONFIG=$ROOT/configs/${ARCH}_${NAME}.yaml
RUN_NAME=${RUN_NAME:-pwm_v1_${NAME}_${ARCH}_bidir}
STEPS=${STEPS:-0}
NPROC=${NPROC:-2}
OVERRIDES=${OVERRIDES:-}

[[ -f "$REPO/train.py" && -f "$REPO/launch/train_collections.sh" ]] || { echo "REPO=$REPO is not a checkout" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "missing recipe $CONFIG (run the prep step first)" >&2; exit 2; }

# ---- environment ------------------------------------------------------------
PWM_CONDA_SH=${PWM_CONDA_SH:-${HOME:-/home/$ME}/miniconda3/etc/profile.d/conda.sh}
[[ -r "$PWM_CONDA_SH" ]] || { echo "cannot read $PWM_CONDA_SH: set PWM_CONDA_SH=<your conda>/etc/profile.d/conda.sh" >&2; exit 2; }
source "$PWM_CONDA_SH"
conda activate "${PWM_CONDA_ENV:-pwm_blackwell}"
export PYTHONUNBUFFERED=1
mkdir -p "$LOG_DIR/slurm" "$SCRATCH/tmp"
cd "$REPO"

BATCH_WORDS=()
if [[ -n "${BATCH:-}" ]]; then BATCH_WORDS=(data.batch_size="$BATCH"); fi

echo "[$RUN_NAME] start $(date -u +%FT%TZ) host=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)" \
     "rev=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) recipe_sha=$(sha256sum "$CONFIG" | cut -c1-12)" \
     "steps=$STEPS nproc=$NPROC batch=${BATCH:-recipe} overrides=[$OVERRIDES]"

# ---- resume on resubmission -------------------------------------------------
CKPT="$LOG_DIR/$RUN_NAME/checkpoints/latest.pt"
if [[ -f "$CKPT" ]]; then
    echo "=== resuming from $CKPT"
    export RESUME="$CKPT"
else
    unset RESUME
fi

# ---- launch -----------------------------------------------------------------
# OVERRIDES is intentionally unquoted: a space-separated list of key=value words.
REPO="$REPO" PYTHON="$(command -v python)" CONFIG="$CONFIG" LOG_DIR="$LOG_DIR" RUN_NAME="$RUN_NAME" \
SCRATCH="$SCRATCH" NPROC="$NPROC" PORT=$((29800 + ${SLURM_JOB_ID:-0} % 100)) \
    bash launch/train_collections.sh --run "${ATTENTION[@]}" "${BATCH_WORDS[@]}" \
        train.stop_after_step="$STEPS" train.nccl_timeout_min=60 log.save_every=20000 log.project=particle_world_model $OVERRIDES
