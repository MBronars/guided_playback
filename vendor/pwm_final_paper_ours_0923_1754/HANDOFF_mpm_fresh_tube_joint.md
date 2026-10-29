# Handoff: mpm_fresh joint-attention run (job 10509494)

Launched 2026-09-20 05:04 EDT. Last updated 2026-09-20 14:55 EDT at step 27,000.

## What this is

A like-for-like twin of Robert's `mpm_fresh` UNet specialist, with one thing changed: the
attention. `model.unet_core=tube_joint` replaces the UNet's temporal-then-spatial block pair
with a single joint attention kernel over each call's whole (rows x tokens) sequence. Same data,
normalization, losses, EMA, LR and timestep schedules, strides, depths, `l1_by_mat`, pool/interp
and validation protocol.

The question it answers: does joint attention beat factorized attention on MPM at matched
parameter count (100.63 M vs 102.74 M)?

## Where everything is

| what | absolute path |
|---|---|
| run output | `/data/group_data/katefgroup/runs_pwm_mbronars/pwm_v1_mpm_fresh_100m_tube_joint_L40s_unet_bidir/` |
| checkpoints | `.../checkpoints/{latest,best,final}.pt` + `step_N.pt` every 20k, 1.61 GB each |
| slurm log (step lines live here) | `/data/group_data/katefgroup/runs_pwm_mbronars/slurm/pwm_v1_mpm_fresh_100m_tube_joint_L40s_10509494.log` |
| recipe | `/data/user_data/mbronars/launch_pwm_data_v1_l40s/configs/unet_mpm_fresh_100m_tube_joint_L40s.yaml` (sha256 `911a354391e3`) |
| launcher | `/data/user_data/mbronars/packages/pwm_final_paper/launch/submit_mpm_fresh_100m_tube_joint_L40s.sh` |
| code | `/data/user_data/mbronars/packages/pwm_final_paper` at commit `3b5e4d6` |
| baseline to compare against | `/data/group_data/katefgroup/runs_pwm_robertre/pwm_v1_mpm_fresh_100m_unet_bidir/` (job 10500135) |

The recipe is deliberately kept OUTSIDE the checkout: that worktree is shared with other agents.
The launcher pins the recipe's sha256 and refuses to submit if it has changed.

## Status

| | |
|---|---|
| state | RUNNING on `babel-q5-16`, 2x L40S, partition `general` |
| progress | step 27,000 of 200,000 after 9 h 48 m |
| rate | **0.81 it/s** median (min 0.32 on validation steps, max 0.85), n=270 |
| peak memory | 24.5 GiB of 46 GB |
| stability | **0 skipped steps, 0 non-finite** since step 0 |
| compile | regional, 32 regions, 0 L0 units left eager |

## Resuming — READ THIS FIRST

The 48 h wall arrives 2026-09-22 ~09:04 UTC at roughly **step 140,000**, so this run does
**not** finish in one job. It needs **one** resubmission:

```
cd /data/user_data/mbronars/packages/pwm_final_paper
bash launch/submit_mpm_fresh_100m_tube_joint_L40s.sh
```

Running the launcher again resumes from `latest.pt` automatically. Two things will stop you:

- It refuses a checkout with **modified tracked files**. Untracked files only warn (the worktree
  is shared; `launch/submit_yikai_100m_L40s.sh` is another agent's and is expected to be there).
- The worktree must still be at commit `3b5e4d6`. The job reads code from this directory live,
  not from a snapshot, so do not check out anything else here while the run is in flight.

## How it compares so far

| | joint (step 26k) | baseline UNet (step 21.5k) |
|---|---|---|
| val_loss | 0.0033 (EMA 0.0030) | 0.0019 (EMA 0.0015) |
| x0mm@t.9 | **1.95 - 2.08 mm** | 2.11 mm |
| params | 100,629,589 | 102.74 M |
| hardware | 2x L40S | 2x RTX PRO 6000 |

Val loss is higher, the mm error is slightly lower. Do not read either as a verdict yet: the two
are at different steps and the loss is not a calibrated quantity across architectures. The mm
number is the one to track.

**The hardware differs, so wall-clock is not comparable.** The baseline runs 0.943 s/step on
RTX PRO 6000; this runs 1.23 s/step on L40S. Only model-level ratios carry over.

## What differs from the baseline recipe

Everything not listed is identical.

| | baseline | joint |
|---|---|---|
| `unet_core` | - | `tube_joint` |
| `unet_tube_sizes` | - | `512,256,0,0` |
| `unet_tube_within_object` | - | `0` (L0 stays in an object; **L1 may cross objects**) |
| `unet_tube_grouping` / `group_draws` | - | `random` / `2` |
| `unet_tube_long_skip` / `dec_mirror` | - | `true` / `true` |
| `unet_dual_ffn` | `true` | dropped (no JointBlock equivalent) |
| `unet_widths` | `64,128,256,512` | `88,264,352,616` |
| `unet_head_dim` | `64` | `88` (1/3/4/7 heads) |
| `unet_token_adaln` | - | `true` (material reaches the BLOCKS, not just the seams) |
| `unet_rope_identity` / `unet_rope_budget` | - | `frame0_pos` / `54,12,12,10` |
| registers | `objs: 1` -> 128 @ L2, 64 @ L3 | `tube_register_objs: 0.5` -> **64 @ L2, 32 @ L3** |
| `unet_tube_register_mode` | - | `per_level` (dropped at every seam; nothing crosses) |
| checkpointing | `from_level 0`, seams, ingraph | **all off** |
| `train.compile_l0` | - | `true` |
| batch x accum | `16` x `1` | `4` x `4` (same 32 windows per optimizer step) |

## Gotchas

- **No activation checkpointing is load-bearing.** It is what makes the step fast, and it
  sidesteps the in-graph-checkpoint + compile gradient bug that killed job 10495865 (1e4x
  gradients on 1-object batches; 27% of mpm_fresh episodes are 1-object). Do not turn
  checkpointing back on without re-running a compiled-vs-eager gradient comparison per object
  count. Eager peaks at 37.5 GiB against 24.5 compiled, so eager debugging is tight on a 46 GB card.
- **`compile_regional` matters enormously on this model family.** Measured 2026-09-20 on the
  action3 sibling run, same config and node: regional 3.5-3.96 it/s vs whole-model 0.37 it/s,
  about 10x. Whole-model compile has to swallow the eager-only regions (`_build_provider`,
  `curve_orders`, `_draw_groups` are all `@torch.compiler.disable`) as graph breaks and re-pays
  that on every shape the data produces. Never set `compile_regional: false` here.
- The provenance line in the log says `dirty=1`. That is the other agent's untracked launcher,
  not a modified tracked file. Code revision `3b5e4d6` is authoritative.
- The recipe's `run_name:` field is ignored: the sbatch template derives the run name as
  `pwm_v1_<JOB>_<ARCH>_bidir`, which is why the directory ends in `_unet_bidir` despite not
  being a UNet.
- Step lines are in the **slurm log**, not in the run directory.

## Provenance of the design

Every number behind the design choices was measured on an L40S at batch 4, compiled, no
checkpointing, and is recorded in the launcher's header. The load-bearing ones:

- L0 tube 512 / L1 tube 256 crossing objects: tubing L1 took the 4-object step from 537 to 462 ms.
  Untubed, L1 is a single 19,456-token call and grows quadratically with object count.
- L0 width 88 rather than 176: halves the cost of the level carrying the most tokens for 1.7 M params.
- Half-object registers: full costs 273 ms per micro-step weighted, half costs 241 ms - 7.5 h over
  the run, for 0.04 M params.
- 20% MFU against the baseline UNet's 12% on the same scene (FlopCounterMode fwd+bwd vs 362 TFLOPS,
  the convention used in the earlier pwm_final profiles). The joint core does 2.45x the arithmetic
  per step, 26.7 vs 10.9 TFLOP.

## Open

- **Projection was optimistic.** I projected 0.96 s/step from single-GPU synthetic timing; the run
  does 1.23 s/step. The gap is DDP all-reduce plus whatever my episode-mix weighting got wrong.
  200k at the real rate is ~68.6 h, not the ~53.5 h in the launcher header.
- No rollout evaluation has been run against the baseline yet. `x0mm@t.9` is the only
  cross-model number so far.
