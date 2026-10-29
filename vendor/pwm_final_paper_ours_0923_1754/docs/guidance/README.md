# Guidance study: stronger baselines and matched ablations

Updated 2026-09-16T19:52:06.208695+00:00. State: **running**. 24 planning runs finished; **10/10 baselines meet the comparison prerequisite**.

[Objective ablations](#objective-ablations) · [Speed and noise optimization](#speed-and-noise-optimization) · [Obstacle pilot](#obstacle-pilot) · [Point-cloud GIFs](#completed-point-cloud-comparisons). Animations are expanded below; each also has an **Open GIF** link.

This replaces the 20-update screening study. Earlier good bottle and teacup trajectories used 64 updates; the new study first reproduces those references, then applies their recipe to the same five pick and five place scenes. The earlier runs remain intact in the [20-update report](../../validation/guidance_study_2026-09-15/REPORT_20_UPDATES.md).

**Shared recipe:** frozen EMA checkpoint `99ec6cc6102a0d65…`; seed 8311; 64 Adam updates; learning rate 0.03; noise-change RMS cap 0.5; prior 0.001; 25-step pyramid sampling; mean position error over the final six frames. Picking uses 68 frames, placing 36. The original ten scenes and their requested destinations stay fixed. The teacup reproduction alone uses its historical leftward goal and is listed separately.

The baseline includes all current geometric/interaction terms. Objective ablations remove only their declared families. Speed/optimizer variants retain the full objective; fast10 alone uses ten sampler steps. Every arm for a scene uses the same saved initial noise and routing.

**Baseline check:** Held position <=20 mm; peak convex overlap <=2 mm; support and joint-limit/speed screens pass. Contact/slip/orientation/full task and independent execution are still reported separately. If baseline64 fails the gate, try 128 updates at weight 1, then 128 at position weight 3. First passing recipe is shared by every arm for that scene. All attempts are retained; if none passes, that scene remains visible and its ablations wait for investigation.

A passing baseline check is not a successful simulator grasp. Full model checks, independent execution, contact failures and finger overtravel remain visible. All attempts and all ten scenes are reported; no failed scene is substituted.

Each GIF shows **original unconditional sample → guided prediction → simulator replay of the generated gripper**, using points only. The same original sample appears across a scene’s comparisons.

Deadline: 2026-09-17T19:57:47.084799+00:00. The approved obstacle pilots precede remaining optimizer variants on GPU 0.

## Obstacle pilot

Purple points show the supplied barrier. Compare the unchanged baseline and avoidance plan in the same obstacle world. The checkpoint does not observe the barrier; only the planning cost changes. Clearance checks include every body and interpolated poses.

| Scene | Plan | State | Held goal error | Minimum clearance | Replay passes | GIF |
|---|---|---|---:|---:|---|---|
| pick_development_02_asset_6079 (1 obstacles) | Unchanged baseline | audited | 16.2 mm | -28.1 mm | False | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_02_asset_6079/obstacle_reference/comparison_points.gif) |
| pick_development_02_asset_6079 (1 obstacles) | Avoidance | running | pending | pending | pending | pending |
| pick_development_02_asset_6079 (2 obstacles) | Unchanged baseline | audited | 16.2 mm | -28.1 mm | False | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_02_asset_6079/obstacle_reference2/comparison_points.gif) |
| pick_development_02_asset_6079 (2 obstacles) | Avoidance | pending | pending | pending | pending | pending |
| place_development_00_asset_4943 (1 obstacles) | Unchanged baseline | audited | 1.1 mm | -37.4 mm | False | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_00_asset_4943/obstacle_reference/comparison_points.gif) |
| place_development_00_asset_4943 (1 obstacles) | Avoidance | pending | pending | pending | pending | pending |

[Pilot settings](../../validation/guidance_study_2026-09-15/upgrade64/obstacle_pilot_pick12.json). Negative clearance means penetration.

## Historical reproduction checks

| Reference | State | Current worst held error | Earlier worst held error |
|---|---|---:|---:|
| pick_reserved_00_asset_8270 | passed | 4.4 mm | 4.8 mm |
| reference_teacup_7803 | passed | 1.6 mm | 1.6 mm |

## Five picks and five places

| Scene | Baseline recipe | Worst held error | Model overlap | Prerequisite |
|---|---|---:|---:|---|
| pick_development_00_asset_5624 | 128 updates, goal weight 3 | 9.8 mm | 1.39 mm | passed |
| place_development_00_asset_4943 | 64 updates, goal weight 1 | 1.1 mm | 1.32 mm | passed |
| pick_development_01_asset_8026 | 128 updates, goal weight 1 | 2.9 mm | 1.70 mm | passed |
| place_development_01_asset_5423 | 128 updates, goal weight 1 | 0.8 mm | 1.75 mm | passed |
| pick_development_02_asset_6079 | 128 updates, goal weight 3 | 16.2 mm | 1.88 mm | passed |
| place_development_02_asset_11421 | 128 updates, goal weight 1 | 1.6 mm | 1.56 mm | passed |
| pick_development_03_asset_5173 | 64 updates, goal weight 1 | 4.6 mm | 1.80 mm | passed |
| place_development_03_asset_7803 | 64 updates, goal weight 1 | 2.5 mm | 1.94 mm | passed |
| pick_reserved_00_asset_8270 | 64 updates, goal weight 1 | 4.4 mm | 0.31 mm | passed |
| place_reserved_00_asset_10736 | 64 updates, goal weight 1 | 2.4 mm | 1.26 mm | passed |

## Objective ablations

**3/30 ablation runs complete; 1/10 scenes have the full four-way comparison.** Completed means audited, not necessarily successful.

Each comparison uses that scene’s selected full-objective baseline: identical checkpoint, target, seed, initial trajectory, routing, horizon, optimizer settings and update budget. Earlier baseline-refinement attempts and obstacle runs are excluded.

| Objective | Terms used during optimization |
|---|---|
| Goal only | Requested position and, for placement, orientation |
| Goal + geometry/joints | Goal plus object–gripper overlap, object/table and gripper/table intrusion, finger limits and speed |
| Geometry/joints + contact | Previous row plus proximity to both fingers during lifted motion; picking also asks for final proximity |
| Full objective | Previous row plus unsupported upward motion and relative slip; placement also includes rest, release and settling |

All arms keep the noise regularizer and RMS cap. These are incremental groups, not individual-term leave-one-out tests. The full independent model and replay checks remain active even when their corresponding optimization terms are removed. These checks are geometric proxies; they do not certify force closure. See [the objective pseudocode](CURRENT_GUIDANCE.tex).

**Coverage across the fixed five picks and five places**

| Scene | Selected full baseline | Goal only | Geometry/joints | + Contact |
|---|---|---|---|---|
| pick_development_00_asset_5624 | 128 updates, weight 3 | completed | completed | completed |
| place_development_00_asset_4943 | 64 updates, weight 1 | running | pending | pending |
| pick_development_01_asset_8026 | 128 updates, weight 1 | pending | pending | pending |
| place_development_01_asset_5423 | 128 updates, weight 1 | pending | pending | pending |
| pick_development_02_asset_6079 | 128 updates, weight 3 | pending | pending | pending |
| place_development_02_asset_11421 | 128 updates, weight 1 | pending | pending | pending |
| pick_development_03_asset_5173 | 64 updates, weight 1 | pending | pending | pending |
| place_development_03_asset_7803 | 64 updates, weight 1 | pending | pending | pending |
| pick_reserved_00_asset_8270 | 64 updates, weight 1 | pending | pending | pending |
| place_reserved_00_asset_10736 | 64 updates, weight 1 | pending | pending | pending |

Held error is the **worst position error over the final six frames**. Overlap is peak object–gripper penetration. Read accuracy together with the failed checks; a small position error can coexist with a poor grasp.

### pick_development_00_asset_5624 — objective comparison

128 updates; goal weight 3; 68 frames; seed 8311. Full objective uses `full128_goal3`.

| Objective | State | Held error (mm) | Overlap (mm) | Model checks | Replay | GIF |
|---|---|---:|---:|---|---|---|
| Full objective (baseline) | completed | 9.8 | 1.39 | Fail: finger proximity | Fail | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full128_goal3/comparison_points.gif) |
| Goal only | completed | 27.1 | 20.07 | Fail: overlap, relative slip, goal position | Fail | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/goal/comparison_points.gif) |
| Goal + geometry/joints | completed | 4.5 | 0.00 | Fail: finger proximity, relative slip | Fail | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/geometry/comparison_points.gif) |
| Geometry/joints + contact | completed | 9.9 | 1.87 | Fail: finger proximity, relative slip | Fail | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/contact/comparison_points.gif) |

The labeled panels show the original unconditional sample, the guided prediction, and simulator replay. Only completed runs are embedded; pending results appear here automatically when ready.

**Full objective (baseline)**

Held error: 9.8 mm. Overlap: 1.39 mm. Model checks: Fail: finger proximity. Replay: Fail.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full128_goal3/comparison_points.gif)

![pick_development_00_asset_5624 — Full objective (baseline)](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full128_goal3/comparison_points.gif)

**Goal only**

Held error: 27.1 mm. Overlap: 20.07 mm. Model checks: Fail: overlap, relative slip, goal position. Replay: Fail.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/goal/comparison_points.gif)

![pick_development_00_asset_5624 — Goal only](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/goal/comparison_points.gif)

**Goal + geometry/joints**

Held error: 4.5 mm. Overlap: 0.00 mm. Model checks: Fail: finger proximity, relative slip. Replay: Fail.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/geometry/comparison_points.gif)

![pick_development_00_asset_5624 — Goal + geometry/joints](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/geometry/comparison_points.gif)

**Geometry/joints + contact**

Held error: 9.9 mm. Overlap: 1.87 mm. Model checks: Fail: finger proximity, relative slip. Replay: Fail.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/contact/comparison_points.gif)

![pick_development_00_asset_5624 — Geometry/joints + contact](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/contact/comparison_points.gif)

### place_development_00_asset_4943 — objective comparison

64 updates; goal weight 1; 36 frames; seed 8311. Full objective uses `full64`.

| Objective | State | Held error (mm) | Overlap (mm) | Model checks | Replay | GIF |
|---|---|---:|---:|---|---|---|
| Full objective (baseline) | completed | 1.1 | 1.32 | Pass | Fail | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_00_asset_4943/full64/comparison_points.gif) |
| Goal only | running | — | — | — | — | — |
| Goal + geometry/joints | pending | — | — | — | — | — |
| Geometry/joints + contact | pending | — | — | — | — | — |

The labeled panels show the original unconditional sample, the guided prediction, and simulator replay. Only completed runs are embedded; pending results appear here automatically when ready.

**Full objective (baseline)**

Held error: 1.1 mm. Overlap: 1.32 mm. Model checks: Pass. Replay: Fail.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_00_asset_4943/full64/comparison_points.gif)

![place_development_00_asset_4943 — Full objective (baseline)](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_00_asset_4943/full64/comparison_points.gif)

## Speed and noise optimization

**3/40 variant runs complete.** All retain the full objective.

| Variant | Change from the full-objective baseline |
|---|---|
| 10 sampler steps | Reduce nominal denoising steps from 25 to 10 |
| Four noise knots | Optimize four temporally interpolated update knots per window, preserving the original Gaussian draw |
| Four noise knots + residual | Add a per-frame residual capped at RMS 0.10; total update remains capped at 0.50 |
| Cosine rate + normalized gradient | Two-update warmup, cosine learning rate and normalized noise gradient |

Budgets match optimization updates, not model calls. Speedup is relative to the selected full baseline on the same scene. Times cover the planning subprocess, including startup, but exclude replay and GIF rendering; `~` marks an output-timestamp estimate. The 10-step sampler shares initial noise and routing but produces a different initial trajectory. Reducing the number of noise-update parameters does not reduce the number of model evaluations.

### pick_development_00_asset_5624 — speed comparison

128 updates; goal weight 3; 68 frames; seed 8311. Full objective uses `full128_goal3`.

| Method | State | Held error (mm) | Overlap (mm) | Minutes | Speedup | Model checks | Replay | GIF |
|---|---|---:|---:|---:|---:|---|---|---|
| Full objective (baseline) | completed | 9.8 | 1.39 | 105.2 | 1.00× | Fail: finger proximity | Fail | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full128_goal3/comparison_points.gif) |
| 10 sampler steps | completed | 33.0 | 3.01 | 62.7 | 1.68× | Fail: finger proximity, overlap, goal position | Fail | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/fast10/comparison_points.gif) |
| Four noise knots + residual | completed | 27.0 | 0.92 | 103.4 | 1.02× | Fail: goal position | Fail | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/coarse_residual/comparison_points.gif) |
| Four noise knots | completed | 5.3 | 2.19 | ~104.5 | ~1.01× | Fail: finger proximity, overlap | Fail | [Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/coarse4/comparison_points.gif) |
| Cosine rate + normalized gradient | pending | — | — | — | — | — | — | — |

## Completed point-cloud comparisons

### reference_teacup_7803

**full64 — 64 updates; goal weight 1**

Model worst held error: 1.6 mm. Full model checks: True. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/reference_teacup_7803/full64/comparison_points.gif)

![reference_teacup_7803 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/reference_teacup_7803/full64/comparison_points.gif)

### pick_development_00_asset_5624

**full64 — 64 updates; goal weight 1**

Model worst held error: 24.7 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.6 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full64/comparison_points.gif)

![pick_development_00_asset_5624 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full64/comparison_points.gif)

**full128 — 128 updates; goal weight 1**

Model worst held error: 8.8 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full128/comparison_points.gif)

![pick_development_00_asset_5624 full128](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full128/comparison_points.gif)

**full128_goal3 — 128 updates; goal weight 3**

Model worst held error: 9.8 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full128_goal3/comparison_points.gif)

![pick_development_00_asset_5624 full128_goal3](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/full128_goal3/comparison_points.gif)

**goal — 128 updates; goal weight 3**

Model worst held error: 27.1 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/goal/comparison_points.gif)

![pick_development_00_asset_5624 goal](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/goal/comparison_points.gif)

**geometry — 128 updates; goal weight 3**

Model worst held error: 4.5 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/geometry/comparison_points.gif)

![pick_development_00_asset_5624 geometry](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/geometry/comparison_points.gif)

**contact — 128 updates; goal weight 3**

Model worst held error: 9.9 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/contact/comparison_points.gif)

![pick_development_00_asset_5624 contact](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/contact/comparison_points.gif)

**fast10 — 128 updates; goal weight 3**

Model worst held error: 33.0 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 9.9 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/fast10/comparison_points.gif)

![pick_development_00_asset_5624 fast10](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/fast10/comparison_points.gif)

**coarse_residual — 128 updates; goal weight 3**

Model worst held error: 27.0 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/coarse_residual/comparison_points.gif)

![pick_development_00_asset_5624 coarse_residual](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/coarse_residual/comparison_points.gif)

**coarse4 — 128 updates; goal weight 3**

Model worst held error: 5.3 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/coarse4/comparison_points.gif)

![pick_development_00_asset_5624 coarse4](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_00_asset_5624/coarse4/comparison_points.gif)

### place_development_00_asset_4943

**full64 — 64 updates; goal weight 1**

Model worst held error: 1.1 mm. Full model checks: True. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_00_asset_4943/full64/comparison_points.gif)

![place_development_00_asset_4943 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_00_asset_4943/full64/comparison_points.gif)

### pick_development_01_asset_8026

**full64 — 64 updates; goal weight 1**

Model worst held error: 3.6 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_01_asset_8026/full64/comparison_points.gif)

![pick_development_01_asset_8026 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_01_asset_8026/full64/comparison_points.gif)

**full128 — 128 updates; goal weight 1**

Model worst held error: 2.9 mm. Full model checks: True. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_01_asset_8026/full128/comparison_points.gif)

![pick_development_01_asset_8026 full128](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_01_asset_8026/full128/comparison_points.gif)

### place_development_01_asset_5423

**full64 — 64 updates; goal weight 1**

Model worst held error: 1.2 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_01_asset_5423/full64/comparison_points.gif)

![place_development_01_asset_5423 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_01_asset_5423/full64/comparison_points.gif)

**full128 — 128 updates; goal weight 1**

Model worst held error: 0.8 mm. Full model checks: True. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_01_asset_5423/full128/comparison_points.gif)

![place_development_01_asset_5423 full128](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_01_asset_5423/full128/comparison_points.gif)

### pick_development_02_asset_6079

**full64 — 64 updates; goal weight 1**

Model worst held error: 44.3 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_02_asset_6079/full64/comparison_points.gif)

![pick_development_02_asset_6079 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_02_asset_6079/full64/comparison_points.gif)

**full128 — 128 updates; goal weight 1**

Model worst held error: 21.7 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_02_asset_6079/full128/comparison_points.gif)

![pick_development_02_asset_6079 full128](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_02_asset_6079/full128/comparison_points.gif)

**full128_goal3 — 128 updates; goal weight 3**

Model worst held error: 16.2 mm. Full model checks: True. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_02_asset_6079/full128_goal3/comparison_points.gif)

![pick_development_02_asset_6079 full128_goal3](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_02_asset_6079/full128_goal3/comparison_points.gif)

### place_development_02_asset_11421

**full64 — 64 updates; goal weight 1**

Model worst held error: 9.8 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_02_asset_11421/full64/comparison_points.gif)

![place_development_02_asset_11421 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_02_asset_11421/full64/comparison_points.gif)

**full128 — 128 updates; goal weight 1**

Model worst held error: 1.6 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_02_asset_11421/full128/comparison_points.gif)

![place_development_02_asset_11421 full128](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_02_asset_11421/full128/comparison_points.gif)

### pick_development_03_asset_5173

**full64 — 64 updates; goal weight 1**

Model worst held error: 4.6 mm. Full model checks: True. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_03_asset_5173/full64/comparison_points.gif)

![pick_development_03_asset_5173 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_development_03_asset_5173/full64/comparison_points.gif)

### place_development_03_asset_7803

**full64 — 64 updates; goal weight 1**

Model worst held error: 2.5 mm. Full model checks: True. Qualified execution: False. Replay finger overtravel: 1.8 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_03_asset_7803/full64/comparison_points.gif)

![place_development_03_asset_7803 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_development_03_asset_7803/full64/comparison_points.gif)

### pick_reserved_00_asset_8270

**full64 — 64 updates; goal weight 1**

Model worst held error: 4.4 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_reserved_00_asset_8270/full64/comparison_points.gif)

![pick_reserved_00_asset_8270 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/pick_reserved_00_asset_8270/full64/comparison_points.gif)

### place_reserved_00_asset_10736

**full64 — 64 updates; goal weight 1**

Model worst held error: 2.4 mm. Full model checks: False. Qualified execution: False. Replay finger overtravel: 0.0 mm.

[Open GIF](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_reserved_00_asset_10736/full64/comparison_points.gif)

![place_reserved_00_asset_10736 full64](../../validation/guidance_study_2026-09-15/upgrade64/artifacts/place_reserved_00_asset_10736/full64/comparison_points.gif)

## Queue and provenance

[Frozen plan](../../validation/guidance_study_2026-09-15/upgrade64/plan.json) · [Queue](../../validation/guidance_study_2026-09-15/upgrade64/queue.json) · [Source snapshot](../../validation/guidance_study_2026-09-15/upgrade64/source_receipt.json) · [Budget](../../validation/guidance_study_2026-09-15/upgrade64/budget.json)
