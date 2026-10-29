# Time budget vs performance: block pushing (DRAFT, not launched)

Written 2026-09-27. Status: plan for review. Only the timing calibration (`sweeps/budget_push/calib.py`, held-out hard seed 0,
GPU 0 of babel-q9-28, server port 8820) has been run.

## 1. Fixed choices

| | choice | why |
|---|---|---|
| model | `/data/user_data/mbronars/packages/pwm_final_clean/runs/push0418_rigid_x0_rope_regpass_mix_cont200k_stream/checkpoints/final.pt` (200k continuation, streaming), adapter `clean_push` | every recent push result and the closed-loop baseline comparison use it. The other 200k continuation, `push0418_particle_x0_rope_regpass_mix_f0knn_cont200k`, is particle-native and has not been run through `clean_push`. |
| test set | PointWorld `scenarios/hard/seed_42`, all 50 episodes (the paper set). Paper video episodes 5, 11, 14, 29, 37 are included and also reported separately. | same problems as the paper table |
| calibration set | `scenarios/hard/seed_0` (held out) | no tuning on seed 42 |
| execution | open loop: one 60-action plan, replayed in SAPIEN set-pose (`gp/replay/sapien_push.py`) | same as the paper number |
| metric | simulated cube-centroid distance to target at frame 60, mm (mean, median, std, fraction <= 20/30/50 mm); imagined distance alongside | |
| time | warm planner compute only: sum of `iteration_seconds` (LNO) or `plan_seconds` (baselines). Compile, server start and replay excluded. **Every method timed on the same GPU model (L40S), one GPU, bf16 where the method supports it.** | the PRO 6000 MPPI run (K26, 21.9 s) is not comparable with L40S times |
| plan selection | member / candidate with the lowest imagined objective, never the simulator | |
| seeds | seed 0 for every method's own randomness | |

## 2. LNO base recipe (user, 2026-09-27)

Population 3, S = 6 denoising steps per frame, 60 transitions, full churn staircase, unit-gradient lr 0.1 constant,
t_freeze 0.8, shell projection on, fixed FPS, 6 iterations.

Measured on an L40S (hard seed 0 ep 0, population 3, t_freeze 0.8): 2.7 s per gradient iteration, 0.8 s for the final
no-grad generation, peak 2.7 GiB. Base recipe = 6 x 2.7 + 0.8 = about 17 s per plan.
Earlier (t_freeze 0.5, lr 0.2): population 13 took 6.6 s per iteration and 10.1 GiB; population 8 took 4.5 s and 6.4 GiB.
Per-population numbers for t_freeze 0.8 are being measured now (`sweeps/budget_push/calib.jsonl`: P = 1, 6, 12, 24, 40).

## 3. Budget axis (user, 2026-09-27)

Ten evenly spaced budgets: **10, 20, 30, 40, 50, 60, 70, 80, 90, 100 s** of warm planner compute. Nothing below 10 s
(no single-sample or near-zero plans). For each budget the cell is the (P, I) whose MEASURED mean time on seed 0 is
closest to it; the plotted x is the measured mean time on the 50 test scenarios, not the nominal budget.

## 4. Three LNO scaling curves

Cost model: time = I x g(P) + f(P), g = seconds per gradient iteration, f = final no-grad generation.
Measured (t_freeze 0.8): g(3) = 2.7 s, f(3) = 0.8 s. The P > 3 values below are ESTIMATES from the t_freeze 0.5 runs
(g(8) = 4.5 s, g(13) = 6.6 s, so g(P) is about 1.2 + 0.42 P); they are replaced by the calibration table before launch.

| budget s | A: iterations (P 3) | B: population (I 6) | C: both (P = I) |
|---|---|---|---|
| 10 | I 3 (8.9 s) | P 1 | (4, 4) |
| 20 | I 7 (19.7) | P 5 | (6, 6) |
| 30 | I 11 (30.5) | P 8 | (7, 7) |
| 40 | I 15 (41.3) | P 12 | (8, 8) |
| 50 | I 18 (49.4) | P 16 | (9, 9) |
| 60 | I 22 (60.2) | P 20 | (10, 10) |
| 70 | I 26 (71.0) | P 24 | (11, 11) |
| 80 | I 29 (79.1) | P 27 | (12, 12) |
| 90 | I 33 (89.9) | P 31 | (13, 13) |
| 100 | I 37 (100.7) | P 35 (about 27 GiB) | (14, 14) |

The base recipe (P 3, I 6, about 17 s) is run once as a reference point but is not one of the ten.
Compile: each distinct population size costs a one-time compile (635 s cold for P 3 on babel-q9-28), cached per node.
Cells are assigned to GPUs by population size so each GPU compiles only its own few sizes.

## 5. Baselines on the same budget axis

All open loop (one 60-action plan), replayed in SAPIEN the same way, timed on L40S.

| baseline | budget knob | notes |
|---|---|---|
| Ours, no guidance (MPPI argmin in the world model) | K candidates per replan (commit 5, 12 replans), `bin/mppi_push_scenario.py` | K 39 on an L40S = 56 s, 33.6 mm on hard42; K 13 calibration exists |
| PointWorld | K (population per replan) | **final-frame cost only**: `closed_loop_baseline_push.py --method pw60` settings (every replan rolls out to task frame 60 and scores only that frame: window cap 6, per-frame weight 1e-9). Not the May `pw` config (dynamic horizon cap 4, per-frame cost). |
| GNS | K per replan | **with real resampling**: every replan draws K FRESH rod-template realizations posed at the imagined rod/block state, rolls each to frame 60, keeps the argmin, commits 5, replans. The existing `pushing_eval_gns_recedhoriz.py` does NOT do this (after replan 0 all K candidates are the winner's continuation, bit-identical to single-shot best-of-N), and `gns_push_plan_from_history.py` is single-shot best-of-256. Needs a new script, `bin/gns_push_resample.py`, reusing `sample_fresh_rod_templates` from the recedhoriz file. |

Each baseline gets its own K grid chosen by a calibration pass on seed 0 so its 10 points land on 10, 20, ..., 100 s.

## 6. Compute and GPUs

Each curve sums to 550 s of planning per scenario (10 + 20 + ... + 100). LNO: 3 curves x 550 s x 50 scenarios = about
23 GPU-hours. Baselines: 3 methods x 550 s x 50 = about 23 GPU-hours. Total about 46 GPU-hours plus compiles, i.e. about
6 h on 8 GPUs. SAPIEN replays: 40 cells x 50 = 2000 LNO plans + 1500 baseline plans at 30-60 s each, CPU-bound; run 8+ in
parallel per node.

Requests (general partition, L40S only so the times are comparable), two jobs of 4:

    sbatch -p general --gres=gpu:L40S:4 --cpus-per-task=32 --mem=256G --time=12:00:00 sweeps/budget_push/node.sbatch   # x2

Plus the current interactive job 10579285 (2 x L40S, babel-q9-28). One server per GPU; each node's queue owns a slice of
(curve, cell) pairs so every cell's 50 scenarios run on one GPU model and one population compile.

## 7. Order of work (each step stops for review)

1. Finish calibration on seed 0 (running). Fill in the (P, I) cells for curves A-C.
2. ONE scenario (hard seed 0, ep 0; not a test scenario) through all 30 LNO cells + publish the time vs distance page.
3. Write and parity-check `bin/gns_push_resample.py` (K fresh templates per replan); calibrate GNS, PW60 and MPPI K grids on seed 0.
4. Reserve the 2 x 4 L40S and run all 50 scenarios, all curves, all baselines.
5. Page + table: sim distance vs measured plan seconds, one curve per method, 50-scenario mean with a std band.

## 8. Open questions for the user

- Curve C: P = I (proposed), or keep the base ratio P 3 : I 6?
- Include the paper video episodes 5/11/14/29/37 in the curve means (proposed: include, also report without)?

## 9. Status, 2026-09-27 (after the user's go-ahead: "test as you are running each cell")

- Decisions taken with the proposed defaults: curve C uses P = I; the video episodes stay in the means (also reported without).
- LNO driver: `sweeps/budget_push/run_cell.py` (compile, time on hard seed 0 eps 0-1, move I / P / n until the measured
  time is within max(10 %, 2 s) of the budget, then the 50 hard seed-42 scenarios with bounded background SAPIEN replays;
  resumable). Queue: `sweeps/budget_push/worker.py` claims cells from `cells.json` (29 cells: C010 dropped, it is the
  same setting as A010) with an atomic mkdir under `<root>/_claims`. Node job: `sweeps/budget_push/node.sbatch`.
- Output root (group storage, not the nearly full home volume): `/data/group_data/katefgroup/runs_pwm_mbronars/guided_playback_budget`
  (`runs/pushing/budget_<cell>_P<P>_I<I>_hard42_epNNN`, `_cells/<cell>.json`).
- Smoke test (GPU 1 of babel-q9-28, root `..._budget_smoke`): A010 = P 3, I 3, calibrated 11.2 s; 2 test scenarios, sim 67.9 / 56.9 mm.
- Timing isolation: at these batch sizes planning is kernel-launch bound, so CPU load changes it. On the 4-CPU interactive
  job a concurrent GNS test and a SAPIEN replay pushed a 11 s plan to 17.8 s and 12.4 s. The node job therefore pins each
  server to 3 dedicated cores and confines workers and replays to the other 20.
- Measured t_freeze 0.8 gradient-iteration time (interactive node, some contention): P 1 2.6 s, P 3 2.7 s, P 6 3.7 s; one
  iteration per run is about 2x slower (included in plan time). P 1 x 6 iterations is about 16 s, so B010 cannot reach 10 s:
  it is kept as the population curve's floor at its measured time.
- GNS with resampling: `bin/gns_push_resample.py` (K fresh rod + cube template realizations per replan, posed at the
  winner's Kabsch-fitted rigid poses over the 6 context frames, final-frame score, commit 5). Held-out test (hard seed 0
  ep 0, K 32): 13.0 s planning, imagined 35.4 mm, SAPIEN 51.4 mm; candidate medians 120 mm vs winners 34-87 mm, so the
  replans do resample. Its per-replan cost falls from 2.1 s (60 frames) to 0.4 s (5 frames).
- NOT yet built: baseline cell drivers (MPPI argmin via `bin/mppi_push_scenario.py --K`, PointWorld `pw60` via
  `imagined_mppi.py --population_size`, GNS via `gns_push_resample.py --K`). Blocked on the PointWorld push server command
  (model_path / data_dir for `PointWorld/pointworld_server.py`), which is not recorded in the baseline logs.
- Job submission (sbatch) is blocked for the agent by the permission guard; the user submits.

## 10. LNO results, 2026-09-27 (jobs 10585223 babel-t5-32, 10585224 babel-t5-28, both COMPLETED in 3:47 / 3:53)

All 29 cells, 50 hard seed-42 scenarios each, 0 replay failures. Cell records: `<root>/_cells/<cell>.json`.
Sim = SAPIEN cube-centroid distance to target, mm, mean; plan s = measured mean on the test scenarios.

| budget | A iterations (P 3): I, s, sim | B population (I 6): P, s, sim | C both (P = I): n, s, sim |
|---|---|---|---|
| 10 | 3, 9.9, 68.0 | - | (= A010) |
| 20 | 6, 18.9, 68.4 | 4, 20.5, 65.9 (P 1: 19.1 s, 100.7) | 5, 19.9, 53.8 |
| 30 | 11, 33.6, 60.1 | 8, 30.7, 44.5 | 7, 32.7, 49.3 |
| 40 | 14, 43.4, 60.9 | 12, 40.8, 44.3 | 8, 39.8, 43.3 |
| 50 | 18, 54.7, 56.9 | 16, 51.3, 41.0 | 9, 49.5, 54.4 |
| 60 | 20, 61.4, 56.0 | 20, 61.3, 45.3 | 10, 58.1, 49.0 |
| 70 | 22, 69.1, 60.5 | 24, 70.4, 41.4 | 11, 68.6, 46.0 |
| 80 | 25, 78.5, 53.4 | 29, 84.7, 38.6 | 12, 79.4, 43.9 |
| 90 | 32, 98.7, 51.2 | 33, 94.2, 42.2 | 13, 90.2, 45.5 |
| 100 | 33, 99.5, 50.8 | 37, 104.4, 40.7 | 14, 104.9, 44.7 |

Population scaling is best at every budget from 30 s; it plateaus at about 40 mm from about 50 s while its imagined
distance keeps falling (33 -> 17 mm). Iterations alone improve slowly (68 -> 51 mm over 10 -> 100 s). Reference, not yet on
this axis: no-guidance MPPI argmin K 39 on an L40S = 33.6 mm in 56 s (earlier run, `runs/pushing_mppi/mppi_hard42_K39_l40s_*`).
