# Why does guided sampling not beat plain sampling on block pushing? (diagnostic study)

Started 2026-09-27 after the budget study (docs/BUDGET_PUSH_PLAN.md section 10). User's ask: make the MPPI (no guidance) result a
full budget curve, and test from the basics whether the guidance formulation, its parameters, or the optimized noise are off.
Target: guidance reaching a given quality FASTER than plain sampling, not necessarily a better asymptote.

Model, scenarios, sampler and replay: as in the budget study (push0418 rigid cont200k_stream, S 6, 60 transitions, full churn,
fixed FPS, SAPIEN set-pose replay). Development set: hard seed 0 (held out). Test set: hard seed 42 (paper), used only to confirm.

## What the budget-study traces already show (no new runs)

Over all 29 cells x 50 scenarios, every member's imagined objective at every iteration is saved (result.json trace_mm).
- 40-46 % of optimizer steps INCREASE a member's objective; mean |change| per step 16-23 mm. Descent should rarely go up:
  the updates act close to a random walk, and the gain comes from keeping the best iterate (random search).
- The same seed and scenario give different iteration-0 objectives across cells: usually 0-5 mm apart, up to 34 mm.
  bf16 compiled numerics alone can flip a plan, so the objective is very rough as a function of the noise.
- unit-gradient lr 0.1 moves EVERY optimized row by 10 % of its expected norm per step.

## Experiments

E1 gradient probe (`sweeps/push_diag/probe_grad.py`, running on babel-p5-20 GPU 0 bf16 / GPU 1 fp32, 20 dev scenarios, 4 members).
   Per scenario and member: J, dJ/dx; steps along -grad vs a random direction of the same size, the optimizer's own per-row
   normalization + shell, alpha 1e-4 .. 0.3, for the noise subsets P0 (initial pose noise), Ehi (bank rows at level >= 0.8, the
   current recipe), Eall (every used bank row), all. Also the first-order forecast and a repeat of J at x (noise floor).
   Question: is the gradient a descent direction at all, at what step size, and on which part of the noise?
E2 sampling vs guidance per model evaluation (dev set, imagined objective, then SAPIEN for the best settings):
   best-of-N plain samples; guided with the settings E1 points to (subset, step size, optimizer, shell); guided MPPI
   (receding horizon with a few gradient steps per replan) if one-shot guidance stays at random-search level.
E3 objective: the final-frame distance has no gradient when the rod never touches the cube. Measure contact vs no contact
   per member, and test a shaped approach term (rod to the push point behind the cube) if E1 shows vanishing gradients.
E4 MPPI budget curve (`sweeps/budget_push/run_cell_mppi.py`, K per replan calibrated to 10..100 s) and the final comparison
   curves on the test set: on the 32-CPU batch nodes only, so wall-clock times match the LNO cells.

## E1 result (probe, babel-p5-20, 18 dev scenarios x 4 members bf16; fp32 12 scenarios, same picture)

Median one-step change of the imagined objective (mm), fraction of members improved, and fraction where the gradient step beats a
random step of the same size (optimizer's per-row normalization + shell):
- Ehi (current recipe, levels >= 0.8), alpha 0.1: -4.2 mm, 67 % improve, beats random 64 %.
- P0 (initial pose noise) alpha 0.1: -2.4 mm, 75 %, 76 %.
- Eall (every churn row) alpha 0.01: -6.8 mm, 72 %, 68 %; alpha 0.03: -9.4, 58 %, 58 %; alpha 0.1: -20.7, 62 %, 53 %.
- Tiny steps (alpha <= 1e-3): changes ~0 and the first-order forecast does not predict them (rough objective at that scale).
Conclusions: the gradient is a real descent direction (not a sign or precision bug; fp32 = bf16). t_freeze 0.8 freezes the churn
rows that matter most. lr 0.1 on all rows is past the scale where the gradient beats a random direction. Next: t_freeze 0 with
lr 0.003-0.03 (queued b25-b29, b53).

## Imagination-to-simulation transfer (all 1450 budget-study plans, no new runs)

corr(imagined, sim) = 0.72. Sim distance by imagined distance of the committed plan:
imagined 0-10 mm -> sim mean 34.0 (median 33.2); 10-20 -> 37.5; 20-30 -> 43.8; 30-50 -> 50.9; 50-80 -> 71.6; >80 -> 100.7.
Below ~20 mm imagined the simulated result floors near 34 mm (model error). MPPI K 39 (imagined 11.4, sim 33.6) sits on the
same floor: guided plans at equal imagined distance transfer as well as sampled ones (no sign of adversarial plans).
So the comparison that matters is compute to reach the floor, i.e. imagined distance vs seconds, with sim confirming.

Infrastructure (2026-09-28): holding jobs 10590675 (babel-p9-20, 4 L40S) and 10590694 (babel-s5-28, 2 L40S), started with
srun --overlap -c N --cpu-bind=none [--mem 80G] bash sweeps/queue/node_queue.sbatch; queue <root>/queue/{pending,running,done,failed,logs}.

## Wave 1 results (finished 2026-09-28, 39 jobs, 0 failures)

MPPI budget curve, test set (hard 42, 50 eps), sim mm / plan s: K4 75.3 @ 7.9 | K12 50.2 @ 23.8 | K16 54.6 @ 32.3 | K24 43.3 @ 37.8 |
K32 40.1 @ 50.0 | K40 38.4 @ 63.3 | K48 36.8 @ 74.9 (and 34.3 @ 74.1) | K56 32.6 @ 88.2 | K64 33.7 @ 100.8.
Dev set (hard 0), ~30 s configs, sim mm (imagined): guidance t_freeze 0 lr 0.1 P8 I6 38.7 (24.4) = best; lr 0.03 41.6; t_freeze 0.5 lr 0.03
47.0; lr 0.3 46.9; base (t_freeze 0.8) 48.8 (38.3); Adam 50.7; no shell 51.0; lr 0.03 (t_freeze 0.8) 55.5; P0 only 57.7; lr 0.01 46.0; lr 0.003 54.6.
P4 I12 (base) 66.9, (t_freeze 0, lr 0.01) 61.4: population beats iterations again. MPPI K16 53.1 @ 31.7 s; K40 35.3 @ 62 s; K8 57.5 @ 16.4 s.
Plain one-shot best-of-N: N4 96.0 @ 0.8 s, N8 62.9 @ 1.3, N16 71.5 @ 2.3, N32 51.3 @ 4.2, N64 50.3 @ 8.0.
Guided MPPI (gradient steps inside every replan) is slow and poor: K4 I2 63.0 @ 70 s, K4 I3 t_freeze 0 53.5 @ 99 s, K8 I2 60.3 @ 140 s.
Up-step fraction stays 0.37-0.44 for every guided variant.
Interactive job 10590046 ended (12 h limit) before the per-iteration curves ran; not repeated (wave 1 answers the question).
Wave 2 queued 2026-09-28 23:10: test-set t_freeze 0 population curve (P 4-32, I 6), best-of-N 16/64/128, sample-then-refine
(prefilter 7 x P8 I6, prefilter 15 x P8 I3); dev: t_freeze 0 I 3 / I 10, lr 0.2, best-of-128, sample-then-refine.

## Closed loop (2026-09-29): what had to be fixed
- Tail windows: the open-loop samplers (push clean_core.Staircase, pick inference.streaming_chain) shrink the last windows to 4-18 rows;
  the frames the objective scores come from truncated windows. Full-window variants: push = prefix sampler with a frame-0 history and 26
  padded noise rows (run_cfg --pad_tail 26); pick = gp/adapters/pick_chain.streaming_chain_stop (noise for 60 + 16, commit 0..60).
  MPPI always used full windows. All open-loop test results are being re-run with full windows (r*, t* jobs).
- Pick closed loop: re-anchoring on the executed frame only (context 0) lost the hand's motion; plans dithered (smoke: closed 27 mm).
  Executed frames 0..c as clean prefix rows (pick_chain.executed_prefix, model's own pose-delta convention, 0.73 mm rebuild error) +
  full sliding windows fixed context, but fresh noise every replan switched plans and knocked the carton over (closed -35 mm).
  Warm start (member 0 continues the committed member's best noise, shifted past the executed rows; session init_noise /
  save_best_noise) -> closed 131 mm on the same problem. Default for pick and push closed loop now (push: pose noise only).
