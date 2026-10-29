# Mould: no-guidance (MPPI argmin) baseline on the paper's cloud_centroid scenarios (seed 42)

Run 2026-09-24. Same model, scenarios, sampler settings and Genesis replay as the guided eval in MOULD_PAPER_EVAL.md;
only the planner differs. Nothing here was computed by hand: `python bin/paper_report.py --summary runs/_paper_eval/mppi42_K6_summary.json`.

## Planner (gp/mppi_argmin_paper.py, POST /plan on the paper-adapter server)
Per replan: draw K = 6 fresh candidates (2 chunks of the compiled batch 3) of ALL remaining transitions from the committed
prefix, with the guided eval's sampler (150 transitions, keep 3, S = 6, full churn eta = 1 on pose and particle rows, fixed
per-member FPS; every model call the full 19-row window -- the window slides back over committed rows near the end instead of
shrinking). Score = dough-centroid distance to the target at the last frame. Keep the ARGMIN candidate, commit its next 5
transitions, repeat: 30 replans. No gradients, no averaging. Seeds 1000003*0 + 1009*replan + chunk. K chosen so one plan
takes ~150 s (user target): K = 3 measured 75.2 s -> K = 6.

## Results, all 50 scenarios (distance closed = target_mm - Genesis centroid-to-target at frame 150)

| planner | closed mean | median | min / max | negative | >= 20 mm | >= 30 mm | plan s (warm) |
|---|---|---|---|---|---|---|---|
| MPPI argmin K = 6, commit 5 | 14.8 mm | 14.2 | -0.8 / 38.1 | 1/50 | 12/50 | 1/50 | 151.5 +- 0.9 (min 151.0, max 156.6) |
| guided (MOULD_PAPER_EVAL.md) | 26.8 mm | 26.0 | 11.3 / 41.9 | 0/50 | 40/50 | 15/50 | 113.8 |

Paired over the same 50 scenarios: guided closes 12.0 mm more on average; MPPI closes more on 7/50. MPPI imagined vs Genesis
gap 1.8 mm mean (max 4.7). First plain sample (no selection) closes -0.1 mm imagined; the MPPI plan 14.9 mm imagined.
Genesis replay mean 88 s (max 147) per scenario (one member, GPU 0, after the pickup queue freed it).

## Artefacts
* summary: /data/user_data/mbronars/packages/guided_playback/runs/_paper_eval/mppi42_K6_summary.json
* runs: /data/user_data/mbronars/packages/guided_playback/runs/mould/mppi42_K6_ep000 .. ep049 (result.json: every candidate's
  objective per replan, seeds, per-replan seconds; rollout.npz; replay_genesis/)
* driver: bin/paper_eval_mppi.py; replay loop: sweeps/mppi42_K6_replay_after_pickup.sh; logs/mppi42_K6_plan.log, logs/mppi42_K6_replay.log
* K = 3 calibration case: runs/mould/mppi42_K3_ep000 (75.2 s, closed -2.6 mm)
