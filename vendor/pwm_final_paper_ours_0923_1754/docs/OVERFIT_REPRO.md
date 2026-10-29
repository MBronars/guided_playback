# Historical overfit reproduction: clean-worktree guide

Updated September 8, 2026. The four joint replay runs are complete. Their input
identity and tested computational agreement are established; historical
retraining-score parity is not. Read the retrospective (`docs/CLEANUP_OVERFIT_RETROSPECTIVE.md`, in git history, last present at `79a3c84`)
before interpreting these runs as a blanket cleanup or memorization pass.

## Recipe and supported scope

Use the retained full Transformer or UNet with the same production overfit
entrypoint, `scripts/unified/overfit_prod.py`. Do not use ordinary `train.py`
defaults to reproduce the overfit schedule.

| Setting | Historical-compatible joint recipe |
| --- | --- |
| Configs | `configs/transformer.yaml` or `configs/unet.yaml` |
| Fixed inputs | B4, C4/F16, all points, original train and holdout per domain |
| Domain order | `mpm,pushing,pickup,cloth` |
| Updates | 40,000 global = 10,000 per domain |
| Optimizer | AdamW, LR 1e-4, weight decay 0.001, betas 0.9/0.999, epsilon 1e-8, gradient clipping 1.0 |
| Schedule | Warmup 800, then constant LR; eval 0/every 2,000/final |
| Training randomness | Seed 42, BF16, dropout 0.1, original context noise and clipping |
| Objective | Velocity flow matching with diffusion forcing; native-pose weight 1.0, no auxiliary losses |
| Disabled | Compilation, EMA, finite spike-skipping thresholds |
| Evaluation RNG | Explicit `--eval_rng legacy`; holdout remains isolated with seed 4242 |
| Sampler | Pyramid, nominal 25 inference steps; actual 40 full-window calls |
| Final seeds | Train 10000–10007; holdout 20000–20007 |
| Score | All valid particles including actuators, all 16 futures, physical L2 in mm |
| Transformer compatibility | Explicit `model.temporal_use_sink=false` |

Cross each model with `xy_p99_per_domain` and `xy_p99_global`: four joint jobs,
16 result cells. The eight original specialist jobs are a separate experiment
and were not rerun in this campaign. This is fixed-window reconstruction, not
a long autoregressive benchmark.

## Inputs and historical sources

The canonical shared input location is:

```text
/data/group_data/katefgroup/VLA/3DPCWM/checkpoints/overfit_v2_repro/
  batches/                  # per-domain profile
  batches_xy_p99_global/     # global profile
  receipts/                 # historical results and selection records
```

Each batch directory contains `mpm`, `pushing`, `pickup`, `cloth`, and their
`_hold` counterparts, with `.pt`/`.json` files and published checksums.
MPM has 3,072 points, pushing 256, pickup 1,183, cloth train 12,675 and cloth
holdout 13,664. MPM uses three 1,024-point objects per episode.

The machine-local prepared fixtures are under
`validation/overfit-repro/inputs/{xy_p99_per_domain,xy_p99_global}/`. The tensor
files are symlinks to the original bytes; local receipt copies add identity
metadata. The preparation verifies full fixture contracts without redrawing,
refitting or rebinding stats. A fresh Git checkout does not contain these assets
or the ignored local preparation wrapper.

The original historical protocol, launch commands and 24-cell table remain in
the original checkout at:

```text
/data/user_data/robertre/particle_world_model/docs/OVERFIT_REPRO.md
/data/user_data/robertre/particle_world_model/reports/unified/overfit_v2/
```

That original document names historical configs and paths, not the retained
clean config names. Historical source is `07c909d`; verified cleanup source was
`12dcb6b` plus its recorded patch. Historical data selection preceded later
exclusion fixes; these holdouts are not a new generalization benchmark.

## Safe use of the retained entrypoint

Inspect the interface without training:

```bash
PYTHONPATH=. python scripts/unified/overfit_prod.py --help
```

For any new historical-compatible replay, supply all original train/holdout
names, the matching prepared batch directory and normalization profile,
`--require_saved_batches`, `--eval_rng legacy`, `--save_final`, the schedule
above, and a new output root/tag. Set the full original model/data overrides,
including seed 42, B4, C4/F16 and the Transformer compatibility flag.
Verify the resolved config against a completed launch receipt before allocating
the model. A changed output tag must not imply a changed numerical recipe.

`--require_saved_batches` fails before corpus/GPU access when any required
tensor or receipt is missing. Without that flag, ordinary diagnostic mode may
draw a fresh batch; such a run is not a reproduction. `--eval_rng` defaults to
`isolated` for new diagnostics. Changing from legacy to isolated RNG is an
explicit experimental change, not a historical-compatible convenience fix.

Check GPU ownership and storage headroom before launching; do not infer a GPU
lease from a momentary idle reading. Do not rerun a completed output tag or
reuse the immutable verification directories as new experiment destinations.

## Completed evidence and interpretation

The full retrospective contains all 16 final
scores and the measured MPM/cloth drift. Main local evidence:

- [Final review](../validation/overfit-repro/FINAL_REVIEW_20260907.md).
- [Closure, hashes and test outcomes](../validation/overfit-repro/FINAL_CLOSURE_20260908.json).
- [GIF gallery](../validation/overfit-repro/VERIFICATION_GALLERY.md).
- [Original protocol audit](../validation/overfit-repro/PROTOCOL_AUDIT_20260907.md).

The ordinary production backward comparisons remain nonexact even in
unchanged-code repeats. Native deterministic controls match exactly in the
tested update sequences. Same-weight inference matches for all eight original
per-domain model/domain cells and the added Transformer/global-cloth case.
Those results do not explain all independently trained final-score differences.

Eight noise seeds quantify variability conditional on one checkpoint, not
retraining uncertainty. Do not replace poor final scores with favorable late
snapshots or claim that all baseline verdicts were retained. Original trained
historical checkpoints were unavailable; paired code checks used the new
replay checkpoint in both implementations.

## Preservation across release commits

Old receipts intentionally preserve their original HEAD, dirty patch and source
hashes. A release commit or documentation update changes repository identity;
do not rewrite historical receipts to make them appear to have run at the new
commit. Validate numerical file identities separately and create fresh receipts
for future experiments. Ignored artifacts must be included separately in any
portable handoff.
