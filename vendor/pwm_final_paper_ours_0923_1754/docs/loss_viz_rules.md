# Rules for plotting losses

Written after the round-2 normalization overfit charts were misread, in this repo, by the
person who made them. Every rule below is here because a specific chart failed.

## What went wrong

The training loss is sampled at a random diffusion timestep each step. For velocity-
parameterized flow matching the loss spans **~370x across t** (0.00072 at t=0.95, 0.266 at
t=0.02), so one logged step is one draw from a very wide distribution. Measured on the
round-2 logs:

| | |
|---|---|
| point-to-point scatter within one curve, last third | **3.4x - 38x** (median 3.9x) |
| difference between the three normalization arms | **1.2x - 3.2x** |

The noise was larger than the effect on 6 of 8 panels. Everything else followed:

1. **A noise band was read as a floor.** "Overfit plateaus at 1e-2" — the true large-t
   value was 7e-4, and the curves were still descending at -0.02 to -0.32 dex/1000 steps.
2. **A trend was asserted that the chart could not support.** "falls and flattens ...
   nothing stalls" — nothing on the chart measured flattening.
3. **The gate was a single sample.** `overfit_smoke` passed on `final loss < 0.05`, one draw
   from a band spanning an order of magnitude. Re-measured at 200 steps with the
   deterministic probe, the old gate got two of four domains backwards: it called cloth
   PASS (0.0494) and mpm FAIL (0.0590), while the deterministic numbers were mpm 0.0073
   (fitted) and cloth 0.0226 (not).
4. **A printf format became the data path.** `print(f"loss {loss:.4f}")` turned 5e-6 into
   `0.0000`; the plot scraped that log, and floored the zero to 1e-7 on the log axis,
   drawing a flat line along the bottom of the panel. Three layers, no complaint from any.
5. **A parser silently drew nothing.** The routed logs use `loss X body_pose Y`; the
   regex expected the smoke format and matched no lines, producing empty panels.
6. **An axis choice erased the result.** The first convergence figure divided by 10K and
   collapsed all three arms onto one line.

The common cause: **nobody ever wrote down what one point on the y-axis was.** With no
stated estimator, nothing forced the question "is this estimator's variance smaller than
the effect I am claiming?" — and it was not.

## The rules

**1. Name the estimator on the chart, or do not ship the chart.**
The y-axis label says what one point is: "running median, 201 steps", "loss at fixed
t >= 0.5, mean of 6 noise draws". If that sentence cannot be written, the chart is not ready.

**2. The number that gates or that goes in a table must be deterministic.**
Fixed batch, fixed timestep grid, fixed noise seeds; re-running reproduces it. A statistic
sampled over a nuisance variable is a sample, not a measurement.
→ `scripts/unified/lossprobe.fixed_t_loss`.

**3. Never average over a nuisance variable that spans an order of magnitude — condition on it.**
For flow matching that variable is `t`. Report loss at fixed `t`, or plot loss-vs-`t`.
A uniform-`t` scalar is a mixture and is not comparable across runs, arms, or checkpoints.

**4. Draw the raw series faint and the estimator bold — never raw alone, never estimator alone.**
Raw alone invites reading trends off noise. Estimator alone hides how noisy the underlying
measurement is.

**5. Print the noise floor next to the effect.**
Each panel carries the point-to-point scatter and the trend. If the effect is smaller than
the scatter, the chart's conclusion is "indistinguishable", and that is what gets said.

**6. Plots read structured data, never scraped stdout.**
Scripts write full-precision JSON; plots read it. No `:.4f` anywhere on a path a parser
follows. Where a training log is genuinely the only record, the parser asserts it matched
the expected number of lines.

**7. Guard before drawing; fail loud.**
Non-positive values, non-finite values, too few points to estimate, arms with mismatched
step counts — all raise instead of drawing.
→ `scripts/unified/lossprobe.guard`, and the arm-length check in `plot_overfit.py`.

**8. Comparisons share the batch, the axes, and the step count.**
Byte-identical batches where the comparison allows it (the same hashed `--batch_dir`),
same x-range, same y-range, same number of steps.

**9. Losses in different normalizations are not comparable in height.**
Say so on the figure. Compare shape, trend, and each arm's distance to its own floor.

**10. Open the rendered image and look at it before showing it to anyone.**
The guards check the data; they do not check for collided labels, a legend over a curve,
or a panel that is empty for a reason the guard did not anticipate.

## What enforces this

| | |
|---|---|
| `scripts/unified/lossprobe.py` | `fixed_t_loss` (rule 2, 3), `tail` (rule 5), `guard` (rule 7) |
| `scripts/unified/overfit_prod.py --fixed_probe` | production objective at fixed times/seeds plus full-precision histories, sampler scores and trivial baselines; explicitly a memorization diagnostic |
| `scripts/unified/plot_overfit.py` | raw + estimator (rule 4), per-panel scatter and slope (rule 5), structured JSON (rule 6), guards (rule 7), identical batch hashes / update counts (rule 8) |
| `tests/test_lossprobe.py` | all of the above, including the printf-zero and the coin-flip gate |
| `tests/test_overfit_plot.py` | mismatched fixtures/normalization, unfinished runs, invalid values and nonempty rendered outputs |

The original failures above are historical examples. Their old scripts/reports
remain on the pre-cleanup branch. Fixed seeds remove sampling variability, not all
GPU kernel nondeterminism; record precision and tolerances. Different prediction
parameterizations also define different loss units: compare physical rollout error,
not velocity-loss height against sample-loss height. A near-static fixture may have
an excellent no-motion baseline even when its random-t training loss is low.
