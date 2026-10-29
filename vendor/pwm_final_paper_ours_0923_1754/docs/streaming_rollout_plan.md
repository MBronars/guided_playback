# Streaming rollouts for the context-predicting rl_push model — agreed plan

**Status 2026-09-20: scoped and agreed, NOT built.** The staged recipe
(`unet_rl_push_ctxgen_100m_L40s.yaml`) still trains on `{independent: 0.6, ramp: 0.4}` and its
logged rollouts still walk the pyramid. This document is what replaces that, and why.

The machinery exists on branch `mb/pick_place` of the same repository (working tree
`/data/user_data/mbronars/packages/pwm_final_clean`), where `docs/streaming_sampling.md` is the
reference. This file records the decisions for THIS tree, which differ.

## 1. Settings (user, 2026-09-20)

| setting | value | note |
|---|---|---|
| window `n` | 19 rows | `context_len` 4 + `future_len` 16 - 1. Same as pickup, so its table transfers unchanged |
| keep `c` | **3 rows** | the 4 context frames the recipe reads. `m = n - c = 16`, so the ramp slope is 1/16 per row and `u = S/16` calls between rows entering |
| given `g` | **0** | frame 0 is the only given state; every transition is generated |
| steps `S` | **sampled, not fixed** | the table's grid, S in 4..50, uniform. S is the setting training varies over |
| draw | **`{independent: 0.6, stream: 0.4}`** | one schedule per sample, both over all 19 rows |
| jitter | 0.048 | the level whose tau/(1-tau) equals a light-noise sigma of 0.05 |
| loss weighting | **unchanged from this tree** | NO `velocity_weight_cap`, `grad_clip` stays 1.0. The pickup stream recipe pairs stream draws with a cap of 400; that is explicitly not adopted here |
| clock conventions | **unchanged from this tree** | no `future_t_low`, no `sampler_full_clock`. See section 4 |

## 2. What a streaming rollout is

A row is one transition. The window always holds `n` rows and slides; at the end of the episode it
slides BACK rather than shrinking, so the clean prefix grows past `c`. Row `j`'s scheduled level at
call `k`, with `g = 0`:

```
level_j(k) = clip(1 + (j*u - k)/S, 0, 1)        u = S/m,  m = n - c = 16
```

A row enters the window at its true level (1), loses `u/S` per call, and is **committed** when its
scheduled level reaches 0; its content is the model's x0 output from that last step. Every call is
classified by how many clean rows its window holds: start-up (`clean == 0`), warm-up
(`0 < clean < 3`), steady (`clean == 3`), tail (`clean > 3`, the window has slid back).

Training draws from a table that enumerates every distinct window such a rollout produces, over the
whole (g, c, S) grid, deduplicated per phase — `data/schedules/stream_states_w19_c4_s4-50.csv.gz`
on `mb/pick_place`, 58,905 windows, whose width already matches this recipe. Phase shares
start-up 0.25 / warm-up 0.10 / steady 0.40 / tail 0.25.

## 3. What the model is shown — no row is special

Every row of the window, context rows included, is noised at its own level, TOLD that level, and
scored against its clean x0. There is no clean-context path and no separate context loss term: the
context rows are folded into one row-count-weighted mean over the 19 rows (see
`utils/trajectory_objective.py`, `train.pose_noise_context`).

Jitter treats the two ends differently: a clean row (level exactly 0) gets `U[0, 0.048]`, one-sided,
so it can only move away from clean; every other row gets `+- U(0, 0.048)`. The noise really is
added at the jittered level. For rigid bodies the context particle rows are rebuilt from the noisy
context POSES (`utils/pose_native.pose_native_inputs`), so a body is rigid at every level.

## 4. The clock ends, and what we are NOT doing about them

| | this tree | pickup stream recipe |
|---|---|---|
| lowest level trained | `low = 1`, i.e. tau = 0.001 | `future_t_low: 0`, tau = 0 trained |
| highest level trained | clamped to `T-1` = 999 | tau = 1 -> 1000 |
| sampler top call | `T-1` = 999 | `sampler_full_clock: true` -> 1000 |
| sampler's finished rows | fed exactly 0 | fed 0, which IS trained there |

**Decision (user, 2026-09-20): do not touch the clock.** Too much other work depends on these
conventions, and the mismatch is confined to one place — the finished rows of a streaming rollout,
which no shared path produces. The new streaming rollout resolves it on the inference side: either
feed committed rows at `t = 1` rather than 0, or give them a lightly noised history and tell the
model that true level, which removes the mismatch instead of papering over it. `stream` training
draws are clamped to `[1, T-1]` like every other draw in this tree.

Everything else stays: training clocks are continuous floats (a row can sit at t = 62.5), the
sampler quantizes to integers, and for the stream matrix that rounding should be `np.rint` rather
than the truncation the other schedules use, which biases every level 0.001 toward clean.

## 5. Work items

| piece | status | work |
|---|---|---|
| `diffusion/stream_states.py` (reference sampler + table generator) | on `mb/pick_place` | port as-is | 
| the 19-row table | exists, width already matches | copy into `data/schedules/` |
| `stream` + `flat_given` draws, `_jitter`, balanced picks | absent | port into `diffusion/timestep_schedules.py`; `sample_window_timesteps` already routes whole windows, so they slot in as `draw(name)` cases |
| restrict the draw to `c = 3, g = 0` | new | `_stream_pick` balances over every (g, c, S) in the table; add a filter. The table itself stays general |
| config keys `stream_states` / `stream_phase_shares` / `stream_jitter`, only-when-used options | absent | mirrors the `velocity_weight_cap` pattern, so the resume contract of existing recipes is untouched |
| one-window stream matrix + `log.stream_rollout_keep` | absent | `generate_scheduling_matrix(n, S, "stream", keep_clean=3)` for the logged/validation rollouts, with `rint` |
| **multi-window streaming rollout** | **exists nowhere** | pickup's `streaming_rollout_native` hard-wires `g = 0, c = C-1` and SHRINKS the window at the end instead of sliding back. This is the real new code, and where the committed-row treatment of section 4 lives |
| actions inside a streaming rollout | new for rl_push | pickup was unconditional; here each window must carry the pusher's recorded pose actions |
| tests | spec exists | each call's levels must equal `window_states(0, 3, S, N, 19)` exactly, for several (S, N) |

## 6. What this changes in the recipe

```yaml
train:
  timestep_schedule: {independent: 0.6, stream: 0.4}
  stream_states: data/schedules/stream_states_w19_c4_s4-50.csv.gz
  stream_phase_shares: {start-up: 0.25, warm-up: 0.10, steady: 0.40, tail: 0.25}
  stream_jitter: 0.048
log:
  stream_rollout_keep: 3        # logged rollouts walk the staircase, not the pyramid
```
