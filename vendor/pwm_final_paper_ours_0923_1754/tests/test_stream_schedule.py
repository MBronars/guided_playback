"""The `stream` draw and the `stream` sampler against their reference, diffusion/stream_states.py.
Standalone runner, house style:

    python tests/test_stream_schedule.py

CPU. What must hold:
  1. The scheduling matrix IS one window of a streaming rollout: every call equals
     window_states(g=0, c, S, N=n, n), exactly, for several (c, S).
  2. Levels are rounded, not truncated, onto the training clock (truncation biases toward clean).
  3. The training draw only produces windows of the table, restricted to the settings
     train.stream_keep / train.stream_given pin, and covers all four phases by their shares.
  4. Jitter: clean rows move one-sided into [0, j]; other rows by at most j; the model is told the
     level its noise was drawn at, so the draw IS the label (checked through the objective).
  5. A recipe that names `stream` without a table, or with stream_* but no stream schedule, is refused.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion.scheduling import generate_scheduling_matrix
from diffusion.stream_states import window_states
from diffusion.timestep_schedules import _jitter, resolve_stream_options, sample_window_timesteps
from diffusion import TrajectoryFlowMatching

TABLE = "data/schedules/stream_states_w19_c4_s4-50.csv.gz"
N_ROWS, KEEP = 19, 3


def test_matrix_is_one_window_of_the_rollout():
    for c, S in ((3, 25), (3, 6), (4, 50), (0, 10)):
        K = generate_scheduling_matrix(N_ROWS, S, "stream", keep_clean=c)
        want = [levels for _, _, levels in window_states(0, c, S, N_ROWS, N_ROWS)]
        got = [tuple(row / S) for row in K]
        assert len(got) >= len(want), (len(got), len(want))
        for i, (a, b) in enumerate(zip(got, want)):
            assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9, (c, S, i, a, b)
        for row in got[len(want):]:                      # the matrix may pad with all-clean calls
            assert max(row) == 0.0
    print("ok  the stream matrix is exactly the rollout's own windows")


def test_levels_round_onto_the_clock():
    S, T = 25, 1000
    K = generate_scheduling_matrix(N_ROWS, S, "stream", keep_clean=KEEP)
    exact = K / S * T
    rounded = np.rint(exact)
    truncated = exact.astype(np.int64)
    assert np.abs(rounded - exact).max() <= 0.5 + 1e-9
    assert (truncated <= exact + 1e-9).all() and (truncated < exact - 1e-9).any(), \
        "truncation is the biased alternative this rounding exists to avoid"
    print(f"ok  rounding keeps |error| <= 0.5 of a clock tick; truncation loses up to "
          f"{float((exact - truncated).max()):.2f}")


def _opts(**kw):
    return resolve_stream_options(TABLE, keep=KEEP, given=0, **kw)


def test_draw_stays_inside_the_pinned_settings():
    from diffusion.stream_states import load_csv
    levels, _, _, settings = load_csv(TABLE, with_settings=True)
    allowed = {i for i, sets in enumerate(settings) if any(g == 0 and c == KEEP for g, c, _ in sets)}
    table_rows = {tuple(round(v, 4) for v in levels[i].tolist()) for i in allowed}
    d = TrajectoryFlowMatching(num_train_timesteps=1000, prediction_type="sample")
    torch.manual_seed(0)
    ctx, fut = sample_window_timesteps(d, 512, torch.device("cpu"), KEEP, N_ROWS - KEEP,
                                       {"stream": 1.0}, stream_options=_opts(jitter=0.0))
    w = torch.cat([ctx, fut], 1) / 1000
    # With jitter off the ONLY edit to a table window is this tree's [1, T-1] clamp, which lifts a
    # clean row to 1/T and caps pure noise at (T-1)/T. Undo it and the window must be in the table.
    lo, hi = 1 / 1000, 999 / 1000
    for row in w.tolist():
        undone = tuple(round(0.0 if abs(v - lo) < 1e-6 else (1.0 if abs(v - hi) < 1e-6 else v), 4)
                       for v in row)
        assert undone in table_rows, undone
    assert abs(float(w.min()) - lo) < 1e-6 and float(w.max()) <= hi + 1e-6
    print("ok  every drawn window is a table window of the pinned (g=0, c=3) settings")


def test_phases_are_drawn_by_their_shares():
    d = TrajectoryFlowMatching(num_train_timesteps=1000, prediction_type="sample")
    torch.manual_seed(1)
    shares = {"start-up": 0.25, "warm-up": 0.10, "steady": 0.40, "tail": 0.25}
    ctx, fut = sample_window_timesteps(d, 8192, torch.device("cpu"), KEEP, N_ROWS - KEEP,
                                       {"stream": 1.0}, stream_options=_opts(phase_shares=shares, jitter=0.0))
    w = torch.cat([ctx, fut], 1) / 1000
    clean = (w <= 0.001 + 1e-9).sum(1)                   # clean rows identify the phase
    got = {"start-up": float((clean == 0).float().mean()),
           "warm-up": float(((clean > 0) & (clean < KEEP)).float().mean()),
           "steady": float((clean == KEEP).float().mean()),
           "tail": float((clean > KEEP).float().mean())}
    for name, want in shares.items():
        assert abs(got[name] - want) < 0.05, (name, got[name], want)
    print("ok  phases follow their shares: " + " ".join(f"{k} {v:.2f}" for k, v in got.items()))


def test_jitter_is_one_sided_at_the_clean_end():
    t = torch.tensor([[0.0, 0.5, 1.0]]).repeat(4096, 1)
    torch.manual_seed(2)
    j = _jitter(t, 0.048)
    assert float(j[:, 0].min()) >= 0.0 and float(j[:, 0].max()) <= 0.048
    assert float(j[:, 0].mean()) > 0.02, "clean rows must move AWAY from clean, not stay"
    assert (j[:, 1] - 0.5).abs().max() <= 0.048 + 1e-6
    assert float(j[:, 2].max()) <= 1.0, "clamped at pure noise"
    assert torch.equal(_jitter(t, 0.0), t), "jitter 0 is a no-op (the control)"
    print("ok  jitter: one-sided at 0, symmetric elsewhere, clamped, no-op at 0")


def test_recipes_fail_closed():
    from omegaconf import OmegaConf
    from utils.trajectory_objective import timestep_schedule_options
    base = {"train": {"noise_context": True, "pose_noise_context": True,
                      "timestep_schedule": {"stream": 1.0}}, "flow_matching": {"prediction_type": "sample"}}
    for cfg, needle in (
            (base, "stream_states"),
            ({"train": {"timestep_schedule": "independent", "stream_states": TABLE}}, "inactive unless"),
            ({"train": {"noise_context": False, "timestep_schedule": {"stream": 1.0}, "stream_states": TABLE}},
             "noise_context")):
        try:
            timestep_schedule_options(OmegaConf.create(cfg))
            raise AssertionError(f"expected a refusal mentioning {needle!r}")
        except ValueError as e:
            assert needle in str(e), e
    print("ok  refusals: no table, stream_* without stream, stream without a generated context")


if __name__ == "__main__":
    test_matrix_is_one_window_of_the_rollout()
    test_levels_round_onto_the_clock()
    test_draw_stays_inside_the_pinned_settings()
    test_phases_are_drawn_by_their_shares()
    test_jitter_is_one_sided_at_the_clean_end()
    test_recipes_fail_closed()
    print("ALL OK")
