"""Streaming window states: every per-row noise-level window a full-window streaming rollout shows the model.

The sampler this enumerates (the target for inference, see docs in the module functions):
  * a rollout is GIVEN g clean transitions and generates the rest;
  * it KEEPS c >= g clean rows at the front of an n-row window: the window starts at
    min(max(0, committed - c), N - n), so it is always n rows and slides back at the end of the episode
    (the context grows past c instead of the window shrinking);
  * transition j (generated index j - g) is scheduled at clip(1 + ((j - g) * u - call) / S, 0, 1) with
    u = S / (n - c), so the ramp after the c clean rows has slope 1 / (n - c) and new rows enter at pure noise;
  * rows slide into the window at pure noise and their first step takes them to the scheduled level
    (inference.streaming_chain enter_full_noise=True): a row's level is the minimum of its schedule over the
    calls it has spent inside the window.

Every model call is classified by the clean rows in its window (clean = committed - window start):
  start-up  no generated transition committed yet (clean = g)
  warm-up   something committed, clean < c
  steady    clean == c
  tail      clean > c (the window has slid back at the end of the episode)

build_states() simulates every (g, c, S, N) on a grid, dedupes the windows per phase and records which settings
(g, c, S) produce each one; save_csv / load_csv round-trip them. Training (diffusion.timestep_schedules, schedule
'stream') draws a phase by its share, then a SETTING balanced over the grid (c ~ U{0..c_max}, g ~ U{0..c},
S ~ U{steps}, restricted to settings that reach that phase), then one of that setting's windows uniformly.

CLI: python -m diffusion.stream_states --window 19 --steps 4-50 --out data/schedules/stream_states_w19_c4_s4-50.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import math
from collections import OrderedDict

PHASES = ("start-up", "warm-up", "steady", "tail")


def window_states(g: int, c: int, S: int, N: int, n: int):
    """Every model call of one full-window streaming rollout: list of (phase, clean_rows, levels tuple of n)."""
    if not 0 <= g <= c < n:
        raise ValueError(f"need 0 <= g <= c < n, got g={g} c={c} n={n}")
    if N < n:
        raise ValueError(f"a full-window rollout needs N >= n transitions, got N={N} n={n}")
    m = n - c
    D = S * m                          # exact arithmetic: level = num / D, num an integer in [0, D]

    def sched(j, call):                # clip(1 + ((j - g) * S/m - call) / S, 0, 1) * D
        if j < g:
            return 0
        return min(D, max(0, D + (j - g) * S - call * m))

    cur = [0 if j < g else D for j in range(N)]
    committed, call, out = g, 0, []
    max_calls = S + math.ceil((N - g) * S / m) + 2
    while committed < N:
        if call > max_calls:
            raise RuntimeError("streaming schedule did not converge (bug)")
        start = min(max(0, committed - c), N - n)
        clean = committed - start
        if committed == g:
            phase = "start-up"
        elif clean < c:
            phase = "warm-up"
        elif clean == c:
            phase = "steady"
        else:
            phase = "tail"
        out.append((phase, clean, tuple(v / D for v in cur[start:start + n])))
        for j in range(start, start + n):
            cur[j] = min(cur[j], sched(j, call + 1))
        call += 1
        while committed < N and sched(committed, call) <= 0:
            committed += 1
    return out


def build_states(n=19, c_max=4, steps=range(4, 51), lengths=None, decimals=4):
    """Unique windows per phase over the grid g in [0, c], c in [0, c_max], S in steps, N in lengths.
    Returns an ordered list of dicts: phase, clean_rows, visits, settings (sorted (g, c, S) tuples that produce it),
    levels (tuple of n floats)."""
    lengths = range(n, 80) if lengths is None else lengths   # one window up to many; >= 40 already saturates
    table = OrderedDict()                                  # (phase, levels) -> [clean, visits, settings set]
    for c in range(c_max + 1):
        for g in range(c + 1):
            for S in steps:
                for N in lengths:
                    for phase, clean, lv in window_states(g, c, S, N, n):
                        key = (phase, tuple(round(v, decimals) for v in lv))
                        rec = table.setdefault(key, [clean, 0, set()])
                        rec[1] += 1
                        rec[2].add((g, c, S))
    rows = [dict(phase=k[0], clean_rows=v[0], visits=v[1], settings=tuple(sorted(v[2])), levels=k[1])
            for k, v in table.items()]
    rows.sort(key=lambda r: (PHASES.index(r["phase"]), r["clean_rows"], r["levels"]))
    return rows


def _open(path, mode):
    """Plain or gzip-compressed (.gz) text file."""
    return gzip.open(path, mode + "t", newline="") if str(path).endswith(".gz") else open(path, mode, newline="")


def save_csv(path, rows, decimals=4):
    n = len(rows[0]["levels"])
    with _open(path, "w") as f:
        w = csv.writer(f)
        w.writerow(["phase", "clean_rows", "visits", "settings"] + [f"t{i}" for i in range(n)])
        for r in rows:
            settings = ";".join(f"{g}:{c}:{S}" for g, c, S in r["settings"])     # g:c:S of every setting reaching it
            w.writerow([r["phase"], r["clean_rows"], r["visits"], settings] + [f"{v:.{decimals}f}" for v in r["levels"]])


def load_csv(path, with_settings=False):
    """-> (levels [K, n] float32 tensor, phase index [K] long tensor, clean_rows [K] long tensor), rows grouped by phase;
    with_settings=True also returns settings: a list of K tuples of (g, c, S)."""
    import torch
    levels, phases, clean, settings = [], [], [], []
    with _open(path, "r") as f:
        r = csv.reader(f)
        header = next(r)
        if header[:4] != ["phase", "clean_rows", "visits", "settings"]:
            raise ValueError(f"{path}: not a stream-states CSV with settings (header {header[:4]}); regenerate it")
        for row in r:
            if row[0] not in PHASES:
                raise ValueError(f"{path}: unknown phase {row[0]!r}")
            phases.append(PHASES.index(row[0]))
            clean.append(int(row[1]))
            settings.append(tuple(tuple(int(x) for x in s.split(":")) for s in row[3].split(";")))
            levels.append([float(v) for v in row[4:]])
    if not levels:
        raise ValueError(f"{path}: no states")
    lv = torch.tensor(levels, dtype=torch.float32)
    ph = torch.tensor(phases, dtype=torch.long)
    order = torch.argsort(ph, stable=True)
    out = (lv[order], ph[order], torch.tensor(clean, dtype=torch.long)[order])
    if with_settings:
        out = out + ([settings[i] for i in order.tolist()],)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=19, help="rows per window = context_len - 1 + future_len (pickup: 3 + 16)")
    p.add_argument("--c_max", type=int, default=4)
    p.add_argument("--steps", default="4-50", help="inclusive range lo-hi of Euler steps per transition")
    p.add_argument("--lengths", default=None, help="inclusive range of rollout lengths in transitions (default: window-79)")
    p.add_argument("--decimals", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    rng = lambda s: range(int(s.split("-")[0]), int(s.split("-")[1]) + 1)
    rows = build_states(a.window, a.c_max, rng(a.steps), rng(a.lengths) if a.lengths else None, a.decimals)
    save_csv(a.out, rows, a.decimals)
    counts = {ph: sum(1 for r in rows if r["phase"] == ph) for ph in PHASES}
    print(f"wrote {a.out}: {len(rows)} unique windows {counts}")


if __name__ == "__main__":
    main()
