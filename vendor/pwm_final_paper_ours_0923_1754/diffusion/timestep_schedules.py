"""Training-time noise-level schedules for the future frames of a window.

Diffusion forcing gives every future frame its own noise level. WHICH joint patterns
of levels training sees is a separate choice from the sampler, and with finite capacity
it decides where the model is accurate:

  independent  iid per frame, the historical default. It is drawn at the call site so
               its RNG stream stays bitwise. Every sampler's states are in its support,
               but almost none of its mass is on the states any one sampler visits.
  flat         ONE level per sample, shared by every future frame: the states of
               df_schedule=full_sequence.
  ramp         the pyramid staircase drawn as a FAMILY. Frame i sits at
               clip(phase + i * slope + jitter). The pyramid sampler with uncertainty
               scale u over S steps feeds the model slope = u / S and phase = 1 - m / S at
               call m, so every state it visits is a member (tests/test_timestep_schedules.py
               records the production sampler and checks this). The phase sweeps from
               "every frame is pure noise" to "only the last frame is unfinished".

`train.timestep_schedule` names ONE schedule or a mixture, a list (equal weights) or a
{name: weight} map, and each SAMPLE of a batch draws one of them. Keeping a share of
`independent` keeps the patterns no staircase contains (a frame noisier than a later one,
which goal inpainting needs) and gives the early frames back the draws a staircase spends
on already finished frames: under `ramp` at n = 16 frames and slope 0.04, frame 0 is
already finished in 36.6% of the draws (frame i in proportion to n - 1 - i, the last frame
never), every frame is strictly between clean and pure noise in 60.9% of them, uniformly
over the levels, and the rest of the time it is still pure noise.

A window whose CONTEXT rows are generated too (train.noise_context with
train.pose_noise_context) is shaped as ONE row sequence: sample_window_timesteps applies the
schedule to [context | future] jointly, so a staircase runs continuously across the window
(the oldest context row least noisy) instead of restarting at the boundary, and `flat` means
one level for the whole window. A clean-context window calls sample_future_timesteps, which
is the same draw with no context rows — the RNG stream of every clean-context recipe is
therefore unchanged.

`stream` is the states of a STREAMING rollout, read from a table rather than parameterised:
diffusion/stream_states.py enumerates every window such a rollout shows the model, over its whole
(given g, keep c, steps S) grid, and the draw picks a phase by its share, a setting balanced over
the grid, and then one of that setting's windows. train.stream_keep / train.stream_given restrict
it to the settings a planned rollout will actually use; S stays sampled, because a rollout may run
at any step count. See docs/streaming_rollout_plan.md.

Ported from mb/pick_place (diffusion/timestep_schedules.py); that branch's `flat_given` draw is not
part of this one.
"""
import math
from collections.abc import Mapping

import torch

SCHEDULES = ("independent", "flat", "ramp", "stream")


_STREAM_CACHE = {}
STREAM_DEFAULT_SHARES = {"start-up": 0.25, "warm-up": 0.10, "steady": 0.40, "tail": 0.25}


def _jitter(t, j):
    """Clean rows (level 0) -> U[0, j]; every other row +- U(0, j); clamped to [0, 1]. t in [0, 1].

    One-sided at the clean end so a committed row can only move AWAY from clean. j = 0.048 is the
    level whose noise-to-signal ratio t/(1-t) equals a light-context-noise sigma of 0.05."""
    if j <= 0:
        return t
    clean = t <= 0.0
    return torch.where(clean, torch.rand_like(t) * j,
                       (t + (torch.rand_like(t) * 2 - 1) * j).clamp(0.0, 1.0))


def resolve_stream_options(states, phase_shares=None, jitter=0.048, keep=None, given=None):
    """train.stream_* -> the options the `stream` draw reads. Fails closed on a bad share or jitter."""
    if not states:
        raise ValueError("timestep_schedule stream needs train.stream_states (a stream-states table)")
    shares = dict(STREAM_DEFAULT_SHARES if phase_shares is None else phase_shares)
    from diffusion.stream_states import PHASES
    unknown = sorted(set(shares) - set(PHASES))
    if unknown:
        raise ValueError(f"unknown train.stream_phase_shares {unknown}; choose from {PHASES}")
    for name, weight in shares.items():
        if _finite(weight, f"train.stream_phase_shares.{name}") < 0:
            raise ValueError("train.stream_phase_shares must be nonnegative")
    if sum(shares.values()) <= 0:
        raise ValueError("train.stream_phase_shares needs one positive share")
    if _finite(jitter, "train.stream_jitter") < 0:
        raise ValueError("train.stream_jitter must be nonnegative")
    for name, value in (("train.stream_keep", keep), ("train.stream_given", given)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be null or a nonnegative whole number of ROWS")
    if keep is not None and given is not None and given > keep:
        raise ValueError("train.stream_given must not exceed train.stream_keep (g <= c)")
    return dict(states=str(states), phase_shares=shares, jitter=float(jitter), keep=keep, given=given)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def resolve_mixture(spec):
    """A schedule name, a list of names (equal weights) or a {name: weight} map
    -> normalized {name: weight} with zero weights dropped, sorted by name."""
    if isinstance(spec, str):
        weights = {spec: 1.0}
    elif isinstance(spec, Mapping):
        weights = {str(name): weight for name, weight in spec.items()}
    elif isinstance(spec, (list, tuple)):
        names = [str(name) for name in spec]
        if len(set(names)) != len(names):
            raise ValueError(f"timestep_schedule lists a name twice: {names}")
        weights = {name: 1.0 for name in names}
    else:
        raise ValueError(f"timestep_schedule must be a name, a list or a mapping, got {spec!r}")
    unknown = sorted(set(weights) - set(SCHEDULES))
    if unknown:
        # A command-line mapping without a space after the colon parses as the KEY 'ramp:0.5'.
        raise ValueError(f"Unknown timestep_schedule {unknown}; choose from {SCHEDULES}. A mapping is "
                         "{name: weight} with a space after each colon, or train.timestep_schedule.<name>=<weight>")
    weights = {name: _finite(weight, f"timestep_schedule weight of {name}") for name, weight in weights.items()}
    if any(weight < 0 for weight in weights.values()):
        raise ValueError(f"timestep_schedule weights must be nonnegative, got {weights}")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("timestep_schedule needs at least one positive weight")
    return {name: weight / total for name, weight in sorted(weights.items()) if weight > 0}


def resolve_slope(value):
    """A scalar (one staircase slope) or [lo, hi] (drawn per sample), in levels per frame."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("ramp_slope range must be [lo, hi]")
        lo, hi = (_finite(v, "ramp_slope") for v in value)
        if not 0.0 <= lo <= hi:
            raise ValueError(f"ramp_slope range must satisfy 0 <= lo <= hi, got {[lo, hi]}")
        return (lo, hi)
    slope = _finite(value, "ramp_slope")
    if slope < 0.0:
        raise ValueError(f"ramp_slope must be nonnegative, got {slope}")
    return slope


def ramp_levels(batch_size, num_frames, slope, jitter, device):
    """[B, n] UNCLAMPED staircase levels, one (slope, phase) pair per sample; 1 = pure noise.

    The phase range [-(n - 1) * slope, 1] is widened past 1 so that the all-noise state,
    the first call of every generation, gets the share it has among the sampler's calls:
    one of 1 / slope + (n - 1), i.e. slope / (1 + (n - 1) * slope). Without the pad it is a
    measure-zero endpoint. The all-finished state is not padded: no sampler queries it.
    jitter=None moves every frame by up to half a sampler step (slope / 2).
    """
    if isinstance(slope, tuple):
        slope = torch.rand((batch_size, 1), device=device) * (slope[1] - slope[0]) + slope[0]
    else:
        slope = torch.full((batch_size, 1), float(slope), device=device)
    span = 1.0 + (num_frames - 1) * slope
    all_noise = slope / span
    pad = all_noise * span / (1.0 - all_noise)
    phase = torch.rand((batch_size, 1), device=device) * (span + pad) - (num_frames - 1) * slope
    frames = torch.arange(num_frames, device=device, dtype=torch.float32)[None]
    width = 0.5 * slope if jitter is None else float(jitter)
    noise = (torch.rand((batch_size, num_frames), device=device) * 2.0 - 1.0) * width
    return phase + frames * slope + noise


def _stream_table(opts, device):
    """The table as sampling blocks: (phase, g, c, S) -> window indices, with the weight of the
    grid's hierarchical draw c ~ U{c values}, g ~ U{0..c}, S ~ U{S values}. train.stream_keep /
    train.stream_given drop every block outside the settings a rollout will use."""
    keep, given = opts.get("keep"), opts.get("given")
    key = (opts["states"], keep, given, str(device))
    if key not in _STREAM_CACHE:
        from collections import defaultdict
        from diffusion.stream_states import load_csv
        levels, phases, _, settings = load_csv(opts["states"], with_settings=True)
        blocks = defaultdict(list)                                     # (phase, g, c, S) -> window indices
        for i, (ph, sets) in enumerate(zip(phases.tolist(), settings)):
            for g, c, S in sets:
                if (keep is not None and c != keep) or (given is not None and g != given):
                    continue
                blocks[(ph, g, c, S)].append(i)
        if not blocks:
            raise ValueError(f"{opts['states']}: no windows with keep={keep} given={given}")
        cs = sorted({k[2] for k in blocks})
        Ss = sorted({k[3] for k in blocks})
        keys = sorted(blocks)
        # P(c) P(g | c) P(S); with keep/given pinned the first two factors are constant and drop out
        n_g = {c: len({k[1] for k in keys if k[2] == c}) for c in cs}
        weight = torch.tensor([1.0 / (len(cs) * n_g[k[2]] * len(Ss)) for k in keys])
        sizes = torch.tensor([len(blocks[k]) for k in keys], dtype=torch.long)
        _STREAM_CACHE[key] = dict(
            levels=levels.to(device),
            members=torch.tensor([i for k in keys for i in blocks[k]], dtype=torch.long).to(device),
            sizes=sizes.to(device),
            starts=torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]]).to(device),
            weight=weight.to(device),
            phase_of=torch.tensor([k[0] for k in keys], dtype=torch.long).to(device),
            steps=Ss, keys=keys)
    return _STREAM_CACHE[key]


def _stream(batch_size, n_rows, opts, T, device):
    """[B, n_rows] clocks in [0, T]: phase by its share, setting balanced over the grid within that
    phase, window uniform within the setting, then jitter."""
    from diffusion.stream_states import PHASES
    tab = _stream_table(opts, device)
    if tab["levels"].shape[1] != n_rows:
        raise ValueError(f"{opts['states']}: {tab['levels'].shape[1]} rows per window, the training window "
                         f"has {n_rows} (context_len - 1 + future_len); regenerate it with "
                         "python -m diffusion.stream_states --window")
    present = torch.bincount(tab["phase_of"], minlength=len(PHASES)) > 0
    w = torch.tensor([opts["phase_shares"].get(p, 0.0) for p in PHASES], device=device) * present
    if float(w.sum()) <= 0:
        raise ValueError("no stream states in any phase with a positive share")
    phase = torch.multinomial(w / w.sum(), batch_size, replacement=True)
    # setting within the phase: rows of a [P, K] weight matrix with out-of-phase blocks zeroed
    Wp = tab["weight"][None] * (tab["phase_of"][None] == torch.arange(len(PHASES), device=device)[:, None])
    Wp = Wp / Wp.sum(1, keepdim=True).clamp(min=1e-12)
    block = torch.multinomial(Wp[phase], 1).squeeze(1)
    off = (torch.rand(batch_size, device=device) * tab["sizes"][block]).long().clamp(max=tab["sizes"][block] - 1)
    idx = tab["members"][tab["starts"][block] + off]
    return _jitter(tab["levels"][idx], float(opts["jitter"])) * T


def sample_future_timesteps(diffusion, batch_size, device, num_frames, schedule, *,
                            ramp_slope=None, ramp_jitter=None, low=1):
    """[B, F] future clocks for a shaped schedule or a mixture, clean context.

    Pure `independent` is drawn at the call site (the historical RNG order); inside a
    mixture it is the same distribution. Staircase levels are clamped to [low, T - 1]:
    T - 1 is the production sampler's top clock, and `low` is the floor of the independent
    draws, so finished frames sit at the clean end of the trained range.
    """
    _, future = sample_window_timesteps(diffusion, batch_size, device, 0, num_frames, schedule,
                                        ramp_slope=ramp_slope, ramp_jitter=ramp_jitter, low=low)
    return future


def sample_window_timesteps(diffusion, batch_size, device, n_context, num_frames, schedule, *,
                            ramp_slope=None, ramp_jitter=None, low=1, stream_options=None):
    """Levels for ONE window: returns (context [B, n_context] or None, future [B, F]).

    The schedule is applied to the joint [context | future] row sequence, so a staircase
    runs continuously across the window and `flat` is one level for every row. n_context=0
    is the clean-context call and draws exactly what sample_future_timesteps always drew.
    Context rows share the future's floor `low` here: under a shaped schedule they are rows
    of the same profile, not the separate low=0 draw a clean-context recipe makes for them.
    """
    if n_context < 0 or num_frames < 1:
        raise ValueError("n_context must be nonnegative and num_frames positive")
    n_rows = n_context + num_frames
    weights = resolve_mixture(schedule)
    if "stream" in weights and stream_options is None:
        raise ValueError("timestep_schedule stream needs its table (train.stream_states)")
    if set(weights) == {"independent"}:
        raise ValueError("independent draws its timesteps at the call site; do not route it here")
    if "ramp" in weights:
        if ramp_slope is None:
            raise ValueError("timestep_schedule ramp needs ramp_slope (the sampler's "
                             "df_uncertainty_scale / inference_steps)")
        if diffusion.time_sampler != "uniform":
            raise ValueError("timestep_schedule ramp requires flow_matching.time_sampler=uniform")
        slope = resolve_slope(ramp_slope)
        if ramp_jitter is not None and _finite(ramp_jitter, "ramp_jitter") < 0:
            raise ValueError("ramp_jitter must be nonnegative")
    T = float(diffusion.num_train_timesteps)

    def draw(name):
        if name == "independent":
            return diffusion.sample_timesteps(batch_size, device, num_frames=n_rows, low=low)
        if name == "flat":
            return diffusion.sample_timesteps(batch_size, device, low=low)[:, None].expand(
                batch_size, n_rows)
        if name == "stream":
            # Clamped like every other draw in this tree: this branch keeps the clock conventions
            # it has (no future_t_low, no sampler_full_clock), so a jittered-to-clean row trains at
            # t = low rather than 0. docs/streaming_rollout_plan.md section 4.
            return _stream(batch_size, n_rows, stream_options, T, device).clamp(float(low), T - 1.0)
        levels = ramp_levels(batch_size, n_rows, slope, ramp_jitter, device)
        return (levels * T).clamp(float(low), T - 1.0)

    names = list(weights)
    timesteps = draw(names[0])
    if len(names) > 1:
        # ONE schedule per sample. Every component is drawn for the whole batch (cheap at
        # these sizes), so each row is a genuine draw of the schedule it was assigned.
        edges = torch.tensor([weights[name] for name in names], device=device).cumsum(0)
        which = torch.bucketize(torch.rand(batch_size, device=device), edges[:-1], right=True)
        for index, name in enumerate(names[1:], start=1):
            timesteps = torch.where((which == index)[:, None], draw(name), timesteps)
    timesteps = timesteps.contiguous()
    if n_context == 0:
        return None, timesteps
    return timesteps[:, :n_context].contiguous(), timesteps[:, n_context:].contiguous()
