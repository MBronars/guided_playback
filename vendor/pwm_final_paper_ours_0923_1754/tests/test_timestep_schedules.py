"""train.timestep_schedule: which joint noise-level patterns training draws.

The claims under test: `ramp` contains every state the PRODUCTION pyramid sampler feeds the
model and its draws stay within one sampler step of those states; `flat` does the same for
full_sequence; independent draws essentially never do; each frame still sees every level
uniformly under `ramp`, with the finished/pure-noise shares the staircase geometry implies;
a mixture assigns one schedule per sample; the default path is untouched.
"""
import math

import pytest
import torch
from omegaconf import OmegaConf

from diffusion import TrajectoryFlowMatching
from diffusion.timestep_schedules import ramp_levels, resolve_mixture, resolve_slope, sample_future_timesteps
from inference import run_diffusion_inference_df
from tests.test_model_scope import _construct
from tests.test_sampler_oracle import GroundTruthOracle, scene
from tests.test_train import create_dummy_batch
from utils.trajectory_objective import trajectory_loss_options, trajectory_objective

T, LOW = 1000, 1
NEW_OPTIONS = ("timestep_schedule", "ramp_slope", "ramp_jitter", "velocity_weight_cap")


class Recorder(GroundTruthOracle):
    """The production sampler's own clocks, call by call."""
    def __init__(self, diffusion):
        super().__init__(diffusion)
        self.cfg.pose_native = False
        self.clocks = []

    def forward(self, batch, noisy, times, **kwargs):
        self.clocks.append(times[0].float().clone())
        return torch.zeros_like(noisy)


def sampler_states(frames, steps, schedule, scale=1.0):
    """[calls, frames] clocks the production single-window sampler feeds the model."""
    diffusion = TrajectoryFlowMatching()
    recorder = Recorder(diffusion)
    batch = create_dummy_batch(B=1, T=2 + frames, N=4, context_len=2)
    run_diffusion_inference_df(recorder, batch, diffusion, num_inference_steps=steps,
                               schedule=schedule, uncertainty_scale=scale)
    return torch.stack(recorder.clocks)


def family_member(state, slope, tolerance=1.0):
    """Is a [frames] clock vector clamp((phase + i * slope) * T, LOW, T - 1) for some phase?
    The sampler labels finished frames 0; the family's floor is LOW."""
    state = state.clamp(LOW, T - 1)
    frames = torch.arange(len(state), dtype=torch.float32)
    active = (state > LOW) & (state < T - 1)
    if active.any():
        phases = state[active] / T - frames[active] * slope
        if float(phases.max() - phases.min()) > 2 * tolerance / T:
            return False
        phase = float(phases.mean())
    else:
        phase = 1.0 if bool((state == T - 1).all()) else float("nan")
    member = ((phase + frames * slope) * T).clamp(LOW, T - 1)
    return bool((member - state).abs().max() <= tolerance)


def nearest_state_distance(draws, states):
    """[B] max-norm distance, in levels, from each draw to its nearest sampler state."""
    states = states.clamp(LOW, T - 1)
    return ((draws[:, None] - states[None]).abs().amax(-1).amin(-1)) / T


@pytest.mark.parametrize("frames,steps,scale", [(16, 25, 1.0), (38, 25, 1.0), (16, 10, 1.0), (16, 50, 1.0),
                                                (16, 30, 1.0), (16, 25, 2.0)])
def test_every_state_of_the_production_pyramid_sampler_is_a_ramp(frames, steps, scale):
    states = sampler_states(frames, steps, "pyramid", scale)
    assert len(states) == steps + int((frames - 1) * scale)
    assert bool((states[0] == T - 1).all())                       # every window starts from all noise
    assert all(family_member(state, scale / steps) for state in states)
    # The check has teeth: the same states are not ramps of another slope, nor is an iid draw.
    assert not all(family_member(state, 2 * scale / steps) for state in states)
    torch.manual_seed(0)
    assert not family_member(TrajectoryFlowMatching().sample_timesteps(1, "cpu", num_frames=frames, low=LOW)[0],
                             scale / steps)


def test_ramp_draws_stay_within_one_sampler_step_of_a_visited_state_and_iid_draws_do_not():
    frames, steps = 16, 25
    diffusion = TrajectoryFlowMatching()
    states = sampler_states(frames, steps, "pyramid")
    torch.manual_seed(1)
    ramp = sample_future_timesteps(diffusion, 20000, "cpu", frames, "ramp", ramp_slope=1 / steps)
    assert float(nearest_state_distance(ramp, states).max()) <= 1 / steps + 2 / T
    iid = diffusion.sample_timesteps(20000, "cpu", num_frames=frames, low=LOW)
    assert float((nearest_state_distance(iid, states) <= 1 / steps).float().mean()) < 1e-3
    # ... and the family reaches every visited state, not just a corner of them.
    nearest = (ramp[:, None] - states.clamp(LOW, T - 1)[None]).abs().amax(-1).argmin(-1)
    assert len(nearest.unique()) == len(states)


def test_flat_draws_stay_within_one_step_of_the_full_sequence_sampler():
    frames, steps = 16, 25
    states = sampler_states(frames, steps, "full_sequence")
    assert len(states) == steps and bool((states == states[:, :1]).all())
    torch.manual_seed(2)
    flat = sample_future_timesteps(TrajectoryFlowMatching(), 20000, "cpu", frames, "flat")
    assert bool((flat == flat[:, :1]).all())
    assert float(nearest_state_distance(flat, states).max()) <= 1 / steps
    assert float(flat.min()) >= LOW and float(flat.max()) < T
    assert abs(float(flat[:, 0].mean()) / T - 0.5) < 0.01


def test_every_frame_sees_every_level_under_ramp_with_the_shares_the_staircase_implies():
    """The early-frame question. Without jitter frame i is finished with probability
    (n - 1 - i) s / L, pure noise with (i s + pad) / L and otherwise uniform over the levels,
    L = (1 + (n - 1) s)^2 / (1 + (n - 2) s) being the padded phase range."""
    frames, slope, draws = 16, 0.04, 400000
    torch.manual_seed(3)
    clocks = sample_future_timesteps(TrajectoryFlowMatching(), draws, "cpu", frames, "ramp",
                                     ramp_slope=slope, ramp_jitter=0.0)
    assert float(clocks.min()) == LOW and float(clocks.max()) == T - 1
    assert bool((clocks[:, 1:] >= clocks[:, :-1]).all())          # earlier frames are never noisier
    span = 1 + (frames - 1) * slope
    length = span ** 2 / (span - slope)
    pad = length - span
    index = torch.arange(frames, dtype=torch.float64)
    finished = (clocks == LOW).double().mean(0)
    noise = (clocks == T - 1).double().mean(0)
    torch.testing.assert_close(finished, (frames - 1 - index) * slope / length, atol=5e-3, rtol=0)
    torch.testing.assert_close(noise, (index * slope + pad) / length, atol=5e-3, rtol=0)
    assert abs(float(finished[0]) - 0.366) < 5e-3 and float(finished[-1]) < 2e-3
    assert abs(float((clocks == T - 1).all(1).double().mean()) - slope / span) < 2e-3   # the all-noise start
    for frame in (0, 7, 15):                                      # uniform over the levels otherwise
        active = clocks[:, frame][(clocks[:, frame] > LOW) & (clocks[:, frame] < T - 1)] / T
        assert abs(len(active) / draws - 1 / length) < 5e-3
        histogram = torch.histc(active, bins=5, min=0.0, max=1.0) / len(active)
        torch.testing.assert_close(histogram, torch.full((5,), 0.2), atol=6e-3, rtol=0)


def test_default_jitter_is_half_a_sampler_step_and_a_slope_range_is_drawn_per_sample():
    torch.manual_seed(4)
    exact = ramp_levels(4096, 16, 0.04, 0.0, "cpu")
    torch.testing.assert_close(exact[:, 1:] - exact[:, :-1], torch.full((4096, 15), 0.04), atol=1e-5, rtol=0)
    torch.manual_seed(4)
    jittered = ramp_levels(4096, 16, 0.04, None, "cpu")
    assert 0.019 < float((jittered - exact).abs().max()) <= 0.02 + 1e-6
    torch.manual_seed(5)
    ranged = ramp_levels(4096, 16, (0.02, 0.08), 0.0, "cpu")
    slopes = (ranged[:, 1:] - ranged[:, :-1]).mean(1)
    assert 0.02 <= float(slopes.min()) < 0.025 and 0.075 < float(slopes.max()) <= 0.08 + 1e-6
    flat_limit = ramp_levels(64, 16, 0.0, 0.0, "cpu")
    assert bool((flat_limit == flat_limit[:, :1]).all()) and 0 <= float(flat_limit.min()) and float(flat_limit.max()) <= 1


def test_a_mixture_assigns_one_schedule_per_sample_with_the_requested_shares():
    frames, draws = 16, 60000
    weights = {"independent": 0.5, "ramp": 0.3, "flat": 0.2}
    torch.manual_seed(6)
    clocks = sample_future_timesteps(TrajectoryFlowMatching(), draws, "cpu", frames, weights,
                                     ramp_slope=0.04, ramp_jitter=0.0)
    equal = (clocks == clocks[:, :1]).all(1)
    monotone = (clocks[:, 1:] >= clocks[:, :-1]).all(1)
    all_noise = equal & (clocks[:, 0] == T - 1)
    flat, ramp = equal & ~all_noise, (monotone & ~equal) | all_noise
    independent = ~monotone                                       # sorted by chance: 1 in 16!
    assert int(flat.sum() + ramp.sum() + independent.sum()) == draws
    for name, mask in (("independent", independent), ("ramp", ramp), ("flat", flat)):
        assert abs(float(mask.float().mean()) - weights[name]) < 4 * math.sqrt(0.25 / draws)
    assert float(clocks[independent].max()) > T - 1               # iid rows keep their own range [LOW, T)
    assert float(clocks[independent].mean()) / T == pytest.approx(0.5, abs=5e-3)


def test_mixture_specifications_normalize_and_fail_closed():
    assert resolve_mixture("ramp") == {"ramp": 1.0}
    assert resolve_mixture(["ramp", "flat"]) == {"flat": 0.5, "ramp": 0.5}
    assert resolve_mixture({"ramp": 3, "independent": 1, "flat": 0}) == {"independent": 0.25, "ramp": 0.75}
    assert resolve_mixture(resolve_mixture({"ramp": 2, "flat": 1})) == resolve_mixture({"ramp": 2, "flat": 1})
    for bad in ("pyramid", ["ramp", "ramp"], {"ramp": -1}, {"ramp": 0}, {"ramp": float("nan")},
                {"ramp": True}, {"ramp": "1"}, 3, [], {"flat_given": 1}):   # stream IS a schedule now
        with pytest.raises(ValueError):
            resolve_mixture(bad)
    assert resolve_slope(0.04) == 0.04 and resolve_slope([0.02, 0.08]) == (0.02, 0.08)
    for bad in (-0.1, [0.08, 0.02], [0.04], float("inf"), "0.04", True):
        with pytest.raises(ValueError):
            resolve_slope(bad)
    diffusion = TrajectoryFlowMatching()
    with pytest.raises(ValueError, match="call site"):
        sample_future_timesteps(diffusion, 2, "cpu", 4, "independent")
    with pytest.raises(ValueError, match="ramp_slope"):
        sample_future_timesteps(diffusion, 2, "cpu", 4, "ramp")
    with pytest.raises(ValueError, match="nonnegative"):
        sample_future_timesteps(diffusion, 2, "cpu", 4, "ramp", ramp_slope=0.04, ramp_jitter=-1)
    with pytest.raises(ValueError, match="time_sampler=uniform"):
        sample_future_timesteps(TrajectoryFlowMatching(time_sampler="beta"), 2, "cpu", 4, "ramp", ramp_slope=0.04)


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("kind", ["deformable", "mixed", "rigid"])
@pytest.mark.parametrize("prediction", ["velocity", "sample"])
def test_the_objective_trains_both_backbones_on_shaped_draws(architecture, kind, prediction):
    batch = scene(kind)
    if kind == "deformable":
        batch.mat_type.zero_()
    model = _construct(architecture, False)
    torch.manual_seed(7)
    loss, _, draw = trajectory_objective(model, batch, TrajectoryFlowMatching(prediction_type=prediction),
                                         clip_deltas=False, timestep_schedule={"ramp": 1.0},
                                         ramp_slope=0.25, ramp_jitter=0.0)
    clocks = draw["timesteps"]
    assert clocks.shape == (batch.batch_size, batch.num_future_frames) and clocks.dtype == torch.float32
    assert float(clocks.min()) >= LOW and float(clocks.max()) <= T - 1
    assert bool((clocks[:, 1:] >= clocks[:, :-1]).all())
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.parameters())


@pytest.mark.parametrize("kind", ["deformable", "mixed", "rigid"])
@pytest.mark.parametrize("prediction", ["velocity", "sample"])
@pytest.mark.parametrize("schedule", ["flat", "ramp", {"independent": 1, "ramp": 1, "flat": 1}])
def test_a_perfect_model_has_zero_loss_under_every_schedule(kind, prediction, schedule):
    diffusion = TrajectoryFlowMatching(prediction_type=prediction)
    torch.manual_seed(8)
    loss, _, _ = trajectory_objective(GroundTruthOracle(diffusion), scene(kind), diffusion, clip_deltas=False,
                                      timestep_schedule=schedule, ramp_slope=0.04)
    assert float(loss) < 1e-8


def test_defaults_resolve_to_the_options_and_the_draw_they_always_had():
    from utils.config import load_config
    options = trajectory_loss_options(load_config("configs/unet.yaml"))
    assert not set(NEW_OPTIONS) & set(options)
    assert not set(NEW_OPTIONS) & set(trajectory_loss_options(load_config(
        "configs/unet.yaml", ["train.timestep_schedule=independent", "train.velocity_weight_cap=0"])))
    # The overfit plots resolve objective contracts from `train` alone: a default like any other.
    assert not set(NEW_OPTIONS) & set(trajectory_loss_options(OmegaConf.create({"train": {}})))
    batch, diffusion = scene("mixed"), TrajectoryFlowMatching()
    results = []
    for extra in ({}, dict(timestep_schedule="independent", velocity_weight_cap=0.0)):
        torch.manual_seed(9)
        results.append(trajectory_objective(GroundTruthOracle(diffusion), batch, diffusion, **extra))
    (loss, metrics, draw), (loss2, metrics2, draw2) = results
    assert draw.pop("batch") is draw2.pop("batch")
    torch.testing.assert_close((loss, metrics, draw), (loss2, metrics2, draw2), rtol=0, atol=0)


def test_recipes_resolve_the_schedule_and_derive_the_slope_from_their_own_sampler():
    from utils.config import load_config
    # The launch template carries overrides as words, so a mixture is written one weight per word.
    dotted = ["train.timestep_schedule.independent=0.5", "train.timestep_schedule.ramp=0.4",
              "train.timestep_schedule.flat=0.1"]
    options = trajectory_loss_options(load_config("configs/unet.yaml", dotted))
    assert options["timestep_schedule"] == pytest.approx({"flat": 0.1, "independent": 0.5, "ramp": 0.4})
    assert options == trajectory_loss_options(load_config(
        "configs/unet.yaml", ["train.timestep_schedule={independent: 0.5, ramp: 0.4, flat: 0.1}"]))
    assert options["ramp_slope"] == pytest.approx(1 / 25) and "ramp_jitter" not in options
    half = trajectory_loss_options(load_config("configs/transformer.yaml", [
        "train.timestep_schedule=[ramp,flat]", "flow_matching.inference_steps=50", "df_uncertainty_scale=2"]))
    assert half["timestep_schedule"] == {"flat": 0.5, "ramp": 0.5} and half["ramp_slope"] == pytest.approx(2 / 50)
    ranged = trajectory_loss_options(load_config("configs/unet.yaml", [
        "train.timestep_schedule=ramp", "train.ramp_slope=[0.02,0.08]", "train.ramp_jitter=0.01"]))
    assert ranged["ramp_slope"] == [0.02, 0.08] and ranged["ramp_jitter"] == 0.01
    assert "ramp_slope" not in trajectory_loss_options(load_config("configs/unet.yaml", ["train.timestep_schedule=flat"]))
    # The resolved options survive the checkpoint's container round trip and run the objective.
    stored = OmegaConf.to_container(OmegaConf.create(dict(options)), resolve=True)
    assert stored == options
    diffusion = TrajectoryFlowMatching()
    chosen = {name: stored[name] for name in NEW_OPTIONS if name in stored}
    loss, _, _ = trajectory_objective(GroundTruthOracle(diffusion), scene("mixed"), diffusion,
                                      clip_deltas=False, **chosen)
    assert float(loss) < 1e-8


@pytest.mark.parametrize("overrides,match", [
    (["train.timestep_schedule=pyramid"], "Unknown timestep_schedule"),
    (["train.timestep_schedule.ramp=-1"], "nonnegative"),
    (["train.timestep_schedule.ramp=0"], "positive weight"),
    (["train.timestep_schedule.ramp=true"], "finite number"),
    (["train.timestep_schedule={ramp:0.5,independent:0.5}"], "space after each colon"),
    (["train.ramp_slope=0.04"], "inactive"),
    (["train.timestep_schedule=flat", "train.ramp_jitter=0.01"], "inactive"),
    (["train.timestep_schedule=ramp", "train.ramp_slope=-0.1"], "nonnegative"),
    (["train.timestep_schedule=ramp", "train.ramp_jitter=-0.1"], "nonnegative"),
    (["train.timestep_schedule=ramp", "flow_matching.time_sampler=beta"], "time_sampler=uniform"),
    (["train.timestep_schdule=ramp"], "Unknown or retired train options"),
])
def test_bad_schedules_fail_at_configuration(overrides, match):
    from utils.config import load_config
    with pytest.raises(ValueError, match=match):
        load_config("configs/unet.yaml", overrides)


def test_shaped_draws_refuse_the_retired_noisy_context():
    diffusion = TrajectoryFlowMatching()
    with pytest.raises(ValueError, match="noise_context"):
        trajectory_objective(GroundTruthOracle(diffusion), scene("mixed"), diffusion,
                             noise_context=True, timestep_schedule="flat")


@pytest.mark.parametrize("change", [["train.timestep_schedule=ramp"],
                                    ["train.timestep_schedule.independent=0.5", "train.timestep_schedule.ramp=0.5"]])
def test_resume_cannot_change_the_schedule(change):
    from utils.checkpoint import validate_resume_configuration
    from utils.config import load_config
    plain = load_config("configs/unet.yaml")
    shaped = load_config("configs/unet.yaml", change)
    for saved, requested in ((plain, shaped), (shaped, plain)):
        with pytest.raises(ValueError, match="training contract"):
            validate_resume_configuration({"cfg": OmegaConf.to_container(saved, resolve=True)}, requested)
    validate_resume_configuration({"cfg": OmegaConf.to_container(shaped, resolve=True)}, shaped)
    other = load_config("configs/unet.yaml", change + ["train.ramp_slope=0.08"])
    with pytest.raises(ValueError, match="training contract"):
        validate_resume_configuration({"cfg": OmegaConf.to_container(shaped, resolve=True)}, other)
