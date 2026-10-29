"""train.velocity_weight_cap: min(1/t^2, cap) per frame on every clean-target error.

Two orthogonal switches: train.pose_loss_space decides the space of the rigid-pose term, the
cap multiplies clean-target terms, so with rigid bodies it requires pose_loss_space=sample.
A model with a KNOWN constant clean-target error makes the loss an exact function of the
drawn clocks, so the weight's values, its per-frame placement and both terms (particles,
rigid poses) are checked against arithmetic, and the uncapped limit is checked against the
production velocity-space pose loss it is meant to bound.
"""
import inspect

import pytest
import torch
from omegaconf import OmegaConf

from diffusion import TrajectoryFlowMatching
from tests.test_sampler_oracle import GroundTruthOracle, scene
from tests.test_training_safety import scalar_batch, setup
from tests.test_unet_std_pose_native import make_batch
from train import train_step, validate
from utils.ema import EMA
from utils.trajectory_objective import capped_velocity_weight, trajectory_loss_options, trajectory_objective

ERROR = 0.01


class OffsetOracle(GroundTruthOracle):
    """Sample head whose clean-target prediction is wrong by ERROR in every coordinate."""
    def field(self, noisy, target, times):
        return target + ERROR


def run(kind, seed=11, **options):
    """Clean-target pose space unless a test asks for the historical velocity space."""
    diffusion = TrajectoryFlowMatching(prediction_type="sample")
    torch.manual_seed(seed)
    options.setdefault("pose_loss_space", "sample")
    return trajectory_objective(OffsetOracle(diffusion), scene(kind), diffusion, clip_deltas=False,
                                clip_pose_deltas=False, **options)


def test_weight_values():
    clocks = torch.tensor([[0.0, 1.0, 10.0, 50.0, 100.0, 500.0, 999.0, 1000.0]])
    weight = capped_velocity_weight(clocks, 1000, 400)
    torch.testing.assert_close(weight, torch.tensor([[400.0, 400.0, 400.0, 400.0, 100.0, 4.0, 1 / 0.999 ** 2, 1.0]]))
    assert bool((capped_velocity_weight(clocks, 1000, 1) == 1).all())        # 1/t^2 >= 1: cap 1 is no weight
    assert float(capped_velocity_weight(clocks, 1000, 1e12).max()) == pytest.approx(1e12)
    assert float(weight.min()) >= 1                                          # noisy frames are never suppressed


@pytest.mark.parametrize("kind,terms", [("deformable", 1), ("rigid", 1), ("mixed", 2)])
@pytest.mark.parametrize("cap", [4.0, 400.0])
def test_a_known_error_is_weighted_per_frame_in_both_terms(kind, terms, cap):
    loss, metrics, draw = run(kind, velocity_weight_cap=cap)
    weight = capped_velocity_weight(draw["timesteps"], 1000, cap)
    assert weight.shape == draw["timesteps"].shape and float(weight.max()) > float(weight.min())
    # Every (sample, frame) holds the same number of scored coordinates, so each term is ERROR^2 * mean weight.
    assert float(loss) == pytest.approx(terms * ERROR ** 2 * float(weight.mean()), rel=1e-4)
    if kind != "deformable":
        assert metrics["loss_pose_native"] == pytest.approx(ERROR ** 2 * float(weight.mean()), rel=1e-4)


@pytest.mark.parametrize("kind,terms", [("deformable", 1), ("rigid", 1), ("mixed", 2)])
def test_cap_one_is_the_flat_clean_target_loss_bit_for_bit(kind, terms):
    """Particles and, under pose_loss_space=sample, rigid poses: a cap of 1 is no weight at all."""
    (loss, metrics, draw), (loss1, metrics1, draw1) = run(kind), run(kind, velocity_weight_cap=1.0)
    assert draw.pop("batch") is not None and draw1.pop("batch") is not None
    torch.testing.assert_close((loss, metrics, draw), (loss1, metrics1, draw1), rtol=0, atol=0)
    assert float(loss) == pytest.approx(terms * ERROR ** 2, rel=1e-5)


def test_the_uncapped_limit_is_the_production_velocity_space_pose_loss():
    """Off, a sample head's pose term is the velocity-space error, i.e. the clean pose error over t.
    The cap bounds exactly that weight, so an unreachable cap must reproduce the historical term."""
    historical, _, draw = run("rigid", pose_loss_space="velocity")
    unreachable, _, draw2 = run("rigid", velocity_weight_cap=1e12)
    torch.testing.assert_close(draw["timesteps"], draw2["timesteps"], rtol=0, atol=0)
    assert float(unreachable) == pytest.approx(float(historical), rel=1e-3)
    capped, _, _ = run("rigid", velocity_weight_cap=400.0)
    flat, _, _ = run("rigid")
    assert float(flat) <= float(capped) <= float(unreachable)
    # Each space reports its own prediction/target pair to the spike diagnostics.
    torch.testing.assert_close(draw2["pose_target"], draw2["pose_x0"], rtol=0, atol=0)
    assert not torch.equal(draw["pose_target"], draw["pose_x0"])


@pytest.mark.parametrize("kind", ["rigid", "mixed"])
def test_with_rigid_bodies_the_cap_requires_the_clean_target_pose_space(kind):
    with pytest.raises(ValueError, match="pose_loss_space=sample"):
        run(kind, pose_loss_space="velocity", velocity_weight_cap=400.0)
    loss, _, _ = run("deformable", pose_loss_space="velocity", velocity_weight_cap=400.0)   # no pose term to score
    assert torch.isfinite(loss)


def test_the_cap_composes_with_a_shaped_schedule():
    loss, _, draw = run("mixed", velocity_weight_cap=400.0, timestep_schedule={"ramp": 0.5, "independent": 0.5},
                        ramp_slope=0.25)
    weight = capped_velocity_weight(draw["timesteps"], 1000, 400.0)
    assert float(loss) == pytest.approx(2 * ERROR ** 2 * float(weight.mean()), rel=1e-4)


def test_the_cap_refuses_heads_and_terms_it_does_not_cover():
    velocity = TrajectoryFlowMatching(prediction_type="velocity")
    with pytest.raises(ValueError, match="sample head"):
        trajectory_objective(GroundTruthOracle(velocity), scene("mixed"), velocity, velocity_weight_cap=400.0)
    sample = TrajectoryFlowMatching(prediction_type="sample")
    for bad in (0.5, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite number >= 1"):
            trajectory_objective(GroundTruthOracle(sample), scene("mixed"), sample, velocity_weight_cap=bad)
    with pytest.raises(ValueError, match="supervise_context"):
        trajectory_objective(GroundTruthOracle(sample), scene("mixed"), sample, velocity_weight_cap=400.0,
                             noise_context=True, supervise_context=True)


def test_recipes_resolve_the_cap_and_resume_cannot_change_it():
    from utils.checkpoint import validate_resume_configuration
    from utils.config import load_config
    sample = ["flow_matching.prediction_type=sample", "train.pose_loss_space=sample"]
    capped = load_config("configs/unet.yaml", sample + ["train.velocity_weight_cap=400"])
    assert trajectory_loss_options(capped)["velocity_weight_cap"] == 400.0
    assert trajectory_loss_options(capped)["pose_loss_space"] == "sample"
    plain = load_config("configs/unet.yaml", sample)
    assert "velocity_weight_cap" not in trajectory_loss_options(plain)
    assert "velocity_weight_cap" not in trajectory_loss_options(load_config(
        "configs/unet.yaml", sample + ["train.velocity_weight_cap=0"]))
    for saved, requested in ((plain, capped), (capped, plain),
                             (capped, load_config("configs/unet.yaml", sample + ["train.velocity_weight_cap=100"]))):
        with pytest.raises(ValueError, match="training contract"):
            validate_resume_configuration({"cfg": OmegaConf.to_container(saved, resolve=True)}, requested)
    validate_resume_configuration({"cfg": OmegaConf.to_container(capped, resolve=True)}, capped)
    for overrides, match in ((["train.velocity_weight_cap=400"], "prediction_type=sample"),
                             (sample[:1] + ["train.velocity_weight_cap=400"], "train.pose_loss_space=sample"),
                             (sample + ["train.velocity_weight_cap=0.5"], "finite number >= 1"),
                             (sample + ["train.velocity_weight_cap=true"], "finite number >= 1"),
                             (sample + ["train.velocity_weight_cap=inf"], "finite number >= 1")):
        with pytest.raises(ValueError, match=match):
            load_config("configs/unet.yaml", overrides)


def test_every_resolved_objective_option_reaches_validation():
    """train.py hands trajectory_loss_options(cfg) to validate(), whose parameter list is explicit:
    an option validation does not name crashes the first validation of a run that enables it."""
    from utils.config import load_config
    # EVERY opt-in option at once, newest included: this list is what stops an option from
    # reaching the objective but not validation, which only shows up at a run's first validation.
    everything = load_config("configs/unet.yaml", [
        "flow_matching.prediction_type=sample", "train.pose_loss_space=sample", "train.velocity_weight_cap=400",
        "train.timestep_schedule.independent=0.5", "train.timestep_schedule.ramp=0.4",
        "train.timestep_schedule.flat=0.1", "train.ramp_slope=[0.02,0.08]", "train.ramp_jitter=0.01",
        "train.noise_context=true", "train.pose_noise_context=true", "model.unet_deformable_context=true"])
    options = trajectory_loss_options(everything)
    assert {"pose_loss_space", "velocity_weight_cap", "timestep_schedule", "ramp_slope", "ramp_jitter",
            "pose_noise_context", "deformable_context"} <= set(options)
    # the stream draw carries a table, so it resolves from its own config
    streaming = trajectory_loss_options(load_config("configs/unet.yaml", [
        "flow_matching.prediction_type=sample", "train.noise_context=true", "train.pose_noise_context=true",
        "train.timestep_schedule.independent=0.6", "train.timestep_schedule.stream=0.4",
        "train.stream_states=data/schedules/stream_states_w19_c4_s4-50.csv.gz",
        "train.stream_keep=3", "train.stream_given=0"]))
    assert "stream_options" in streaming
    for extra in (options, streaming):
        assert set(extra) <= set(inspect.signature(validate).parameters), sorted(set(extra) - set(inspect.signature(validate).parameters))
        assert set(extra) <= set(inspect.signature(trajectory_objective).parameters)
    assert set(options) <= set(inspect.signature(validate).parameters)
    assert set(options) <= set(inspect.signature(trajectory_objective).parameters)
    # ... and the production validation runs the SAME weighted, shaped objective as the training step.
    model, optimizer, accelerator, diffusion = setup(pose=True)
    optimizer.param_groups[0]["lr"] = 0.0
    batch = make_batch()
    chosen = {name: options[name] for name in ("pose_loss_space", "velocity_weight_cap", "timestep_schedule",
                                               "ramp_slope", "ramp_jitter")}
    torch.manual_seed(12345)
    loss, _ = train_step(model, batch, diffusion, optimizer, accelerator,
                         grad_norm_spike_threshold=float("inf"), clip_deltas=False, **chosen)
    values = validate(model, [batch], diffusion, EMA(model), torch.device("cpu"), accelerator, num_batches=1,
                      use_ema=False, noise_context=False, clip_deltas=False, **chosen)
    assert values["loss"] == pytest.approx(loss, rel=1e-6)


@pytest.mark.parametrize("options", [dict(velocity_weight_cap=400.0),
                                     dict(velocity_weight_cap=400.0, timestep_schedule={"ramp": 0.5, "independent": 0.5},
                                          ramp_slope=0.04)])
def test_the_production_step_converges_to_the_clean_target_under_the_weight(options):
    """The weight changes how errors are counted, never where the minimum is."""
    model, optimizer, accelerator, diffusion = setup()
    for group in optimizer.param_groups:
        group["lr"] = 2e-3                                        # the weighted loss is up to 400x steeper
    batch = scalar_batch(target=1.0)
    expected = float(batch.delta_stats.normalize(batch.prediction_x0).mean())
    torch.manual_seed(12)
    for _ in range(400):
        # The toy's gradient is up to 400x steeper than its unweighted one, past the absolute
        # grad-norm gate's default of 50: the gate is a recipe setting, not part of this contract.
        _, metrics = train_step(model, batch, diffusion, optimizer, accelerator, grad_clip=None,
                                grad_norm_spike_threshold=float("inf"), clip_deltas=False, **options)
        assert metrics["optimizer_step"]
    assert float(model.weight) == pytest.approx(expected, abs=1e-3)
