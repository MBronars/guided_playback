"""Unconditional actuator (train.action_dropout): the model is never shown the hidden
future motion, validation scores what it deploys, and train.init_from starts a new
experiment from the same network in the same coordinates."""
import dataclasses

import pytest
import torch
from accelerate import Accelerator

from diffusion.flow_matching import TrajectoryFlowMatching
from models.unet_std import UNetWrapper
from tests.test_unet_std_pose_native import _model_cfg, make_batch
from utils.conditioning import hide_actions
from utils.ema import EMA
from utils.trajectory_objective import trajectory_objective


def _unet(**config):
    torch.manual_seed(29417)
    model = UNetWrapper(_model_cfg(**config)).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.025)
    return model


def _other_future(batch):
    """The same scene with a different given actuator future (the pose-token stream)."""
    moved = batch.pose_actions + 0.01 * batch.pose_action_mask[:, None, :, None]
    return dataclasses.replace(batch, pose_actions=moved)


def test_hiding_replaces_only_the_pose_action_stream():
    batch = make_batch()
    assert batch.pose_action_mask.any() and batch.pose_actions.abs().sum() > 0
    assert hide_actions(batch, 0.0) is batch
    hidden = hide_actions(batch, 1.0)
    assert not hidden.pose_action_mask.any()
    assert not hidden.pose_actions.any()
    for name in ("positions", "actions", "action_mask", "body_poses", "pose_deltas_gt", "mat_type"):
        assert torch.equal(getattr(hidden, name), getattr(batch, name)), name


def test_partial_hiding_is_per_element_and_follows_the_ambient_seed():
    batch = make_batch()
    torch.manual_seed(5)
    first = hide_actions(batch, 0.5).pose_action_mask
    torch.manual_seed(5)
    assert torch.equal(hide_actions(batch, 0.5).pose_action_mask, first)
    for element in range(batch.batch_size):
        kept = first[element][batch.pose_action_mask[element]]
        assert bool(kept.all()) or not bool(kept.any())       # whole elements, never single bodies


@pytest.mark.parametrize("dropout,invariant", [(1.0, True), (0.0, False)])
def test_objective_never_shows_the_hidden_future_motion(dropout, invariant):
    model, batch, diffusion = _unet(), make_batch(), TrajectoryFlowMatching()
    losses = []
    for candidate in (batch, _other_future(batch)):
        torch.manual_seed(11)
        losses.append(trajectory_objective(model, candidate, diffusion, action_dropout=dropout)[0])
    assert torch.equal(losses[0], losses[1]) is invariant


@pytest.mark.parametrize("dropout,invariant", [(1.0, True), (0.5, False), (0.0, False)])
def test_sampler_validation_generates_the_actuator_only_for_an_unconditional_recipe(dropout, invariant):
    from train import validate_sampler
    model, diffusion, accelerator = _unet(), TrajectoryFlowMatching(), Accelerator(cpu=True)
    batch = make_batch()
    scores = [validate_sampler(model, [candidate], diffusion, EMA(model), torch.device("cpu"), accelerator,
                               num_batches=1, num_inference_steps=2, use_ema=False,
                               action_dropout=dropout)["sample_mm"]
              for candidate in (batch, _other_future(batch))]
    assert (scores[0] == scores[1]) is invariant


def test_validation_loss_and_its_probe_hide_the_actions_for_an_unconditional_recipe():
    from train import validate
    model, diffusion, accelerator = _unet(), TrajectoryFlowMatching(), Accelerator(cpu=True)
    batch = make_batch()
    results = [validate(model, [candidate], diffusion, EMA(model), torch.device("cpu"), accelerator,
                        num_batches=1, use_ema=False, action_dropout=1.0)
               for candidate in (batch, _other_future(batch))]
    assert "x0_mm_t09" in results[0]
    assert results[0] == results[1]


@pytest.mark.parametrize("bidirectional", [False, True])
def test_clean_context_unet_initialization_preserves_the_sampled_function(bidirectional):
    """Disabling historical context noise changes metadata from 0 to None. The
    UNet maps both to t=0, so identical weights/noise still sample identically.
    Exercise the full sampler with nonzero modulation/output weights, not just
    the timestep helper or a zero-initialized network.
    """
    from inference import run_diffusion_inference_df
    model = _unet(unet_bidirectional_temporal=bidirectional)
    batch, diffusion = hide_actions(make_batch(), 1.0), TrajectoryFlowMatching()
    outputs = []
    for context_timestep in (0, None, 1):
        model._clean_context_timestep = context_timestep
        model._ctx_timesteps_trained = context_timestep is not None
        torch.manual_seed(818)
        outputs.append(run_diffusion_inference_df(model, batch, diffusion, num_inference_steps=3))
    for before, after in zip(*outputs[:2]):
        assert before.abs().sum() > 0
        torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert any(not torch.equal(a, b) for a, b in zip(outputs[0], outputs[2]))


def test_config_accepts_the_unconditional_unet_recipe_and_rejects_the_rest(tmp_path):
    from utils.config import load_config
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"")
    base = ["data.domains=[pickup]"]
    cfg = load_config("configs/unet.yaml", [*base, "train.action_dropout=1.0", f"train.init_from={checkpoint}"])
    assert cfg.train.action_dropout == 1.0 and cfg.train.init_from == str(checkpoint)
    for overrides, match in (
        (["train.action_dropout=1.5"], "action_dropout"),
        (["train.action_dropout=1", "model.unet_actions=true"], "action_dropout"),
        (["train.init_from=relative/source.pt"], "init_from"),
        ([f"train.init_from={tmp_path / 'missing.pt'}"], "init_from"),
    ):
        with pytest.raises(ValueError, match=match):
            load_config("configs/unet.yaml", [*base, *overrides])
    with pytest.raises(ValueError, match="action_dropout"):
        load_config("configs/transformer.yaml", [*base, "train.action_dropout=1"])


def test_initialization_requires_the_same_network_and_coordinates():
    from omegaconf import OmegaConf
    from utils.checkpoint import validate_initialization
    from utils.config import load_config
    base = ["data.domains=[pickup]", "model.articulated_bodies=true"]
    bidirectional = [*base, "model.unet_bidirectional_temporal=true"]
    state = {"cfg": OmegaConf.to_container(load_config("configs/unet.yaml", bidirectional), resolve=True)}
    validate_initialization(state, load_config("configs/unet.yaml", [*bidirectional, "train.lr=5e-5",
                                                                      "train.action_dropout=1"]))
    with pytest.raises(ValueError, match="unet_bidirectional_temporal"):
        validate_initialization(state, load_config("configs/unet.yaml", base))
    with pytest.raises(ValueError):
        validate_initialization(state, load_config("configs/unet.yaml",
                                                   [*bidirectional, "normalization.profile=xy_p99_global"]))


@pytest.mark.parametrize("hidden", [False, True], ids=["given", "hidden"])
def test_native_zero_shot_window_cannot_read_hidden_actions(hidden):
    """scripts/dev/native_zero_shot.sample_window with actions hidden is bitwise invariant to the
    given actuator future; with actions given, the same change moves the prediction."""
    from scripts.dev.native_zero_shot import sample_window
    model, batch, diffusion = _unet(), make_batch(), TrajectoryFlowMatching()
    predictions = []
    for candidate in (batch, _other_future(batch)):
        torch.manual_seed(109)
        with torch.no_grad():
            predictions.append(sample_window(model, diffusion, candidate, 2, "pyramid", 1.0,
                                             hide_given_actions=hidden)[0])
    assert torch.equal(predictions[0], predictions[1]) == hidden
