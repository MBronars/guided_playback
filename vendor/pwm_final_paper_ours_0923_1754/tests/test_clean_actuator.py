"""model.unet_pose_action_mode=clean on the factorized UNet: a GIVEN actuated body is shown unnoised
(no action vector), its tokens are told t = 0, and which bodies are given follows train.action_dropout.

Every behavioural test carries a control that must FAIL without the mode, and the zero-init pieces are
woken first -- otherwise the assertions pass vacuously.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import test_unet_std_pose_native as PN                     # the pose-native fixtures of this tree

from diffusion import TrajectoryFlowMatching
from models.unet_std import UNetWrapper
from utils.trajectory_objective import trajectory_objective


def wrapper(mode, **kw):
    torch.manual_seed(0)
    return UNetWrapper(PN._model_cfg(pose=True, unet_pose_action_mode=mode, **kw)).eval()


def diffusion():
    return TrajectoryFlowMatching(num_train_timesteps=1000, shift=1.0, prediction_type="sample")


def batch():
    b = PN.make_batch(deform_n=0)                 # all-rigid, like pickup / rl_push / yikai
    ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
    act = b.pose_action_mask[:, ridx]
    assert bool(act.any()) and not bool(act.all()), "fixture needs a given AND a passive rigid body"
    return b


def fed_pose(model, b, mode, dropout=0.0, seed=0):
    """Run the TRAINING objective and record the noisy pose the model was actually handed."""
    seen = {}
    orig = model.forward
    def spy(*a, **k):
        seen["noisy_pose"] = k.get("noisy_pose")
        return orig(*a, **k)
    model.forward = spy
    try:
        torch.manual_seed(seed)
        _, _, draw = trajectory_objective(model, b, diffusion(), pose_action_mode=mode,
                                          action_dropout=dropout, noise_context=False)
    finally:
        model.forward = orig
    return seen["noisy_pose"], draw["pose_x0"], draw["batch"]


# ----- construction -----------------------------------------------------------------------------------

def test_default_is_skip_and_keeps_the_action_vector():
    w = wrapper("skip")
    assert w.unet_cfg.pose_action_mode == "skip"
    assert w.model.pose_action_skip is not None and w.model.pose_no_action_token is not None


def test_clean_builds_no_action_vector():
    w = wrapper("clean")
    assert w.model.pose_action_skip is None and w.model.pose_no_action_token is None
    assert not any("pose_action_skip" in k or "pose_no_action" in k for k in w.state_dict())


# ----- training: the given body arrives unnoised -------------------------------------------------------

def test_given_body_is_unnoised_and_passive_is_not():
    b = batch()
    ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
    noisy, x0, eff = fed_pose(wrapper("clean"), b, "clean")
    given = eff.pose_action_mask[:, ridx]                                   # [B, Kr]
    g = given[:, None, :, None].expand_as(noisy)
    assert torch.equal(noisy[g], x0[g]), "a given body's pose rows were noised"
    assert not torch.allclose(noisy[~g], x0[~g]), "passive bodies were not noised"


def test_control_skip_mode_noises_everything():
    """The same draw under skip mode must noise the actuated body -- or the test above is vacuous."""
    b = batch()
    ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
    noisy, x0, eff = fed_pose(wrapper("skip"), b, "skip")
    g = eff.pose_action_mask[:, ridx][:, None, :, None].expand_as(noisy)
    assert not torch.allclose(noisy[g], x0[g]), "skip mode left the actuated body clean"


def test_given_set_follows_action_dropout():
    """dropout 1 hides every action: nothing is given, so every pose row is noised."""
    b = batch()
    noisy, x0, eff = fed_pose(wrapper("clean"), b, "clean", dropout=1.0)
    assert not bool(eff.pose_action_mask.any()), "dropout 1 should hide every actuated body"
    assert not torch.isclose(noisy, x0).all(), "nothing was noised under full dropout"


# ----- the t = 0 substitution -------------------------------------------------------------------------

def test_given_object_is_modulated_at_t0():
    """A given object's modulation must be adaln(t_emb0) whatever its frame's level is. Asserted at the
    Block, where it happens: with every token of one object given, the block's output cannot depend on
    t_emb. The control -- same object NOT given -- must depend on it."""
    w = wrapper("clean")
    blk = next(m for m in w.model.modules() if hasattr(m, "adaln") and hasattr(m, "adaln_obj"))
    with torch.no_grad():                                                     # wake adaLN-Zero
        blk.adaln.weight.normal_(0, 0.05); blk.adaln.bias.normal_(0, 0.5)
    Bt, T, n, wd = 2, 3, 6, blk.norm1.normalized_shape[0] if hasattr(blk, "norm1") else None
    wd = blk.adaln.out_features // (12 if blk.dual_ffn else 9)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(Bt, T, n, wd, generator=g)
    pos = torch.randn(Bt, T, n, 3, generator=g)
    tdim = blk.adaln.in_features
    tA, tB = torch.randn(Bt, T, tdim, generator=g), torch.randn(Bt, T, tdim, generator=g)
    t0 = torch.randn(Bt, 1, tdim, generator=g)
    idx = torch.zeros(Bt, n, dtype=torch.long)                                # one object, slot 0
    def run(t_emb, given):
        with torch.no_grad():
            return blk(x, pos, t_emb, None, obj_index=idx,
                       obj_given=torch.full((Bt, 1), given), t_emb0=t0)
    assert torch.allclose(run(tA, True), run(tB, True), atol=1e-6), \
        "a given object's output still depends on its frame's timestep"
    assert not torch.allclose(run(tA, False), run(tB, False), atol=1e-6), \
        "control is vacuous: the output does not depend on the timestep at all"


# ----- sampling ---------------------------------------------------------------------------------------

def test_sampler_pins_given_rows_every_step():
    from inference import run_diffusion_inference_df
    b = batch()
    ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
    clean = b.pose_delta_stats.normalize(b.pose_deltas_gt[:, :, ridx], clip=False)
    given = b.pose_action_mask[:, ridx][:, None, :, None].expand_as(clean)
    w = wrapper("clean")
    torch.manual_seed(0)
    with torch.no_grad():
        _, x_pose = run_diffusion_inference_df(w, b, diffusion(), num_inference_steps=4,
                                               schedule="full_sequence")
    assert torch.allclose(x_pose[given], clean[given].to(x_pose), atol=1e-6), \
        "the sampler did not hold the given body at its clean pose"
    # control: a skip-mode model is not pinned
    w2 = wrapper("skip")
    torch.manual_seed(0)
    with torch.no_grad():
        _, x2 = run_diffusion_inference_df(w2, b, diffusion(), num_inference_steps=4,
                                           schedule="full_sequence")
    assert not torch.allclose(x2[given], clean[given].to(x2), atol=1e-6), "skip mode was pinned"


# ----- config -----------------------------------------------------------------------------------------

def _policy_cfg(**model):
    """A minimal config that reaches utils.config._validate_training_policy's action checks."""
    from omegaconf import OmegaConf
    from utils.config import DEFAULTS
    cfg = OmegaConf.create(DEFAULTS)
    cfg.model = OmegaConf.merge(cfg.model, dict(architecture="unet", **model))
    return cfg


def test_config_rejects_clean_with_an_action_channel():
    """clean passes NO action vector; with unet_actions=true the model would read the given motion twice."""
    from utils.config import _validate_training_policy
    with pytest.raises(ValueError, match="unet_actions=false"):
        _validate_training_policy(_policy_cfg(unet_pose_action_mode="clean", unet_actions=True))


def test_config_rejects_an_unknown_mode():
    from utils.config import _validate_training_policy
    with pytest.raises(ValueError, match="skip or clean"):
        _validate_training_policy(_policy_cfg(unet_pose_action_mode="claen"))


def test_mode_typo_is_caught_where_the_model_is_built():
    """A typo must not silently train skip mode, even on a path that skips load_config."""
    with pytest.raises(AssertionError, match="skip or clean"):
        UNetWrapper(PN._model_cfg(pose=True, unet_pose_action_mode="claen"))
