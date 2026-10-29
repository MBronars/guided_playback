"""Pose-noise semantics checked through independent geometry and production boundaries."""
import dataclasses

import pytest
import torch
from omegaconf import OmegaConf

from tests.test_articulated_bodies import gripper
from tests.test_training_safety import setup
from tests.test_unet_std_pose_native import make_batch
from train import train_step, validate
from utils.config import load_config
from utils.ema import EMA
from utils.pose_native import (future_pose_deltas, jitter_rigid_history, materialize_pose_native,
                               pose_sequence_deltas)
from utils.pose_ops import so3_exp, so3_log
from utils.trajectory_objective import trajectory_objective, trajectory_loss_options


def test_pose_delta_convention_has_an_independent_noncommuting_oracle():
    # Known world increments, integrated independently; a body-frame/right convention fails.
    increments = torch.tensor([[.2, 0., 0.], [0., -.3, 0.], [0., 0., .1]])
    dt = torch.tensor([[.01, .02, 0.], [-.03, 0., .01], [0., -.02, .04]])
    poses = torch.eye(4).repeat(1, 4, 1, 1, 1)
    poses[0, 0, 0, :3, :3] = so3_exp(torch.tensor([.3, .2, -.1]))
    for f in range(3):
        poses[0, f+1, 0, :3, :3] = so3_exp(increments[f]) @ poses[0, f, 0, :3, :3]
        poses[0, f+1, 0, :3, 3] = poses[0, f, 0, :3, 3] + dt[f]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = pose_sequence_deltas(poses)
    torch.testing.assert_close(actual[0, :, 0], torch.cat([dt, increments], -1), atol=1e-6, rtol=0)


@pytest.mark.parametrize("target", ["restore", "carry"])
def test_moved_revolute_tree_per_element_masks_and_complete_materialization(target):
    batch, _, _ = gripper(second="revolute")
    # First element: move the whole tree, keep object exact. Second: the reverse.
    mask = torch.tensor([[False, False, False, True], [True, False, False, False]])
    batch = dataclasses.replace(batch, pose_action_mask=mask,
                                pose_actions=batch.pose_actions * mask[:, None, :, None])
    before = batch.positions.clone()
    noisy = jitter_rigid_history(batch, .05, anchor=True, target=target,
                                 generator=torch.Generator().manual_seed(7))
    C = batch.context_len
    for e, stationary in [(0, [3]), (1, [0, 1, 2])]:
        assert torch.equal(noisy.body_poses[e, :, stationary], batch.body_poses[e, :, stationary])
        rows = torch.isin(batch.particle_body[e], torch.tensor(stationary))
        assert torch.equal(noisy.positions[e, :, rows], batch.positions[e, :, rows])
    for child in [1, 2]:
        def relative(b):
            R = b.body_poses[:, :C, 0, :3, :3].transpose(-1, -2)
            return (R @ b.body_poses[:, :C, child, :3, :3],
                    (R @ (b.body_poses[:, :C, child, :3, 3]-b.body_poses[:, :C, 0, :3, 3])[..., None])[..., 0])
        for a, b in zip(relative(noisy), relative(batch)):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=0)
    assert torch.equal(noisy.pose_deltas_gt[:, :, 1:3], batch.pose_deltas_gt[:, :, 1:3])
    expected = batch.future if target == "restore" else noisy.future
    positions, R, t, _ = materialize_pose_native(
        noisy, None, noisy.pose_delta_stats.normalize(noisy.pose_deltas_gt, clip=False))
    torch.testing.assert_close(positions, expected, atol=2e-6, rtol=0)
    expected_poses = (batch if target == "restore" else noisy).body_poses[:, C:]
    torch.testing.assert_close(R, expected_poses[..., :3, :3], atol=2e-6, rtol=0)
    torch.testing.assert_close(t, expected_poses[..., :3, 3], atol=2e-6, rtol=0)
    torch.testing.assert_close(future_pose_deltas(noisy), noisy.pose_deltas_gt, atol=2e-6, rtol=0)
    assert torch.equal(batch.positions, before)
    if target == "carry":
        assert noisy.pose_deltas_gt is batch.pose_deltas_gt
        assert not torch.equal(noisy.future, batch.future)
        # Free root keeps a constant RIGHT/local offset, rather than left/world offset.
        clean = batch.body_poses[0, C-1:, 0, :3, :3]
        moved = noisy.body_poses[0, C-1:, 0, :3, :3]
        local = clean.transpose(-1, -2) @ moved
        torch.testing.assert_close(local, local[:1].expand_as(local), atol=2e-6, rtol=0)
    else:
        assert torch.equal(noisy.pose_deltas_gt[:, 1:], batch.pose_deltas_gt[:, 1:])
        assert torch.equal(noisy.future, batch.future)


def test_passive_history_scope_and_generator_do_not_move_anchor_or_ambient_rng():
    batch, _, _ = gripper()
    state = torch.random.get_rng_state().clone()
    noisy = jitter_rigid_history(batch, .05, scope="passive", generator=torch.Generator().manual_seed(8))
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(noisy.body_poses[:, :, :3], batch.body_poses[:, :, :3])
    assert torch.equal(noisy.positions[:, batch.context_len-1:], batch.positions[:, batch.context_len-1:])
    assert not torch.equal(noisy.positions, batch.positions)


def test_backward_integration_has_the_drawn_forward_transitions():
    batch, _, _ = gripper(articulated=False)
    B, C, K = batch.batch_size, batch.context_len, batch.body_type.shape[1]
    g = torch.Generator().manual_seed(13)
    half, move = batch.pose_delta_stats.scale/2, ~batch.pose_action_mask
    eta = torch.randn(B, C-1, K, 6, generator=g) * .05 * half * move[:, None, :, None]
    eta_anchor = torch.randn(B, K, 6, generator=g) * .05 * half * move[..., None]
    noisy = jitter_rigid_history(batch, .05, anchor=True, generator=torch.Generator().manual_seed(13))
    R = batch.body_poses[:, :C, :, :3, :3]
    Rn = noisy.body_poses[:, :C, :, :3, :3]
    expected_rotation = so3_exp(eta[..., 3:]) @ R[:, 1:] @ R[:, :-1].transpose(-1, -2)
    torch.testing.assert_close(Rn[:, 1:] @ Rn[:, :-1].transpose(-1, -2), expected_rotation, atol=2e-6, rtol=0)
    t, tn = batch.body_poses[:, :C, :, :3, 3], noisy.body_poses[:, :C, :, :3, 3]
    torch.testing.assert_close(tn[:, 1:]-tn[:, :-1], t[:, 1:]-t[:, :-1]+eta[..., :3], atol=1e-7, rtol=0)
    torch.testing.assert_close(tn[:, -1]-t[:, -1], eta_anchor[..., :3], atol=1e-7, rtol=0)


def test_carry_keeps_deformable_rows_and_given_actions_bitwise_unchanged():
    batch = make_batch()
    noisy = jitter_rigid_history(batch, .05, anchor=True, target="carry", generator=torch.Generator().manual_seed(3))
    deform = batch.body_type.gather(1, batch.particle_body) == 0
    stationary = deform | batch.action_mask
    rows = stationary[:, None, :, None].expand_as(batch.positions)
    assert torch.equal(noisy.positions[rows], batch.positions[rows])
    assert torch.equal(noisy.actions, batch.actions) and torch.equal(noisy.pose_actions, batch.pose_actions)
    torch.testing.assert_close(noisy.anchor[:, None]+noisy.future_deltas.cumsum(1), noisy.future, atol=1e-6, rtol=0)


@pytest.mark.parametrize("overrides", [["train.light_pose_noise_target=carry"],
    ["train.light_pose_noise_target=unknown"], ["train.light_pose_noise_scope=unknown"],
    ["train.light_pose_noise_std=.05", "train.light_pose_noise_anchor=true", "train.light_pose_noise_scope=all"]])
def test_invalid_pose_noise_policies_fail_before_training(overrides):
    with pytest.raises(ValueError, match="light_pose_noise"):
        load_config("configs/unet.yaml", overrides)


def test_restore_and_carry_match_context_and_downstream_noise_draws():
    model, _, _, diffusion = setup(pose=True)
    batch = make_batch()
    draws, states = [], []
    for target in ["restore", "carry"]:
        torch.manual_seed(13)
        _, _, draw = trajectory_objective(model, batch, diffusion, light_pose_noise_std=.05,
            light_pose_noise_anchor=True, light_pose_noise_target=target, clip_pose_deltas=False)
        draws.append(draw)
        states.append(torch.random.get_rng_state().clone())
    a, b = draws
    assert torch.equal(a["batch"].context, b["batch"].context)
    assert torch.equal(a["timesteps"], b["timesteps"]) and torch.equal(states[0], states[1])
    # Training uses the same Gaussian draws despite different pose targets.
    torch.testing.assert_close(a["pose_target"] + a["pose_x0"], b["pose_target"] + b["pose_x0"], atol=2e-7, rtol=0)


@pytest.mark.parametrize("anchor,target,scope", [(False, "restore", "all"),
    (False, "restore", "passive"), (True, "restore", "passive"), (True, "carry", "passive")])
def test_train_validate_parity_and_probe_use_effective_batch(anchor, target, scope):
    model, optimizer, accelerator, diffusion = setup(pose=True)
    optimizer.param_groups[0]["lr"] = 0
    seen = []
    def forward(batch, x, timesteps, **kwargs):
        seen.append((batch, x.clone(), timesteps.clone(), kwargs["noisy_pose"].clone()))
        return model.weight.expand_as(x), model.weight.expand_as(kwargs["noisy_pose"])
    model.forward = forward
    batch = make_batch()
    options = dict(noise_context=False, light_pose_noise_std=.05, light_pose_noise_anchor=anchor,
                   light_pose_noise_target=target, light_pose_noise_scope=scope,
                   clip_pose_deltas=False, action_dropout=.5)
    torch.manual_seed(12345)
    loss, _ = train_step(model, batch, diffusion, optimizer, accelerator, **options)
    state = torch.random.get_rng_state().clone()
    metrics = validate(model, [batch], diffusion, EMA(model), torch.device("cpu"), accelerator,
                       num_batches=1, use_ema=False, **options)
    assert len(seen) == 3 and seen[1][0] is seen[2][0]  # objective and large-t diagnostic
    assert seen[1][0] is not batch
    assert torch.equal(seen[0][0].positions, seen[1][0].positions)
    assert torch.equal(seen[0][0].pose_action_mask, seen[1][0].pose_action_mask)
    for a, b in zip(seen[0][1:], seen[1][1:]):
        assert torch.equal(a, b)
    assert loss == pytest.approx(metrics["loss"], abs=0)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("value", ["1", "oops", '"false"', "null"])
def test_anchor_flag_requires_a_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        load_config("configs/unet.yaml", ["train.light_pose_noise_std=.05", "train.clip_pose_deltas=false",
                                          "train.light_pose_noise_anchor="+value])
    cfg = OmegaConf.create({"train": {"light_pose_noise_anchor": OmegaConf.from_dotlist(["x="+value]).x}})
    with pytest.raises(ValueError, match="must be a boolean"):
        trajectory_loss_options(cfg)


@pytest.mark.parametrize("target", ["restore", "carry"])
def test_moved_anchor_refuses_pose_clipping_in_config_and_objective(target):
    overrides = ["train.light_pose_noise_std=.05", "train.light_pose_noise_anchor=true",
                 "train.light_pose_noise_target="+target]
    with pytest.raises(ValueError, match="clip_pose_deltas=false"):
        load_config("configs/unet.yaml", overrides)
    model, _, _, diffusion = setup(pose=True)
    with pytest.raises(ValueError, match="clip_pose_deltas=false"):
        trajectory_objective(model, make_batch(), diffusion, light_pose_noise_std=.05,
                              light_pose_noise_anchor=True, light_pose_noise_target=target)
    # Particle clipping remains a separate setting: only pose clipping binds this invariant.
    cfg = load_config("configs/unet.yaml", overrides+["train.clip_pose_deltas=false"])
    assert trajectory_loss_options(cfg)["clip_deltas"] is True


def test_boundary_restore_target_reaches_recording_through_unclipped_objective():
    batch, _, _ = gripper(second="prismatic")
    poses, positions, C, k = batch.body_poses.clone(), batch.positions.clone(), batch.context_len, 3
    for f in range(C, positions.shape[1]):
        x = poses[:, C-1, k, 0, 3] + .0198 * (f-C+1)
        shift = x - poses[:, f, k, 0, 3]
        poses[:, f, k, 0, 3] = x
        positions[:, f, batch.particle_body[0] == k, 0] += shift[:, None]
    batch = dataclasses.replace(batch, positions=positions, body_poses=poses)
    batch = dataclasses.replace(batch, pose_deltas_gt=future_pose_deltas(batch))
    model, _, _, diffusion = setup(pose=True)
    torch.manual_seed(0)
    _, _, draw = trajectory_objective(model, batch, diffusion, light_pose_noise_std=.05,
        light_pose_noise_anchor=True, clip_pose_deltas=False)
    assert draw["pose_x0"][:, 0, k, 0].max() > 1  # Correction crosses the old clipping bound.
    predicted = materialize_pose_native(draw["batch"], None, draw["pose_x0"])[0]
    torch.testing.assert_close(predicted, batch.future, atol=2e-6, rtol=0)


def test_restore_target_feeds_the_sample_space_pose_loss():
    """The 2026-09-19 protocol trains x0 heads with train.pose_loss_space=sample. With the moved anchor and the restore
    target, the pose term compares the prediction with the PATCHED clean target, flat in t, and that target still
    lands on the recorded future."""
    batch, _, _ = gripper(second="prismatic")
    model, _, _, diffusion = setup(pose=True)
    torch.manual_seed(0)
    loss, metrics, draw = trajectory_objective(model, batch, diffusion, light_pose_noise_std=.05,
        light_pose_noise_anchor=True, light_pose_noise_target="restore", clip_pose_deltas=False,
        pose_loss_space="sample")
    assert torch.equal(draw["pose_target"], draw["pose_x0"])                      # clean target, not noise - x0
    assert not torch.equal(draw["batch"].body_poses[:, batch.context_len - 1], batch.body_poses[:, batch.context_len - 1])
    predicted = materialize_pose_native(draw["batch"], None, draw["pose_x0"])[0]
    torch.testing.assert_close(predicted, batch.future, atol=2e-6, rtol=0)
    square = (draw["pose_prediction"].float() - draw["pose_x0"].float()) ** 2
    mask = draw.get("pose_mask")
    expected = square.mean() if mask is None else (square * mask).sum() / (mask.sum() * square.shape[0] * square.shape[1])
    assert metrics["loss_pose_native"] == pytest.approx(float(expected.detach()), rel=1e-6)
    assert torch.isfinite(loss)


def test_random_walk_noise_accumulates_and_pins_the_oldest_frame():
    """train.light_pose_noise_type=random_walk is GNS's noise: increments at std/sqrt(C-1) summed
    across the window, so transition noise is correlated and the anchor offset is its integral.
    With a moved anchor the oldest frame keeps its recorded translation exactly, the mirror of the
    iid recipe, which draws the anchor independently and leaves the oldest frame furthest away."""
    batch, _, _ = gripper(second="prismatic")
    C, OBJ = batch.context_len, 3
    obj = (batch.particle_body == OBJ)[0]

    def offsets(**kw):
        torch.manual_seed(0)
        n = jitter_rigid_history(batch, 0.5, **kw)
        d = (n.positions[:, :C, obj] - batch.positions[:, :C, obj]).norm(dim=-1).mean(dim=(0, 2))
        return [round(float(v), 6) for v in d]

    walk = offsets(anchor=True, target="carry", noise_type="random_walk")
    iid = offsets(anchor=True, target="carry")
    assert walk[-1] > 0.0 and iid[-1] > 0.0 and iid[0] > iid[-1]   # iid: own anchor draw, oldest worst

    # A ROTATING passive body is the case that matters. Deriving the anchor by summing the angular
    # perturbations is wrong to first order once a body turns, because backward integration
    # conjugates the accumulated error by each rotation increment; the forward construction is not.
    # The gripper fixture's object only translates, so it cannot see this.
    rot = make_batch()
    Cr = rot.context_len
    k = int(((rot.body_type[0] == 1) & ~rot.pose_action_mask[0]).nonzero(as_tuple=True)[0][0])
    turn = so3_log((rot.body_poses[:, 1:Cr, k, :3, :3]
                    @ rot.body_poses[:, :Cr - 1, k, :3, :3].transpose(-1, -2)).reshape(-1, 3, 3)).norm(dim=-1)
    assert float(turn.min()) > 0.02, "the fixture body must actually rotate for this to test anything"
    for std in (0.05, 0.4):                       # linear in std would betray a first-order error
        torch.manual_seed(0)
        n = jitter_rigid_history(rot, std, anchor=True, target="carry", noise_type="random_walk")
        torch.testing.assert_close(n.body_poses[:, 0, k], rot.body_poses[:, 0, k], atol=1e-6, rtol=0)
        assert not torch.equal(n.body_poses[:, Cr - 1, k], rot.body_poses[:, Cr - 1, k])


def test_random_walk_matches_iid_magnitude_at_the_most_recent_transition():
    """`std` must mean the same thing in both recipes or a carry-versus-iid A/B confounds
    magnitude with correlation. GNS scales its per-step draws by 1/sqrt(num_velocities)."""
    batch, _, _ = gripper(second="prismatic")
    C, OBJ = batch.context_len, 3

    def last_transition_rms(noise_type, trials=128):
        out = []
        for seed in range(trials):
            torch.manual_seed(seed)
            n = jitter_rigid_history(batch, 0.5, anchor=True, target="carry", noise_type=noise_type)
            clean = batch.body_poses[:, C - 1, OBJ, :3, 3] - batch.body_poses[:, C - 2, OBJ, :3, 3]
            noisy = n.body_poses[:, C - 1, OBJ, :3, 3] - n.body_poses[:, C - 2, OBJ, :3, 3]
            out.append((noisy - clean).pow(2).mean().sqrt())
        return float(torch.stack(out).mean())

    iid, walk = last_transition_rms("iid"), last_transition_rms("random_walk")
    # Tight on purpose: dropping the 1/sqrt(C-1) scaling measures 1.46 here, which a loose band
    # would wave through while the two arms of an A/B carried different noise magnitudes.
    assert 0.88 < walk / iid < 1.20, (iid, walk)


def test_random_walk_is_opt_in_and_validated():
    """The default recipe is untouched, and a bad value fails in the config and the objective."""
    batch, _, _ = gripper(second="prismatic")
    torch.manual_seed(0)
    default = jitter_rigid_history(batch, 0.5, anchor=True)
    torch.manual_seed(0)
    explicit = jitter_rigid_history(batch, 0.5, anchor=True, noise_type="iid")
    assert torch.equal(default.positions, explicit.positions)
    assert torch.equal(default.body_poses, explicit.body_poses)
    assert torch.equal(default.pose_deltas_gt, explicit.pose_deltas_gt)
    with pytest.raises(ValueError, match="iid or random_walk"):
        jitter_rigid_history(batch, 0.5, noise_type="brownian")
    with pytest.raises(ValueError, match="iid or random_walk"):
        load_config("configs/unet.yaml", ["train.light_pose_noise_std=0.05",
                                          "train.light_pose_noise_type=brownian"])
    cfg = load_config("configs/unet.yaml", [
        "train.light_pose_noise_std=0.2", "train.light_pose_noise_anchor=true",
        "train.light_pose_noise_target=carry", "train.light_pose_noise_scope=passive",
        "train.light_pose_noise_type=random_walk", "train.clip_pose_deltas=false"])
    assert trajectory_loss_options(cfg)["light_pose_noise_type"] == "random_walk"
