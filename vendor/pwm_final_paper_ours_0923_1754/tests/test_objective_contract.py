"""Training and validation must measure the same mathematical objective."""
import pytest
import torch

from tests.test_training_safety import setup, scalar_batch
from tests.test_train import create_dummy_batch
from train import train_step, validate, validate_sampler
from utils.ema import EMA
from utils.trajectory_objective import compute_aux_losses, trajectory_objective, trajectory_loss_options


def test_spread_axes_can_be_enabled_on_its_own():
    pred = torch.randn(1, 4, 8, 3, requires_grad=True)
    gt = torch.zeros_like(pred)
    losses = compute_aux_losses(pred, gt, torch.ones_like(pred, dtype=torch.bool),
                               torch.zeros(1, 8, dtype=torch.long), {"spread_axes": 1.0})
    assert set(losses) == {"spread_axes"}
    assert losses["spread_axes"] > 0
    losses["spread_axes"].backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


@pytest.mark.parametrize("padding_z", [-10.0, float("nan")])
def test_floor_excludes_padding_from_loss_and_gradients(padding_z):
    from utils.phys_losses import floor_loss
    batch = create_dummy_batch(B=1, T=3, N=3, context_len=2)
    batch.positions.zero_()
    batch.positions[..., 2] = torch.tensor([-0.2, 0.1, padding_z])
    batch.valid_lengths[:] = 2
    predicted = torch.zeros_like(batch.future_deltas, requires_grad=True)
    loss = floor_loss(batch, predicted, floor_z=0)
    assert loss.item() == pytest.approx(0.02)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
    assert predicted.grad[0, 0, 2].abs().sum() == 0
    assert predicted.grad[0, 0, 0, 2].item() == pytest.approx(-0.2)


def test_floor_uses_objective_deformable_mask_and_handles_empty_mask():
    from utils.phys_losses import floor_loss
    batch = create_dummy_batch(B=1, T=3, N=3, context_len=2)
    batch.positions.zero_()
    batch.positions[..., 2] = torch.tensor([0.1, -1.0, float("nan")])
    batch.valid_lengths[:] = 2
    predicted = torch.zeros_like(batch.future_deltas, requires_grad=True)
    deformable = torch.tensor([[[True, False, False]]])
    losses = compute_aux_losses(predicted, torch.zeros_like(predicted),
        deformable[..., None].expand_as(predicted), batch.object_ids,
        {"floor": 1.0, "floor_z": 0.0}, batch=batch)
    assert losses["floor"].item() == 0
    losses["floor"].backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() == 0
    predicted.grad = None
    empty_loss = floor_loss(batch, predicted, floor_z=0, valid_mask=torch.zeros_like(deformable))
    assert empty_loss.requires_grad and empty_loss.item() == 0
    empty_loss.backward()
    assert torch.isfinite(predicted.grad).all() and predicted.grad.abs().sum() == 0


@pytest.mark.parametrize("seed", [0, 7, 31])
def test_cumulative_strain_common_context_displacement_cancels(seed):
    from types import SimpleNamespace
    from utils.pose_ops import so3_exp
    from utils.phys_losses import strain_cum_loss
    generator = torch.Generator().manual_seed(seed)
    first = torch.randn(1, 48, 3, generator=generator) * 0.1
    rotations = so3_exp(torch.tensor([[0., 0., 0.], [0., 0., 0.9],
                                      [0.1, 0.2, 1.1], [-0.2, 0.1, 1.4]]))
    positions = torch.einsum("tij,bnj->btni", rotations, first)
    batch = SimpleNamespace(positions=positions, context=positions[:, :2],
        object_ids=torch.zeros(1, 48, dtype=torch.long), valid_lengths=torch.tensor([48]),
        mat_type=torch.zeros(1, 48, dtype=torch.long))
    anchor, target = positions[:, 1], positions[:, -1]
    predicted = torch.einsum("ij,bnj->bni", so3_exp(torch.tensor([-.17, .13, 1.43])), first)
    context_displacement = anchor - first
    gt, pred = target - anchor, predicted - anchor
    assert strain_cum_loss(batch, gt, gt).item() == 0
    future_only = strain_cum_loss(batch, pred, gt)
    full_displacement = strain_cum_loss(batch, pred + context_displacement, gt + context_displacement)
    assert future_only > 0
    torch.testing.assert_close(future_only, full_displacement, rtol=2e-5, atol=2e-7)


def test_retired_offset_objective_is_rejected_explicitly():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"model": {}, "train": {}, "f0off_delta_loss": True})
    with pytest.raises(ValueError, match="Retired objective settings"):
        trajectory_loss_options(cfg)
    model, _, _, diffusion = setup()
    batch = scalar_batch()
    batch.prediction_target = "frame0_offset"
    with pytest.raises(ValueError, match="delta targets"):
        trajectory_objective(model, batch, diffusion)


@pytest.mark.parametrize("timestep", [50, 500])
def test_sample_pose_loss_retains_its_distinct_time_weight(timestep):
    from tests.test_unet_std_pose_native import make_batch
    model, _, _, diffusion = setup(pose=True)
    batch = make_batch()
    # A controlled x0 error of 0.1 in both spaces. This is an objective oracle,
    # intentionally allowed to read GT (real model non-leakage is tested separately).
    def forward(batch, noisy_future, timesteps=None, **kwargs):
        rigid = batch.body_type[0] == 1
        return (batch.delta_stats.normalize(batch.future_deltas) + 0.1,
                batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, rigid]) + 0.1)
    model.forward = forward
    diffusion.sample_timesteps = lambda B, device, **k: torch.full((B,), timestep, device=device)
    loss, _, _ = trajectory_objective(model, batch, diffusion)
    assert loss.item() == pytest.approx(0.01 + 0.01 / (timestep / 1000) ** 2, rel=2e-5)


@pytest.mark.parametrize("clip", [False, True])
def test_validation_respects_training_target_clipping(clip):
    model, optimizer, accelerator, diffusion = setup()
    optimizer.param_groups[0]["lr"] = 0.0
    batch = scalar_batch(3.0)
    torch.manual_seed(12345)
    loss, _ = train_step(model, batch, diffusion, optimizer, accelerator, clip_deltas=clip)
    values = validate(model, [batch], diffusion, EMA(model), torch.device("cpu"), accelerator,
                      num_batches=1, use_ema=False, noise_context=False, clip_deltas=clip)
    assert values["loss"] == pytest.approx(loss, abs=1e-7)


def test_validation_uses_the_same_per_frame_context_noise_draw():
    model, optimizer, accelerator, diffusion = setup()
    optimizer.param_groups[0]["lr"] = 0.0
    model.cfg.all_relative = True
    batch = create_dummy_batch(B=1, T=6, N=8, context_len=4)
    captured = []
    forward = model.forward

    def recording(*args, **kwargs):
        captured.append(kwargs["context_timesteps"].detach().clone())
        return forward(*args, **kwargs)

    model.forward = recording
    torch.manual_seed(12345)
    train_step(model, batch, diffusion, optimizer, accelerator,
               diffusion_forcing=True, noise_context=True)
    expected = captured[0]
    captured.clear()
    validate(model, [batch], diffusion, EMA(model), torch.device("cpu"), accelerator,
             num_batches=1, use_ema=False, diffusion_forcing=True, noise_context=True)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    assert expected.shape == (1, 3)


def test_validation_preserves_callers_mode_and_rng():
    model, _, accelerator, diffusion = setup()
    model.cfg.all_relative = True
    model.eval()
    batch = scalar_batch()
    before = torch.random.get_rng_state().clone()
    validate(model, [batch], diffusion, EMA(model), torch.device("cpu"), accelerator,
             num_batches=1, use_ema=False, noise_context=False)
    assert not model.training
    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("sampler", [False, True])
def test_cpu_validation_and_large_t_probe_never_touch_cuda_rng(monkeypatch, sampler):
    from types import SimpleNamespace
    from diffusion import TrajectoryFlowMatching
    from tests.test_sampler_oracle import GroundTruthOracle, scene
    fm, batch = TrajectoryFlowMatching(), scene("deformable")
    model = GroundTruthOracle(fm)
    accelerator = SimpleNamespace(reduce=lambda values, reduction: values)
    before = torch.random.get_rng_state().clone()

    def no_cuda(*args, **kwargs):
        pytest.fail("CPU validation may not initialize or reset CUDA generators")

    for name in ("manual_seed", "manual_seed_all", "device", "get_rng_state", "get_rng_state_all",
                 "set_rng_state", "set_rng_state_all"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    function = validate_sampler if sampler else validate
    result = function(model, [batch], fm, None, torch.device("cpu"), accelerator,
                      num_batches=1, use_ema=False)
    assert result["sample_mm" if sampler else "x0_mm_t09"] < .002
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("kind", ["deformable", "mixed", "rigid"])
def test_objective_metadata_unwrap_preserves_nested_forward_and_gradients(architecture, kind):
    from diffusion import TrajectoryFlowMatching
    from tests.test_model_scope import _construct
    from tests.test_sampler_oracle import MetadataWrapper, scene
    model, batch = _construct(architecture, False), scene(kind)
    wrapper = MetadataWrapper(MetadataWrapper(model, "_orig_mod"))
    fm, results = TrajectoryFlowMatching(), []
    for forward in (model, wrapper):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(219)
        loss, metrics, _ = trajectory_objective(forward, batch, fm, noise_context=True)
        loss.backward()
        gradients = {name: None if p.grad is None else p.grad.detach().clone()
                     for name, p in model.named_parameters()}
        results.append((loss.detach(), metrics, gradients))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    assert wrapper.calls == wrapper.module.calls == 1


def test_light_context_noise_leaves_rigid_history_on_its_pose():
    """Rigid rows are template @ pose (ISSUES.md ISSUE-001): light history noise spares them, so
    the UNet places every rigid particle's context frames and anchor where its clean pose puts
    it, while deformable history is still jittered."""
    from diffusion import TrajectoryFlowMatching
    from models.unet_std import UNetWrapper
    from tests.test_unet_std_pose_native import _model_cfg, make_batch
    model, batch = UNetWrapper(_model_cfg()).eval(), make_batch()
    captured = {}
    hook = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: captured.update(pos=kwargs["pos"] * model._geo_r), with_kwargs=True)
    try:
        torch.manual_seed(0)
        trajectory_objective(model, batch, TrajectoryFlowMatching(), light_context_noise_std=0.5)
    finally:
        hook.remove()
    C = batch.context_len
    # Token starts: frames 0..C-2 of the history, then the anchor C-1 where the future begins.
    error = (captured["pos"][:, :C] - batch.positions[:, :C]).norm(dim=-1).transpose(1, 2)  # [B, N, C]
    rigid = batch.body_type.gather(1, batch.particle_body) == 1
    assert error[rigid].max() < 1e-5                   # metres
    assert error[~rigid][:, 1:].mean() > 1e-3          # frame 0 is clean for every row


def test_light_pose_noise_moves_rigid_history_as_whole_bodies():
    """train.light_pose_noise_std: rigid history takes whole-body pose noise integrated back from the
    clean anchor. Every rigid particle keeps its body-frame coordinates under the perturbed pose, jointed
    fingers keep their joint coordinates, and the anchor and the future are untouched."""
    from tests.test_articulated_bodies import gripper
    from utils.pose_native import context_pose_deltas, jitter_rigid_history
    batch, _, _ = gripper(second="prismatic")
    C = batch.context_len
    assert jitter_rigid_history(batch, 0.0) is batch
    torch.manual_seed(0)
    noisy = jitter_rigid_history(batch, 0.5)
    assert torch.equal(noisy.positions[:, C - 1:], batch.positions[:, C - 1:])
    assert torch.equal(noisy.body_poses[:, C - 1:], batch.body_poses[:, C - 1:])

    def body_frame(b):                                   # R^T (x - t) of every particle's own body
        P = b.body_poses[:, :C - 1][:, :, b.particle_body[0]]
        return torch.einsum("bfnji,bfnj->bfni", P[..., :3, :3], b.positions[:, :C - 1] - P[..., :3, 3])
    torch.testing.assert_close(body_frame(noisy), body_frame(batch), atol=1e-5, rtol=0)
    rigid_idx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    clean6, noisy6 = context_pose_deltas(batch, rigid_idx), context_pose_deltas(noisy, rigid_idx)
    torch.testing.assert_close(noisy6[:, :, 1:3], clean6[:, :, 1:3], atol=1e-6, rtol=0)   # finger joints
    assert (noisy6[:, :, [0, 3]] - clean6[:, :, [0, 3]]).abs().amax() > 1e-3             # hand, object


def test_light_pose_noise_reaches_the_model_and_is_off_by_default():
    """The UNet reads the perturbed rigid history, integrates it back to the clean anchor, and leaves
    deformable history alone; a zero or absent setting is bit-identical to the objective without it."""
    from diffusion import TrajectoryFlowMatching
    from models.unet_std import UNetWrapper
    from tests.test_unet_std_pose_native import _model_cfg, make_batch
    model, batch = UNetWrapper(_model_cfg()).eval(), make_batch()
    captured = []
    hook = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: captured.append(kwargs["pos"] * model._geo_r), with_kwargs=True)
    losses = []
    try:
        for options in ({}, {"light_pose_noise_std": 0.0}, {"light_pose_noise_std": 0.5}):
            torch.manual_seed(0)
            losses.append(trajectory_objective(model, batch, TrajectoryFlowMatching(), **options)[0])
    finally:
        hook.remove()
    assert torch.equal(losses[0], losses[1]) and torch.equal(captured[0], captured[1])
    C = batch.context_len
    error = (captured[2][:, :C] - batch.positions[:, :C]).norm(dim=-1).transpose(1, 2)  # [B, N, C]
    rigid = batch.body_type.gather(1, batch.particle_body) == 1
    assert error[rigid][:, C - 1].max() < 1e-5          # the anchor stays on its clean pose
    assert error[rigid][:, :C - 1].mean() > 1e-3        # the history moved
    assert error[~rigid].max() < 1e-5                   # deformable history untouched
    with pytest.raises(ValueError, match="retired noise_context"):
        trajectory_objective(model, batch, TrajectoryFlowMatching(), noise_context=True, light_pose_noise_std=0.5)


def test_future_pose_deltas_reproduces_the_stored_target():
    """The recomputation train.light_pose_noise_anchor corrects the target with must agree with the
    loader's own construction (data/unified_dataset._augment_pose_channels), or a perturbed anchor
    would silently ship a target in a different convention."""
    from tests.test_articulated_bodies import gripper
    from utils.pose_native import future_pose_deltas
    for second in ("prismatic", "revolute"):
        batch, _, _ = gripper(second=second)
        torch.testing.assert_close(future_pose_deltas(batch), batch.pose_deltas_gt, atol=1e-6, rtol=0)


def test_light_pose_noise_anchor_moves_passive_bodies_and_relands_the_targets():
    """train.light_pose_noise_anchor: the passive body's anchor moves with its history, the actuated
    bodies do not move at all, and both targets are rebuilt from the perturbed anchor so they still
    land on the recorded future. That relanding is the whole point of the option: the model has to
    return to the recorded trajectory instead of carrying the perturbation forward."""
    from tests.test_articulated_bodies import gripper
    from utils.pose_native import jitter_rigid_history
    from utils.pose_ops import so3_exp
    batch, _, _ = gripper(second="prismatic")
    C, OBJ = batch.context_len, 3                      # bodies 0-2 are the actuated gripper, 3 the object
    assert batch.pose_action_mask[0].tolist() == [True, True, True, False]
    torch.manual_seed(0)
    n = jitter_rigid_history(batch, 0.5, anchor=True)

    # The recording itself is untouched, and so is every actuated body, at every frame.
    assert torch.equal(n.positions[:, C:], batch.positions[:, C:])
    assert torch.equal(n.body_poses[:, C:], batch.body_poses[:, C:])
    assert torch.equal(n.body_poses[..., :OBJ, :, :], batch.body_poses[..., :OBJ, :, :])
    gripper_rows = (batch.particle_body < OBJ)[:, None, :, None].expand_as(batch.positions)
    assert torch.equal(n.positions[gripper_rows], batch.positions[gripper_rows])

    # The object's anchor moved, and it moved as a rigid body.
    assert not torch.equal(n.body_poses[:, C - 1, OBJ], batch.body_poses[:, C - 1, OBJ])
    obj = (batch.particle_body == OBJ)[0]
    for f in range(C):
        a, b = batch.positions[:, f, obj].double(), n.positions[:, f, obj].double()
        torch.testing.assert_close(torch.cdist(a - a.mean(1, keepdim=True), a - a.mean(1, keepdim=True),
                                               compute_mode="donot_use_mm_for_euclid_dist"),
                                   torch.cdist(b - b.mean(1, keepdim=True), b - b.mean(1, keepdim=True),
                                               compute_mode="donot_use_mm_for_euclid_dist"),
                                   atol=2e-7, rtol=0)      # 0.2 microns: float32 pose composition,
        #  against a signal of millimetres. Distances are taken in float64 on centred points because
        #  cdist's matmul kernel cancels at metre scale and invents about 0.3 mm of deformation.

    # Particle target: only the first transition changes, and only for the moved rows.
    assert torch.equal(n.prediction_x0[:, 1:], batch.prediction_x0[:, 1:])
    assert not torch.equal(n.prediction_x0[:, 0, obj], batch.prediction_x0[:, 0, obj])
    assert torch.equal(n.prediction_x0[:, 0, ~obj], batch.prediction_x0[:, 0, ~obj])
    # and integrating it from the perturbed anchor still lands on the recorded future
    torch.testing.assert_close(n.anchor[:, None] + n.prediction_x0.cumsum(dim=1),
                               batch.positions[:, C:], atol=1e-6, rtol=0)

    # Pose target: the actuated bodies' rows and the given actions are untouched, and integrating the
    # object's rebuilt deltas from its perturbed anchor pose reproduces its recorded poses.
    assert torch.equal(n.pose_deltas_gt[..., :OBJ, :], batch.pose_deltas_gt[..., :OBJ, :])
    assert torch.equal(n.pose_actions, batch.pose_actions)
    assert not torch.equal(n.pose_deltas_gt[:, 0, OBJ], batch.pose_deltas_gt[:, 0, OBJ])
    assert torch.equal(n.pose_deltas_gt[:, 1:, OBJ], batch.pose_deltas_gt[:, 1:, OBJ])
    R, t = n.body_poses[:, C - 1, OBJ, :3, :3].float(), n.body_poses[:, C - 1, OBJ, :3, 3].float()
    for f in range(n.pose_deltas_gt.shape[1]):
        d6 = n.pose_deltas_gt[:, f, OBJ].float()
        R, t = so3_exp(d6[:, 3:]) @ R, t + d6[:, :3]
        torch.testing.assert_close(R, batch.body_poses[:, C + f, OBJ, :3, :3].float(), atol=1e-5, rtol=0)
        torch.testing.assert_close(t, batch.body_poses[:, C + f, OBJ, :3, 3].float(), atol=1e-6, rtol=0)


def test_light_pose_noise_anchor_leaves_the_history_only_mode_alone():
    """anchor=False keeps the committed behaviour: every rigid body's history moves, the anchor and
    the targets do not, and the stored pose target is not even rebuilt."""
    from tests.test_articulated_bodies import gripper
    from utils.pose_native import jitter_rigid_history
    batch, _, _ = gripper(second="prismatic")
    C = batch.context_len
    torch.manual_seed(0)
    n = jitter_rigid_history(batch, 0.5)
    assert torch.equal(n.positions[:, C - 1:], batch.positions[:, C - 1:])
    assert torch.equal(n.body_poses[:, C - 1:], batch.body_poses[:, C - 1:])
    assert n.pose_deltas_gt is batch.pose_deltas_gt
    assert torch.equal(n.prediction_x0, batch.prediction_x0)
    assert not torch.equal(n.positions[:, :C - 1], batch.positions[:, :C - 1])
    assert jitter_rigid_history(batch, 0.0, anchor=True) is batch


def test_light_pose_noise_anchor_refuses_an_actuated_child_of_a_passive_body():
    """A jointed body takes its parent's correction, so an actuated child of a passive parent would be
    moved away from the anchor its given pose_actions were measured from."""
    import dataclasses
    from tests.test_articulated_bodies import gripper
    from utils.pose_native import jitter_rigid_history
    batch, _, _ = gripper(second="prismatic")
    flipped = batch.pose_action_mask.clone()
    flipped[:, 0] = False                              # the hand becomes passive, its fingers stay actuated
    with pytest.raises(ValueError, match="jointed child of a passive"):
        jitter_rigid_history(dataclasses.replace(batch, pose_action_mask=flipped), 0.5, anchor=True)
    bare = dataclasses.replace(batch, pose_action_mask=None)
    with pytest.raises(ValueError, match="pose_action_mask"):
        jitter_rigid_history(bare, 0.5, anchor=True)


def test_light_pose_noise_anchor_on_a_mixed_scene_leaves_deformables_alone():
    """A mixed rigid and deformable batch, C=3, no articulation: only the passive rigid body moves.
    Deformable rows are train.light_context_noise_std's business, and the actuated gripper links keep
    the anchor their pose_actions were measured from."""
    from tests.test_unet_std_pose_native import make_batch
    from utils.pose_native import jitter_rigid_history
    batch = make_batch()
    C = batch.context_len
    rigid = batch.body_type[0] == 1
    passive = (rigid & ~batch.pose_action_mask[0]).nonzero(as_tuple=True)[0]
    assert C == 3 and passive.numel() == 1 and bool((~rigid).any())
    torch.manual_seed(0)
    n = jitter_rigid_history(batch, 0.5, anchor=True)
    k = int(passive[0])
    deform_rows = (batch.body_type.gather(1, batch.particle_body) != 1)[0]
    assert torch.equal(n.positions[:, :, deform_rows], batch.positions[:, :, deform_rows])
    assert torch.equal(n.positions[:, C:], batch.positions[:, C:])
    assert not torch.equal(n.positions[:, C - 1, batch.particle_body[0] == k],
                           batch.positions[:, C - 1, batch.particle_body[0] == k])
    others = rigid.nonzero(as_tuple=True)[0]
    others = others[others != k]
    assert torch.equal(n.body_poses[:, :, others], batch.body_poses[:, :, others])
    assert torch.equal(n.pose_deltas_gt[:, :, others], batch.pose_deltas_gt[:, :, others])
    assert torch.equal(n.prediction_x0[:, 1:], batch.prediction_x0[:, 1:])
    torch.testing.assert_close(n.anchor[:, None] + n.prediction_x0.cumsum(dim=1),
                               batch.positions[:, C:], atol=1e-6, rtol=0)


def test_light_pose_noise_anchor_needs_a_magnitude():
    """The flag alone does nothing, so the objective says so instead of training a silent no-op."""
    from diffusion import TrajectoryFlowMatching
    from models.unet_std import UNetWrapper
    from tests.test_unet_std_pose_native import _model_cfg, make_batch
    model, batch = UNetWrapper(_model_cfg()).eval(), make_batch()
    with pytest.raises(ValueError, match="needs a positive"):
        trajectory_objective(model, batch, TrajectoryFlowMatching(), light_pose_noise_anchor=True)


@pytest.mark.parametrize("sampler", [False, True])
def test_empty_validation_cannot_become_a_best_checkpoint(sampler):
    model, _, accelerator, diffusion = setup()
    function = validate_sampler if sampler else validate
    with pytest.raises(ValueError, match="no batches"):
        function(model, [], diffusion, EMA(model), torch.device("cpu"), accelerator, use_ema=False)


def test_nonfinite_validation_is_an_error():
    model, _, accelerator, diffusion = setup(failure="loss")
    with pytest.raises(RuntimeError, match="Non-finite validation"):
        validate(model, [scalar_batch()], diffusion, EMA(model), torch.device("cpu"),
                 accelerator, use_ema=False, noise_context=False)


@pytest.mark.parametrize("nonfinite", [False, True])
def test_sampler_validation_restores_mode_and_rng_even_on_error(monkeypatch, nonfinite):
    model, _, accelerator, diffusion = setup()
    model.eval()
    batch = scalar_batch()

    def sample(*args, **kwargs):
        out = torch.randn_like(batch.prediction_x0)
        return out * float("nan") if nonfinite else out

    monkeypatch.setattr("inference.run_diffusion_inference_df", sample)
    state = torch.random.get_rng_state().clone()
    if nonfinite:
        with pytest.raises(RuntimeError, match="Non-finite validation"):
            validate_sampler(model, [batch], diffusion, EMA(model), torch.device("cpu"),
                             accelerator, use_ema=False)
    else:
        result = validate_sampler(model, [batch], diffusion, EMA(model), torch.device("cpu"),
                                  accelerator, use_ema=False)
        assert result["sample_mm"] > 0
    assert not model.training
    assert torch.equal(torch.random.get_rng_state(), state)


@pytest.mark.parametrize("clip_pose", [True, False])
def test_pose_targets_follow_their_own_clip_switch(clip_pose):
    """clip_pose_deltas=False trains on the faithful pose tail: a rigid body moving faster
    than the frozen scale gets an un-clipped target, and the objective sees it. With the
    default (True) the historical clipped target is unchanged."""
    from tests.test_unet_std_pose_native import make_batch
    model, _, _, diffusion = setup(pose=True)
    batch = make_batch()
    rigid = batch.body_type[0] == 1
    batch.pose_deltas_gt[:, :, rigid, :3] *= 50.0                   # far beyond the scale: clipped target != faithful target
    seen = {}
    def forward(batch, noisy_future, timesteps=None, **kwargs):
        seen["pose_x0_norm"] = kwargs.get("pose_x0") if "pose_x0" in kwargs else None
        return (batch.delta_stats.normalize(batch.future_deltas),
                batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, rigid], clip=False))
    model.forward = forward
    diffusion.sample_timesteps = lambda B, device, **k: torch.full((B,), 500, device=device)
    loss, _, _ = trajectory_objective(model, batch, diffusion, clip_pose_deltas=clip_pose)
    # the oracle returns the FAITHFUL x0; with clipped targets the loss carries the clip gap, without it it vanishes
    if clip_pose:
        assert loss.item() > 1.0
    else:
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
    values = validate(model, [batch], diffusion, EMA(model), torch.device("cpu"), setup(pose=True)[2],
                      num_batches=1, use_ema=False, noise_context=False, clip_pose_deltas=clip_pose)
    assert values["loss"] == pytest.approx(loss.item(), abs=1e-5)
