"""Articulated rigid bodies (utils/pose_native.py JOINT_*): forward kinematics, joint channels, the
loss and sampler masks, and the loaders' finger coupling, on a synthetic gripper with known joints."""
import pytest
import torch

from data.batch import TrajectoryBatch
from data.dataset import NormalizationStats
from data.unified_dataset import _articulate_fingers, _augment_pose_channels, _rigid_sidecar_from_particles
from inference import run_diffusion_inference_df
from utils.pose_native import (JOINT_FIXED, JOINT_FREE, JOINT_PRISMATIC, JOINT_REVOLUTE, compose_joints,
                               joint_layout, layout_from, materialize_pose_native, pose_channel_mask,
                               pose_native_inputs)
from utils.pose_ops import so3_exp, so3_log

B, C, F = 2, 3, 6
T = C + F
E_X, E_Y = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])
PIVOT = torch.tensor([-0.02, 0.0, 0.0])


def gripper(second="revolute", articulated=True, seed=0):
    """A moving hand (body 0), two fingers (1, 2) and a free object (3). Finger 1 slides along the
    hand's x axis; finger 2 slides along -x, or swings about the hand's y axis through PIVOT.
    Returns the batch and the true joint coordinates [B, T] of both fingers."""
    g = torch.Generator().manual_seed(seed)
    tt = torch.arange(T, dtype=torch.float32)

    def box(n, lo, hi):
        return torch.rand(B, n, 3, generator=g) * (torch.tensor(hi) - torch.tensor(lo)) + torch.tensor(lo)

    hand = box(40, [-0.06, -0.015, 0.0], [0.06, 0.015, 0.03])
    f1 = box(20, [0.02, -0.01, -0.05], [0.034, 0.01, 0.0])
    f2 = box(20, [-0.034, -0.01, -0.05], [-0.02, 0.01, 0.0])
    obj = box(30, [-0.02, -0.02, -0.09], [0.02, 0.02, -0.06])
    q1 = 0.003 * tt[None] * torch.linspace(0.5, 1.5, B)[:, None]            # slide of finger 1 (m)
    q2 = 0.06 * tt[None] * torch.linspace(1.0, 2.0, B)[:, None]             # finger 2: angle (rad) or slide / 20
    R_h = so3_exp(0.05 * torch.randn(B, 1, 3, generator=g) * tt[None, :, None])   # R_h(0) = I
    t_h = 0.1 * torch.randn(B, 1, 3, generator=g) + 0.01 * tt[None, :, None] * torch.randn(B, 1, 3, generator=g)
    loc1 = f1[:, None] + q1[..., None, None] * E_X
    if second == "revolute":
        loc2 = torch.einsum("btij,bnj->btni", so3_exp(q2[..., None] * E_Y), f2 - PIVOT) + PIVOT
    else:
        q2 = q2 / 20
        loc2 = f2[:, None] - q2[..., None, None] * E_X

    def world(loc):
        return torch.einsum("btij,btnj->btni", R_h, loc) + t_h[:, :, None]

    X = torch.cat([world(hand[:, None].expand(-1, T, -1, -1)), world(loc1), world(loc2),
                   obj[:, None] + 0.002 * tt[None, :, None, None]], dim=2)
    N = X.shape[2]
    pb = torch.tensor([0] * 40 + [1] * 20 + [2] * 20 + [3] * 30)[None].expand(B, -1).clone()
    sc = _rigid_sidecar_from_particles(X, pb, 4)
    sc["particle_body"] = pb.clone()
    if articulated:
        second_axis = -E_X if second == "prismatic" else E_X
        sc = _articulate_fingers(sc, [(0, 1), (0, 2)], torch.stack([E_X, second_axis])[None].expand(B, -1, -1))
        if second == "revolute":                     # the loaders only make sliders; set the hinge here
            sc["joint_type"][:, 2] = JOINT_REVOLUTE
            sc["joint_axis"][:, 2] = E_Y
            sc["joint_pivot"][:, 2] = PIVOT - hand.mean(1)                    # in the hand's body frame
            sc["body_poses"] = _rigid_sidecar_from_particles(X, pb, 4)["body_poses"].where(
                torch.arange(4)[None, None, :, None, None] == 2, sc["body_poses"])
    actuated = pb <= 2
    pose_stats = NormalizationStats(min_val=torch.tensor([-0.02] * 3 + [-0.3] * 3),
                                    max_val=torch.tensor([0.02] * 3 + [0.3] * 3))
    sc = _augment_pose_channels(sc, actuated, C, pose_stats)
    batch = TrajectoryBatch(
        positions=X, actions=(X[:, 1:] - X[:, :-1])[:, C - 1:] * actuated[:, None, :, None],
        action_mask=actuated, object_ids=(pb <= 2).long(), valid_lengths=torch.full((B,), N),
        E=torch.full((B, N), 1e9), nu=torch.full((B, N), 0.3), rho=torch.full((B, N), 750.0),
        mat_type=torch.where(actuated, 4, 3), context_len=C,
        delta_stats=NormalizationStats(min_val=torch.full((3,), -0.03), max_val=torch.full((3,), 0.03)),
        position_stats=NormalizationStats(min_val=torch.full((3,), -0.8), max_val=torch.full((3,), 0.8)),
        prediction_target="delta", **sc)
    return batch, q1, q2


def _x0(batch):
    return batch.pose_delta_stats.normalize(batch.pose_deltas_gt, clip=False)


def _noisy(batch, seed=3):
    z = torch.randn(_x0(batch).shape, generator=torch.Generator().manual_seed(seed)) * 0.3
    return _x0(batch) + z * pose_channel_mask(batch, torch.arange(4))


def _relative(R, t, parent, child):
    RpT = R[..., parent, :, :].transpose(-1, -2)
    return RpT @ R[..., child, :, :], (RpT @ (t[..., child, :] - t[..., parent, :])[..., None])[..., 0]


@pytest.mark.parametrize("second", ["prismatic", "revolute"])
def test_gt_pose_channels_rebuild_every_rigid_point(second):
    batch, _, _ = gripper(second)
    pred, *_ = materialize_pose_native(batch, None, _x0(batch))
    assert float((pred - batch.future).norm(dim=-1).max()) < 2e-6


def test_joint_channels_carry_the_joint_coordinate():
    batch, q1, q2 = gripper("revolute")
    d = batch.pose_deltas_gt                                             # [B, F, 4, 6] over C-1 .. T-1
    torch.testing.assert_close(d[:, :, 1, 0], q1[:, C:] - q1[:, C - 1:-1], atol=1e-6, rtol=0)
    torch.testing.assert_close(d[:, :, 2, 3], q2[:, C:] - q2[:, C - 1:-1], atol=1e-5, rtol=0)
    mask = pose_channel_mask(batch, torch.arange(4))
    assert mask.tolist() == [[1.0] * 6, [1.0, 0, 0, 0, 0, 0], [0, 0, 0, 1.0, 0, 0], [1.0] * 6]
    assert float((d * (1 - mask)).abs().max()) == 0.0
    assert float((batch.pose_actions * (1 - mask)).abs().max()) == 0.0


def test_noisy_pose_values_never_move_a_child_off_its_joint():
    batch, _, _ = gripper("revolute")
    _, R, t, _ = materialize_pose_native(batch, None, _noisy(batch))
    Ra, ta = batch.body_poses[:, C - 1, :, :3, :3], batch.body_poses[:, C - 1, :, :3, 3]
    # slider: fixed relative rotation, offset moves only along its axis
    R1, t1 = _relative(R, t, 0, 1)
    R1a, t1a = _relative(Ra, ta, 0, 1)
    assert float(so3_log(R1 @ R1a[:, None].transpose(-1, -2)).norm(dim=-1).max()) < 1e-5
    assert float(torch.linalg.cross(t1 - t1a[:, None], E_X.expand_as(t1)).norm(dim=-1).max()) < 1e-6
    # hinge: the pivot stays put in the hand frame and the turn is about the hinge axis only
    R2, t2 = _relative(R, t, 0, 2)
    R2a, t2a = _relative(Ra, ta, 0, 2)
    pivot = batch.joint_pivot[:, 2]
    p_child = (R2a.transpose(-1, -2) @ (pivot - t2a)[..., None])[..., 0]
    assert float(((R2 @ p_child[:, None, :, None])[..., 0] + t2 - pivot[:, None]).norm(dim=-1).max()) < 1e-6
    turn = so3_log(R2 @ R2a[:, None].transpose(-1, -2))
    assert float(torch.linalg.cross(turn, E_Y.expand_as(turn)).norm(dim=-1).max()) < 1e-5
    assert float(turn.norm(dim=-1).max()) > 0.01                        # it does turn


def test_input_rewrite_and_materialization_share_the_kinematics():
    batch, _, _ = gripper("revolute")
    noisy = _noisy(batch)
    patched, _ = pose_native_inputs(batch, torch.zeros_like(batch.future), noisy)
    pred, *_ = materialize_pose_native(batch, None, noisy)
    rebuilt = batch.positions[:, C - 1:C] + batch.delta_stats.denormalize(patched).cumsum(dim=1)
    assert float((rebuilt - pred).norm(dim=-1).max()) < 1e-5


def test_layout_rules():
    parent, jt = torch.tensor([-1, 0, 1]), torch.tensor([JOINT_FREE, JOINT_FIXED, JOINT_PRISMATIC])
    axis = torch.zeros(1, 3, 3)
    lay = layout_from(parent, jt, axis, axis, torch.arange(3))
    assert lay[0].tolist() == [-1, 0, 1]
    with pytest.raises(ValueError, match="free body"):
        layout_from(torch.tensor([-1, 0]), torch.tensor([JOINT_FREE, JOINT_FREE]), axis[:, :2], axis[:, :2],
                    torch.arange(2))
    with pytest.raises(ValueError, match="not among"):
        layout_from(parent, jt, axis, axis, torch.tensor([0, 2]))
    cyclic = (torch.tensor([1, 0]), torch.tensor([JOINT_FIXED, JOINT_FIXED]), axis[:, :2], axis[:, :2])
    eye = torch.eye(3).expand(1, 1, 2, 3, 3)
    with pytest.raises(ValueError, match="cycle"):
        compose_joints(eye, torch.zeros(1, 1, 2, 3), eye[:, 0], torch.zeros(1, 2, 3), torch.zeros(1, 1, 2, 6), cyclic)


def test_unarticulated_batches_keep_six_free_channels():
    batch, _, _ = gripper("prismatic", articulated=False)
    assert joint_layout(batch, torch.arange(4)) is None
    assert pose_channel_mask(batch, torch.arange(4)) is None


def test_objective_noises_and_scores_only_the_channels_a_body_uses():
    from tests.test_training_safety import setup
    from utils.trajectory_objective import trajectory_objective
    batch, _, _ = gripper("revolute")
    mask = pose_channel_mask(batch, torch.arange(4))
    model, _, _, diffusion = setup(pose=True)
    seen = []

    def forward(garbage):
        def f(batch, noisy_future, timesteps=None, **kwargs):
            seen.append(kwargs["noisy_pose"])
            return None, _x0(batch) + garbage * (1 - mask)
        return f
    losses = []
    for garbage in (0.0, 5.0):
        model.forward = forward(garbage)
        torch.manual_seed(0)
        losses.append(float(trajectory_objective(model, batch, diffusion)[0]))
    assert losses[0] == losses[1]
    assert all(float((n * (1 - mask)).abs().max()) == 0.0 for n in seen)


def test_sampler_holds_the_unused_channels_at_zero():
    from tests.test_training_safety import setup
    from diffusion.flow_matching import TrajectoryFlowMatching
    batch, _, _ = gripper("revolute")
    mask = pose_channel_mask(batch, torch.arange(4))
    model, _, _, _ = setup(pose=True)
    model.forward = lambda batch, x_t, t, **kw: (None, torch.randn_like(kw["noisy_pose"]))
    _, x_pose = run_diffusion_inference_df(model, batch, TrajectoryFlowMatching(), num_inference_steps=4)
    assert float((x_pose * (1 - mask)).abs().max()) == 0.0
    assert float((x_pose * mask).abs().max()) > 0.0


@pytest.mark.integration
@pytest.mark.parametrize("domain", ["pickup", "cloth"])
def test_corpus_grippers_round_trip_through_their_joints(domain):
    from data.unified_dataset import build_domain_dataset
    ds = build_domain_dataset(domain, split="val", training=False, context_len=4, normalization="xy_p99_per_domain")
    ds.emit_sidecar = True
    ds.articulated = True
    b = next(iter(ds))
    ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
    assert set(b.joint_type[0, ridx].tolist()) == {JOINT_FREE, JOINT_PRISMATIC}
    pred, *_ = materialize_pose_native(b, None, b.pose_delta_stats.normalize(b.pose_deltas_gt[:, :, ridx], clip=False))
    rigid = (b.body_type.gather(1, b.particle_body) == 1) & b.get_valid_mask()
    assert float((pred - b.future).norm(dim=-1)[:, :, rigid[0]].max()) < 5e-5


def _tiny_model(architecture, articulated):
    from models.transformer import ModelConfig, TrajectoryTransformer
    from models.unet_std import UNetWrapper
    from tests.test_unet_std_pose_native import _model_cfg
    if architecture == "transformer":
        return TrajectoryTransformer(ModelConfig(
            dim=32, depth=2, heads=4, num_latents=24, k_neighbors=8, dropout=0, context_len=C,
            all_relative=True, num_mat_types=6, pose_native=True, articulated_bodies=articulated,
            use_stratified_encoding=True, use_stratified_decoding=True, bypass_compression=False,
            per_frame_knn=True, concat_anchor_z=True, embed_actuatable=True, causal_temporal=True))
    return UNetWrapper(_model_cfg(articulated_bodies=articulated))


@pytest.mark.parametrize("architecture", ["unet", "transformer"])
def test_plain_models_refuse_articulated_batches(architecture):
    """Without model.articulated_bodies a model must not read joint channels as world deltas."""
    model = _tiny_model(architecture, articulated=False)
    batch, _, _ = gripper("revolute")
    with pytest.raises(ValueError, match="articulated"):
        model(batch, torch.randn(B, F, batch.num_particles, 3), torch.randint(1, 1000, (B, F)),
              noisy_pose=torch.zeros(B, F, 4, 6))


@pytest.mark.parametrize("architecture", ["unet", "transformer"])
def test_articulated_models_train_and_sample(architecture):
    """Both backbones with model.articulated_bodies: the joint embedding reaches the pose output,
    gradients reach it, and the production sampler leaves the unused channels at zero."""
    from diffusion.flow_matching import TrajectoryFlowMatching
    torch.manual_seed(0)
    model = _tiny_model(architecture, articulated=True)
    with torch.no_grad():                                   # open the zero-initialized gates and heads
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.025)
    batch, _, _ = gripper("revolute")
    mask = pose_channel_mask(batch, torch.arange(4))
    noisy = torch.randn(B, F, 4, 6) * mask
    _, pose_pred = model(batch, torch.randn(B, F, batch.num_particles, 3), torch.randint(1, 1000, (B, F)),
                         noisy_pose=noisy)
    assert pose_pred.shape == (B, F, 4, 6)
    (pose_pred * mask).square().mean().backward()
    joint = [(n, p.grad) for n, p in model.named_parameters() if "pose_joint" in n]
    assert joint and all(g is not None and float(g.abs().sum()) > 0 for _, g in joint), [n for n, _ in joint]
    _, x_pose = run_diffusion_inference_df(model.eval(), batch, TrajectoryFlowMatching(), num_inference_steps=3)
    assert float((x_pose * (1 - mask)).abs().max()) == 0.0
