"""Partial final windows integrate from the committed state, including native FK."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import inference
from diffusion import TrajectoryFlowMatching
from tests.test_articulated_bodies import gripper
from tests.test_model_scope import _construct
from tests.test_rollout_batch import sequence
from utils.pose_native import materialize_pose_native


def long_batch(kind, horizon):
    b = gripper(kind)[0].slice(1) if kind else sequence().get_window(0, 4, 6).slice(1)
    C, N = b.context_len, b.num_particles
    fields = dict(positions=torch.cat([b.context, b.anchor[:, None].expand(-1, horizon, -1, -1)], 1),
                  actions=torch.zeros(1, horizon, N, 3))
    if kind:
        K = b.body_type.shape[1]
        fields.update(body_poses=torch.cat([b.body_poses[:, :C], b.body_poses[:, C-1:C].expand(-1, horizon, -1, -1, -1)], 1),
                      pose_deltas_gt=torch.zeros(1, horizon, K, 6), pose_actions=torch.zeros(1, horizon, K, 6))
    return replace(b, **fields)


@pytest.mark.parametrize("window", [1, 16])
def test_deep_joint_rollout_does_not_amplify_rotation_roundoff(monkeypatch, window):
    from utils.pose_native import JOINT_FREE, JOINT_REVOLUTE
    batch = long_batch("revolute", 160)
    batch.body_parent = torch.tensor([[-1, 0, 1, 2]])
    batch.joint_type = torch.tensor([[JOINT_FREE, JOINT_REVOLUTE, JOINT_REVOLUTE, JOINT_REVOLUTE]])
    batch.joint_axis = torch.tensor([[[0., 0., 1.]] * 4])
    batch.joint_pivot = torch.zeros(1, 4, 3)
    # A few ulps of imperfect orthogonality are normal after float32 pose
    # composition. A stationary hand must not turn that into finger motion.
    batch.body_poses = batch.body_poses.clone()
    batch.body_poses[..., :3, :3] *= 1.000002

    def stationary(model, win, diffusion, **kw):
        return None, win.pose_delta_stats.normalize(torch.zeros_like(win.pose_deltas_gt), clip=False)

    monkeypatch.setattr(inference, "run_diffusion_inference_df", stationary)
    model = SimpleNamespace(cfg=SimpleNamespace(pose_native=True))
    out = inference.windowed_rollout(model, None, batch, horizon=160, future_len=window)
    err = (out["positions"][:, batch.context_len:] - batch.future).norm(dim=-1).max()
    assert float(err) < 2e-5, f"Stationary articulated body drifted {float(err)*1000} mm"
    R = out["R"][:, batch.context_len:]
    torch.testing.assert_close(R.transpose(-1, -2) @ R,
                               torch.eye(3).expand_as(R), rtol=0, atol=1e-6)


def test_rollout_stabilization_keeps_unjointed_materialization_bitwise():
    batch = gripper(articulated=False)[0]
    pose = batch.pose_delta_stats.normalize(batch.pose_deltas_gt, clip=False)
    before = materialize_pose_native(batch, None, pose)
    after = materialize_pose_native(batch, None, pose, stabilize_joints=True)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize('kind', [None, 'prismatic', 'revolute'])
@pytest.mark.parametrize('horizon', [1, 16, 17, 32, 36, 45])
def test_rollout_equals_integration_of_all_sampled_deltas(monkeypatch, kind, horizon):
    batch = long_batch(kind, horizon)
    C = batch.context_len
    calls, deltas = [], []

    def sample(model, win, diffusion, **kw):
        # The last old-style overlap window sees a different velocity/rotation.
        # A GT oracle or identical predictions across windows would miss the bug.
        scale = len(calls)+1
        calls.append(win.num_future_frames)
        if kind:
            assert not win.pose_action_mask.any()
            raw = torch.zeros_like(win.pose_deltas_gt)
            raw[:, :, [0, 3], 0] = .001*scale
            raw[:, :, [0, 3], 5] = .02*scale
            raw[:, :, 1, 0] = .0002*scale
            raw[:, :, 2, 0 if kind == 'prismatic' else 3] = .0003*scale
            deltas.append(raw)
            return None, win.pose_delta_stats.normalize(raw, clip=False)
        raw = torch.zeros_like(win.actions)
        raw[..., 0] = .001*scale
        deltas.append(raw)
        return win.delta_stats.normalize(raw, clip=False)

    monkeypatch.setattr(inference, 'run_diffusion_inference_df', sample)
    out = inference.windowed_rollout(SimpleNamespace(cfg=SimpleNamespace(pose_native=bool(kind))),
        None, batch, horizon=horizon, future_len=16, hide_given_actions=bool(kind))
    raw = torch.cat(deltas, dim=1)
    assert raw.shape[1] == horizon
    assert calls == [16]*(horizon//16) + ([horizon % 16] if horizon % 16 else [])
    torch.testing.assert_close(out['positions'][:, :C], batch.context, rtol=0, atol=0)
    if kind:
        positions, R, T, _ = materialize_pose_native(batch, None, batch.pose_delta_stats.normalize(raw, clip=False))
        torch.testing.assert_close(out['R'][:, C:], R, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(out['T'][:, C:], T, rtol=1e-5, atol=2e-6)
    else:
        positions = batch.predictions_to_positions(batch.delta_stats.denormalize(batch.delta_stats.normalize(raw, clip=False)))
    torch.testing.assert_close(out['positions'][:, C:], positions, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize('architecture', ['transformer', 'unet'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('tail', [1, 4, 13])
def test_backbones_sample_partial_native_windows(architecture, device, tail):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    # Use the model-scope layout to exercise the production model factories and
    # real attention at the shortened time dimension, without trained weights.
    from tests.test_unet_std_pose_native import make_batch
    b = make_batch().slice(1)
    C, N, K = b.context_len, b.num_particles, b.body_type.shape[1]
    b = replace(b, positions=torch.cat([b.context, b.anchor[:, None].expand(-1, tail, -1, -1)], 1),
                actions=torch.zeros(1, tail, N, 3),
                body_poses=torch.cat([b.body_poses[:, :C], b.body_poses[:, C-1:C].expand(-1, tail, -1, -1, -1)], 1),
                pose_deltas_gt=torch.zeros(1, tail, K, 6), pose_actions=torch.zeros(1, tail, K, 6)).to(device)
    model = _construct(architecture, reference=False).to(device).eval()
    result = inference.run_diffusion_inference_df(model, b, TrajectoryFlowMatching(), num_inference_steps=2)
    positions = materialize_pose_native(b, *result)[0]
    assert positions.shape == b.future.shape and torch.isfinite(positions).all()
