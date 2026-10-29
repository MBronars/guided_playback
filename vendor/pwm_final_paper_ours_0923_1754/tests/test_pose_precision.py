"""Rigid geometry keeps FP32 arithmetic inside a mixed-precision model forward."""

import pytest
import torch

from tests.test_articulated_bodies import gripper
from tests.test_model_scope import _construct
from tests.test_unet_std_pose_native import make_batch
from utils.pose_native import context_pose_deltas, materialize_pose_native, pose_native_inputs


@pytest.fixture(params=[("cpu", torch.bfloat16), ("cuda", torch.bfloat16),
                       ("cuda", torch.float16)], ids=["cpu-bf16", "cuda-bf16", "cuda-fp16"])
def amp(request):
    device, dtype = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")
    return torch.device(device), dtype


@pytest.fixture(params=["mixed", "rigid", "prismatic", "revolute"])
def scene(request, amp):
    if request.param in ("prismatic", "revolute"):
        batch, _, _ = gripper(request.param)
    else:
        batch = make_batch(**({"deform_n": 0} if request.param == "rigid" else {}))
    return batch.to(amp[0])


def _inputs(batch):
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    pose = batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, ridx], clip=False)
    # Nonzero deformable inputs also verify that rewriting rigid rows preserves them.
    particle = batch.delta_stats.normalize(batch.future_deltas, clip=False) + 0.17
    return particle, pose, ridx


@pytest.mark.parametrize("noisy", [False, True])
def test_amp_geometry_matches_fp32(scene, amp, noisy):
    batch, (device, dtype) = scene, amp
    particle, pose, ridx = _inputs(batch)
    if noisy:
        pose = pose + 0.15
    reference = (pose_native_inputs(batch, particle, pose), context_pose_deltas(batch, ridx),
                 materialize_pose_native(batch, particle, pose))
    with torch.autocast(device.type, dtype=dtype):
        actual = (pose_native_inputs(batch, particle, pose), context_pose_deltas(batch, ridx),
                  materialize_pose_native(batch, particle, pose))
        assert torch.is_autocast_enabled(device.type)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    patched, pack = actual[0]
    assert patched.dtype == particle.dtype
    assert pack["centers"].dtype == actual[1].dtype == actual[2][1].dtype == torch.float32
    rigid = batch.body_type.gather(1, batch.particle_body) == 1
    torch.testing.assert_close(patched[:, :, ~rigid[0]], particle[:, :, ~rigid[0]], rtol=0, atol=0)
    rebuilt = batch.anchor[:, None] + batch.delta_stats.denormalize(patched).cumsum(1)
    torch.testing.assert_close(rebuilt[:, :, rigid[0]], actual[2][0][:, :, rigid[0]],
                               rtol=0, atol=2e-6)


@pytest.mark.parametrize("enabled", [False, True])
def test_low_precision_pose_is_promoted_before_denormalizing(amp, enabled):
    device, dtype = amp
    batch = make_batch().to(device)
    particle, pose, _ = _inputs(batch)
    pose = (pose + 0.15).to(dtype)
    expected_input, expected_pack = pose_native_inputs(batch, particle, pose.float())
    expected_output = materialize_pose_native(batch, particle, pose.float())
    with torch.autocast(device.type, dtype=dtype, enabled=enabled):
        actual_input, actual_pack = pose_native_inputs(batch, particle, pose)
        actual_output = materialize_pose_native(batch, particle, pose)
    torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)
    torch.testing.assert_close(actual_pack["centers"], expected_pack["centers"], rtol=0, atol=0)
    assert actual_pack["noisy_pose"] is pose  # The learned pose-token input keeps its dtype.
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


def test_amp_geometry_preserves_input_gradients(scene, amp):
    device, dtype = amp
    particle, pose, _ = _inputs(scene)
    results = []
    for enabled in (False, True):
        p = particle.detach().clone().requires_grad_()
        q = (pose + 0.15).detach().clone().requires_grad_()
        with torch.autocast(device.type, dtype=dtype, enabled=enabled):
            patched, _ = pose_native_inputs(scene, p, q)
            positions, rotations, centers, _ = materialize_pose_native(scene, p, q)
            # Unequal coordinate weights exercise rotation gradients as well as translation.
            weights = positions.new_tensor([0.7, -0.4, 1.3])
            loss = (patched * weights).square().mean() + (positions * weights).square().mean()
            loss = loss + (rotations * weights).sum() * 0.01 + centers.square().mean()
        grads = torch.autograd.grad(loss, (p, q))
        assert all(torch.isfinite(g).all() for g in grads)
        assert grads[1].abs().sum() > 0
        results.append(grads)
    torch.testing.assert_close(results[0], results[1], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
def test_backbone_keeps_amp_and_gradients_after_geometry(amp, architecture):
    device, dtype = amp
    batch = make_batch().to(device)
    model = _construct(architecture, reference=False).to(device).train()
    particle, pose, _ = _inputs(batch)
    pose = pose.detach().requires_grad_()
    observed_dtypes = []
    handles = [module.register_forward_hook(lambda _m, _args, out: observed_dtypes.append(out.dtype))
               for module in model.modules() if isinstance(module, torch.nn.Linear)]
    try:
        with torch.autocast(device.type, dtype=dtype):
            output = model(batch, particle, torch.full(pose.shape[:2], 500, device=device), noisy_pose=pose)
            loss = sum(value.float().square().mean() for value in output if value is not None)
            assert torch.is_autocast_enabled(device.type)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert observed_dtypes and set(observed_dtypes) == {dtype}
    assert pose.grad is not None and torch.isfinite(pose.grad).all() and pose.grad.abs().sum() > 0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
