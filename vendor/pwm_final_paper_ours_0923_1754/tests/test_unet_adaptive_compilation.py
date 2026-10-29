"""Opt-in CUDA/Inductor parity across adaptive shapes (can take minutes cold).

RUN_UNET_COMPILE_TESTS=1 python -m pytest tests/test_unet_adaptive_compilation.py -s
"""
import os

import pytest
import torch

from models.unet_std import UNetWrapper
from tests.test_unet_adaptive_allocation import wrapper_config
from tests.test_unet_std_pose_native import _noisy_pose_gt, make_batch, open_gates
from utils.compilation import prepare_compilation


@pytest.mark.skipif(os.environ.get("RUN_UNET_COMPILE_TESTS") != "1", reason="opt-in Inductor compile check")
def test_regional_compilation_matches_eager_across_adaptive_shapes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(727)
    torch.set_num_threads(2)
    # Match deterministic inference/trunk arithmetic. Eager and compiled dropout
    # kernels consume RNG differently and cannot define a same-mask parity test.
    cfg = wrapper_config(dropout=0, unet_ckpt_from_level=0, unet_ckpt_ingraph=True, unet_ckpt_seams=True)
    eager = UNetWrapper(cfg)
    open_gates(eager)
    compiled = UNetWrapper(cfg)
    compiled.load_state_dict(eager.state_dict(), strict=True)
    eager.cuda()
    compiled.cuda()
    keys = list(compiled.state_dict())
    prepare_compilation(compiled, dict(compile=True, compile_regional=True, compile_dynamic=False))
    # Binding cap, smaller non-binding scene, then return to the first shape.
    for deform_n, rigid_n in ((128, 96), (24, 16), (128, 96)):
        batch = make_batch(deform_n=deform_n, rigid=((1, rigid_n), (2, rigid_n))).to("cuda")
        noisy, pose = torch.randn_like(batch.future), _noisy_pose_gt(batch) + .1
        times = torch.full(noisy.shape[:2], 500, device="cuda", dtype=torch.long)
        eager.zero_grad(set_to_none=True)
        compiled.zero_grad(set_to_none=True)
        outputs = []
        for model in (eager, compiled):
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = model(batch, noisy, times, noisy_pose=pose)
                loss = sum(v.float().square().mean() for v in values if v is not None)
            loss.backward()
            outputs.append(values)
        torch.testing.assert_close(outputs[1], outputs[0], atol=.003, rtol=.03)
        for (name, param), (other_name, other) in zip(eager.named_parameters(), compiled.named_parameters()):
            assert name == other_name and (param.grad is None) == (other.grad is None)
            if param.grad is not None:
                assert torch.isfinite(other.grad).all(), name
                torch.testing.assert_close(other.grad, param.grad, atol=.003, rtol=.05, msg=name)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            eager.eval()
            compiled.eval()
            torch.testing.assert_close(compiled(batch, noisy, times, noisy_pose=pose),
                                       eager(batch, noisy, times, noisy_pose=pose), atol=.003, rtol=.03)
        assert list(compiled.state_dict()) == keys
        print(f"compiled train/backward/eval parity: deformable={deform_n}, rigid={rigid_n} x2", flush=True)
