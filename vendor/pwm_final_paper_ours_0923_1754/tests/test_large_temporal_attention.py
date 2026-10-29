import pytest
import torch
import torch.nn.functional as F

from models.components.attention import batched_sdpa


@pytest.mark.parametrize("mask_kind", ["none", "shared", "per_batch", "causal"])
def test_chunked_sdpa_matches_outputs_and_gradients(monkeypatch, mask_kind):
    torch.manual_seed(1916)
    q, k, v = [torch.randn(7, 2, 4, 8, requires_grad=True) for _ in range(3)]
    mask = None
    if mask_kind == "shared":
        mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    elif mask_kind == "per_batch":
        mask = torch.rand(7, 1, 4, 4) > .3
        mask[..., 0] = True
    causal = mask_kind == "causal"
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
    rg = torch.autograd.grad(ref.square().sum(), (q, k, v), retain_graph=True)
    monkeypatch.setattr("models.components.attention._SDPA_BATCH_LIMIT", 3)
    actual = batched_sdpa(q, k, v, attn_mask=mask, is_causal=causal)
    ag = torch.autograd.grad(actual.square().sum(), (q, k, v))
    torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-6)
    for a, b in zip(ag, rg):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-6)


def test_ordinary_sdpa_remains_bitwise_identical():
    torch.manual_seed(0)
    q, k, v = [torch.randn(8, 2, 4, 16) for _ in range(3)]
    assert torch.equal(batched_sdpa(q, k, v, is_causal=True),
                       F.scaled_dot_product_attention(q, k, v, is_causal=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA launch-grid regression")
def test_native_folding_batch_beyond_cuda_grid_limit():
    q, k, v = [torch.randn(69478, 1, 4, 8, device="cuda", dtype=torch.bfloat16,
                          requires_grad=True) for _ in range(3)]
    out = batched_sdpa(q, k, v, is_causal=True)
    assert torch.isfinite(out).all()
    out.float().square().mean().backward()
    assert all(torch.isfinite(t.grad).all() for t in (q, k, v))
    ref = F.scaled_dot_product_attention(q[:32], k[:32], v[:32], is_causal=True)
    torch.testing.assert_close(out[:32], ref)
