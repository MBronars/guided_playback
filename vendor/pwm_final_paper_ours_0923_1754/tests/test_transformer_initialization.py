"""AdaLN-Zero identity startup, gate learning and saved-weight compatibility."""
from dataclasses import asdict

import pytest
import torch

from diffusion import TrajectoryFlowMatching
from inference import load_model_and_diffusion
from models.components.attention import TransformerBlock
from models.components.embeddings import RotaryPositionEncoding
import models.transformer as transformer
from tests.test_model_scope import _options, _reference_module
from tests.test_unet_std_pose_native import _noisy_pose_gt, make_batch
from utils.trajectory_objective import trajectory_objective


@pytest.fixture(params=[("cpu", None), ("cpu", torch.bfloat16), ("cuda", None),
                       ("cuda", torch.bfloat16)], ids=["cpu-fp32", "cpu-bf16", "cuda-fp32", "cuda-bf16"])
def precision(request):
    device, dtype = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 is unavailable")
    return torch.device(device), dtype


def _autocast(precision):
    device, dtype = precision
    return torch.autocast(device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None)


def _modulations(block):
    return [getattr(block, f"norm{i}").linear for i in range(1, 5)]


def _block_inputs(device, *, per_token=True, conditioned=True):
    torch.manual_seed(17)
    B, F, L, C, d = 2, 4, 5, 3, 32
    x = torch.randn(B, F, L, d, device=device)
    context = torch.randn(B, C, L, d, device=device)
    actions = torch.randn_like(x) if conditioned else None
    t_emb = torch.randn((B, F, L, d) if per_token else (B, F, d), device=device)
    ctx_t_emb = torch.randn((B, C, L, d) if per_token else (B, C, d), device=device)
    rope = RotaryPositionEncoding(dim=d // 4).to(device)
    temporal_q = rope(torch.arange(C, C + F, device=device)[None].expand(B * L, -1).float())
    temporal_kv = rope(torch.arange(C + (F if conditioned else 0), device=device)[None].expand(B * L, -1).float())
    spatial = rope(torch.randn(B * F, L, device=device))
    mask = torch.zeros(F, F, device=device).masked_fill(
        torch.ones(F, F, device=device, dtype=torch.bool).triu(1), float("-inf"))
    return dict(x=x, context=context, actions=actions, t_emb=t_emb,
                ctx_t_emb=ctx_t_emb if conditioned else None,
                rotary_emb=dict(temporal_q=temporal_q, temporal_kv=temporal_kv, spatial=spatial),
                temporal_attn_mask=mask)


@pytest.mark.parametrize("per_token,conditioned", [(False, False), (False, True), (True, True)])
def test_fresh_block_is_exact_identity(precision, per_token, conditioned):
    torch.manual_seed(29)
    block = TransformerBlock(dim=32, heads=4, dropout=0.1).to(precision[0]).train()
    for projection in _modulations(block):
        assert torch.count_nonzero(projection.weight) == 0
        assert torch.count_nonzero(projection.bias) == 0
    inputs = _block_inputs(precision[0], per_token=per_token, conditioned=conditioned)
    with _autocast(precision):
        output = block(**inputs)
    torch.testing.assert_close(output, inputs["x"], rtol=0, atol=0)


def test_gates_learn_then_attention_and_feedforward_learn(precision):
    torch.manual_seed(29)
    block = TransformerBlock(dim=32, heads=4, dropout=0).to(precision[0])
    inputs = _block_inputs(precision[0])
    target = torch.randn_like(inputs["x"])
    branches = (block.cross_attn, block.temporal_attn, block.spatial_attn, block.ffn)
    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with _autocast(precision):
            output = block(**inputs)
            loss = (output.float() - target).square().mean()
        loss.backward()
        for projection in _modulations(block):
            for grad in (projection.weight.grad, projection.bias.grad):
                assert grad is not None and torch.isfinite(grad).all()
                assert torch.count_nonzero(grad[2 * block.dim:]) > 0
                if step == 0:
                    assert torch.count_nonzero(grad[:2 * block.dim]) == 0
        for branch in branches:
            grads = [p.grad for p in branch.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            if step == 0:
                assert all(torch.count_nonzero(g) == 0 for g in grads)
            else:
                assert any(torch.count_nonzero(g) > 0 for g in grads)
        optimizer.step()
    assert not torch.equal(output, inputs["x"])


def _legacy_model(monkeypatch, cfg, seed):
    # The frozen block has the former random modulation initializer. Other
    # modules stay current, isolating initialization from earlier geometry fixes.
    with monkeypatch.context() as patch:
        patch.setattr(transformer, "TransformerBlock",
                      _reference_module("models/components/attention.py").TransformerBlock)
        torch.manual_seed(seed)
        return transformer.TrajectoryTransformer(cfg)


@pytest.mark.parametrize("seed", [0, 31])
def test_only_modulation_parameters_change_at_construction(monkeypatch, seed):
    cfg = _options("transformer")
    legacy = _legacy_model(monkeypatch, cfg, seed)
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(seed)
    fresh = transformer.TrajectoryTransformer(cfg)
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    modulation_names = {f"blocks.{i}.norm{j}.linear.{kind}" for i in range(cfg.depth)
                        for j in range(1, 5) for kind in ("weight", "bias")}
    expected = legacy.state_dict()
    assert fresh.state_dict().keys() == expected.keys()
    for name, value in fresh.state_dict().items():
        if name in modulation_names:
            assert torch.count_nonzero(value) == 0
            assert torch.count_nonzero(expected[name]) > 0
        else:
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("scene", ["mixed", "rigid"])
def test_fresh_model_trains_through_zero_gates(precision, scene):
    torch.manual_seed(43)
    model = transformer.TrajectoryTransformer(_options("transformer")).to(precision[0]).train()
    batch = make_batch(**({"deform_n": 0} if scene == "rigid" else {})).to(precision[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(59)
        with _autocast(precision):
            loss, _, _ = trajectory_objective(model, batch, TrajectoryFlowMatching(), noise_context=False)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    # Pure-rigid training first learns the zero pose head, then the gates, then
    # branch weights. Inspect actual objective gradients without manually opening anything.
    for block in model.blocks:
        for projection in _modulations(block):
            assert projection.weight.grad is not None and projection.weight.grad.abs().sum() > 0
        for branch in (block.cross_attn, block.temporal_attn, block.spatial_attn, block.ffn):
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters())


@pytest.mark.parametrize("ema", [False, True])
def test_loading_saved_nonzero_modulation_preserves_weights_and_outputs(precision, monkeypatch, tmp_path, ema):
    cfg = _options("transformer")
    legacy = _legacy_model(monkeypatch, cfg, seed=71).to(precision[0]).eval()
    with torch.no_grad():
        legacy.pose_v_head.weight.normal_(0, 0.025)  # Exercise both output streams.
    batch = make_batch().to(precision[0])
    torch.manual_seed(73)
    noisy = torch.randn_like(batch.future)
    pose = _noisy_pose_gt(batch) + 0.15
    times = torch.full(noisy.shape[:2], 500, device=precision[0], dtype=torch.long)
    saved = {name: value.detach().clone() for name, value in legacy.state_dict().items()}
    config = dict(model=dict(asdict(cfg), architecture="transformer"),
                  data=dict(context_len=cfg.context_len, future_len=batch.num_future_frames,
                            format="unified", stats_location="unified://xy_p99"),
                  train=dict(noise_context=False), denoise_method="flow_matching",
                  flow_matching=dict(num_train_timesteps=1000, shift=1.0,
                                     prediction_type="velocity", inference_steps=4))
    weights = dict(legacy.named_parameters()) if ema else saved
    path = tmp_path / "checkpoint.pt"
    torch.save(dict(cfg=config, **{"ema" if ema else "model": weights}), path)
    loaded, _, _, _ = load_model_and_diffusion(path, precision[0])
    torch.testing.assert_close(loaded.state_dict(), saved, rtol=0, atol=0)
    with torch.no_grad(), _autocast(precision):
        expected = legacy(batch, noisy, times, noisy_pose=pose)
        actual = loaded(batch, noisy, times, noisy_pose=pose)
    assert all(torch.isfinite(value).all() and value.abs().sum() > 0 for value in expected)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Repeated forwards must never reset the loaded gates.
    torch.testing.assert_close(loaded.state_dict(), saved, rtol=0, atol=0)
