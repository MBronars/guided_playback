"""
Correctness tests for Diffusion Forcing implementation.

Three groups:
  Group 1: Backward compatibility (must match current behavior exactly)
  Group 2: Per-frame correctness (new functionality works correctly)
  Group 3: Scheduling matrix & end-to-end
"""

import numpy as np
import torch

from data.batch import TrajectoryBatch
from data.dataset import NormalizationStats
from diffusion import TrajectoryFlowMatching
from diffusion.scheduling import generate_scheduling_matrix
from models.transformer import TrajectoryTransformer, ModelConfig
from models.components.attention import TransformerBlock


# =============================================================================
# Helpers
# =============================================================================

def create_dummy_batch(
    B: int = 2, T: int = 10, N: int = 64, context_len: int = 2, device: str = "cpu",
) -> TrajectoryBatch:
    """Create a dummy TrajectoryBatch for testing."""
    F = T - context_len
    positions = torch.randn(B, T, N, 3, device=device)
    actions = torch.randn(B, F, N, 3, device=device)
    num_grippers = max(1, N // 10)
    action_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    action_mask[:, :num_grippers] = True
    valid_lengths = torch.full((B,), N, dtype=torch.long, device=device)
    E = torch.rand(B, N, device=device) * 1e6
    nu = torch.rand(B, N, device=device) * 0.5
    rho = torch.rand(B, N, device=device) * 1000
    object_ids = torch.zeros(B, N, dtype=torch.long, device=device)
    # Identity normalization stats (no-op): normalize/denormalize are ~identity
    dummy_stats = NormalizationStats(
        min_val=torch.tensor([-1.0, -1.0, -1.0]),
        max_val=torch.tensor([1.0, 1.0, 1.0]),
    )
    return TrajectoryBatch(
        positions=positions, actions=actions, action_mask=action_mask,
        valid_lengths=valid_lengths, E=E, nu=nu, rho=rho, context_len=context_len,
        object_ids=object_ids, delta_stats=dummy_stats, position_stats=dummy_stats,
    )


def make_small_model_cfg():
    return ModelConfig(
        dim=64, depth=2, heads=4, num_latents=32, k_neighbors=8,
        dropout=0.0, context_len=2,
    )


def _activate_modulation(blocks):
    """Conditioning tests need active branches beyond identity initialization."""
    for block in blocks:
        for norm in (block.norm1, block.norm2, block.norm3, block.norm4):
            norm.linear.reset_parameters()


# =============================================================================
# Group 1: Backward Compatibility
# =============================================================================

def test_flow_matching_add_noise_backward_compat():
    """New add_noise (while ndim) must be bit-identical to old (view) for [B] timesteps."""
    print("\n" + "=" * 60)
    print("TEST 1: Flow matching add_noise backward compat")
    print("=" * 60)

    K = 1000
    B, F, N = 4, 8, 32
    x_0 = torch.randn(B, F, N, 3)
    noise = torch.randn_like(x_0)
    timesteps = torch.randint(0, K, (B,))

    # Old path (view)
    t_old = timesteps.float() / K
    t_old = t_old.view(-1, *([1] * (x_0.ndim - 1)))
    x_t_old = (1 - t_old) * x_0 + t_old * noise

    # New path (while ndim)
    fm = TrajectoryFlowMatching(num_train_timesteps=K)
    x_t_new = fm.add_noise(x_0, noise, timesteps)

    assert torch.equal(x_t_old, x_t_new), "Must be bit-identical"
    print("PASSED")




def test_adaln_reshape_trick_equivalence():
    """Reshape trick with uniform t_emb must match old repeat_interleave path."""
    print("\n" + "=" * 60)
    print("TEST 3: AdaLN reshape trick equivalence")
    print("=" * 60)

    B, F, L, C, d = 2, 8, 32, 2, 64
    torch.manual_seed(42)
    block = TransformerBlock(dim=d, heads=4, dropout=0.0)
    _activate_modulation([block])
    block.eval()

    x = torch.randn(B, F, L, d)
    context = torch.randn(B, C, L, d)
    actions = torch.randn(B, F, L, d)

    # The new code expects [B, F, d]. To test equivalence, we pass uniform t_emb.
    t_emb_scalar = torch.randn(B, d)
    t_emb_perframe = t_emb_scalar.unsqueeze(1).expand(-1, F, -1).contiguous()  # [B, F, d]

    with torch.no_grad():
        # New path (per-frame, but all frames same value)
        out_new = block(x, context, actions, t_emb_perframe)
        out_per_token = block(x, context, actions,
                              t_emb_perframe[:, :, None].expand(-1, -1, L, -1))

    assert out_new.shape == (B, F, L, d)
    assert torch.isfinite(out_new).all(), "Output must be finite"
    assert not torch.allclose(out_new, x), "The reshape comparison must exercise active branches"
    torch.testing.assert_close(out_new, out_per_token, rtol=1e-5, atol=1e-6)
    print("PASSED (active per-frame and per-token conditioning agree)")


def test_full_model_backward_compat():
    """Model with [B] timestep must match [B,F] timestep (all frames same value)."""
    print("\n" + "=" * 60)
    print("TEST 4: Full model backward compat")
    print("=" * 60)

    torch.manual_seed(42)
    cfg = make_small_model_cfg()
    model = TrajectoryTransformer(cfg)
    _activate_modulation(model.blocks)
    model.eval()

    B, T, N = 2, 10, 64
    batch = create_dummy_batch(B=B, T=T, N=N, context_len=cfg.context_len)
    F = batch.num_future_frames
    noisy_future = torch.randn(B, F, N, 3)

    timestep_scalar = torch.tensor([100, 500])  # [B]
    timestep_expanded = timestep_scalar.unsqueeze(1).expand(-1, F).contiguous()  # [B, F]

    with torch.no_grad():
        out_scalar = model(batch, noisy_future.clone(), timestep_scalar)
        out_expanded = model(batch, noisy_future.clone(), timestep_expanded)

    assert torch.allclose(out_scalar, out_expanded, atol=1e-5), \
        f"Max diff: {(out_scalar - out_expanded).abs().max():.2e}"
    print("PASSED")


# =============================================================================
# Group 2: Per-Frame Correctness
# =============================================================================

def test_per_frame_noise_independence_flow_matching():
    """Per-frame noising with [B,F] must match noising each frame independently."""
    print("\n" + "=" * 60)
    print("TEST 5a: Per-frame noise independence (flow matching)")
    print("=" * 60)

    fm = TrajectoryFlowMatching(num_train_timesteps=1000)
    B, F, N = 3, 6, 16
    x_0 = torch.randn(B, F, N, 3)
    noise = torch.randn_like(x_0)
    timesteps_2d = torch.randint(0, 1000, (B, F))

    # Batch path
    x_t_batch = fm.add_noise(x_0, noise, timesteps_2d)

    # Per-frame path
    x_t_manual = torch.zeros_like(x_0)
    for f in range(F):
        t_f = timesteps_2d[:, f]  # [B]
        t_norm = t_f.float() / 1000
        t_norm = t_norm[:, None, None]  # [B, 1, 1]
        x_t_manual[:, f] = (1 - t_norm) * x_0[:, f] + t_norm * noise[:, f]

    assert torch.allclose(x_t_batch, x_t_manual, atol=1e-6), \
        f"Max diff: {(x_t_batch - x_t_manual).abs().max():.2e}"
    print("PASSED")




def test_model_per_frame_timesteps_shape_and_gradients():
    """Model handles [B,F] timesteps: correct shape, gradients flow."""
    print("\n" + "=" * 60)
    print("TEST 6: Model per-frame timesteps shape and gradients")
    print("=" * 60)

    cfg = make_small_model_cfg()
    model = TrajectoryTransformer(cfg)

    B, T, N = 2, 10, 64
    batch = create_dummy_batch(B=B, T=T, N=N, context_len=cfg.context_len)
    F = batch.num_future_frames
    noisy_future = torch.randn(B, F, N, 3, requires_grad=True)
    timesteps = torch.randint(0, 1000, (B, F))  # per-frame

    out = model(batch, noisy_future, timesteps)
    assert out.shape == (B, F, N, 3), f"Expected {(B, F, N, 3)}, got {out.shape}"

    loss = out.sum()
    loss.backward()
    assert noisy_future.grad is not None, "Gradients should flow to input"
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Model should have gradients"
    print("PASSED")


def test_per_frame_timesteps_affect_output():
    """Different per-frame timestep patterns must produce different outputs."""
    print("\n" + "=" * 60)
    print("TEST 7: Per-frame timesteps affect output")
    print("=" * 60)

    torch.manual_seed(42)
    cfg = make_small_model_cfg()
    model = TrajectoryTransformer(cfg)
    _activate_modulation(model.blocks)
    model.eval()

    B, T, N = 2, 10, 64
    batch = create_dummy_batch(B=B, T=T, N=N, context_len=cfg.context_len)
    F = batch.num_future_frames
    noisy_future = torch.randn(B, F, N, 3)

    timesteps_a = torch.full((B, F), 100, dtype=torch.long)
    timesteps_b = torch.linspace(100, 900, F).long().unsqueeze(0).expand(B, -1).contiguous()

    with torch.no_grad():
        out_a = model(batch, noisy_future.clone(), timesteps_a)
        out_b = model(batch, noisy_future.clone(), timesteps_b)

    assert not torch.allclose(out_a, out_b, atol=1e-3), \
        "Different per-frame timesteps should produce different outputs"
    print("PASSED")


# =============================================================================
# Group 3: Scheduling Matrix & End-to-End
# =============================================================================

def test_scheduling_matrix_full_sequence():
    """full_sequence: all frames same timestep, monotonically decreasing to 0."""
    print("\n" + "=" * 60)
    print("TEST 8: Scheduling matrix full_sequence")
    print("=" * 60)

    F, S = 8, 50
    K = generate_scheduling_matrix(F, S, schedule="full_sequence")
    assert K.shape == (S + 1, F), f"Expected {(S + 1, F)}, got {K.shape}"

    # All columns identical at each row
    for m in range(K.shape[0]):
        assert np.all(K[m] == K[m, 0]), f"Row {m}: frames should have same noise level"

    # Monotonically decreasing
    for m in range(K.shape[0] - 1):
        assert K[m, 0] >= K[m + 1, 0], f"Row {m}: should be monotonically decreasing"

    # First row = max noise, last row = 0
    assert K[0, 0] == S, f"First row should be {S}, got {K[0, 0]}"
    assert K[-1, 0] == 0, f"Last row should be 0, got {K[-1, 0]}"
    print("PASSED")


def test_scheduling_matrix_pyramid():
    """pyramid: later frames noisier, all reach 0."""
    print("\n" + "=" * 60)
    print("TEST 9: Scheduling matrix pyramid")
    print("=" * 60)

    F, S = 8, 50
    K = generate_scheduling_matrix(F, S, schedule="pyramid", uncertainty_scale=1.0)

    # Later frames have >= noise level at each step
    for m in range(K.shape[0]):
        for t in range(F - 1):
            assert K[m, t] <= K[m, t + 1], \
                f"Row {m}: frame {t} ({K[m, t]}) should be <= frame {t+1} ({K[m, t+1]})"

    # All frames reach 0 by the last row
    assert np.all(K[-1] == 0), f"Last row should be all zeros, got {K[-1]}"

    # First frame starts at S
    assert K[0, 0] == S, f"First frame at row 0 should be {S}, got {K[0, 0]}"

    # Last frame starts higher than first frame
    assert K[0, -1] >= K[0, 0], "Last frame should start at higher noise"

    # More rows than full_sequence (because later frames need extra steps)
    K_full = generate_scheduling_matrix(F, S, schedule="full_sequence")
    assert K.shape[0] > K_full.shape[0], "Pyramid should have more rows than full_sequence"

    print("PASSED")


def test_train_step_with_diffusion_forcing():
    """End-to-end: train step with DF produces finite loss and gradients."""
    print("\n" + "=" * 60)
    print("TEST 10: Train step with diffusion forcing")
    print("=" * 60)

    torch.manual_seed(42)
    cfg = make_small_model_cfg()
    model = TrajectoryTransformer(cfg)

    B, T, N = 2, 10, 64
    batch = create_dummy_batch(B=B, T=T, N=N, context_len=cfg.context_len)
    F = batch.num_future_frames

    fm = TrajectoryFlowMatching(num_train_timesteps=1000)

    # Per-frame timesteps
    timesteps = fm.sample_timesteps(B, torch.device("cpu"), num_frames=F)
    assert timesteps.shape == (B, F), f"Expected {(B, F)}, got {timesteps.shape}"

    # Forward pass
    x_0 = batch.future_deltas
    noise = torch.randn_like(x_0)
    x_t = fm.add_noise(x_0, noise, timesteps)
    pred = model(batch, x_t, timesteps)
    target = fm.get_target(x_0, noise)
    loss = ((pred - target) ** 2).mean()

    assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"

    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Model should have gradients"

    print(f"Loss: {loss.item():.4f}")
    print("PASSED")
