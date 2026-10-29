"""Shared synthetic batch fixture and retained FM training/checkpoint contracts."""

import torch
import pytest

from data.batch import TrajectoryBatch
from data.dataset import NormalizationStats


def create_dummy_batch(
    B: int = 2,
    T: int = 10,
    N: int = 64,
    context_len: int = 2,
    device: str = "cpu",
) -> TrajectoryBatch:
    """Create a dummy TrajectoryBatch for testing."""
    F = T - context_len

    positions = torch.randn(B, T, N, 3, device=device)
    actions = torch.randn(B, F, N, 3, device=device)

    num_grippers = max(1, N // 10)
    action_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    action_mask[:, :num_grippers] = True

    valid_lengths = torch.full((B,), N, dtype=torch.long, device=device)

    object_ids = torch.zeros(B, N, dtype=torch.long, device=device)

    E = torch.rand(B, N, device=device) * 1e6
    nu = torch.rand(B, N, device=device) * 0.5
    rho = torch.rand(B, N, device=device) * 1000

    # Identity normalization stats (denormalize is a no-op), required by forward.
    ident = NormalizationStats(
        min_val=torch.tensor([-1.0, -1.0, -1.0], device=device),
        max_val=torch.tensor([1.0, 1.0, 1.0], device=device),
    )

    return TrajectoryBatch(
        positions=positions,
        actions=actions,
        action_mask=action_mask,
        object_ids=object_ids,
        valid_lengths=valid_lengths,
        E=E,
        nu=nu,
        rho=rho,
        context_len=context_len,
        delta_stats=ident,
        position_stats=ident,
    )


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
def test_fm_optimizer_ema_checkpoint_continuation(architecture, tmp_path):
    from diffusion import TrajectoryFlowMatching
    from tests.test_model_scope import _construct
    from tests.test_unet_std_pose_native import make_batch
    from utils.ema import EMA
    from utils.trajectory_objective import trajectory_objective

    def construct():
        model = _construct(architecture, False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
        return model, optimizer, scheduler, EMA(model, decay=0.9)

    batch = make_batch()
    diffusion = TrajectoryFlowMatching()
    model, optimizer, scheduler, ema = construct()

    def update(model, optimizer, scheduler, ema, seed):
        torch.manual_seed(seed)
        optimizer.zero_grad(set_to_none=True)
        loss, _, _ = trajectory_objective(model, batch, diffusion,
            diffusion_forcing=True, noise_context=True, clip_deltas=False)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update()
        return loss.detach()

    before = {name: value.clone() for name, value in model.state_dict().items()}
    for seed in (312, 313, 314):
        update(model, optimizer, scheduler, ema, seed)
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    path = tmp_path / "checkpoint.pt"
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), ema=ema.state_dict()), path)
    restored, opt2, sched2, ema2 = construct()
    state = torch.load(path, map_location="cpu", weights_only=True)
    restored.load_state_dict(state["model"], strict=True)
    opt2.load_state_dict(state["optimizer"])
    sched2.load_state_dict(state["scheduler"])
    ema2.load_state_dict(state["ema"])
    loss1 = update(model, optimizer, scheduler, ema, 315)
    loss2 = update(restored, opt2, sched2, ema2, 315)
    torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict(), restored.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), opt2.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(ema.state_dict(), ema2.state_dict(), rtol=0, atol=0)
    assert scheduler.state_dict() == sched2.state_dict()
