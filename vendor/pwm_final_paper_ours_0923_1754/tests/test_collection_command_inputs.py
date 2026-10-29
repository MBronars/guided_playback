"""Command amplitudes must reach the network even beyond achieved-motion p99."""
from dataclasses import replace

import pytest
import torch

from tests.test_mpm_drag_adapter import adapter, sample


@pytest.mark.parametrize("domain", ["mpm", "mpm_drag"])
def test_unet_preserves_new_collection_commands_and_legacy_convention(domain):
    from models.transformer import ModelConfig
    from models.unet_std import UNetWrapper
    batch = adapter()._process_sample(sample()).slice(1)
    # Deliberately exceed normalization bounds without confusing zero holds and
    # releases. This isolates conditioning from the learned output head.
    commands = torch.zeros_like(batch.actions)
    commands[..., 2] = batch.delta_stats.max_val[2] * 1.4
    commands *= batch.action_active[..., None]
    batch = replace(batch, domain=domain, actions=commands)
    cfg = ModelConfig(dim=32, depth=1, heads=4, num_latents=8, k_neighbors=4,
        dropout=0., context_len=4, unet_actions=True, all_relative=True,
        unet_points=0, unet_strides="1,2", unet_widths="16,32,64",
        unet_enc_depths="1,1,1", unet_dec_depths="1,1", unet_head_dim=8,
        unet_t_dim=32, unet_l1_by_mat="0:16,1:16,2:16", unet_pool_k=4,
        unet_interp_k=3, unet_geo_r_m=.1, unet_fps_random_start=False)
    model = UNetWrapper(cfg).eval()
    seen = []
    hook = model.model.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(kwargs["feats"].detach().clone()),
        with_kwargs=True)
    with torch.no_grad():
        model(batch=batch, noisy_future=torch.zeros_like(batch.actions),
              timestep=torch.full((1, 16), .5))
    hook.remove()
    expected = batch.delta_stats.normalize(commands, clip=domain != "mpm_drag")
    torch.testing.assert_close(seen[0][:, 3:, :, -4:-1], expected, atol=0, rtol=0)
    torch.testing.assert_close(seen[0][:, 3:, :, -1], batch.action_active.float(), atol=0, rtol=0)
    assert expected.abs().max() > 1 if domain == "mpm_drag" else expected.abs().max() <= 1


@pytest.mark.parametrize("domain", ["pickup", "cloth", "pushing", "rl_push", "folding", "yikai", "dex"])
def test_transformer_pose_centers_preserve_new_collection_coordinates(domain, monkeypatch):
    from tests.test_model_scope import _construct
    from tests.test_unet_std_pose_native import make_batch
    from utils.pose_native import pose_native_inputs
    batch = replace(make_batch(), domain=domain)
    model = _construct("transformer", reference=False)
    ridx = (batch.body_type[0] == 1).nonzero(as_tuple=True)[0]
    particle = batch.delta_stats.normalize(batch.future_deltas)
    pose = batch.pose_delta_stats.normalize(batch.pose_deltas_gt[:, :, ridx])
    _, pack = pose_native_inputs(batch, particle, pose)
    assert pack["centers"].abs().max() > .8  # Regression must reach the old clamp.
    observed = []
    original = model._compute_rotary_emb

    def capture(positions, *args, **kwargs):
        observed.append(positions.detach().clone())
        return original(positions, *args, **kwargs)

    monkeypatch.setattr(model, "_compute_rotary_emb", capture)
    with torch.no_grad():
        model(batch=batch, noisy_future=particle, noisy_pose=pose,
              timestep=torch.full((batch.batch_size, batch.num_future_frames), .5))
    centers = observed[-1][:, :, -len(ridx):]
    unclipped = domain in ("folding", "yikai", "dex")
    expected = batch.position_stats.normalize(pack["centers"], clip=not unclipped) * model.cfg.rope_position_scale
    torch.testing.assert_close(centers, expected, atol=0, rtol=0)
    assert centers.abs().max() > 1 if unclipped else centers.abs().max() <= 1
