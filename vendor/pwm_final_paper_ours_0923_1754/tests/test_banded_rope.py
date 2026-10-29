"""Unit tests for the banded-RoPE additions to RotaryPositionEncoding3D.

Covers: legacy-equivalence when off (backward compat), explicit log-band frequencies,
partial-rotary identity pass-through tail, and patch-pool (forward_pooled) magnitude.
"""
import math
import torch

from models.components.embeddings import RotaryPositionEncoding3D


def _recover_axis0_freqs(rope, c=0.01):
    """Recover axis-0 frequencies via a small coordinate (no phase wrap)."""
    dpa = rope.dim_per_axis
    xyz = torch.zeros(1, 1, 3)
    xyz[0, 0, 0] = c
    cos, sin = rope(xyz)
    # axis-0 occupies [0:dpa], interleaved (c0,c0,c1,c1,...) -> stride-2 are the freqs
    return torch.atan2(sin[0, 0, 0:dpa:2], cos[0, 0, 0:dpa:2]) / c


def test_legacy_default_unchanged():
    """Defaults (no rotary_dim, no band) reproduce the legacy max_period frequencies."""
    dim, mp = 64, 0.1
    rope = RotaryPositionEncoding3D(dim=dim, max_period=mp)
    assert rope.f_min is None and rope.f_max is None
    assert rope.feature_dim == (dim // 6) * 6 and rope.pad_length == dim - rope.feature_dim
    cos, sin = rope(torch.randn(2, 10, 3))
    assert cos.shape == (2, 10, dim) and sin.shape == (2, 10, dim)
    # pure rotation -> unit magnitude per channel
    assert torch.allclose(cos ** 2 + sin ** 2, torch.ones_like(cos), atol=1e-5)
    # legacy frequency formula
    dpa = rope.dim_per_axis
    exp = torch.exp(-math.log(mp) * torch.arange(0, dpa, 2, dtype=torch.float32) / dpa)
    assert torch.allclose(_recover_axis0_freqs(rope), exp, rtol=1e-3)


def test_explicit_band_frequencies():
    """f_min/f_max give log-spaced frequencies and override max_period."""
    dim, f_min, f_max = 96, 0.5, 64.0
    rope = RotaryPositionEncoding3D(dim=dim, max_period=0.1, f_min=f_min, f_max=f_max)
    n_freq = rope.dim_per_axis // 2
    exp = torch.exp(torch.linspace(math.log(f_min), math.log(f_max), n_freq))
    assert torch.allclose(_recover_axis0_freqs(rope), exp, rtol=1e-3)


def test_partial_rotary_passthrough():
    """rotary_dim < dim -> the tail channels are identity (cos=1, sin=0)."""
    dim, rotary_dim = 64, 48  # 48 divisible by 6 -> feature_dim 48, pad 16
    rope = RotaryPositionEncoding3D(dim=dim, rotary_dim=rotary_dim, f_min=0.5, f_max=64.0)
    assert rope.feature_dim == 48 and rope.pad_length == 16
    cos, sin = rope(torch.randn(2, 5, 3))
    assert cos.shape == (2, 5, dim)
    assert torch.allclose(cos[..., 48:], torch.ones(2, 5, 16))
    assert torch.allclose(sin[..., 48:], torch.zeros(2, 5, 16))


def test_forward_pooled_magnitude():
    """Patch-pool: tight patch -> magnitude ~1; spread patch -> magnitude < 1; tail identity."""
    dim = 96
    rope = RotaryPositionEncoding3D(dim=dim, rotary_dim=72, f_min=0.5, f_max=64.0)
    fd = rope.feature_dim
    pt = torch.tensor([0.3, -0.1, 0.2])
    tight = pt.view(1, 1, 1, 3).expand(2, 4, 8, 3).contiguous()  # [B,N,k,3] identical members
    cos_t, sin_t = rope.forward_pooled(tight)
    assert cos_t.shape == (2, 4, dim)
    mag_t = (cos_t[..., :fd] ** 2 + sin_t[..., :fd] ** 2).sqrt()
    assert mag_t.mean() > 0.99  # tight patch -> ~unit magnitude
    # pad tail still identity
    assert torch.allclose(cos_t[..., fd:], torch.ones_like(cos_t[..., fd:]))
    assert torch.allclose(sin_t[..., fd:], torch.zeros_like(sin_t[..., fd:]))
    # spread patch -> attenuated magnitude
    spread = torch.randn(2, 4, 8, 3)
    cos_s, sin_s = rope.forward_pooled(spread)
    mag_s = (cos_s[..., :fd] ** 2 + sin_s[..., :fd] ** 2).sqrt()
    assert mag_s.mean() < mag_t.mean()
    assert (mag_s <= 1.0 + 1e-5).all()


def test_unet_rope_f_min_default_and_historical_checkpoint():
    """The fresh-run default is the 5.58 m band; a checkpoint that omits the key keeps 0.5.

    The default moved 0.5 -> 0.15 on 2026-09-20 (measured: mpm_fresh exceeds the 1.67 m unique range
    in 598 of 600 windows under its own ruler). Checkpoints saved before that never wrote the key, so
    models.factory pins them at the value they trained with instead of re-banding them silently.
    """
    from models.config import ModelConfig
    from models.factory import parse_model_config

    assert ModelConfig().unet_rope_f_min == 0.15
    fresh = {"architecture": "unet"}
    assert parse_model_config(fresh)[1].unet_rope_f_min == 0.15
    assert parse_model_config(dict(fresh), checkpoint=True)[1].unet_rope_f_min == 0.5
    stated = {"architecture": "unet", "unet_rope_f_min": 0.3}
    assert parse_model_config(dict(stated), checkpoint=True)[1].unet_rope_f_min == 0.3

    r = 0.133155
    assert abs(2 * math.pi * r / 0.15 - 5.578) < 0.01      # fresh runs: 5.58 m per axis
    assert abs(2 * math.pi * r / 0.5 - 1.673) < 0.01       # historical checkpoints: 1.67 m


def test_resolved_config_always_names_the_rope_band():
    """A recipe written before the change still SAVES 0.15, so its own resume does not fall back to 0.5."""
    from omegaconf import OmegaConf
    from utils.config import resolve_config

    base = OmegaConf.to_container(OmegaConf.load("configs/unet.yaml"), resolve=True)
    base["run_name"] = "rope-band-probe"
    assert base["model"]["unet_rope_f_min"] == 0.15                     # the recipe base states it
    old = {**base, "model": {k: v for k, v in base["model"].items() if k != "unet_rope_f_min"}}
    assert resolve_config(old).model.unet_rope_f_min == 0.15            # stamped for the checkpoint
    stated = {**base, "model": {**base["model"], "unet_rope_f_min": 0.3}}
    assert resolve_config(stated).model.unet_rope_f_min == 0.3          # an explicit value is kept
