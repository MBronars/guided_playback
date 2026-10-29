"""Unified-merge flags on models/unet_std.py (robert/unified-unet-combined, 2026-08-28):

  1. unet_actions: +4 per-particle input channels [commanded delta(3), actuated(1)].
     At init the model is BIT-IDENTICAL to the action-free model built from the same
     seed (zero-init stem columns), and the action columns receive gradient.
  2. unet_l1_rule="pow2_quarter": per-object L1 count from the object's OWN count,
     n1 = clamp(pow2_round(n0/4), min, max); a material table, when given, wins for
     the materials it lists. Provider sizes + a full forward/backward on the rule.

    python tests/test_unet_std_unified_flags.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.unet_std import GraphProvider, UNet, UNetWrapper
from tests.test_unet_std_stratified import _model_cfg, _traj_batch, make_batch, make_cfg, open_gates


def _wrapper(seed, **kw):
    torch.manual_seed(seed)
    return UNetWrapper(_model_cfg(**kw))          # float32: pytorch3d kNN/FPS kernels


def test_actions_identity_at_init_and_trainable():
    b = _traj_batch((64, 40), (3, 4))          # rigid + gripper (actuated rows exist); L1 table 3:32,4:32
    b.actions = 0.01 * torch.randn(b.positions.shape[0], b.future.shape[1], b.positions.shape[2], 3)
    b.action_mask = (b.object_ids == 1)
    base = _wrapper(0, unet_actions=False)
    act = _wrapper(1, unet_actions=True)
    s_b, s_a = base.model.stem[0].weight, act.model.stem[0].weight
    assert s_a.shape[1] == s_b.shape[1] + 4 and bool((s_a[:, -4:] == 0).all())
    # a different in_dim changes the RNG stream, so copy base's parameters into the
    # action model (shared slices); the 4 stem columns stay exactly zero
    sd = {k: v.clone() for k, v in base.state_dict().items()}
    w = torch.zeros_like(s_a); w[:, :s_b.shape[1]] = sd["model.stem.0.weight"]
    sd["model.stem.0.weight"] = w
    act.load_state_dict(sd)
    open_gates(base.model); open_gates(act.model)
    nf = torch.randn(b.positions.shape[0], b.future.shape[1], b.positions.shape[2], 3)
    t = torch.rand(b.positions.shape[0])
    torch.manual_seed(1); y0 = base(b, nf, t)
    torch.manual_seed(1); y1 = act(b, nf, t)
    # zero columns contribute exactly 0 to each stem dot product; only the kernel's
    # summation order can differ, hence allclose at float precision, not equal.
    assert torch.allclose(y0, y1, atol=1e-6, rtol=1e-5), "action channels must be silent at init"
    y1.square().mean().backward()
    g = act.model.stem[0].weight.grad[:, -4:]
    assert torch.isfinite(g).all() and g.abs().sum() > 0, "action columns must receive gradient"


def test_l1_rule_sizes():
    cfg = make_cfg(l1_by_mat={}, l1_rule="pow2_quarter", l1_rule_min=4, l1_rule_max=64,
                   strides=(1, 2))
    # n0 -> pow2_round(n0/4) clamped to [4, 64]
    assert [cfg.l1_for_object(0, n) for n in (8, 16, 40, 100, 1000, 8600)] == [4, 4, 8, 32, 64, 64]
    # a table wins for the materials it lists; the rule fills the rest
    cfg2 = make_cfg(l1_by_mat={2: 4}, l1_rule="pow2_quarter", l1_rule_min=4, l1_rule_max=64,
                    strides=(1, 2))
    assert cfg2.l1_for_object(2, 1000) == 4 and cfg2.l1_for_object(0, 1000) == 64
    assert cfg.stratified and cfg.level_sizes == [-1, 64, 32]
    feats, pos, pos0, obj_id, mat_id, t = make_batch()            # counts 40/24/16
    prov = GraphProvider(cfg, pos, pos0, obj_id, mat_id)
    assert prov.sizes[1] == [8, 8, 4] and prov.sizes[2] == [4, 4, 2]
    # bounds must be powers of two, min divisible by the sub-stride product
    try:
        make_cfg(l1_by_mat={}, l1_rule="pow2_quarter", l1_rule_min=6, l1_rule_max=64)
        raise AssertionError("expected an assert on a non-power-of-two min")
    except AssertionError as e:
        assert "powers of two" in str(e)


def test_l1_rule_forward_backward():
    cfg = make_cfg(l1_by_mat={}, l1_rule="pow2_quarter", l1_rule_min=4, l1_rule_max=64,
                   strides=(1, 2))
    torch.manual_seed(0)
    m = UNet(cfg); open_gates(m)
    feats, pos, pos0, obj_id, mat_id, t = make_batch()
    y = m(feats, pos, pos0, obj_id, mat_id, t, n_context=2)
    assert y.shape == feats.shape[:3] + (3,) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    for fn in (test_actions_identity_at_init_and_trainable, test_l1_rule_sizes,
               test_l1_rule_forward_backward):
        fn(); print("ok", fn.__name__)
