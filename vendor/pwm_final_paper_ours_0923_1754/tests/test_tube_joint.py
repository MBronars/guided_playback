"""Behavioural tests for the joint-attention core (model.unet_core=tube_joint, models/tube_joint.py).

Each test pins a property the flags CLAIM, not just that a forward runs. Every test that exercises a
zero-init path (token adaLN) wakes the weights first and carries a control that must FAIL without the
flag, so a silently-dead knob cannot pass.
"""
import math

import pytest
import torch

from models.config import ModelConfig
from models.unet_std import build_unet_config, UNetWrapper
from models.tube_joint import TubeJointCore
from models.tube_groups import random_cells, label_slots

BASE = dict(unet_points=0, unet_strides="4,2", unet_widths="44,88,176",
            unet_enc_depths="2,2,2", unet_dec_depths="2,2", unet_head_dim=44,
            unet_l1_by_mat="0:64,1:64,2:64,3:32,4:32,5:64",
            unet_pool_mode="pointnet", unet_unpool_mode="pointnet", unet_pool_k=8,
            unet_fps_random_start=False, unet_geo_r_m=0.133155, all_relative=True, num_mat_types=7)


def make_cfg(**over):
    kw = dict(BASE); kw.update(over)
    return build_unet_config(ModelConfig(**kw))


def wake_blocks(core):
    """Every JointBlock is adaLN-ZERO: at init both gates are 0, so the attention and the FFN contribute
    NOTHING and the output is blind to everything the blocks do. Any test that probes block behaviour must
    wake the per-row modulation first or it passes vacuously. adaln_obj (the per-token material projection)
    is deliberately left at zero -- the token-adaLN test wakes that one itself."""
    with torch.no_grad():
        for m in core.modules():
            if isinstance(m, torch.nn.Linear) and getattr(m, "_is_block_adaln", False):
                m.weight.normal_(0, 0.02); m.bias.normal_(0, 0.5)
    return core


def make_core(cfg=None, tubes=(64, 0, 0), wake=True, **kw):
    torch.manual_seed(0)
    kw.setdefault("within_object_levels", [0])
    kw.setdefault("rope_identity", "frame0_pos")
    core = TubeJointCore(cfg or make_cfg(), tube_sizes=list(tubes), **kw)
    for blk in [b for lvl in list(core.enc) + list(core.dec) for b in lvl]:
        blk.adaln._is_block_adaln = True
    return wake_blocks(core) if wake else core


def make_batch(B=2, T=4, counts=(128, 96, 128), mats=(1, 3, 0), seed=1):
    n = sum(counts)
    g = torch.Generator().manual_seed(seed)
    obj = torch.cat([torch.full((c,), o) for o, c in enumerate(counts)])[None].expand(B, n).contiguous()
    mat = torch.cat([torch.full((c,), m) for m, c in zip(mats, counts)])[None].expand(B, n).contiguous()
    pos0 = torch.rand(B, n, 3, generator=g) * 0.5
    pos = pos0[:, None] + 0.01 * torch.randn(B, T, n, 3, generator=g).cumsum(1)
    return dict(feats=torch.randn(B, T, n, 10, generator=g), pos=pos, pos0=pos0, obj_id=obj, mat_id=mat,
                t=torch.rand(B, T, generator=g), n_context=1)


# ---------------------------------------------------------------------------------------------------
# the base UNet path is untouched


def test_default_core_is_the_plain_unet():
    """model.unet_core defaults to 'unet' and then builds exactly the production UNet -- no joint blocks,
    no register-mode plumbing, and none of the new keys read."""
    from models.unet_std import UNet
    w = UNetWrapper(ModelConfig(**BASE))
    assert w._unet_core == "unet"
    assert isinstance(w.model, UNet) and not isinstance(w.model, TubeJointCore)
    assert not any("adaln_obj" in k for k in w.state_dict())


def test_token_adaln_is_accepted_on_both_cores():
    """unet_token_adaln works on the factorized UNet too (per OBJECT there, per token in the joint
    blocks -- the same values, since the conditioning is constant within an object). It used to raise
    on the UNet path; it now builds a projection on both."""
    for core in ("unet", "tube_joint"):
        extra = dict(unet_core="tube_joint", unet_tube_sizes="64,0,0",
                     unet_rope_identity="frame0_pos") if core == "tube_joint" else {}
        w = UNetWrapper(ModelConfig(unet_token_adaln=True, **BASE, **extra))
        assert any("adaln_obj" in k for k in w.state_dict()), f"{core}: no per-object projection built"


# ---------------------------------------------------------------------------------------------------
# grouping


def test_random_cells_partition_every_token_exactly_once():
    X = torch.rand(3, 100, 3)
    lab, k, cap = random_cells(X, 32)
    assert k == math.ceil(100 / 32) and cap == math.ceil(100 / k)
    for g in range(3):
        assert torch.bincount(lab[g], minlength=k).max() <= cap
        assert int(lab[g].min()) >= 0 and int(lab[g].max()) < k
    assert lab.shape == (3, 100)


def test_label_slots_round_trips_every_token():
    """src lays tokens into the group layout and back reads them out: the composition is the identity,
    and every slot that is not a real token reads the pad row."""
    X = torch.rand(2, 100, 3)
    lab, k, cap = random_cells(X, 32)
    src, back = label_slots(lab, k, cap)
    tok = torch.arange(100).float()[None, :, None].expand(2, 100, 1)
    ext = torch.cat([tok, torch.full((2, 1, 1), -1.0)], dim=1)          # [tokens | pad row]
    laid = ext.gather(1, src[..., None])                                 # [B, k*cap, 1]
    got = laid.gather(1, back[..., None])[..., 0]
    assert torch.equal(got, tok[..., 0]), "a token did not come back to its own slot"
    assert int((laid[..., 0] == -1).sum(1)[0]) == k * cap - 100, "pad slots do not account for the padding"


def test_random_grouping_is_redrawn_every_forward():
    core = make_core(grouping="random", group_draws=2).eval()
    b = make_batch()
    torch.manual_seed(1); a = core(**b)
    torch.manual_seed(2); c = core(**b)
    assert not torch.allclose(a, c), "the random grouping did not change between forwards"


def test_within_object_groups_never_straddle_objects():
    """At a within-object tubed level a token's L0 encoder output must not depend on another object's
    features. The control -- the same level without within_object -- must fail this."""
    cfg = make_cfg()
    counts = (128, 96, 128)
    b = make_batch(counts=counts)

    def l0_enc(core):
        prov = core._build_provider(cfg, b["pos"], b["pos0"], b["obj_id"], b["mat_id"], None)
        t_emb = core.t_mlp(core.t_embed(b["t"].reshape(-1)).reshape(b["t"].shape[0], b["t"].shape[1], -1))
        x = core.stem(b["feats"]) + core.mat_emb(b["mat_id"])[:, None]
        orders = core.curve_orders(prov.pos0[0], core.K,
                                   obj=core._level_obj(prov, 0) if 0 in core.within_object else None)
        core._fwd_cache = {}
        out = core._run_level(0, core.enc[0], x, prov.pos[0], prov.pos0[0], orders, t_emb, 0,
                              [n for _, n in prov.segs[0]] if 0 in core.within_object else None, None)
        core._fwd_cache = None
        return out

    for within, expect_isolated in ((([0]), True), ([], False)):
        core = make_core(cfg=cfg, within_object_levels=within).eval()
        torch.manual_seed(3)
        a = l0_enc(core)
        b["feats"] = b["feats"].clone()
        b["feats"][:, :, counts[0]:] += 5.0                      # perturb objects 1 and 2 only
        torch.manual_seed(3)
        c = l0_enc(core)
        own = torch.allclose(a[:, :, :counts[0]], c[:, :, :counts[0]], atol=1e-5)
        b["feats"][:, :, counts[0]:] -= 5.0
        assert own is expect_isolated, \
            ("within_object leaked another object into the group" if expect_isolated
             else "the control is vacuous: objects were isolated WITHOUT within_object")


# ---------------------------------------------------------------------------------------------------
# registers


def test_per_level_registers_do_not_cross_the_seams():
    """unet_tube_register_mode=per_level: one learned set per level, no seam projections at all, and
    every set is alive (a gradient reaches it)."""
    cfg = make_cfg(unet_register_from_level=1, unet_register_count=8)
    core = make_core(cfg=cfg, register_mode="per_level").train()
    assert not hasattr(core, "reg_up") and not hasattr(core, "reg_down"), \
        "per_level registers must not build the carried seam projections"
    assert sorted(core.reg_init.keys()) == ["1", "2"]
    assert all(core.reg_init[k].shape == (8, cfg.widths[int(k)]) for k in core.reg_init)
    core(**make_batch()).square().mean().backward()
    for k, p in core.reg_init.items():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, \
            f"level {k}'s register set is dead"


def test_carried_registers_build_the_seam_projections():
    cfg = make_cfg(unet_register_from_level=1, unet_register_count=8)
    core = make_core(cfg=cfg, register_mode="carried").train()
    assert list(core.reg_up.keys()) == ["1"] and list(core.reg_down.keys()) == ["1"]
    assert core.reg_init.shape == (8, cfg.widths[1])
    core(**make_batch()).square().mean().backward()
    assert core.reg_init.grad is not None and core.reg_init.grad.abs().sum() > 0


def test_registers_change_the_output_in_both_modes():
    """A register set that nothing reads would leave the output untouched."""
    cfg = make_cfg(unet_register_from_level=1, unet_register_count=8)
    b = make_batch()
    for mode in ("per_level", "carried"):
        core = make_core(cfg=cfg, register_mode=mode).eval()
        with torch.no_grad():
            a = core(**b).clone()
            if mode == "per_level":
                core.reg_init["2"] += 3.0
            else:
                core.reg_init += 3.0
            c = core(**b)
        assert not torch.allclose(a, c, atol=1e-5), f"{mode} registers do not reach the output"


def test_registers_are_refused_on_a_tubed_level():
    cfg = make_cfg(unet_register_from_level=1, unet_register_count=8)
    with pytest.raises(AssertionError, match="whole-level calls"):
        make_core(cfg=cfg, tubes=(64, 64, 0), register_mode="per_level")


# ---------------------------------------------------------------------------------------------------
# per-token adaLN


def test_token_adaln_carries_material_into_the_blocks():
    """adaln_obj is zero-init, so a fresh model cannot show the effect: wake it, then changing a token's
    MATERIAL alone must change the output. The control (token_adaln off, same woken-shaped model) must
    not -- the base UNet conditions per row only."""
    cfg = make_cfg()
    B, T = 2, 4
    b = make_batch(B=B, T=T)
    phys = torch.randn(B, b["feats"].shape[2], 3, generator=torch.Generator().manual_seed(4))
    # Remap every material to a DIFFERENT one with the SAME l1_by_mat budget (0->5, 1->2, 3->4 are all
    # 64 / 64 / 32): a remap that changed the budget would rebuild the anchor ladder, and the test would
    # pass on that instead of on the conditioning.
    alt = b["mat_id"].clone()
    for src, dst in ((0, 5), (1, 2), (3, 4)):
        alt[b["mat_id"] == src] = dst

    def probe(token_adaln, wake):
        core = make_core(cfg=cfg, token_adaln=token_adaln).eval()   # row adaLN woken by make_core
        if wake:
            with torch.no_grad():
                for m in core.modules():
                    if getattr(m, "adaln_obj", None) is not None:
                        m.adaln_obj.weight.normal_(0, 0.05)
        kw = {"phys": phys} if token_adaln else {}
        with torch.no_grad():
            # mat_emb also feeds the stem, so hold the stem input fixed and vary ONLY the adaLN material
            a = core(**{**b, "mat_id": b["mat_id"]}, **kw)
            core.mat_emb.weight.zero_()                        # kill the stem path: adaLN is the only route left
            a = core(**{**b, "mat_id": b["mat_id"]}, **kw)
            c = core(**{**b, "mat_id": alt}, **kw)
        return not torch.allclose(a, c, atol=1e-6)

    assert probe(True, wake=True), "unet_token_adaln does not carry material into the blocks"
    assert not probe(False, wake=False), \
        "the control is vacuous: material reached the blocks WITHOUT unet_token_adaln"
    assert not probe(True, wake=False), \
        "a fresh (zero-init) adaln_obj must be exactly the identity"


# ---------------------------------------------------------------------------------------------------
# long skips


def test_long_skip_pairs_each_decoder_block_with_its_encoder_block():
    cfg = make_cfg()
    core = make_core(cfg=cfg, long_skip=True)
    for l in range(len(cfg.strides)):
        for blk in core.dec[l]:
            assert blk.skip_linear is not None and blk.skip_linear.in_features == 2 * cfg.widths[l]
    for l in range(len(cfg.enc_depths)):
        for blk in core.enc[l]:
            assert blk.skip_linear is None
    core.train()
    core(**make_batch()).square().mean().backward()
    for l in range(len(cfg.strides)):
        for blk in core.dec[l]:
            assert blk.skip_linear.weight.grad.abs().sum() > 0, "a long skip is never read"


def test_long_skip_pairing_needs_matched_depths():
    """UNetConfig already forces enc_depths[:-1] == dec_depths for the UNet's own long skips, so the joint
    core's dec <= enc guard can never be the first to fire. Pin that, so a future config relaxation does not
    silently leave the joint decoder blocks unpaired."""
    with pytest.raises(AssertionError, match="one-to-one long-skip pairing"):
        make_cfg(unet_enc_depths="2,2,2", unet_dec_depths="3,2")


# ---------------------------------------------------------------------------------------------------
# RoPE budget


def test_explicit_rope_budget_must_add_up_to_the_head_dim():
    cfg = make_cfg()
    make_core(cfg=cfg, rope_budget_dims=[18, 6, 12, 8])                    # 44 = head dim, fine
    with pytest.raises(AssertionError, match="!= head dim"):
        make_core(cfg=cfg, rope_budget_dims=[18, 6, 12, 4])                # 40


def test_frame0_identity_is_zero_offset_between_the_rows_of_a_particle():
    """The point of unet_rope_identity=frame0_pos: a particle's identity coordinate is its CLEAN frame-0
    position, so it is the same in every row whatever the noisy positions do."""
    core = make_core(rope_identity="frame0_pos", rope_budget_dims=[18, 6, 12, 8])
    b = make_batch()
    prov = core._build_provider(core.cfg, b["pos"], b["pos0"], b["obj_id"], b["mat_id"], None)
    coords = core._coords(prov.pos[0], prov.pos0[0], 0)                    # [B, T, n, 7]
    ident = coords[..., 4:]
    assert ident.shape[-1] == 3
    assert torch.allclose(ident, ident[:, :1].expand_as(ident)), "the identity coordinate drifts across rows"
    assert not torch.allclose(coords[..., :3], coords[:, :1, :, :3].expand_as(coords[..., :3])), \
        "the control is vacuous: the noisy positions did not move"
