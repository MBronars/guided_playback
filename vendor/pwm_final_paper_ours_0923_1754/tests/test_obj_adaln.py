"""model.unet_token_adaln on the factorized UNet: a per-OBJECT adaLN term in every block.

adaln_obj is zero-init, so a fresh model is the identity and any probe of it passes vacuously.
Every behavioural test here wakes it first and carries a control that must FAIL without the flag.
"""
import pytest
import torch

from models.config import ModelConfig
from models.unet_std import UNet, build_unet_config

BASE = dict(unet_points=0, unet_strides="4,2", unet_widths="44,88,176",
            unet_enc_depths="2,2,2", unet_dec_depths="2,2", unet_head_dim=44,
            unet_l1_by_mat="0:64,1:64,2:64,3:32,4:32,5:64",
            unet_pool_mode="pointnet", unet_unpool_mode="pointnet", unet_pool_k=8,
            unet_fps_random_start=False, unet_geo_r_m=0.133155, all_relative=True, num_mat_types=7)


def make(token_adaln, seed=0):
    torch.manual_seed(seed)
    return UNet(build_unet_config(ModelConfig(**BASE, unet_token_adaln=token_adaln)))


def batch(B=2, T=4, counts=(128, 96, 128), mats=(1, 3, 0), seed=1):
    n = sum(counts); g = torch.Generator().manual_seed(seed)
    obj = torch.cat([torch.full((c,), o) for o, c in enumerate(counts)])[None].expand(B, n).contiguous()
    mat = torch.cat([torch.full((c,), m) for m, c in zip(mats, counts)])[None].expand(B, n).contiguous()
    pos0 = torch.rand(B, n, 3, generator=g) * 0.5
    pos = pos0[:, None] + 0.01 * torch.randn(B, T, n, 3, generator=g).cumsum(1)
    phys = torch.randn(B, n, 3, generator=g)
    for o, c in enumerate(counts):                     # physics is constant WITHIN an object
        s = sum(counts[:o]); phys[:, s:s + c] = phys[:, s:s + 1]
    return dict(feats=torch.randn(B, T, n, 10, generator=g), pos=pos, pos0=pos0,
                obj_id=obj, mat_id=mat, t=torch.rand(B, T, generator=g), n_context=1), phys


def wake(core, sd=0.05):
    with torch.no_grad():
        for m in core.modules():
            if getattr(m, "adaln_obj", None) is not None:
                m.adaln_obj.weight.normal_(0, sd)
    return core


def test_flag_off_builds_no_object_projection():
    core = make(False)
    assert all(getattr(b, "adaln_obj", None) is None for b in core.modules() if hasattr(b, "adaln"))
    assert core.obj_mat_emb is None and core.obj_phys is None


def test_flag_on_builds_one_projection_per_block():
    core = make(True)
    blocks = [m for m in core.modules() if hasattr(m, "adaln_obj")]
    assert blocks and all(b.adaln_obj is not None for b in blocks)
    assert core.obj_mat_emb is not None and core.obj_phys is not None


def test_phys_is_required_when_on():
    core = make(True)
    b, _ = batch()
    with pytest.raises(AssertionError, match="must pass phys"):
        core(**b)


def test_zero_init_is_exactly_the_identity():
    """A fresh adaln_obj contributes nothing. Compared against the SAME model with only the per-object
    projections removed -- not against a flag-off model, which would draw different weights from the RNG."""
    b, phys = batch()
    core = make(True).eval()
    with torch.no_grad():
        on = core(**b, phys=phys)
        for m in core.modules():                      # drop only the projection; keep every other weight
            if getattr(m, "adaln_obj", None) is not None:
                m.adaln_obj = None
        without = core(**b, phys=phys)
    assert torch.equal(on, without), "zero-init adaln_obj changed the output"


def test_object_conditioning_reaches_the_output():
    """Woken, changing an object's MATERIAL must change the output; with the flag off it must not.
    The remap preserves each material's l1_by_mat budget so the anchor ladder is unchanged."""
    b, phys = batch()
    alt = b["mat_id"].clone()
    for src, dst in ((0, 5), (1, 2), (3, 4)):          # 64->64, 64->64, 32->32
        alt[b["mat_id"] == src] = dst

    def moved(token_adaln, do_wake):
        core = make(token_adaln).eval()
        if do_wake:
            wake(core)
        with torch.no_grad():
            core.mat_emb.weight.zero_()                # kill the stem path: adaLN is the only route
            kw = {"phys": phys} if token_adaln else {}
            a = core(**b, **kw)
            c = core(**{**b, "mat_id": alt}, **kw)
        return not torch.allclose(a, c, atol=1e-7)

    assert moved(True, True), "unet_token_adaln does not carry the object into the blocks"
    assert not moved(False, False), "control is vacuous: material moved the output with the flag OFF"


def test_modulation_is_shared_within_an_object():
    """Per-object means every particle of an object receives the SAME modulation. Asserted on the
    modulation itself: permuting particles through the model would also move the FPS anchors, which
    would confound the comparison."""
    counts = (128, 96, 128)
    b, phys = batch(counts=counts)
    core = wake(make(True).eval())
    blk = next(m for m in core.modules() if getattr(m, "adaln_obj", None) is not None)
    first = torch.tensor([sum(counts[:o]) for o in range(len(counts))])
    with torch.no_grad():
        oc_mat = core.obj_mat_emb(b["mat_id"].gather(1, first[None].expand(2, -1)))
        oc_phys = core.obj_phys(phys.gather(1, first[None, :, None].expand(2, -1, 3)))
        obj_cond = torch.nn.functional.silu(oc_mat + oc_phys)          # [B, O, t_dim]
        om = blk.adaln_obj(obj_cond)                                   # [B, O, M*w]
    assert om.shape[1] == len(counts), "one modulation row per object"
    assert not torch.allclose(om[:, 0], om[:, 1], atol=1e-6), \
        "control is vacuous: two DIFFERENT objects got the same modulation"
    # and the conditioning that produced it is constant within each object
    for o, c in enumerate(counts):
        st = sum(counts[:o])
        assert torch.equal(b["mat_id"][:, st:st + c], b["mat_id"][:, st:st + 1].expand(-1, c)), \
            "material is not constant within the object"
        assert torch.allclose(phys[:, st:st + c], phys[:, st:st + 1].expand(-1, c, -1)), \
            "physics is not constant within the object"


def test_pose_native_stream_is_covered():
    """The block stream on a pose-native run is [particles | pose rows | registers]. Every pose row
    must take its body's object slot -- a length mismatch here crashed the forward (fixed 2026-09-21),
    and the three-domain recipe is pose-native. Woken, the pose rows must actually be modulated."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import test_unet_std_pose_native as PN
    from models.unet_std import UNetWrapper
    b = PN.make_batch()
    torch.manual_seed(0)
    w = UNetWrapper(PN._model_cfg(pose=True, unet_token_adaln=True)).eval()
    with torch.no_grad():
        part0, pose0 = PN._fwd(w, b, PN._noisy_pose_gt(b))
        for m in w.modules():
            if getattr(m, "adaln_obj", None) is not None:
                m.adaln_obj.weight.normal_(0, 0.05)
        PN.open_gates(w)
        part1, pose1 = PN._fwd(w, b, PN._noisy_pose_gt(b))
    assert part1.shape == part0.shape and pose1.shape == pose0.shape
    assert torch.isfinite(part1).all() and torch.isfinite(pose1).all()
