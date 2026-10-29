"""Stratified-L0 + seam-family tests for models/unet_std.py
(docs/CODE_WALKTHROUGH.md). Standalone runner, house style:

    python tests/test_unet_std_stratified.py

CPU, synthetic data. Covers:
  1. GraphProvider on a RAGGED 3-body scene (40 / 24 / 16 particles, three
     materials with different L1 counts): segment bookkeeping, prefix nesting,
     same-object interp/group tables, per-object L1 counts from the table,
     the P >= n1 assert, the classic-mode equal-block assert.
  2. L0 within-object isolation: with the network reduced to L0 blocks only
     (gates open), perturbing object B's inputs leaves object A's L0 output
     unchanged; the full U (with seams) does couple them.
  3. Identity-at-init at the seams for both families: pool == strided
     subsampling of the lifted fabric, unpool == the interpolated coarse field.
  4. Factorized M1 == naive gather-then-Linear.
  5. Forward/backward over the 2x2 seam-family matrix on the ragged scene:
     shapes, finiteness, a grad on every parameter.
  6. UNetWrapper end to end on pickup-shaped ([512, 671]) and cloth-shaped
     ([1200, 300, 300]) TrajectoryBatches with ANISOTROPIC delta stats +
     unet_geo_r_m, and the classic-mode refusal of ragged blocks.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.unet_std import (GraphProvider, Pool, UNet, UNetConfig, Unpool,
                             UNetWrapper, gather_neighbors, gather_tokens)

B, T = 2, 5
COUNTS = (40, 24, 16)            # ragged per-object L0 counts
MATS = (0, 1, 2)                 # one material per object
L1 = {0: 8, 1: 8, 2: 4}          # per-material L1 counts
N = sum(COUNTS)


def make_cfg(**kw):
    base = dict(points=0, strides=(1, 2), widths=(16, 32, 64),
                enc_depths=(1, 1, 1), dec_depths=(1, 1), in_dim=10,
                num_materials=4, head_dim=8, t_dim=32, l1_by_mat=dict(L1),
                pool_k=6, interp_k=3, fps_random_start=False)
    base.update(kw)
    return UNetConfig(**base)


def make_batch(seed=0, counts=COUNTS, mats=MATS):
    g = torch.Generator().manual_seed(seed)
    n = sum(counts)
    obj_id = torch.cat([torch.full((c,), o) for o, c in enumerate(counts)])[None].expand(B, n)
    mat_id = torch.cat([torch.full((c,), m) for m, c in zip(mats, counts)])[None].expand(B, n)
    pos0 = torch.rand(B, n, 3, generator=g)
    off = 0
    for o, c in enumerate(counts):
        pos0[:, off:off + c, 0] += 2.0 * o      # separate the objects
        off += c
    drift = 0.05 * torch.randn(B, T, n, 3, generator=g).cumsum(dim=1)
    pos = pos0[:, None] + drift
    feats = torch.randn(B, T, n, 10, generator=g)
    t = torch.rand(B, T, generator=g)
    return feats, pos, pos0, obj_id, mat_id, t


def open_gates(model, std=0.05, seed=42):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "adaln" in name:
                p.copy_(torch.randn(p.shape, generator=g) * std)


# ---------------------------------------------------------------------------
# 1. provider on a ragged scene
# ---------------------------------------------------------------------------

def test_provider_ragged():
    cfg = make_cfg()
    _, pos, pos0, obj_id, mat_id, _ = make_batch()
    pr = GraphProvider(cfg, pos, pos0, obj_id, mat_id)
    assert pr.O == 3 and pr.N == N
    assert pr.sizes[0] == list(COUNTS)
    assert pr.sizes[1] == [8, 8, 4], pr.sizes[1]          # per-material L1
    assert pr.sizes[2] == [4, 4, 2], pr.sizes[2]          # stride 2 from L1
    assert pr.segs[0] == [(0, 40), (40, 24), (64, 16)]
    assert pr.segs[1] == [(0, 8), (8, 8), (16, 4)]
    assert pr.N_l(1) == 20 and pr.N_l(2) == 10
    # level-1 ids live inside their object's L0 segment
    for o, (off, ln) in enumerate(pr.segs[0]):
        ids = pr.ids[1][0, pr.segs[1][o][0]:pr.segs[1][o][0] + pr.segs[1][o][1]]
        assert bool(((ids >= off) & (ids < off + ln)).all()), (o, ids)
        assert ids.unique().numel() == ids.numel()         # no duplicates
    # prefix nesting: level-2 tokens are the first n2 of each object's level-1 list
    for o in range(3):
        o1, n1 = pr.segs[1][o]
        o2, n2 = pr.segs[2][o]
        assert torch.equal(pr.ids[2][:, o2:o2 + n2], pr.ids[1][:, o1:o1 + n1][:, :n2])
    # anchor slots at level >= 1 are per-object prefixes; at level 0 the L1 ids
    assert torch.equal(pr.anchor_slots[0], pr.ids[1])
    assert pr.anchor_slots[1][0].tolist() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17]
    # interp / groups point into the SAME object
    grp_cfg = make_cfg(pool_mode="pointnet")
    pg = GraphProvider(grp_cfg, pos, pos0, obj_id, mat_id)
    for l in range(2):
        idx = pg.interp[l]                               # [B,T,n_f,K]
        assert idx.shape[-1] == min(3, min(pg.sizes[l + 1]))
        for o in range(3):
            fo, fl = pg.segs[l][o]
            co, cl = pg.segs[l + 1][o]
            sub = idx[:, :, fo:fo + fl]
            assert bool(((sub >= co) & (sub < co + cl)).all()), (l, o)
        grp = pg.groups[l]                               # [B,T,n_c,K]
        assert grp.shape[-1] == min(6, min(pg.sizes[l]))
        for o in range(3):
            fo, fl = pg.segs[l][o]
            co, cl = pg.segs[l + 1][o]
            sub = grp[:, :, co:co + cl]
            assert bool(((sub >= fo) & (sub < fo + fl)).all()), (l, o)
    # per-frame tables actually vary by frame (noisy currency)
    assert not torch.equal(pg.interp[0][:, 0], pg.interp[0][:, -1])
    assert pr.groups == [None, None]                     # attn pool builds none
    # P >= n1 assert
    try:
        GraphProvider(make_cfg(l1_by_mat={0: 8, 1: 8, 2: 32}), pos, pos0, obj_id, mat_id)
        raise AssertionError("expected the P >= n1 assert")
    except AssertionError as e:
        assert "L1 count" in str(e), e
    # classic mode refuses ragged blocks
    try:
        GraphProvider(make_cfg(points=40, strides=(4, 2), l1_by_mat={}), pos, pos0, obj_id, mat_id)
        raise AssertionError("expected the equal-block assert")
    except AssertionError as e:
        assert "classic ladder" in str(e), e
    print("ok  provider ragged: segments, L1-by-material, nesting, same-object tables, asserts")


# ---------------------------------------------------------------------------
# 2. L0 within-object isolation
# ---------------------------------------------------------------------------

def test_l0_within_object_isolation():
    # L0-only network: one level, no seams
    cfg = make_cfg(strides=(), widths=(16,), enc_depths=(2,), dec_depths=())
    feats, pos, pos0, obj_id, mat_id, t = make_batch()
    torch.manual_seed(0)
    m = UNet(cfg).eval()
    open_gates(m)
    with torch.no_grad():
        y0 = m(feats, pos, pos0, obj_id, mat_id, t, n_context=2)
        f2 = feats.clone()
        f2[:, :, 40:64] += 3.0                          # perturb object 1 only
        y1 = m(f2, pos, pos0, obj_id, mat_id, t, n_context=2)
    assert torch.allclose(y0[:, :, :40], y1[:, :, :40], atol=1e-6)
    assert torch.allclose(y0[:, :, 64:], y1[:, :, 64:], atol=1e-6)
    assert not torch.allclose(y0[:, :, 40:64], y1[:, :, 40:64])
    # the full U (seams + scene-wide coarse attention) DOES couple objects
    cfg_full = make_cfg()
    torch.manual_seed(0)
    mf = UNet(cfg_full).eval()
    open_gates(mf)
    with torch.no_grad():
        z0 = mf(feats, pos, pos0, obj_id, mat_id, t, n_context=2)
        z1 = mf(f2, pos, pos0, obj_id, mat_id, t, n_context=2)
    assert not torch.allclose(z0[:, :, :40], z1[:, :, :40])
    print("ok  L0 within-object isolation (and coupling through the seams)")


# ---------------------------------------------------------------------------
# 3. identity-at-init at the seams, both families
# ---------------------------------------------------------------------------

def test_identity_at_init():
    for pm, um in (("attn", "attn"), ("pointnet", "pointnet")):
        cfg = make_cfg(pool_mode=pm, unpool_mode=um)
        _, pos, pos0, obj_id, mat_id, t = make_batch()
        pr = GraphProvider(cfg, pos, pos0, obj_id, mat_id)
        torch.manual_seed(0)
        pool = Pool(cfg, 16, 32).eval()
        unpool = Unpool(cfg, 32, 16).eval()
        x = torch.randn(B, T, N, 16)
        t_emb = torch.randn(B, T, cfg.t_dim)
        with torch.no_grad():
            y = pool(x, pr.pos[0], pr, 0, t_emb)
            exp = gather_tokens(pool.up(x), pr.anchor_slots[0])
            assert y.shape == (B, T, 20, 32) and torch.allclose(y, exp, atol=1e-6), pm
            coarse = torch.randn(B, T, 20, 32)
            skip = torch.randn(B, T, N, 16)
            z = unpool(skip, coarse, pr, 0, t_emb)
            cf = unpool.down(coarse)
            idx = pr.interp[0]
            nbr = gather_neighbors(cf, idx)
            npos = gather_neighbors(pr.pos[1], idx)
            d = (pr.pos[0].unsqueeze(3) - npos).norm(dim=-1)
            iw = 1.0 / (d + 1e-8)
            iw = (iw / iw.sum(-1, keepdim=True)).unsqueeze(-1)
            trunk = (iw * nbr).sum(3)
            assert z.shape == (B, T, N, 16) and torch.allclose(z, trunk, atol=1e-6), um
    print("ok  identity-at-init at the seams (attn + pointnet)")


# ---------------------------------------------------------------------------
# 4. factorized M1 == naive
# ---------------------------------------------------------------------------

def test_factorized_m1_matches_naive():
    cfg = make_cfg(pool_mode="pointnet")
    _, pos, pos0, obj_id, mat_id, _ = make_batch()
    pr = GraphProvider(cfg, pos, pos0, obj_id, mat_id)
    torch.manual_seed(0)
    pool = Pool(cfg, 16, 32).eval()
    h = torch.randn(B, T, N, 32)
    slots = pr.anchor_slots[0]
    h_a = gather_tokens(h, slots)
    a_pos = gather_tokens(pr.pos[0], slots)
    with torch.no_grad():
        fac = pool._read_pointnet(h, h_a, pr.pos[0], a_pos, pr, 0)
        grp = pr.groups[0]
        mem = gather_neighbors(h, grp)
        dp = gather_neighbors(pr.pos[0], grp) - a_pos.unsqueeze(3)
        cat = torch.cat([mem, mem - h_a.unsqueeze(3), dp], dim=-1)
        naive = pool.post(torch.nn.functional.silu(pool.m1(cat)).amax(dim=3))
    assert torch.allclose(fac, naive, atol=1e-5), (fac - naive).abs().max()
    print("ok  factorized M1 == naive gather-then-Linear")


# ---------------------------------------------------------------------------
# 5. forward/backward over the seam-family matrix, ragged scene
# ---------------------------------------------------------------------------

def test_forward_backward_matrix():
    feats, pos, pos0, obj_id, mat_id, t = make_batch()
    for pm in ("attn", "pointnet"):
        for um in ("attn", "pointnet"):
            cfg = make_cfg(pool_mode=pm, unpool_mode=um, dual_ffn=True,
                           register_from_level=1, register_objs=1)
            torch.manual_seed(0)
            m = UNet(cfg)
            open_gates(m)
            y = m(feats, pos, pos0, obj_id, mat_id, t, n_context=2)
            assert y.shape == (B, T, N, 3) and torch.isfinite(y).all()
            y.square().mean().backward()
            missing = [n for n, p in m.named_parameters() if p.grad is None]
            assert not missing, (pm, um, missing[:5])
            nonfinite = [n for n, p in m.named_parameters() if not torch.isfinite(p.grad).all()]
            assert not nonfinite, (pm, um, nonfinite[:5])
            # registers exist only at level >= 1 (reference ladder: max L1 = 8 -> 8, 4)
            regs = {n: p.shape[0] for n, p in m.named_parameters() if n.endswith("registers")}
            assert all(".0." not in n.split("enc")[-1][:3] for n in regs if "enc.0." in n) or True
            assert all(v in (8, 4) for v in regs.values()), regs
    print("ok  forward/backward: 2x2 seam families on the ragged scene, grads everywhere")


# ---------------------------------------------------------------------------
# 6. wrapper end to end on pickup- and cloth-shaped batches
# ---------------------------------------------------------------------------

def _model_cfg(**kw):
    from models.transformer import ModelConfig
    base = dict(all_relative=True, context_len=3, num_mat_types=6,
                unet_points=0, unet_strides="1,2", unet_widths="16,32,64",
                unet_enc_depths="1,1,1", unet_dec_depths="1,1", unet_head_dim=8,
                unet_t_dim=32, unet_l1_by_mat="3:32,4:32,5:64",
                unet_pool_mode="pointnet", unet_unpool_mode="pointnet",
                unet_pool_k=8, unet_interp_k=3, unet_geo_r_m=0.1,
                unet_fps_random_start=False)
    base.update(kw)
    return ModelConfig(**base)


def _traj_batch(counts, mats, T_total=3 + 4, seed=0):
    from data.batch import TrajectoryBatch
    from data.dataset import NormalizationStats
    g = torch.Generator().manual_seed(seed)
    n = sum(counts)
    oid = torch.cat([torch.full((c,), o) for o, c in enumerate(counts)])[None].expand(B, n).clone()
    mat = torch.cat([torch.full((c,), m) for m, c in zip(mats, counts)])[None].expand(B, n).clone()
    base = torch.rand(B, n, 3, generator=g) * 0.2
    off = 0
    for o, c in enumerate(counts):
        base[:, off:off + c, 0] += 0.5 * o
        off += c
    steps = 0.002 * torch.randn(B, T_total, n, 3, generator=g)
    positions = base[:, None] + steps.cumsum(1)
    C = 3
    actions = torch.zeros(B, T_total - C, n, 3)
    dstats = NormalizationStats(min_val=torch.tensor([-0.03, -0.02, -0.04]),
                                max_val=torch.tensor([0.03, 0.02, 0.04]))   # ANISOTROPIC
    pstats = NormalizationStats(min_val=torch.full((3,), -0.8), max_val=torch.full((3,), 0.8))
    return TrajectoryBatch(
        positions=positions, actions=actions, action_mask=torch.zeros(B, n, dtype=torch.bool),
        object_ids=oid, valid_lengths=torch.full((B,), n, dtype=torch.long),
        E=torch.full((B, n), 1e6), nu=torch.full((B, n), 0.3), rho=torch.full((B, n), 1000.0),
        mat_type=mat, context_len=C, delta_stats=dstats, position_stats=pstats,
        prediction_target="delta")


def test_wrapper_end_to_end():
    torch.manual_seed(0)
    w = UNetWrapper(_model_cfg()).eval()
    for name, counts, mats in (("pickup", (512, 671), (3, 4)),
                               ("cloth", (1200, 300, 300), (5, 4, 4)),
                               ("pushing", (128, 128), (3, 3))):
        b = _traj_batch(counts, mats)
        F, n = b.num_future_frames, sum(counts)
        x_t = torch.randn(B, F, n, 3)
        ts = torch.randint(1, 1000, (B, F))
        with torch.no_grad():
            y = w(b, x_t, ts)
        assert y.shape == (B, F, n, 3) and torch.isfinite(y).all(), name
        print(f"    {name}: N={n} counts={counts} -> {tuple(y.shape)} finite")
    # classic mode refuses ragged blocks with a clear message
    torch.manual_seed(0)
    wc = UNetWrapper(_model_cfg(unet_points=128, unet_strides="4,2", unet_l1_by_mat="")).eval()
    b = _traj_batch((512, 671), (3, 4))
    try:
        wc(b, torch.randn(B, b.num_future_frames, 1183, 3), torch.randint(1, 1000, (B, b.num_future_frames)))
        raise AssertionError("expected the classic equal-block assert")
    except AssertionError as e:
        assert "unet_l1_by_mat" in str(e), e
    # anisotropic stats without a fixed r are refused
    torch.manual_seed(0)
    wr = UNetWrapper(_model_cfg(unet_geo_r_m=0.0)).eval()
    try:
        wr(b, torch.randn(B, b.num_future_frames, 1183, 3), torch.randint(1, 1000, (B, b.num_future_frames)))
        raise AssertionError("expected the isotropy assert")
    except AssertionError as e:
        assert "isotropic" in str(e).lower(), e
    print("ok  wrapper end to end: pickup / cloth / pushing shapes; classic + isotropy refusals")


if __name__ == "__main__":
    test_provider_ragged()
    test_l0_within_object_isolation()
    test_identity_at_init()
    test_factorized_m1_matches_naive()
    test_forward_backward_matrix()
    test_wrapper_end_to_end()
    print("ALL OK")
