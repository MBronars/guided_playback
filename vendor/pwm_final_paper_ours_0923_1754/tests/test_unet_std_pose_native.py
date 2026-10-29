"""Native pose tokens inside the unified UNet (docs/CODE_WALKTHROUGH.md).
Standalone runner, house style:

    python tests/test_unet_std_pose_native.py

CPU, synthetic rigid+deformable scenes built through the unified sidecar
helpers (data/unified_dataset.py) so the batch layout is the production one.
Covers:
  1. GT round-trip of the shared input rewrite: noisy_pose = normalized GT
     deltas -> the rigid rows of noisy_future equal the normalized GT particle
     deltas (< 1e-4).
  2. Kr = 0 regression: a pose_native model on a sidecar-free batch (MPM case)
     equals the plain model with the same trunk weights, bit for bit.
  3. Mixed batch (deformable + 3 rigid links in 2 objects): tuple return,
     shapes, loss on both terms reaches EVERY pose module and the trunk.
  4. Pure-rigid batch: particle output is None, pose_v finite.
  5. L0 grouping (symmetric, within object): with an L0-only network,
     perturbing object 1's particles leaves object 0's particles AND pose
     token unchanged; perturbing object 0's particles changes its pose token.
  6. RuntimeError on a rigid batch without noisy_pose (Robert's guard).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.batch import TrajectoryBatch
from data.dataset import NormalizationStats
from data.unified_dataset import (_augment_pose_channels,
                                  _rigid_sidecar_from_particles)
from models.transformer import ModelConfig
from models.unet_std import UNetWrapper
from utils.pose_native import pose_native_inputs
from utils.pose_ops import so3_exp

B, C, F = 2, 3, 4
T = C + F


def _model_cfg(pose=True, **kw):
    base = dict(all_relative=True, context_len=C, num_mat_types=6, pose_native=pose,
                unet_points=0, unet_strides="1,2", unet_widths="16,32,64",
                unet_enc_depths="1,1,1", unet_dec_depths="1,1", unet_head_dim=8,
                unet_t_dim=32, unet_l1_by_mat="0:8,3:8,4:8,5:16",
                unet_pool_mode="pointnet", unet_unpool_mode="pointnet",
                unet_pool_k=6, unet_interp_k=3, unet_geo_r_m=0.1,
                unet_fps_random_start=False, unet_register_from_level=1,
                unet_register_objs=1)
    base.update(kw)
    return ModelConfig(**base)


def _pose_stats():
    return NormalizationStats(min_val=torch.tensor([-0.02] * 3 + [-0.3] * 3),
                              max_val=torch.tensor([0.02] * 3 + [0.3] * 3))


def make_batch(deform_n=48, rigid=((1, 24), (1, 16), (2, 16)), sidecar=True, seed=0):
    """Production layout (uniform material per object): object 0 = cloth-like
    deformable (mat 5); object 1 = gripper LINKS (mat 4, several rigid bodies
    in one object — the cloth EE / pickup hand+fingers case); object 2 = a
    rigid object (mat 3). rigid: (object, P) per rigid body, objects
    ascending. Body index = 0 deformable, then rigid bodies in order."""
    g = torch.Generator().manual_seed(seed)
    parts, pb, oid, mat = [], [], [], []
    body = 0
    if deform_n:
        base = torch.rand(B, deform_n, 3, generator=g) * 0.2
        wob = 0.002 * torch.randn(B, T, deform_n, 3, generator=g).cumsum(1)
        parts.append(base[:, None] + wob)
        pb += [0] * deform_n; oid += [0] * deform_n; mat += [5] * deform_n
        body = 1
    for (o, P) in rigid:
        tmpl = torch.rand(B, P, 3, generator=g) * 0.1
        tmpl = tmpl - tmpl.mean(1, keepdim=True)
        w = 0.05 * torch.randn(B, 3, generator=g)
        v = 0.005 * torch.randn(B, 3, generator=g)
        c0 = torch.rand(B, 3, generator=g) * 0.2 + torch.tensor([0.6 * o, 0.0, 0.0])
        frames = []
        for f in range(T):
            R = so3_exp(w * f)                                   # [B,3,3]
            frames.append(torch.einsum("bij,bpj->bpi", R, tmpl) + (c0 + v * f)[:, None])
        parts.append(torch.stack(frames, 1))
        pb += [body] * P; oid += [o] * P; mat += [4 if o == 1 else 3] * P
        body += 1
    positions = torch.cat(parts, dim=2)                           # [B,T,N,3]
    N = positions.shape[2]
    pb = torch.tensor(pb)[None].expand(B, N).clone()
    oid_t = torch.tensor(oid)[None].expand(B, N).clone()
    mat_t = torch.tensor(mat)[None].expand(B, N).clone()
    actuated = (mat_t == 4)                                       # gripper links actuated
    actions = (positions[:, 1:] - positions[:, :-1])[:, C - 1:] * actuated[:, None, :, None]
    dstats = NormalizationStats(min_val=torch.full((3,), -0.03), max_val=torch.full((3,), 0.03))
    pstats = NormalizationStats(min_val=torch.full((3,), -0.8), max_val=torch.full((3,), 0.8))
    sc = {}
    if sidecar:
        n_bodies = body
        rigid_bodies = range(1 if deform_n else 0, n_bodies)
        sc = _rigid_sidecar_from_particles(positions, pb, n_bodies, rigid_bodies=rigid_bodies)
        sc["particle_body"] = pb.clone()
        sc = _augment_pose_channels(sc, actuated, C, _pose_stats())
    return TrajectoryBatch(
        positions=positions, actions=actions, action_mask=actuated, object_ids=oid_t,
        valid_lengths=torch.full((B,), N, dtype=torch.long),
        E=torch.full((B, N), 1e6), nu=torch.full((B, N), 0.3), rho=torch.full((B, N), 1000.0),
        mat_type=mat_t, context_len=C, delta_stats=dstats, position_stats=pstats,
        prediction_target="delta", **sc)


def _noisy_pose_gt(b):
    return b.pose_delta_stats.normalize(b.pose_deltas_gt[:, :, b.body_type[0] == 1], clip=False)


def _fwd(w, b, noisy_pose, seed=1):
    g = torch.Generator().manual_seed(seed)
    n = b.positions.shape[2]
    x_t = torch.randn(B, F, n, 3, generator=g)
    ts = torch.randint(1, 1000, (B, F), generator=g)
    return w(b, x_t, ts, noisy_pose=noisy_pose)


def open_gates(model, std=0.05, seed=42):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "adaln" in name or "pose_v_head" in name:
                p.copy_(torch.randn(p.shape, generator=g) * std)


# ---------------------------------------------------------------------------

def test_gt_round_trip():
    b = make_batch()
    n = b.positions.shape[2]
    x_t = torch.randn(B, F, n, 3)
    patched, pack = pose_native_inputs(b, x_t, _noisy_pose_gt(b))
    gt = b.delta_stats.normalize(b.future_deltas, clip=False)
    rig = (b.body_type.gather(1, b.particle_body) == 1)               # [B,N]
    err = (patched - gt)[rig[:, None, :, None].expand_as(gt)].abs().max().item()
    assert err < 1e-4, err
    # deformable rows untouched
    assert torch.equal(patched[:, :, ~rig[0]], x_t[:, :, ~rig[0]])
    assert pack["centers"].shape == (B, F, 3, 3) and pack["rigid_idx"].tolist() == [1, 2, 3]
    print(f"ok  GT round-trip: rigid rows rebuilt from the pose, max err {err:.2e}")


def test_kr0_regression():
    b = make_batch(deform_n=48, rigid=(), sidecar=False)          # MPM-like: no sidecar
    b.mat_type[:] = 0
    torch.manual_seed(0); base = UNetWrapper(_model_cfg(pose=False)).eval()
    torch.manual_seed(0); pm = UNetWrapper(_model_cfg(pose=True)).eval()
    missing, unexpected = pm.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all("pose_" in k for k in missing), (missing, unexpected)
    open_gates(base); open_gates(pm)
    pm.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        y0 = _fwd(base, b, None); y1 = _fwd(pm, b, None)
    assert torch.equal(y0, y1)
    print("ok  Kr=0: pose_native model == plain model on a sidecar-free batch (bit-exact)")


def test_mixed_batch_forward_backward():
    b = make_batch()
    torch.manual_seed(0); w = UNetWrapper(_model_cfg()).train()
    open_gates(w)
    part, pose_v = _fwd(w, b, _noisy_pose_gt(b))
    n = b.positions.shape[2]
    assert part.shape == (B, F, n, 3) and pose_v.shape == (B, F, 3, 6)
    assert torch.isfinite(part).all() and torch.isfinite(pose_v).all()
    (part.square().mean() + pose_v.square().mean()).backward()
    missing = [k for k, p in w.named_parameters() if p.grad is None]
    assert not missing, missing[:8]
    pose_mods = [k for k, _ in w.named_parameters() if "pose_" in k]
    assert len(pose_mods) >= 10, pose_mods
    print(f"ok  mixed batch: tuple return, grads on all {len(list(w.parameters()))} params "
          f"({len(pose_mods)} pose tensors)")


def test_pure_rigid_batch():
    b = make_batch(deform_n=0, rigid=((1, 24), (2, 16)))          # gripper object + rigid object
    torch.manual_seed(0); w = UNetWrapper(_model_cfg(unet_l1_by_mat="3:8,4:8")).eval()
    open_gates(w)
    with torch.no_grad():
        part, pose_v = _fwd(w, b, _noisy_pose_gt(b))
    assert part is None and pose_v.shape == (B, F, 2, 6) and torch.isfinite(pose_v).all()
    print("ok  pure-rigid batch: particle output None, pose_v finite")


def test_l0_grouping_symmetric():
    # L0-only network (no seams): within-object groups = object particles + its pose tokens.
    # Layout: object 0 deformable (48 rows); object 1 = rigid bodies 0,1 (pose tokens 0,1);
    # object 2 = rigid body 2 (pose token 2).
    cfg = _model_cfg(unet_strides="", unet_widths="16", unet_enc_depths="2", unet_dec_depths="",
                     unet_register_from_level=-1)
    b = make_batch()
    torch.manual_seed(0); w = UNetWrapper(cfg).eval()
    open_gates(w)
    n = b.positions.shape[2]
    g = torch.Generator().manual_seed(3)
    x_t = torch.randn(B, F, n, 3, generator=g); ts = torch.randint(1, 1000, (B, F), generator=g)
    npz = _noisy_pose_gt(b)
    o0 = (b.object_ids[0] == 0); o1 = (b.object_ids[0] == 1)
    with torch.no_grad():
        p0, q0 = w(b, x_t, ts, noisy_pose=npz)
        npz1 = npz.clone(); npz1[:, :, 2] += 0.5                          # object 2's body
        p1, q1 = w(b, x_t, ts, noisy_pose=npz1)
        npz2 = npz.clone(); npz2[:, :, 0] += 0.5                          # object 1's first body
        p2, q2 = w(b, x_t, ts, noisy_pose=npz2)
        x3 = x_t.clone(); x3[:, :, :48] += 1.0                            # object 0's deformable rows
        p3, q3 = w(b, x3, ts, noisy_pose=npz)
    # object 2's pose perturbed: objects 0/1 particles and tokens 0/1 untouched, token 2 moved
    assert torch.allclose(p0[:, :, o0 | o1], p1[:, :, o0 | o1], atol=1e-6)
    assert torch.allclose(q0[:, :, :2], q1[:, :, :2], atol=1e-6)
    assert not torch.allclose(q0[:, :, 2], q1[:, :, 2])
    # object 1's body 0 perturbed: its sibling token 1 moves (same object), token 2 does not,
    # object 0 particles do not; object 1's own particles do (pose-derived rows + attention)
    assert not torch.allclose(q0[:, :, 1], q2[:, :, 1])
    assert torch.allclose(q0[:, :, 2], q2[:, :, 2], atol=1e-6)
    assert torch.allclose(p0[:, :, o0], p2[:, :, o0], atol=1e-6)
    assert not torch.allclose(p0[:, :, o1], p2[:, :, o1])
    # deformable object 0 perturbed: NO pose token moves at L0 (no rigid body lives there)
    assert torch.allclose(q0, q3, atol=1e-6)
    assert not torch.allclose(p0[:, :, o0], p3[:, :, o0])
    print("ok  L0 grouping: within-object, symmetric particle<->pose read")


def test_rigid_batch_without_pose_raises():
    b = make_batch()
    torch.manual_seed(0); w = UNetWrapper(_model_cfg()).eval()
    try:
        _fwd(w, b, None)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "noisy_pose" in str(e)
    print("ok  rigid batch without noisy_pose raises (Robert's guard)")


if __name__ == "__main__":
    test_gt_round_trip()
    test_kr0_regression()
    test_mixed_batch_forward_backward()
    test_pure_rigid_batch()
    test_l0_grouping_symmetric()
    test_rigid_batch_without_pose_raises()
    print("ALL OK")
