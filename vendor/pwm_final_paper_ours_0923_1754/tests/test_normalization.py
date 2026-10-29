"""Supported physical-unit normalizers and real-data representation contracts."""
import os
import pickle

import numpy as np
import pytest
import torch

from data.normalization import (NORMALIZATION_PROFILES, NormalizationStats,
                                domain_normalizers, normalization_profile)

SUPPORTED = NORMALIZATION_PROFILES
RIGID = ("pickup", "cloth", "pushing")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_linear_arithmetic_is_checkpoint_compatible(dtype):
    stats = NormalizationStats(torch.tensor([-0.03, -0.02, -0.1]),
                               torch.tensor([0.04, 0.02, 0.1]))
    x = torch.randn(2, 4, 8, 3).to(dtype) * 0.005
    expected = 2 * (x - stats.min_val) / (stats.max_val - stats.min_val + 1e-8) - 1
    torch.testing.assert_close(stats.normalize(x), expected, rtol=0, atol=0)
    y = stats.normalize(x)
    expected_back = (y + 1) / 2 * (stats.max_val - stats.min_val) + stats.min_val
    torch.testing.assert_close(stats.denormalize(y), expected_back, rtol=0, atol=0)


def test_clipping_is_explicit_and_raw_units_round_trip():
    stats = NormalizationStats(-torch.ones(3), torch.ones(3))
    x = torch.tensor([[2.0, -2.0, 0.03]])
    assert stats.normalize(x).abs().max() > 1
    torch.testing.assert_close(stats.normalize(x, clip=True), x.clamp(-1, 1))
    torch.testing.assert_close(stats.denormalize(stats.normalize(x)), x)


def test_legacy_pickle_class_path_still_resolves():
    assert pickle.loads(b"cdata.dataset\nNormalizationStats\n.") is NormalizationStats


def test_load_all_returns_only_present_representations(tmp_path):
    path = tmp_path / "linear.npz"
    np.savez(path, min_delta=-np.ones(3), max_delta=np.ones(3))
    delta, position, offset = NormalizationStats.load_all(path)
    assert isinstance(delta, NormalizationStats)
    assert position is None and offset is None


def test_nonlinear_native_assets_cannot_silently_become_linear(tmp_path):
    path = tmp_path / "retired.npz"
    np.savez(path, min_delta=-np.ones(3), max_delta=np.ones(3), delta_norm_kind="asinh")
    with pytest.raises(ValueError, match="Retired normalization.*55c386c"):
        NormalizationStats.load_all(path)


@pytest.mark.parametrize("name", ["per-domain", "iso_p9999", "global_iso_p9999",
                                  "unified_asinh", "pd_asinh"])
def test_retired_sets_fail_with_recovery_instructions(name):
    with pytest.raises(ValueError, match="Retired unified stats.*55c386c"):
        normalization_profile("unified://" + name)


def test_sets_are_registered():
    for s in SUPPORTED:
        assert normalization_profile(f"unified://{s}") == s


def _have_data():
    from data.unified_dataset import (CLOTH_WDS_ROOT, MPM_VAL_ROOT, PICKUP_ROOTS,
                                      PUSHING_ROOT)
    return all(os.path.isdir(p) for p in [CLOTH_WDS_ROOT, MPM_VAL_ROOT, PUSHING_ROOT,
                                          *PICKUP_ROOTS])


@pytest.mark.parametrize("profile", SUPPORTED)
def test_xy_p99_bounds_share_x_and_y_and_are_symmetric(profile):
    for dom in ("mpm", *RIGID):
        particle, _, pose = domain_normalizers(profile, dom)
        assert torch.equal(particle.min_val, -particle.max_val)
        assert particle.max_val[0] == particle.max_val[1]
        assert particle.max_val[2] > 0
        if pose is not None:
            assert torch.equal(pose.min_val, -pose.max_val)
            h = pose.max_val
            assert h[0] == h[1] and h[3] == h[4]


# ------------------------------------------------------- 4. a real batch through the model
@pytest.mark.skipif(not _have_data(), reason="external unified research corpora are unavailable")
@pytest.mark.parametrize("setname,domain", [(s, d) for s in SUPPORTED
                                            for d in ("pushing", "mpm")])
@pytest.mark.integration
def test_forward_backward_on_a_real_unified_batch(setname, domain):
    from data.unified_dataset import build_domain_dataset
    from diffusion import TrajectoryFlowMatching
    from models.factory import build_particle_model
    from train import trajectory_loss_options, trajectory_objective
    from utils.config import load_config

    cfg = load_config("configs/transformer.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = build_particle_model(cfg).to(device)
    ds = build_domain_dataset(domain, "val", training=False, normalization=setname,
                              context_len=cfg.data.context_len, future_len=cfg.data.future_len)
    ds.batch_size = 1
    ds.emit_sidecar = domain != "mpm"
    batch = next(iter(ds)).to(device)
    diffusion = TrajectoryFlowMatching(prediction_type=cfg.flow_matching.prediction_type)
    loss, _, _ = trajectory_objective(model, batch, diffusion, **trajectory_loss_options(cfg))
    assert torch.isfinite(loss) and loss > 0, f"{setname}/{domain}: invalid objective"
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients), f"{setname}: non-finite grad"
    assert any(torch.count_nonzero(g) > 0 for g in gradients), "Zero-head square loss is not a backward test"


# ------------------------------------- 5. the rigid double representation stays consistent
# A rigid body is written down twice: as its particles (delta_stats, INPUT ONLY -- the
# particle loss is masked off rigid rows) and as a 6-D pose (pose_delta_stats, the target).
# They cannot drift, because the particle rows are a pure function of the pose on both
# sides -- utils.pose_native.pose_native_inputs on the way in, eval_unified.
# materialize_pose_native on the way out. These two tests pin that.

@pytest.mark.skipif(not _have_data(), reason="external unified research corpora are unavailable")
@pytest.mark.parametrize("setname", SUPPORTED)
@pytest.mark.integration
def test_rigid_rows_are_rendered_from_the_pose_and_nothing_else(setname):
    """Rendering the rigid rows from the clean pose must reproduce the stored particle
    deltas, and must leave every deformable row untouched.

    The pose is normalized with clip=False here on purpose: this test is about the two
    representations being CONSISTENT (the particle rows are a pure function of the pose),
    not about the scale being wide enough. The second thing is what
    test_pose_scale_can_represent_the_windows_evaluation_draws measures, and mixing them
    hides a real inconsistency behind an expected clip.
    """
    from data.unified_dataset import build_domain_dataset
    from utils.pose_native import pose_native_inputs
    for dom, tol_mm in (("pushing", 0.01), ("pickup", 0.01), ("cloth", 0.01)):
        ds = build_domain_dataset(dom, "val", training=False, normalization=setname)
        ds.batch_size, ds.emit_sidecar = 4, True
        b = next(iter(ds))
        ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
        valid = b.get_valid_mask()
        rigid = torch.zeros_like(valid)
        for e in range(b.positions.shape[0]):
            for kk in ridx.tolist():
                rigid[e] |= (b.particle_body[e] == kk) & valid[e]
        prev = torch.cat([b.anchor[:, None], b.positions[:, b.context_len:-1]], 1)
        gt = b.delta_stats.normalize(b.future - prev, clip=True)
        pose = b.pose_delta_stats.normalize(b.pose_deltas_gt[:, :, ridx], clip=False)
        patched, _ = pose_native_inputs(b, gt.clone(), pose)
        m = rigid[:, None, :, None].expand_as(gt)
        err = (b.delta_stats.denormalize(patched) - (b.future - prev))[m].abs().max() * 1e3
        assert float(err) < tol_mm, f"{setname}/{dom}: rendered rigid rows off by {err:.4f} mm"
        assert torch.equal(patched[~m], gt[~m]), f"{setname}/{dom}: deformable rows were touched"


@pytest.mark.skipif(not _have_data(), reason="external unified research corpora are unavailable")
@pytest.mark.parametrize("setname", SUPPORTED)
@pytest.mark.integration
def test_pose_scale_can_represent_the_windows_evaluation_draws(setname):
    """Clipped-target reconstruction: normalize(clip=True) -> denormalize ->
    integrate, on the window distribution EVAL uses, and check where the body lands.

    This is one reconstruction error, not a lower bound for arbitrary predictions.
    It caught a real defect: cloth's training loader samples window starts in proportion to
    cloth motion (UnifiedClothDataset._motion_window, training only), which suppresses
    gripper ROTATION by 15-83x while leaving every particle channel within 10%. A pose scale
    fitted only to that stream clipped 61% of cloth's rotation components and left 9.7 deg
    reconstruction error after 16 frames. Both the frozen profiles and the new
    data.stats_fitting pipeline take pose bounds over the selected training and
    prefix policies, without fitting validation episodes.
    """
    from data.unified_dataset import build_domain_dataset
    from utils.pose_ops import so3_exp, so3_log
    for dom in RIGID:
        ds = build_domain_dataset(dom, "val", training=False, normalization=setname)
        ds.batch_size, ds.emit_sidecar = 8, True
        b = next(iter(ds))
        ridx = (b.body_type[0] == 1).nonzero(as_tuple=True)[0]
        gt = b.pose_deltas_gt[:, :, ridx].float()
        back = b.pose_delta_stats.denormalize(
            b.pose_delta_stats.normalize(gt, clip=True)).float()
        B, F, Kr, _ = gt.shape
        Rt = torch.eye(3).expand(B, Kr, 3, 3).clone()
        Rc, tt, tc = Rt.clone(), torch.zeros(B, Kr, 3), torch.zeros(B, Kr, 3)
        for f in range(F):
            Rt = so3_exp(gt[:, f, :, 3:].reshape(-1, 3)).reshape(B, Kr, 3, 3) @ Rt
            Rc = so3_exp(back[:, f, :, 3:].reshape(-1, 3)).reshape(B, Kr, 3, 3) @ Rc
            tt, tc = tt + gt[:, f, :, :3], tc + back[:, f, :, :3]
        mm = float(((tt - tc).norm(dim=-1) * 1e3).mean())
        deg = float(torch.rad2deg(
            so3_log((Rt @ Rc.transpose(-1, -2)).reshape(-1, 3, 3)).norm(dim=-1)).mean())
        # generous thresholds: a p99 scale clips ~2-3% by design. What must never come back
        # is the 9.7 deg / 2.5 mm regime a window-skewed scale produced.
        assert mm < 2.0, f"{setname}/{dom}: {mm:.3f} mm clipped-target reconstruction error"
        assert deg < 1.5, f"{setname}/{dom}: {deg:.4f} deg clipped-target reconstruction error"
