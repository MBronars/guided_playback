"""PhysCtrl-style physics aux losses for delta-target particle world models.

Three terms with historical calibration on an MPM-only UNet run (2026-08-04).
That calibration is provenance, not a recommendation for unified loss weights.

  strain_inc  L1 on per-frame incremental deformation gradients, SPATIAL form:
              per-particle 3x3 affine A fitted by ridge least-squares over
              same-object kNN neighborhoods whose indices AND gauge offsets
              come from the GT PREVIOUS-frame configuration (well-posed for
              granular flow — MPM's own current-configuration grad-v). All
              materials. Supervises deformation RATES (impact compression,
              rebound timing, sand spreading rate).

  strain_cum  Same fit on the context-end -> final-frame displacement supplied
              by the objective, over the FIRST context-frame graph. Gated to materials whose
              deformation-from-reference is physically meaningful (default
              elastic+plasticine; sand's plastic flow forgets its reference).
              The anti-shrivel term: a per-frame contraction bias too small
              for per-frame losses (~0.6%/frame) integrates to an order-one
              cumulative strain error.

  floor       Hinge relu(floor_z - z)^2 on integrated predicted positions
              (anchor + cumsum of denormalized deltas). The low-level default
              0.091 belongs to historical unshifted MPM data; fresh unified
              recipes must explicitly choose floor_z=0 for the canonical plane.

All terms are computed in CLEAN space: GT geometry for graphs/offsets, the
model's x0 prediction for the supervised field — noise never enters the loss.
Targets are (near) mode-invariant, so they are applied uniformly over t.

Requires the mpm_multi layout: contiguous equal-size per-object particle
blocks, all particles valid, prediction_target == "delta".
"""

from __future__ import annotations

import torch

K_NN = 16
RIDGE = 1e-4          # scale-aware: multiplied by mean squared neighbor offset
DEFAULT_FLOOR_Z = 0.091
DEFAULT_STRAIN_CUM_MATS = (0, 1)   # elastic, plasticine


def _object_layout(batch):
    """Contiguous equal-block object layout (asserted) -> (O, P)."""
    B, _, N, _ = batch.positions.shape
    obj = batch.object_ids
    O = int(torch.unique(obj[0]).numel())
    P = N // O
    assert O * P == N, f"N={N} not divisible into {O} objects"
    blocks = obj.reshape(B, O, P)
    assert (blocks == blocks[:, :, :1]).all(), \
        "phys aux losses require contiguous equal-size per-object blocks"
    assert (batch.valid_lengths == N).all(), \
        "phys aux losses do not support padded batches"
    return O, P


def _ls_system(points, k):
    """points [M,P,3] -> (idx [M,P,k], dp [M,P,k,3], chol [M,P,3,3]).

    Same-group kNN (self excluded) + Cholesky of the ridge-regularized
    normal matrix. Ridge is scaled by the neighborhood size so the
    regularization is sampling-density-invariant."""
    from pytorch3d.ops import knn_points

    M, P, _ = points.shape
    _, idx, _ = knn_points(points, points, K=k + 1)
    idx = idx[:, :, 1:]
    gather = idx.reshape(M, P * k, 1).expand(-1, -1, 3)
    dp = points.gather(1, gather).reshape(M, P, k, 3) - points[:, :, None]
    cov = torch.einsum("mpki,mpkj->mpij", dp, dp)
    # Ridge: relative (density-invariant) + ABSOLUTE floor. The floor keeps
    # cov strictly PD even for fully-degenerate neighborhoods (>= k+1 exactly
    # coincident particles — observed in compacted GT sand; killed job
    # 9726012 via a cholesky not-positive-definite hard error). Degenerate
    # neighborhoods then fit A=0 on both sides: zero loss, no gradient — the
    # correct treatment, since coincident points carry no strain information.
    lam = (RIDGE * dp.pow(2).sum(-1).mean(-1) + 1e-8)[..., None, None]
    cov = cov + lam * torch.eye(3, device=points.device, dtype=points.dtype)
    return idx, dp, torch.linalg.cholesky(cov.float())


def _fit_affine(disp, idx, dp, chol):
    """disp [M,P,3] on the graph (idx,dp,chol) -> A [M,P,3,3] minimizing
    || (disp_j - disp_i) - A dp_j ||^2 + ridge."""
    M, P, _ = disp.shape
    k = idx.shape[-1]
    nbrd = disp.gather(1, idx.reshape(M, P * k, 1).expand(-1, -1, 3)) \
        .reshape(M, P, k, 3)
    dd = nbrd - disp[:, :, None]
    cross = torch.einsum("mpki,mpkj->mpij", dd.float(), dp.float())
    A = torch.cholesky_solve(cross.transpose(-1, -2).reshape(-1, 3, 3),
                             chol.reshape(-1, 3, 3))
    return A.transpose(-1, -2).reshape(M, P, 3, 3)


def strain_inc_loss(batch, pred_deltas, gt_deltas=None):
    """Spatial-form incremental strain L1, all materials.

    pred_deltas [B,F,N,3]: predicted per-frame displacements in METERS.
    gt_deltas defaults to batch.future_deltas.
    Graphs and gauge offsets from GT frame s-1 positions (ctx end for s=0)."""
    O, P = _object_layout(batch)
    pos_prev = torch.cat([batch.context[:, -1:], batch.future[:, :-1]], dim=1)
    B, F, N, _ = pos_prev.shape
    p = pos_prev.reshape(B, F, O, P, 3).reshape(B * F * O, P, 3)
    idx, dp, chol = _ls_system(p, K_NN)

    def per_frame(disp):
        return _fit_affine(
            disp.reshape(B, F, O, P, 3).reshape(B * F * O, P, 3), idx, dp, chol)

    A_pred = per_frame(pred_deltas)
    A_gt = per_frame(batch.future_deltas if gt_deltas is None else gt_deltas)
    return (A_pred - A_gt).abs().mean()


def strain_cum_loss(batch, pred_disp, gt_disp, mats=DEFAULT_STRAIN_CUM_MATS):
    """Frame-0-referenced cumulative strain L1 at the final frame, gated to
    materials with a physically meaningful reference (default elastic+plast).

    pred_disp/gt_disp [B,N,3]: accumulated displacement fields in METERS.
    The unified objective supplies future-delta sums (context-end to final).
    The first-context graph is intentional: its affine fit is linear in the
    displacement, so adding the same context motion to prediction and target
    cancels from A_pred - A_gt. Correct rigid forecasts incur zero loss.
    Returns a zero scalar (grad-connected) if no gated object is present."""
    O, P = _object_layout(batch)
    B, N, _ = pred_disp.shape
    p0 = batch.context[:, 0].reshape(B * O, P, 3)
    idx, dp, chol = _ls_system(p0, K_NN)

    def cum(disp):
        return _fit_affine(disp.reshape(B * O, P, 3), idx, dp, chol)

    mat_obj = batch.mat_type.reshape(B, O, P)[:, :, 0].reshape(B * O)
    mask = torch.zeros(B * O, dtype=torch.bool, device=pred_disp.device)
    for m in mats:
        mask |= mat_obj == int(m)
    if not mask.any():
        return pred_disp.sum() * 0.0
    A_pred = cum(pred_disp)[mask]
    A_gt = cum(gt_disp)[mask]
    return (A_pred - A_gt).abs().mean()


def floor_loss(batch, pred_raw, floor_z=DEFAULT_FLOOR_Z, valid_mask=None):
    """Hinge on valid predicted particle positions below the floor plane.

    The objective passes its deformable-only [B,F,N] mask; direct callers
    default to the batch's prefix-valid particles. Select before nonlinear
    arithmetic so NaNs in padding never create NaN gradients.
    """
    pos_pred = batch.predictions_to_positions(pred_raw)
    if valid_mask is None:
        valid_mask = batch.get_valid_mask()[:, None].expand(pos_pred.shape[:3])
    if valid_mask.shape != pos_pred.shape[:3] or valid_mask.dtype != torch.bool:
        raise ValueError("floor valid_mask must be boolean [B,F,N]")
    z = pos_pred[..., 2][valid_mask]
    if z.numel() == 0:
        return z.sum()  # zero, still connected to the predicted displacement
    return (torch.relu(floor_z - z) ** 2).mean()
