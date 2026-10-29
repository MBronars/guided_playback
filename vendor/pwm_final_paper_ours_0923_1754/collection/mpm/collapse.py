"""Per-object saved-state collapse diagnostics, independent of constitutive J.

Provisional quarantine thresholds, not material calibration. Translation and
rotation do not count as contraction; becoming flat does not imply point collapse.
Plastic return mapping means det(F_trial) is NOT total physical volume change.
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CollapsePolicy:
    version: str = 'saved-object-collapse-v1'
    minimum_radius_ratio: float = .10
    warning_radius_ratio: float = .50
    point_radius_m: float = 1e-5
    minimum_det: float = 1e-3
    minimum_singular_value: float = 1e-4
    degenerate_particle_fraction: float = .10

    def __post_init__(self):
        values = list(asdict(self).values())[1:]
        if (not np.isfinite(values).all() or self.version != 'saved-object-collapse-v1'
                or not 0 < self.minimum_radius_ratio < self.warning_radius_ratio < 1
                or not 0 < self.point_radius_m < .001
                or not 0 < self.minimum_det < 1 or not 0 < self.minimum_singular_value < 1
                or not 0 < self.degenerate_particle_fraction <= 1):
            raise ValueError('Invalid collapse policy')


def analyze(x, object_ids, F_trial=None, policy=None):
    policy = policy or CollapsePolicy()
    x = np.asarray(x, dtype=np.float64)
    ids = np.asarray(object_ids)
    if (x.ndim != 3 or x.shape[-1] != 3 or len(x) < 2 or ids.shape != x.shape[1:2]
            or ids.dtype.kind not in 'iu' or not np.isfinite(x).all()
            or not np.array_equal(np.unique(ids), np.arange(len(np.unique(ids))))):
        raise ValueError('Invalid collapse input')
    if F_trial is not None:
        F_trial = np.asarray(F_trial, dtype=np.float64)
        if F_trial.shape != (*x.shape[:2], 3, 3) or not np.isfinite(F_trial).all():
            raise ValueError('Invalid collapse deformation input')
    objects, reasons, warnings = [], [], []
    for oid in np.unique(ids):
        points = x[:, ids == oid]
        if points.shape[1] < 2:
            raise ValueError('Collapse measurement requires at least two particles per object')
        centered = points - points.mean(axis=1, keepdims=True)
        rms = np.sqrt(np.mean(np.sum(centered ** 2, axis=-1), axis=1))
        # A handful of distant outliers must not hide a collapsed majority.
        radial = np.linalg.norm(points - np.median(points, axis=1, keepdims=True), axis=-1)
        q90 = np.quantile(radial, .90, axis=1)
        axes = np.sqrt(np.maximum(0, np.linalg.eigvalsh(
            np.einsum('tni,tnj->tij', centered, centered) / points.shape[1])))
        if rms[0] <= policy.point_radius_m or q90[0] <= policy.point_radius_m:
            raise ValueError('Degenerate initial object size')
        rms_ratio, core_ratio = rms / rms[0], q90 / q90[0]
        collapsed = (rms_ratio < policy.minimum_radius_ratio) | (core_ratio < policy.minimum_radius_ratio)
        point = q90 <= policy.point_radius_m
        warn = (rms_ratio < policy.warning_radius_ratio) | (core_ratio < policy.warning_radius_ratio)
        row = dict(object_id=int(oid), rms_radius_m=rms.tolist(), core90_radius_m=q90.tolist(),
                   rms_radius_ratio=rms_ratio.tolist(), core90_radius_ratio=core_ratio.tolist(),
                   principal_rms_extents_m=axes.tolist(),
                   severe_contraction_states=np.flatnonzero(collapsed).tolist(),
                   point_collapse_states=np.flatnonzero(point).tolist(),
                   contraction_warning_states=np.flatnonzero(warn).tolist())
        if collapsed.any(): reasons.append(f'object_{oid}:extreme_contraction')
        if point.any(): reasons.append(f'object_{oid}:point_collapse')
        if warn.any(): warnings.append(f'object_{oid}:contraction_review')
        if F_trial is not None:
            f = F_trial[:, ids == oid]
            det = np.linalg.det(f)
            smallest = np.linalg.svd(f, compute_uv=False)[..., -1]
            fraction = np.mean((det < policy.minimum_det) | (smallest < policy.minimum_singular_value), axis=1)
            row.update(minimum_det_F_trial=det.min(axis=1).tolist(),
                       minimum_singular_value=smallest.min(axis=1).tolist(),
                       degenerate_F_fraction=fraction.tolist())
            if np.any(fraction >= policy.degenerate_particle_fraction):
                reasons.append(f'object_{oid}:widespread_degenerate_deformation')
        objects.append(row)
    return dict(policy=asdict(policy), objects=objects, reasons=reasons, warnings=warnings,
                eligible=not reasons, calibrated=False, observation='every saved state; not every native substep',
                limitation='Point geometry and algebraic F checks; not true solid volume or calibrated material limits')
