"""Versioned scene witnesses; not a physical contact or calibration certificate."""
import numpy as np
from .recipes import INTERACTION_VERSIONS


def positive_stencil_cells(points, n_grid=100, inv_dx=64.):
    """Occupied positive quadratic B-spline support, matching positive-grid P2G.

    Shared support witnesses potential grid coupling at a saved state, not a
    continuous collision event, contact force, or solid surface intersection.
    """
    points = np.asarray(points, dtype=np.float32)
    grid = points * np.float32(inv_dx)
    base = np.trunc(grid - np.float32(.5)).astype(np.int64)
    fx = grid - base.astype(np.float32)
    w = np.stack((.5 * (1.5 - fx)**2, .75 - (fx - 1.)**2, .5 * (fx - .5)**2), axis=1)
    offset = np.array([(i, j, k) for i in range(3) for j in range(3) for k in range(3)])
    nodes = base[:, None, :] + offset[None, :, :]
    valid = np.all((nodes >= 0) & (nodes < n_grid), axis=-1)
    weights = w[:, offset[:, 0], 0] * w[:, offset[:, 1], 1] * w[:, offset[:, 2], 2]
    if not valid.all():
        raise ValueError('Unsafe stencil in interaction witness')
    nodes = nodes[weights > 0]
    return np.unique((nodes[:, 0] * n_grid + nodes[:, 1]) * n_grid + nodes[:, 2])


def analyze(manifest, arrays):
    geometry = manifest.get('geometry_provenance', {})
    if geometry.get('recipe_version') not in INTERACTION_VERSIONS:
        raise ValueError('Interaction qualification requires its declared sampler version')
    x = np.asarray(arrays['x'], dtype=np.float32)
    ids = arrays['object_ids']
    if len(x) != 49 or not np.isfinite(x).all():
        raise ValueError('Complete finite 49-state trajectory required')
    n = manifest['n_objects']
    centroids = np.stack([x[:, ids == oid].astype(float).mean(axis=1) for oid in range(n)], axis=1)
    requested = geometry['requested_pairs']
    pairs = []
    future_shared = {tuple(p): [] for p in requested}
    for frame in range(4, len(x)):
        needed = set(v for pair in requested for v in pair)
        cells = {oid: positive_stencil_cells(x[frame, ids == oid]) for oid in needed}
        for a, b in requested:
            count = len(np.intersect1d(cells[a], cells[b], assume_unique=True))
            if count:
                future_shared[(a, b)].append(dict(state=frame, shared_positive_nodes=count))
    for a, b in requested:
        axis = 0 if manifest['scenario'] == 'collision' else 2
        separation = np.abs(centroids[:, a, axis] - centroids[:, b, axis])
        approach = float(separation[0] - separation[4:].min())
        pairs.append(dict(a=a, b=b, saved_future_shared_stencils=future_shared[(a, b)],
                          approach_m=approach, witness=bool(future_shared[(a, b)] and approach >= .03)))
    movement = np.linalg.norm(centroids[4:] - centroids[3], axis=-1).max(axis=0)
    reasons = []
    if requested and not all(p['witness'] for p in pairs):
        reasons.append('requested_pair_has_no_future_approach_and_shared_stencil_witness')
    if not requested and manifest['scenario'] == 'scatter':
        direction = centroids[0, :, :2] - np.array([.75, .75])
        norm = np.linalg.norm(direction, axis=-1)
        if np.any(norm == 0):
            raise ValueError('Scatter radial direction undefined')
        outward = np.sum((centroids[4:, :, :2] - centroids[3, :, :2]) * (direction / norm[:, None]), axis=-1).max(axis=0)
        if not np.all(outward >= .05):
            reasons.append('scatter_missing_50mm_future_radial_travel')
    if not requested and manifest['scenario'] == 'random' and movement.max() < .03:
        reasons.append('random_missing_30mm_future_motion')
    if manifest['scenario'] in ('drop', 'translate'):
        if n != 1 or requested:
            raise ValueError('Single-object family contract violated')
        if movement[0] < .03:
            reasons.append('single_object_missing_30mm_future_motion')
        if manifest['scenario'] == 'drop' and centroids[3, 0, 2] - centroids[4:, 0, 2].min() < .03:
            reasons.append('drop_missing_30mm_future_descent')
        if manifest['scenario'] == 'translate' and np.linalg.norm(
                centroids[4:, 0, :2] - centroids[3, 0, :2], axis=-1).max() < .03:
            reasons.append('translate_missing_30mm_future_horizontal_motion')
    if manifest['scenario'] in ('collision', 'stack', 'drop_on') and not requested:
        reasons.append('missing_declared_interaction_pairs')
    delta = np.linalg.norm(np.diff(x.astype(float), axis=0), axis=-1).mean(axis=1)
    return dict(schema='mpm-interaction-witness-v1', pairs=pairs,
                max_future_centroid_displacement_m=movement.tolist(),
                future_window_mean_point_step_mm=[float(delta[start + 3:start + 19].mean() * 1000) for start in range(30)],
                semantic_eligible=not reasons, reasons=reasons, accepted_for_training=False,
                limitation='Saved-state positive P2G support and approach, not continuous contact forces or calibrated solid physics')
