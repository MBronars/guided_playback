"""Explicit rigid initial placement; material rest and particle identity unchanged."""
import numpy as np
from .contracts import finite_array
from .inputs import validate_input


def quarter_turn_input(arrays, *, turns, center_xy):
    """Rotate by exact quarter turns around the input XY bounding-box center.

    Return a new source input and transform receipt. Height, topology, source UV,
    areas, mass, rest dihedrals and all non-position arrays remain byte-identical.
    Collision/table qualification is separate; no failed outcome is repaired.
    """
    validate_input(arrays)
    if type(turns) is not int or turns not in (0, 1, 2, 3):
        raise ValueError('Explicit quarter turn 0..3 required')
    center = finite_array(center_xy, 'target XY center', (2,)).astype(np.float64)
    rotations = (np.array([[1., 0.], [0., 1.]]), np.array([[0., -1.], [1., 0.]]),
                 np.array([[-1., 0.], [0., -1.]]), np.array([[0., 1.], [-1., 0.]]))
    original = np.asarray(arrays['vertices_world_m'])
    source_center = (original[:, :2].astype(np.float64).min(0) + original[:, :2].astype(np.float64).max(0)) / 2
    placed = original.copy()
    placed[:, :2] = (original[:, :2].astype(np.float64) - source_center) @ rotations[turns].T + center
    if not np.isfinite(placed).all():
        raise ValueError('Placed vertices overflowed source storage precision')
    result = {key: value.copy() for key, value in arrays.items()}
    result['vertices_world_m'] = placed
    validate_input(result)
    return result, dict(quarter_turns=turns, source_center_xy_m=source_center.tolist(),
        target_center_xy_m=center.tolist(), rotation_xy=rotations[turns].tolist(),
        bounds_m=[placed.min(0).tolist(), placed.max(0).tolist()],
        non_position_fields_unchanged=True, height_unchanged=True, collision_qualified=False)


def swept_drop_aabb_clear(points, world_robot_meshes, *, floor_center_z=.808, radius=.008):
    """Conservative no-overlap sufficient test for a nominal vertical rigid drop.

    Includes radius inflation. Failure means possibly overlapping, not proven
    collision. A pass concerns this static robot and nominal rigid sweep only,
    not subsequent deformed cloth motion or robot motion.
    """
    q = finite_array(points, 'cloth', (None, 3)).astype(np.float64)
    if not len(q) or not np.isfinite(floor_center_z) or not np.isfinite(radius) or radius <= 0:
        raise ValueError('Nonempty geometry and finite positive radius required')
    if q[:, 2].min() < floor_center_z:
        raise ValueError('Nominal drop cannot raise cloth to its declared floor endpoint')
    if not world_robot_meshes:
        raise ValueError('Explicit nonempty robot mesh inventory required')
    low, high = q.min(0) - radius, q.max(0) + radius
    low[2] = floor_center_z - radius
    overlaps = []
    for item in world_robot_meshes:
        v = finite_array(item['vertices'], 'actual robot mesh', (None, 3))
        if not len(v):
            raise ValueError('Empty robot mesh')
        if np.all(v.max(0) >= low) and np.all(v.min(0) <= high):
            overlaps.append(item['shape_id'])
    return dict(clear=not overlaps, possibly_overlapping_shape_ids=overlaps,
        swept_bounds_m=[low.tolist(), high.tolist()], radius_m=radius,
        scope='static robot mesh AABBs and radius-inflated nominal rigid vertical sweep; not actual deformed trajectory')
