"""Small CPU-only candidate contracts for a new SIM1 release.

Not imported by live SIM1 or the frozen training loader. These checks establish
specific schema/kinematic invariants, never claim garment physics are validated.
All lengths are metres, time seconds, rotations XYZW unit quaternions.
"""
from __future__ import annotations
import math
import numpy as np


def finite_array(value, name, shape=None):
    array = np.asarray(value)
    if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError(f'{name}: finite numeric array required')
    if shape is not None and (array.ndim != len(shape) or
            any(n is not None and n != actual for n, actual in zip(shape, array.shape))):
        raise ValueError(f'{name}: shape {array.shape} does not match {shape}')
    return array


def unit_quaternions(value, name='quaternions'):
    q = finite_array(value, name)
    if q.shape[-1] != 4 or np.any(np.abs(np.linalg.norm(q, axis=-1) - 1) > 2e-4):
        raise ValueError(f'{name}: XYZW unit quaternions required')
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def slerp(q0, q1, fraction):
    q0, q1 = unit_quaternions(q0), unit_quaternions(q1)
    dot = float(q0 @ q1)
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = (1 - fraction) * q0 + fraction * q1
        return q / np.linalg.norm(q)
    angle = np.arccos(np.clip(dot, -1, 1))
    return (np.sin((1-fraction)*angle)*q0 + np.sin(fraction*angle)*q1) / np.sin(angle)


def resample_both_arms(left, right, openness, *, max_step_m=0.004,
                       max_angle_rad=0.05, max_open_step=0.025, max_frames=20000):
    """Bound both arms' translation/rotation and both jaw commands per frame.

    Inserts shared synchronized states; jaw commands are interpolated, not
    jumped immediately to the destination during a subdivided motion. Shortest
    quaternion interpolation handles q/-q without the zero-norm lerp failure.
    Returns original waypoint fractional indices to preserve phase provenance.
    It does not establish IK tracking, collision freedom or a successful grasp.
    """
    left = finite_array(left, 'left', (None, 7))
    right = finite_array(right, 'right', left.shape)
    openness = finite_array(openness, 'openness', (len(left), 2))
    if not len(left) or np.any((openness < 0) | (openness > 1)):
        raise ValueError('Nonempty poses and openness in [0,1] required')
    if type(max_frames) is not int or max_frames<1:
        raise ValueError('Explicit positive integer frame budget required')
    for x in (max_step_m, max_angle_rad, max_open_step):
        if not np.isfinite(x) or x <= 0:
            raise ValueError('Positive finite resampling limits required')
    lq, rq = unit_quaternions(left[:, 3:]), unit_quaternions(right[:, 3:])
    factors = []
    for t in range(1, len(left)):
        step = max(np.linalg.norm(left[t,:3]-left[t-1,:3]),
                   np.linalg.norm(right[t,:3]-right[t-1,:3]))
        angle = max(2*np.arccos(np.clip(abs(lq[t]@lq[t-1]), 0, 1)),
                    2*np.arccos(np.clip(abs(rq[t]@rq[t-1]), 0, 1)))
        jaw = np.max(np.abs(openness[t]-openness[t-1]))
        factors.append(max(1, math.ceil(step/max_step_m), math.ceil(angle/max_angle_rad),
                           math.ceil(jaw/max_open_step)))
    if sum(factors) + 1 > max_frames:
        raise ValueError('Resampling would exceed the explicit frame budget')
    L, R, O, I = [left[0].copy()], [right[0].copy()], [openness[0].copy()], [0.]
    for t, n in enumerate(factors, 1):
        for k in range(1, n+1):
            a = k/n
            for output, arm in ((L,left), (R,right)):
                pose = (1-a)*arm[t-1] + a*arm[t]
                pose[3:] = slerp(arm[t-1,3:], arm[t,3:], a)
                output.append(pose)
            O.append((1-a)*openness[t-1] + a*openness[t])
            I.append(t-1+a)
    return tuple(np.asarray(a) for a in (L,R,O,I))


def snap_material_vertex(points, proposed_xy, *, jitter_xy=(0.,0.),
                         max_snap_m=0.05, support_radius_m=0.015, min_support=3):
    """Snap AFTER jitter to a stable actual cloth vertex, never bbox corners.

    This is only geometric candidacy. A multilayer garment additionally requires
    a chosen target panel/surface and collision-based grasp confirmation.
    """
    q = finite_array(points, 'material_points', (None,3))
    target = finite_array(proposed_xy, 'target_xy', (2,)).astype(float)
    target = target + finite_array(jitter_xy, 'jitter_xy', (2,))
    if (not len(q) or not np.isfinite(max_snap_m) or max_snap_m <= 0 or
            not np.isfinite(support_radius_m) or support_radius_m <= 0 or
            type(min_support) is not int or min_support < 1):
        raise ValueError('Nonempty material and positive support limits required')
    d = np.linalg.norm(q[:,:2]-target, axis=1)
    index = int(np.argmin(d))
    neighbors = np.linalg.norm(q-q[index], axis=1) <= support_radius_m
    if d[index] > max_snap_m:
        raise ValueError('No material within the allowed grasp snap radius')
    if neighbors.sum() < min_support:
        raise ValueError('Selected material vertex lacks local cloth support')
    return dict(particle_id=index, point_m=q[index].copy(), proposed_xy_m=target,
                snap_distance_m=float(d[index]), support_count=int(neighbors.sum()),
                grasp_success_verified=False)


def integer_state_times(frame_indices, fps):
    frame_indices = finite_array(frame_indices, 'frame_indices', (None,))
    if frame_indices.dtype.kind not in 'iu' or not len(frame_indices):
        raise ValueError('Nonempty integer state indices required')
    if frame_indices[0] != 0 or np.any(np.diff(frame_indices) != 1):
        raise ValueError('State indices must be complete, consecutive and start at zero')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Positive finite fps required')
    return frame_indices.astype(np.float64) / float(fps)


def require_complete_usd_timeline(samples, expected_indices, timecodes_per_second, fps):
    """No clamp-to-first or USD interpolation may silently invent saved states."""
    samples = finite_array(samples, 'USD saved timecodes', (None,))
    expected_indices = finite_array(expected_indices, 'expected indices', (None,))
    if expected_indices.dtype.kind not in 'iu' or not len(expected_indices):
        raise ValueError('Explicit nonempty integer state indices required')
    if not np.isfinite(timecodes_per_second) or timecodes_per_second <= 0:
        raise ValueError('Positive timeCodesPerSecond required')
    if not np.isfinite(fps) or fps <= 0 or np.any(np.diff(expected_indices) <= 0):
        raise ValueError('Positive fps and increasing state indices required')
    expected = expected_indices.astype(float) * timecodes_per_second / fps
    if samples.shape != expected.shape or not np.allclose(samples, expected, atol=1e-9, rtol=0):
        raise ValueError('USD timeline is incomplete or does not match the declared states')
    return expected


def require_link_membership(link_ids, n_points, *, expected_links=3, contiguous=True):
    ids = finite_array(link_ids, 'ee_link_ids', (n_points,))
    if ids.dtype.kind not in 'iu' or ids.min(initial=0)<0 or ids.max(initial=-1)>=expected_links:
        raise ValueError('Integer gripper link labels out of range')
    counts = np.bincount(ids, minlength=expected_links)
    if np.any(counts == 0):
        raise ValueError('Every gripper link must have points')
    if contiguous and not np.array_equal(ids, np.repeat(np.arange(expected_links), counts)):
        raise ValueError('Consumer requires contiguous per-link point ordering, not just a histogram')
    return counts


def points_from_recorded_links(local_points, link_ids, body_poses):
    """Materialize actual saved link states, not FK of unachieved commands."""
    p = finite_array(local_points, 'template', (None,3))
    ids = finite_array(link_ids, 'link_ids', (len(p),))
    poses = finite_array(body_poses, 'body_poses', (None,3,7))
    require_link_membership(ids, len(p))
    normalized=unit_quaternions(poses[:,:,3:], 'body_poses')
    q = normalized[:,ids]
    v = np.broadcast_to(p, (len(poses), *p.shape))
    cross = 2*np.cross(q[:,:,:3], v)
    with np.errstate(over='ignore',invalid='ignore'):
        output=(v + q[:,:,3,None]*cross + np.cross(q[:,:,:3],cross) + poses[:,ids,:3]).astype(np.float32)
    if not np.isfinite(output).all():raise ValueError('Materialized gripper points overflow finite float32')
    return output


def principal_stretches(points, faces, inverse_material_rest):
    q = finite_array(points, 'positions', (None,3))
    f = finite_array(faces, 'faces', (None,3))
    inv = finite_array(inverse_material_rest, 'inverse_material_rest', (len(f),2,2))
    if f.dtype.kind not in 'iu' or not len(f) or f.min()<0 or f.max()>=len(q):
        raise ValueError('Nonempty in-range triangular topology required')
    tri = q[f]
    ds = np.stack((tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]), axis=-1)
    return np.linalg.svd(ds @ inv, compute_uv=False)


def geometry_diagnostics(positions, times, *, table_top_z, table_xy_bounds,
                         faces=None, inverse_material_rest=None):
    """Every frame, mesh-only strain, real table footprint, actual dt.

    No default numerical threshold implies scientific acceptance. Initial/final
    rest, gripper collision, seam contact and self-intersection are separate gates.
    """
    q = finite_array(positions, 'cloth_positions', (None,None,3))
    t = finite_array(times, 'state_times_s', (len(q),))
    bounds = finite_array(table_xy_bounds, 'table_xy_bounds', (2,2))
    if len(q)<2 or not q.shape[1] or np.any(np.diff(t)<=0) or np.any(bounds[:,0]>=bounds[:,1]):
        raise ValueError('At least two increasing states and valid table bounds required')
    if not np.isfinite(table_top_z):
        raise ValueError('Finite recorded table surface required')
    speeds, penetrations, strain = [], [], []
    outside_counts = []
    for index, frame in enumerate(q):
        over = np.all((frame[:,:2]>=bounds[:,0]) & (frame[:,:2]<=bounds[:,1]), axis=1)
        penetrations.append(float(np.maximum(table_top_z-frame[over,2], 0).max(initial=0)))
        outside_counts.append(int((~over).sum()))
        if index:
            speeds.append(float((np.linalg.norm(frame-q[index-1], axis=-1)/(t[index]-t[index-1])).max()))
        if faces is not None and inverse_material_rest is not None:
            s = principal_stretches(frame, faces, inverse_material_rest)
            strain.append(float(np.abs(s-1).max()))
    return dict(max_particle_speed_m_s=max(speeds),
                max_table_penetration_m=max(penetrations),
                per_frame_table_penetration_m=penetrations,
                per_frame_outside_table_vertices=outside_counts,
                max_abs_principal_strain=max(strain) if strain else None,
                examined_frames=len(q), accepted_for_training=False,
                pending=['calibrated phase-aware thresholds', 'rest/kinetic settling',
                         'contact and collision coverage', 'validated garment seam',
                         'grasp/release/fold success'])
