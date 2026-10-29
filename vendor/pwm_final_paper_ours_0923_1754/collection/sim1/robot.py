"""Explicit recorded-home initialization and native gripper collision filtering."""
import numpy as np

def read_home(path, env, model):
    with np.load(path, allow_pickle=False) as z:
        q, openness = z['joint_q'][1].copy(), z['openness'][1].copy()
    if q.shape != (model.joint_coord_count,) or openness.shape != (2,) or not np.isfinite(q).all():
        raise ValueError('Home recording schema differs from the actual robot')
    if not np.isfinite(openness).all() or np.any((openness < 0) | (openness > 1)):
        raise ValueError('Invalid home jaw command')
    lo, hi = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
    for side, ids in enumerate((env.left_gripper_joint_indices, env.right_gripper_joint_indices)):
        q[ids] = lo[ids] + openness[side] * (hi[ids] - lo[ids])
    if np.any(q < lo - 1e-5) or np.any(q > hi + 1e-5):
        raise ValueError('Home is outside actual joint limits')
    return q.astype(np.float32), openness.astype(np.float32)


def gripper_filter(env, model, openness, *, baseline_flags=None):
    """Filter gripper particle contacts; provide original flags for dynamic use.

    The default retains the historical one-shot home behavior. A manipulation
    loop must use GripperCollisionFilter, which retains the unfiltered flags so
    closing a previously opened jaw restores its original collision eligibility.
    """
    flag = 1 << 2
    values = model.shape_flags.numpy().copy()
    openness = np.asarray(openness)
    if openness.shape != (2,) or not np.isfinite(openness).all() or np.any((openness < 0) | (openness > 1)):
        raise ValueError('Two finite jaw commands in [0,1] required')
    baseline = values if baseline_flags is None else np.asarray(baseline_flags)
    if baseline.shape != values.shape or baseline.dtype.kind not in 'iu':
        raise ValueError('Original integer model shape flags required')
    sides = []
    children = model.joint_child.numpy()
    for names, ee, op in (({'left_joint17', 'left_joint18'}, env.lee_index, openness[0]),
                          ({'right_joint27', 'right_joint28'}, env.ree_index, openness[1])):
        bodies = [int(children[i]) for i, k in enumerate(model.joint_key) if k in names] + [ee]
        ids = sorted({int(s) for b in bodies for s in model.body_shapes.get(b, [])})
        if not ids:
            raise ValueError('Gripper collider mapping is empty')
        # Restore only the original particle-contact bit, not unrelated flags
        # and not shapes whose particle contacts were disabled before filtering.
        values[ids] = (values[ids] & ~flag) | (baseline[ids] & flag)
        if op > .6:
            values[ids] &= ~flag
        sides.append(ids)
    model.shape_flags.assign(values)
    return dict(left_shapes=sides[0], right_shapes=sides[1],
                disabled_when_home_openness_above=.6,
                shape_flags_after_filter=values.tolist())


class GripperCollisionFilter:
    """Construct once before opening jaws; then update for every jaw command."""
    def __init__(self, env, model):
        self.env, self.model = env, model
        self.baseline = model.shape_flags.numpy().copy()

    def apply(self, openness):
        return gripper_filter(self.env, self.model, openness, baseline_flags=self.baseline)
