"""Post-release joint-path retiming with sampled FK speed/acceleration checks.

The recorded path is retained by interpolation; only its clock changes. Bounds
apply to sampled command FK, not continuous-time or achieved-physics guarantees.
"""
import numpy as np


def metrics(q, poses, fps=60):
    from scipy.spatial.transform import Rotation
    q, poses = np.asarray(q), np.asarray(poses)
    if poses.shape != (len(q), 2, 7) or not np.isfinite(poses).all():
        raise ValueError('Two finite FK poses per command required')
    linear = np.diff(poses[:, :, :3], axis=0) * fps
    angular = []
    for hand in range(2):
        r = Rotation.from_quat(poses[:, hand, 3:])
        angular.append((r[1:] * r[:-1].inv()).as_rotvec() * fps)
    angular = np.stack(angular, axis=1)
    joint = np.diff(q, axis=0) * fps
    # q[0:2] are preceding unchanged commands: bound the new transition and its
    # acceleration from the previous velocity, not the unchanged prefix itself.
    return dict(linear_speed=float(np.linalg.norm(linear[1:], axis=-1).max()),
        linear_acceleration=float(np.linalg.norm(np.diff(linear, axis=0) * fps, axis=-1).max()),
        angular_speed=float(np.linalg.norm(angular[1:], axis=-1).max()),
        angular_acceleration=float(np.linalg.norm(np.diff(angular, axis=0) * fps, axis=-1).max()),
        joint_speed=float(np.abs(joint[1:]).max()),
        joint_acceleration=float(np.abs(np.diff(joint, axis=0) * fps).max()))


def retime_return(commands, metadata, fk, *, speed=.30, acceleration=2., maximum_frames=2400):
    """Retain pre-return commands exactly; smoothly reduce the source-clock rate."""
    if not np.isfinite([speed, acceleration]).all() or not 0 < speed <= .5 or not 0 < acceleration <= 5:
        raise ValueError('Explicit bounded positive return speed/acceleration required')
    q, op = commands['command_joint_q'], commands['command_openness']
    suffix = metadata['suffix_frames']
    if suffix < 2 or len(q) < suffix + 5:
        raise ValueError('Recorded return and terminal hold required')
    last = len(q) - suffix - 1
    not_open = np.flatnonzero((op[:last + 1] < .999).any(axis=1))
    if not len(not_open): raise ValueError('No release event identifies the final return')
    start = int(not_open[-1]) + 1
    if not 2 <= start < last - 1 or np.any(op[start:] < .999):
        raise ValueError('Final continuously open return segment required')
    limits = dict(linear_speed=speed, linear_acceleration=acceleration,
        angular_speed=2., angular_acceleration=8., joint_speed=1.5, joint_acceleration=20.)
    scale = 2.
    history = []
    for attempt in range(12):
        u, fractions = float(start - 1), []
        while u < last:
            t = len(fractions)
            blend = .5 * (1 + np.cos(np.pi * min(t / 24, 1)))
            rate = 1 / scale + (1 - 1 / scale) * blend
            u = min(float(last), u + rate)
            fractions.append(u)
            if start + len(fractions) + suffix > maximum_frames:
                raise ValueError(f'Retimed return exceeds explicit frame/RAM planning budget; trials={history}')
        fractions = np.asarray(fractions)
        lo = np.floor(fractions).astype(int); hi = np.minimum(lo + 1, last)
        alpha = (fractions - lo)[:, None]
        tail_q = (q[lo] * (1 - alpha) + q[hi] * alpha).astype(np.float32)
        tail_op = (op[lo] * (1 - alpha) + op[hi] * alpha).astype(np.float32)
        check_q = np.concatenate((q[start-2:start], tail_q, q[-1:]))
        values = metrics(check_q, np.stack([fk(row) for row in check_q]))
        history.append(dict(scale=scale, **values))
        if all(values[k] <= bound for k, bound in limits.items()): break
        ratio = max(max(values[k] / limits[k] for k in limits if k.endswith('speed')),
                    max(np.sqrt(values[k] / limits[k]) for k in limits if k.endswith('acceleration')))
        scale *= max(1.15, min(ratio * 1.05, 2.))
    else: raise ValueError('Could not satisfy the sampled return-motion bounds')
    # Exact terminal-preserving 30 Hz export, with at most one additional held frame.
    pad = (start + len(tail_q) + suffix) % 2
    new_q = np.concatenate((q[:start], tail_q, np.tile(q[-1], (suffix + pad, 1))))
    new_op = np.concatenate((op[:start], tail_op, np.tile(op[-1], (suffix + pad, 1))))
    if len(new_q) > maximum_frames: raise ValueError('Terminal padding exceeds frame budget')
    source = np.r_[commands['source_command_index'][:start], np.full(len(tail_q), -3), np.full(suffix + pad, -2)]
    plan = dict(command_joint_q=new_q, command_openness=new_op, source_command_index=source,
        original_command_fraction=np.r_[np.arange(start), fractions, np.full(suffix + pad, last)])
    meta = dict(metadata, original_replay_frames=len(q), replay_frames=len(new_q), suffix_frames=suffix + pad,
        policy=metadata['policy'].replace('; no retiming', '') + '; post-release return retimed',
        source_index_sentinels={**metadata['source_index_sentinels'], '-3': 'retimed original joint path'},
        return_retime=dict(start_original_command=start, unchanged_prefix_commands=start,
            original_start_time_s=start / 60, final_clock_slowdown=scale,
            original_return_frames=last - start + 1, retimed_return_frames=len(tail_q),
            sampled_command_limits=limits, sampled_command_metrics=values,
            ramp_seconds=.4, added_terminal_hold_frames=pad,
            scope='final both-open return only; no IK or path shortcut; sampled FK bounds require achieved-state verification'))
    return plan, meta
