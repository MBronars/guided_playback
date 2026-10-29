"""Bounded physical command replay; quiet motion is diagnostic, not a task gate.

Replay a recorded joint/openness sequence with the same native controller and
source-rest garment. No IK, cloth attachment, position reset, or hidden settling.
"""
import argparse
import os
from pathlib import Path
import subprocess
import numpy as np

SCHEMA = 'sim1-command-replay-v1'


def save_failure(out, frame_index, rows, arrays):
    """Retain failure evidence without asserting that a prefix is a full episode."""
    from .qualify import save_arrays
    from collection.common import digest, write_once
    if not any(row['guard'] for row in rows):
        raise ValueError('Failure evidence requires an actual guard failure')
    out = Path(out)
    path = out / 'FAILED_FRAME.npz'
    save_arrays(path, arrays)
    write_once(out / 'FAILURE.json', dict(transition_index=frame_index,
        failed_substeps=[r for r in rows if r['guard']], frame_substeps=rows,
        artifact=path.name, sha256=digest(path), complete_episode=False,
        accepted_for_training=False))


def shared_memory_check(total_mib, used_mib, own_mib, *, starting=False):
    """Admission/observation guard, NOT a CUDA allocator cap or reservation."""
    if not all(type(x) is int and x >= 0 for x in (total_mib, used_mib, own_mib)):
        raise ValueError('Finite nonnegative integer GPU memory counters required')
    free = total_mib - used_mib
    # Reserve headroom for the other workload and an additional startup allowance.
    required = 32768 if starting else 16384
    if used_mib > total_mib or own_mib > used_mib or own_mib > 16384 or free < required:
        raise RuntimeError('Shared GPU memory guard: stop only the folding worker')
    return dict(total_mib=total_mib, used_mib=used_mib, own_mib=own_mib,
                free_mib=free, required_free_mib=required)


def shared_gpu_status(uuid, *, allow_own=False, memory_limit_mib=16384):
    """Explicit SIM1-only concurrency; do not weaken the exclusive MPM default."""
    inventory = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
        '--query-gpu=uuid,memory.total,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10)
    rows = [r.strip().split(',') for r in inventory.splitlines() if r.strip()]
    if len(rows) != 1 or rows[0][0].strip() != uuid:
        raise RuntimeError('Requested GPU identity is missing or ambiguous')
    processes = subprocess.check_output(['nvidia-smi', '--id=' + uuid,
        '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True, timeout=10)
    counters = [list(map(int, r.split(','))) for r in processes.splitlines() if r.strip()]
    own = sum(memory for pid, memory in counters if pid == os.getpid())
    result = shared_memory_check(int(rows[0][1]), int(rows[0][2]), own, starting=not allow_own)
    return dict(result, uuid=uuid, process_ids=[pid for pid, _ in counters],
                sharing=True, memory_is_observed_not_reserved=True)


def save_chunk(recorder, start, out):
    """Preserve a committed actual-state prefix; never a physical-resume claim."""
    from .qualify import save_arrays
    from collection.common import digest, write_once
    stop = len(recorder.commands)
    if not 0 <= start < stop:
        raise ValueError('Nonempty consecutive replay chunk required')
    arrays = {key: np.stack([s[i] for s in recorder.states[start:stop + 1]])
              for i, key in enumerate(('particle_q', 'particle_qd', 'body_q', 'actual_joint_q'))}
    arrays.update(command_joint_q=np.stack([c[0] for c in recorder.commands[start:stop]]),
        command_openness=np.stack([c[1] for c in recorder.commands[start:stop]]),
        state_index=np.arange(start, stop + 1), transition_index=np.arange(start, stop),
        state_time_s=np.arange(start, stop + 1) / recorder.fps, fps=np.array(recorder.fps))
    path = Path(out) / f'frames-{start:06d}-{stop:06d}.npz'
    save_arrays(path, arrays)
    write_once(path.with_suffix('.json'), dict(sha256=digest(path), first_state=start,
        last_state=stop, complete_episode=False, physical_resume_supported=False))
    return stop


def command_sequence(raw, home, lower, upper, jaw_ids, *, prefix_frames=60, suffix_frames=120):
    from .contracts import finite_array
    home = finite_array(home, 'home', (None,)).astype(np.float32)
    lower = finite_array(lower, 'lower', home.shape).astype(np.float32)
    upper = finite_array(upper, 'upper', home.shape).astype(np.float32)
    q = finite_array(raw['joint_q'], 'recorded joint_q', (None, len(home))).astype(np.float32)
    op = finite_array(raw['openness'], 'recorded openness', (len(q), 2)).astype(np.float32)
    if len(q) < 2 or np.any(lower > upper):
        raise ValueError('Nonempty motion and valid joint limits required')
    for n in (prefix_frames, suffix_frames):
        if type(n) is not int or not 0 <= n <= 120:
            raise ValueError('Explicit bounded prefix/suffix required')
    if 'base_transform' in raw:
        base = finite_array(raw['base_transform'], 'base_transform', (len(q), 7))
        if not np.array_equal(base, np.broadcast_to(base[0], base.shape)):
            raise ValueError('Moving recorded bases are not supported by this fixed-base replay')
    ids = np.asarray(jaw_ids)
    if ids.shape != (2, 2) or ids.dtype.kind not in 'iu' or len(np.unique(ids)) != 4 or ids.min() < 0 or ids.max() >= len(home):
        raise ValueError('Two distinct actual jaw coordinates per hand required')
    overshoot = float(np.maximum(-op, op - 1).max(initial=0))
    if overshoot > .01:
        raise ValueError('Recorded openness overshoot exceeds explicit 1% repair bound')
    effective_op = np.clip(op, 0., 1.)
    effective_q = q.copy()
    for side in (0, 1):
        effective_q[:, ids[side]] = lower[ids[side]] + effective_op[:, side, None] * (upper[ids[side]] - lower[ids[side]])
    violation = float(np.maximum(lower - effective_q, effective_q - upper).max(initial=0))
    if violation > 1e-4:
        raise ValueError(f'Recorded arm joint limit violation {violation} exceeds numerical tolerance')
    effective_q = np.clip(effective_q, lower, upper)
    if np.max(np.abs(effective_q[1] - home)) > 1e-5:
        raise ValueError('Recorded sequence and initialized home differ')
    if np.max(np.abs(np.diff(effective_q, axis=0))) > .25:
        raise ValueError('Discontinuous recorded joint commands')
    commands = np.concatenate((np.tile(home, (prefix_frames, 1)), effective_q[1:],
                               np.tile(effective_q[-1], (suffix_frames, 1))))
    openness = np.concatenate((np.tile(effective_op[1], (prefix_frames, 1)), effective_op[1:],
                               np.tile(effective_op[-1], (suffix_frames, 1))))
    source_index = np.r_[np.full(prefix_frames, -1), np.arange(1, len(q)), np.full(suffix_frames, -2)]
    if len(commands) > 2400 or len(commands) % 2:
        raise ValueError('Replay must fit 2400 frames and terminal-preserving 30 Hz export')
    return dict(command_joint_q=commands, command_openness=openness, source_command_index=source_index), dict(
        recorded_frames=len(q), replay_frames=len(commands), prefix_frames=prefix_frames, suffix_frames=suffix_frames,
        source_index_sentinels={'-1': 'recorded home hold', '-2': 'last command hold'},
        discarded_source_indices=[0], home_source_index=1,
        openness_clipped_values=int(np.count_nonzero(op != effective_op)), maximum_openness_overshoot=overshoot,
        maximum_joint_limit_projection=violation,
        policy='Openness explicitly clipped within 1%; jaws derived from actual limits; arm roundoff <=1e-4 rad projected; no retiming',
        achieved_task_verified=False)


def main():
    from .qualify import run
    from collection.lease import gpu_lease
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'home', 'commands', 'sim1-source', 'assets-root', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda:0'], required=True)
    p.add_argument('--uuid')
    p.add_argument('--shared-gpu', action='store_true',
                   help='Allow other GPU workloads with explicit free-memory and own-memory guards')
    p.add_argument('--retime-return', action='store_true', help='Bound final both-open return speed/acceleration using FK')
    p.add_argument('--mode', choices=['eager', 'captured'], default='captured')
    p.add_argument('--build-only', action='store_true')
    p.add_argument('--self-contact-distance', type=float, choices=[.0005, .001, .002], default=.002)
    p.add_argument('--wall-seconds', type=int, default=3600)
    args = p.parse_args()
    args.frames, args.substeps, args.probe_vertex = 1, 24, []
    if args.device == 'cpu':
        run(args)
    else:
        with gpu_lease(args.uuid):
            run(args)


if __name__ == '__main__':
    main()
