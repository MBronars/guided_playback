"""Durable saved-state prefixes, never a mid-episode physical restart promise."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from collection.common import digest, fsync_dir, write_once
from .contracts import Clock

FIELDS = {'x', 'v', 'F_trial', 'C', 'wall_first_grid_step', 'wall_first_particle_step',
          'wall_covered_substeps', 'wall_simulation_steps'}


class FrameJournal:
    def __init__(self, root, *, initial_sha256, manifest_sha256, clock):
        self.root = Path(root)
        self.root.mkdir(exist_ok=False)
        self.clock = clock
        self.count = 0
        self.previous = None
        write_once(self.root / 'HEADER.json', dict(schema='mpm-saved-frame-journal-v1',
            initial_sha256=initial_sha256, manifest_sha256=manifest_sha256,
            clock=dict(substep_dt_s=clock.substep_dt_s,
                       substeps_per_transition=clock.substeps_per_transition, transitions=clock.transitions),
            physical_restart_supported=False))

    def append(self, step, arrays):
        if set(arrays) != FIELDS or step != self.count * self.clock.substeps_per_transition:
            raise ValueError('Journal fields or saved-state clock mismatch')
        if self.count > self.clock.transitions:
            raise ValueError('Journal exceeds declared recording')
        name = f'frame-{self.count:05d}'
        fd, temporary = tempfile.mkstemp(prefix='.' + name + '-', suffix='.partial', dir=self.root)
        with os.fdopen(fd, 'wb') as f:
            np.savez_compressed(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        target = self.root / (name + '.npz')
        os.link(temporary, target)
        Path(temporary).unlink()  # Only this writer's successfully published link.
        fsync_dir(self.root)
        receipt = self.root / (name + '.json')
        write_once(receipt, dict(index=self.count, step=int(step),
            time_s=float(self.clock.state_time_s[self.count]), payload_sha256=digest(target),
            previous_receipt_sha256=self.previous))
        self.previous = digest(receipt)
        self.count += 1


def verify_journal(root, *, initial_sha256, manifest_sha256, complete=False, raw_path=None):
    root = Path(root)
    header = json.loads((root / 'HEADER.json').read_text())
    if (header['schema'] != 'mpm-saved-frame-journal-v1' or header['initial_sha256'] != initial_sha256
            or header['manifest_sha256'] != manifest_sha256 or header['physical_restart_supported'] is not False):
        raise ValueError('Journal identity mismatch')
    clock = Clock(**header['clock'])
    previous = None
    receipts = sorted(root.glob('frame-*.json'))
    for index, path in enumerate(receipts):
        row = json.loads(path.read_text())
        if (path.name != f'frame-{index:05d}.json' or row['index'] != index
                or row['step'] != index * clock.substeps_per_transition
                or index > clock.transitions or row['time_s'] != float(clock.state_time_s[index])
                or row['previous_receipt_sha256'] != previous):
            raise ValueError('Journal gap, clock or receipt chain mismatch')
        payload = path.with_suffix('.npz')
        if payload.is_symlink() or digest(payload) != row['payload_sha256']:
            raise ValueError('Journal payload changed')
        with np.load(payload, allow_pickle=False) as data:
            if set(data.files) != FIELDS or int(data['wall_simulation_steps']) != row['step']:
                raise ValueError('Journal state/evidence mismatch')
            if raw_path is not None:
                import h5py
                with h5py.File(raw_path, 'r') as raw:
                    for key in FIELDS:
                        np.testing.assert_array_equal(data[key], raw[key][index])
        previous = digest(path)
    if complete and len(receipts) != clock.transitions + 1:
        raise ValueError('Incomplete saved-frame journal')
    return dict(states=len(receipts), last_receipt_sha256=previous,
                header_sha256=digest(root / 'HEADER.json'), complete=len(receipts) == clock.transitions + 1,
                orphan_payloads=len(list(root.glob('frame-*.npz'))) - len(receipts),
                physical_restart_supported=False)
