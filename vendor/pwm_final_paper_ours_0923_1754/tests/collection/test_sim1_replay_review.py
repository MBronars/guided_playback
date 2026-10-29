"""Synthetic storage-contract tests, not physical manipulation evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from collection.common import digest, write_once
from collection.sim1.qualify import save_arrays
from collection.sim1.recording import StateRecorder
from collection.sim1.replay import SCHEMA, save_chunk
from collection.sim1.replay_review import review
from collection.sim1.settling import SettlingGate


class ReplayReviewTests(unittest.TestCase):
    def fixture(self, root, *, bad_flags=False, bad_trace=False):
        q = np.array([[.4, 0, .81], [.5, 0, .81], [.4, .1, .81]], np.float32)
        b = np.array([[.4, 0, .9, 0, 0, 0, 1], [.4, .1, .9, 0, 0, 0, 1]], np.float32)
        state = dict(particle_q=q, particle_qd=np.zeros_like(q), body_q=b, joint_q=np.zeros(6))
        r = StateRecorder(60, 3, 2, 6, max_states=5)
        r.append_initial(**state)
        for i in range(4):
            r.append_transition(i, command_joint_q=np.zeros(6), command_openness=np.zeros(2), **state)
        a = r.arrays(4)
        save_arrays(root / 'actual_states.npz', a)
        save_chunk(r, 0, root)
        save_arrays(root / 'commands.npz', {k: a[k] for k in ('command_joint_q', 'command_openness')})
        save_arrays(root / 'finalized_model.npz', dict(particle_q=q, home=np.zeros(6),
            tri_indices=np.array([[0, 1, 2]]), tri_poses=np.eye(2)[None]))
        flags = np.full((4, 2), 5, np.int32)
        if bad_flags: flags[2, 0] = 1
        save_arrays(root / 'collision_flags.npz', dict(shape_flags=flags))
        gate = SettlingGate(1 / 1440)
        rows = [gate.observe(i + 1, q, np.zeros_like(q)) for i in range(96)]
        if bad_trace: rows[23]['min_height_m'] += .1
        write_once(root / 'SUBSTEPS.json', rows)
        write_once(root / 'SOURCES.json', {})
        # A dummy archive is sufficient for the hash-only storage-contract check.
        write_once(root / 'executed_sources.tar.gz', dict(synthetic_fixture=True))
        write_once(root / 'PROTOCOL.json', dict(schema=SCHEMA, frames=4, substeps=24, fps=60,
            dt_s=1 / 1440, source_hashes={}, gripper_original_flags=[5, 5],
            gripper_filter=dict(left_shapes=[0], right_shapes=[1]), end_effector_body_ids=[0, 1],
            finalized_sha256=digest(root / 'finalized_model.npz'),
            command_plan_sha256=digest(root / 'commands.npz')))
        write_once(root / 'RESULT.json', dict(status='complete_diagnostic', frames=5, substeps=96,
            accepted_for_training=False, artifacts={p.name: digest(p) for p in root.iterdir()}))

    def test_valid_storage_does_not_claim_fold_success(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d))
            result = review(d)
            self.assertEqual(result['status'], 'replay_integrity_passed')
            self.assertFalse(result['fold_verified'])
            self.assertFalse(result['dense_substep_states_independently_verified'])
            self.assertEqual(result['chunks_verified'], 1)

    def test_dynamic_flag_error_rejected_even_when_hashes_match(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d), bad_flags=True)
            with self.assertRaises(AssertionError): review(d)

    def test_endpoint_error_rejected_even_when_hashes_match(self):
        with tempfile.TemporaryDirectory() as d:
            self.fixture(Path(d), bad_trace=True)
            with self.assertRaisesRegex(ValueError, 'endpoint'): review(d)


if __name__ == '__main__': unittest.main()
