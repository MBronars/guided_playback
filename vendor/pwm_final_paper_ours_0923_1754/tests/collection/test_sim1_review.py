import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from collection.common import digest
from collection.sim1.recording import StateRecorder
from collection.sim1.review import review
from collection.sim1.settling import SettlingGate


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.q = np.array([[.5, 0, .81], [.51, 0, .81], [.5, .01, .81]], np.float32)
        self.v = np.zeros_like(self.q)
        body = np.array([[0, 0, 0, 0, 0, 0, 1]], np.float32)
        joint = np.zeros(1, np.float32)
        openness = np.ones(2, np.float32)
        np.savez(self.root / 'finalized_model.npz', particle_q=self.q, home=joint,
            openness=openness, tri_indices=np.array([[0, 1, 2]], np.int32),
            tri_poses=np.array([np.eye(2) * 100], np.float32))
        recorder = StateRecorder(60, 3, 1, 1, max_states=2)
        state = dict(particle_q=self.q, particle_qd=self.v, body_q=body, joint_q=joint)
        recorder.append_initial(**state)
        recorder.append_transition(0, command_joint_q=joint, command_openness=openness, **state)
        np.savez(self.root / 'actual_states.npz', **recorder.arrays(1))
        self.substeps = {key: np.repeat(value[None], 24, axis=0) for key, value in
            dict(particle_q=self.q, particle_qd=self.v, body_q=body, actual_joint_q=joint).items()}
        for prefix in ('vertex_colliding_triangles', 'edge_colliding_edges'):
            self.substeps[prefix + '_count'] = np.zeros((24, 3), np.int32)
            self.substeps[prefix + '_buffer_sizes'] = np.full((24, 3), 32, np.int32)
        np.savez(self.root / 'substeps-0000.npz', **self.substeps)
        gate = SettlingGate(1 / 1440)
        self.rows = [gate.observe(i + 1, self.q, self.v) for i in range(24)]
        self.json('SUBSTEPS.json', self.rows)
        self.json('PROTOCOL.json', dict(schema='sim1-matched-home-execution-v1',
            frames=1, substeps=24, fps=60, dt_s=1 / 1440, numpy_version=np.__version__))
        self.json('SOURCES.json', {})
        # Synthetic fixture: hashes are exercised; no simulator archive is executed.
        (self.root / 'executed_sources.tar.gz').write_bytes(b'test fixture')
        self.receipt()

    def json(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def receipt(self):
        self.json('RESULT.json', dict(status='complete_diagnostic', accepted_for_training=False,
            frames=2, substeps=24, artifacts={p.name: digest(p) for p in self.root.iterdir()
                                             if p.name != 'RESULT.json'}))

    def test_independent_review_does_not_grant_physics_approval(self):
        result = review(self.root)
        self.assertEqual(result['substeps'], 24)
        self.assertEqual(result['maximum_substep_table_penetration_m'], 0)
        self.assertFalse(result['motion_gate_pass_observed'])
        self.assertFalse(result['accepted_for_training'])
        self.assertFalse(result['physics_qualified'])

    def test_changed_artifact_rejected(self):
        self.rows[0]['rms_speed_m_s'] = 1.
        self.json('SUBSTEPS.json', self.rows)
        with self.assertRaisesRegex(ValueError, 'Artifact changed'):
            review(self.root)

    def test_rehashed_inconsistent_trace_rejected(self):
        self.rows[0]['rms_speed_m_s'] = 1.
        self.json('SUBSTEPS.json', self.rows); self.receipt()
        with self.assertRaisesRegex(ValueError, 'trace disagrees'):
            review(self.root)

    def test_rehashed_missing_substep_rejected(self):
        self.json('SUBSTEPS.json', self.rows[:-1]); self.receipt()
        with self.assertRaisesRegex(ValueError, 'Incomplete physical'):
            review(self.root)

    def test_rehashed_frame_endpoint_disagreement_rejected(self):
        self.substeps['particle_q'][-1, 0, 0] += .01
        np.savez(self.root / 'substeps-0000.npz', **self.substeps); self.receipt()
        with self.assertRaises(AssertionError): review(self.root)

    def test_overflow_cannot_be_hidden_by_trace(self):
        self.substeps['vertex_colliding_triangles_count'][0, 0] = 33
        np.savez(self.root / 'substeps-0000.npz', **self.substeps); self.receipt()
        with self.assertRaisesRegex(ValueError, 'trace disagrees'):
            review(self.root)

    def test_missing_hash_membership_rejected(self):
        result = json.loads((self.root / 'RESULT.json').read_text())
        del result['artifacts']['substeps-0000.npz']
        self.json('RESULT.json', result)
        with self.assertRaisesRegex(ValueError, 'omitted'):
            review(self.root)

    def test_different_reduction_runtime_rejected_before_numeric_comparison(self):
        protocol = json.loads((self.root / 'PROTOCOL.json').read_text())
        protocol['numpy_version'] = 'different'
        self.json('PROTOCOL.json', protocol); self.receipt()
        with self.assertRaisesRegex(ValueError, 'recorded NumPy version'):
            review(self.root)


if __name__ == '__main__': unittest.main()
