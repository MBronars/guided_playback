import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from collection.sim1.replay import command_sequence, save_chunk, shared_memory_check, save_failure
from collection.sim1.recording import StateRecorder


class ReplayTests(unittest.TestCase):
    def test_failure_retained_even_with_nonfinite_endpoint(self):
        with tempfile.TemporaryDirectory() as d:
            rows = [dict(step=1, guard='nonfinite_state')]
            save_failure(d, 0, rows, dict(particle_q=np.full((24, 2, 3), np.nan)))
            proof = json.loads((Path(d) / 'FAILURE.json').read_text())
            self.assertEqual(proof['failed_substeps'][0]['guard'], 'nonfinite_state')
            self.assertFalse(proof['complete_episode'])
            with np.load(Path(d) / 'FAILED_FRAME.npz') as z:
                self.assertTrue(np.isnan(z['particle_q']).all())

    def test_shared_gpu_memory_guards(self):
        shared_memory_check(97887, 13000, 0, starting=True)
        shared_memory_check(97887, 17000, 4000)
        for total, used, own, starting in ((97887, 70000, 0, True),
                (97887, 84000, 4000, False), (97887, 50000, 17000, False),
                (100, 101, 0, False), (97887, 100, 200, False)):
            with self.assertRaises(RuntimeError):
                shared_memory_check(total, used, own, starting=starting)

    def setUp(self):
        self.home = np.zeros(6, np.float32)
        self.lower = np.zeros(6, np.float32)
        self.upper = np.ones(6, np.float32)
        self.ids = np.array([[2, 3], [4, 5]])
        self.raw = dict(joint_q=np.zeros((5, 6), np.float32),
                        openness=np.zeros((5, 2), np.float32),
                        base_transform=np.tile([0, 0, .8, 0, 0, 0, 1], (5, 1)))

    def plan(self, **kwargs):
        return command_sequence(self.raw, self.home, self.lower, self.upper, self.ids, **kwargs)

    def test_hold_clock_and_effective_jaws(self):
        self.raw['openness'][2:] = .1
        arrays, meta = self.plan()
        self.assertEqual(arrays['command_joint_q'].shape, (184, 6))
        np.testing.assert_array_equal(arrays['source_command_index'][60:64], [1, 2, 3, 4])
        np.testing.assert_array_equal(arrays['command_joint_q'][61, 2:], np.float32(.1))
        self.assertFalse(meta['achieved_task_verified'])

    def test_explicit_small_limit_projection(self):
        self.raw['joint_q'][3, 0] = -2e-5
        a, m = self.plan()
        self.assertGreater(m['maximum_joint_limit_projection'], 1e-5)
        self.assertEqual(a['command_joint_q'][62, 0], 0.)
        self.raw['joint_q'][3, 0] = -.001
        with self.assertRaises(ValueError): self.plan()

    def test_bounded_openness_projection(self):
        self.raw['openness'][3, 0] = -.005
        _, m = self.plan()
        self.assertEqual(m['openness_clipped_values'], 1)
        self.raw['openness'][3, 0] = -.02
        with self.assertRaises(ValueError): self.plan()

    def test_reject_moving_base_nonfinite_discontinuity_home(self):
        self.raw['base_transform'][2, 0] = .1
        with self.assertRaises(ValueError): self.plan()
        self.raw['base_transform'][2, 0] = 0
        self.raw['joint_q'][3, 0] = np.nan
        with self.assertRaises(ValueError): self.plan()
        self.raw['joint_q'][3, 0] = .3
        with self.assertRaises(ValueError): self.plan()
        self.raw['joint_q'][3, 0] = 0
        self.raw['joint_q'][1, 0] = .01
        with self.assertRaises(ValueError): self.plan()

    def test_reject_odd_clock_and_bad_jaw_mapping(self):
        with self.assertRaises(ValueError): self.plan(prefix_frames=59)
        self.ids[1] = self.ids[0]
        with self.assertRaises(ValueError): self.plan()

    def test_chunks_preserve_actual_states_and_outgoing_commands(self):
        recorder = StateRecorder(60, 2, 1, 6, max_states=5)
        def state(i):
            return dict(particle_q=np.full((2, 3), i, np.float32), particle_qd=np.zeros((2, 3)),
                        body_q=np.array([[0, 0, 0, 0, 0, 0, 1]]), joint_q=np.full(6, i))
        recorder.append_initial(**state(0))
        with tempfile.TemporaryDirectory() as d:
            for i in range(2):
                recorder.append_transition(i, command_joint_q=np.full(6, i + .1),
                                           command_openness=np.zeros(2), **state(i + 1))
            end = save_chunk(recorder, 0, d)
            for i in range(2, 4):
                recorder.append_transition(i, command_joint_q=np.full(6, i + .1),
                                           command_openness=np.zeros(2), **state(i + 1))
            self.assertEqual(save_chunk(recorder, end, d), 4)
            full = recorder.arrays(4)
            for lo, hi in ((0, 2), (2, 4)):
                p = Path(d) / f'frames-{lo:06d}-{hi:06d}.npz'
                with np.load(p) as z:
                    for key in z.files:
                        target = full[key] if key == 'fps' else full[key][lo:hi + (0 if key.startswith('command_') or key == 'transition_index' else 1)]
                        np.testing.assert_array_equal(z[key], target)
                self.assertFalse(json.loads(p.with_suffix('.json').read_text())['physical_resume_supported'])


if __name__ == '__main__':
    unittest.main()
