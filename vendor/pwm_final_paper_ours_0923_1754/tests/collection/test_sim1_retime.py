import unittest
import importlib.util
import numpy as np
from collection.sim1.retime import retime_return, metrics
from collection.sim1.recording import StateRecorder


class RetimeTests(unittest.TestCase):
    def test_corrected_tee_retimed_recorder_fits_explicit_two_gib(self):
        with self.assertRaises(ValueError): StateRecorder(60,12859,22,19,max_states=2101)
        r=StateRecorder(60,12859,22,19,max_states=2101,max_bytes=2*1024**3)
        self.assertLess(r.estimated_bytes,2*1024**3)

    @staticmethod
    def fk(q):
        return np.array([[q[0], 0, 0, 0, 0, 0, 1], [q[1], 0, 0, 0, 0, 0, 1]])

    def setup_plan(self):
        q = np.zeros((240, 2), np.float32)
        t = np.linspace(0, 1, 60)
        q[60:120] = (.4 * (10*t**3 - 15*t**4 + 6*t**5))[:, None]
        q[120:] = .4
        op = np.ones((240, 2), np.float32); op[:55] = 0
        return dict(command_joint_q=q, command_openness=op, source_command_index=np.arange(240)), dict(
            suffix_frames=120, policy='no retiming', source_index_sentinels={'-2':'hold'})

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'Requires optional collection SciPy environment')
    def test_prefix_path_terminal_and_speed(self):
        c, m = self.setup_plan()
        new, meta = retime_return(c, m, self.fk)
        start = meta['return_retime']['unchanged_prefix_commands']
        np.testing.assert_array_equal(new['command_joint_q'][:start], c['command_joint_q'][:start])
        np.testing.assert_array_equal(new['command_openness'][:start], c['command_openness'][:start])
        np.testing.assert_array_equal(new['command_joint_q'][-1], c['command_joint_q'][-1])
        self.assertEqual(len(new['command_joint_q']) % 2, 0)
        self.assertGreater(len(new['command_joint_q']), len(c['command_joint_q']))
        self.assertTrue(np.all(np.diff(new['original_command_fraction']) >= 0))
        for k, bound in meta['return_retime']['sampled_command_limits'].items():
            self.assertLessEqual(meta['return_retime']['sampled_command_metrics'][k], bound)

    def test_refuse_gripper_still_closed(self):
        c, m = self.setup_plan(); c['command_openness'][-1] = 0
        with self.assertRaises(ValueError): retime_return(c, m, self.fk)

    def test_refuse_budget_and_invalid_limit(self):
        c, m = self.setup_plan()
        with self.assertRaises(ValueError): retime_return(c, m, self.fk, maximum_frames=240)
        with self.assertRaises(ValueError): retime_return(c, m, self.fk, speed=np.nan)

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'Requires optional collection SciPy environment')
    def test_quaternion_sign_does_not_create_angular_spike(self):
        q = np.zeros((5,2)); poses=np.stack([self.fk(row) for row in q]);poses[2,:,3:]*=-1
        result=metrics(q,poses)
        self.assertEqual(result['angular_speed'],0)


if __name__ == '__main__': unittest.main()
