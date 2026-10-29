import json
import unittest
import numpy as np
from collection.sim1.settling import SettlingGate, SettlingAndHold


class SettlingTests(unittest.TestCase):
    def test_all_substeps_required(self):
        gate = SettlingGate(.01)
        with self.assertRaises(ValueError):
            gate.observe(2, np.array([[.65, 0, .81]]), np.zeros((1, 3)))

    def test_overflow_is_permanent(self):
        gate = SettlingGate(.01)
        q = np.array([[.65, 0, .81]])
        gate.observe(1, q, np.zeros_like(q), 1)
        for step in range(2, 101):
            row = gate.observe(step, q, np.zeros_like(q))
            self.assertFalse(row['passed'])

    def test_hold_failure_does_not_restart(self):
        hold = SettlingAndHold(.1, 1., .3)
        for step in range(1, 7):
            row = dict(step=step, passed=step >= 3, low_motion=step != 4, guard=None)
            result = hold.observe(row)
        self.assertEqual(result['status'], 'motion_gate_passed_hold_failed')
        self.assertEqual(result['post_gate_failed_substeps'], 1)
        self.assertTrue(result['post_hold_observation_complete'])
        with self.assertRaises(ValueError): hold.observe(dict(row, step=7))

    def test_late_pass_gets_full_hold(self):
        hold = SettlingAndHold(.1, 1., .3)
        for step in range(1, 14):
            result = hold.observe(dict(step=step, passed=step >= 10, low_motion=True, guard=None))
            self.assertEqual(result['stopping'], step == 13)
        self.assertEqual(result['status'], 'motion_gate_and_hold_passed')

    def test_large_finite_velocity_is_not_a_json_or_quiet_pass(self):
        row = SettlingGate(.01).observe(1, np.array([[.65, 0, .81]], np.float32), np.full((1, 3), 1e30, np.float32))
        self.assertFalse(row['low_motion'])
        self.assertEqual(row['guard'], 'extreme_speed_or_position')
        json.dumps(row, allow_nan=False)
