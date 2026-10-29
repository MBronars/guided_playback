import unittest
import numpy as np
from collection.sim1.recording import StateRecorder


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.rec=StateRecorder(60,2,3,4,max_states=10)
        b=np.zeros((3,7)); b[:,6]=1
        self.s=dict(particle_q=np.zeros((2,3)),particle_qd=np.zeros((2,3)),
                    body_q=b,joint_q=np.zeros(4))
        self.c=dict(command_joint_q=np.ones(4),command_openness=np.ones(2))

    def test_realized_states_and_commands_are_distinct(self):
        self.rec.append_initial(**self.s)
        self.rec.append_transition(0,**self.c,**self.s)
        a=self.rec.arrays(1)
        np.testing.assert_array_equal(a['actual_joint_q'],0)
        np.testing.assert_array_equal(a['command_joint_q'],1)
        self.assertEqual(a['body_q'].shape[0],2)
        self.assertEqual(a['command_joint_q'].shape[0],1)
        np.testing.assert_allclose(a['state_time_s'],[0,1/60])

    def test_incomplete_replay_is_not_complete(self):
        self.rec.append_initial(**self.s)
        with self.assertRaises(ValueError): self.rec.arrays(1)
        self.rec.append_transition(0,**self.c,**self.s)
        with self.assertRaises(ValueError): self.rec.arrays(2)

    def test_duplicate_initial_rejected(self):
        self.rec.append_initial(**self.s)
        with self.assertRaises(ValueError): self.rec.append_initial(**self.s)

    def test_no_missing_initial_or_skipped_transition(self):
        with self.assertRaises(ValueError): self.rec.append_transition(0,**self.c,**self.s)
        self.rec.append_initial(**self.s)
        with self.assertRaises(ValueError): self.rec.append_transition(1,**self.c,**self.s)

    def test_buffers_are_copied(self):
        self.rec.append_initial(**self.s)
        self.s['particle_q'][:]=3
        self.rec.append_transition(0,**self.c,**self.s)
        self.s['particle_q'][:]=9
        a=self.rec.arrays(1)
        np.testing.assert_array_equal(a['particle_q'][0],0)
        np.testing.assert_array_equal(a['particle_q'][1],3)

    def test_nonfinite_and_bad_quaternion_rejected(self):
        self.s['particle_q'][0,0]=np.nan
        with self.assertRaises(ValueError): self.rec.append_initial(**self.s)
        self.s['particle_q'][0,0]=0; self.s['body_q'][0,6]=0
        with self.assertRaises(ValueError): self.rec.append_initial(**self.s)

    def test_finite_float64_state_overflow_does_not_advance(self):
        self.s['particle_q'][0,0]=1e100
        with self.assertRaises(ValueError):self.rec.append_initial(**self.s)
        self.assertEqual(self.rec.states,[])

    def test_finite_float64_command_overflow_does_not_advance(self):
        self.rec.append_initial(**self.s);self.c['command_joint_q'][0]=1e100
        with self.assertRaises(ValueError):self.rec.append_transition(0,**self.c,**self.s)
        self.assertEqual(len(self.rec.states),1);self.assertEqual(self.rec.commands,[])

    def test_failed_transition_does_not_advance_recorder(self):
        self.rec.append_initial(**self.s); self.c['command_openness'][:]=2
        with self.assertRaises(ValueError): self.rec.append_transition(0,**self.c,**self.s)
        self.assertEqual(len(self.rec.states),1); self.assertEqual(len(self.rec.commands),0)

    def test_recorder_has_memory_and_state_budgets(self):
        with self.assertRaises(ValueError): StateRecorder(60,1000000,3,4,max_bytes=1024)
        self.rec=StateRecorder(60,2,3,4,max_states=1); self.rec.append_initial(**self.s)
        with self.assertRaises(ValueError): self.rec.append_transition(0,**self.c,**self.s)


if __name__=='__main__': unittest.main()
