from pathlib import Path
import tempfile
import unittest
import numpy as np
from collection.sim1.export import check_export,qualify_files,outgoing_state_deltas


class ExportTests(unittest.TestCase):
    def setUp(self):
        pose=np.zeros((3,7));pose[:,6]=1
        self.a=dict(cloth_points=np.zeros((3,4,3),np.float32),
            left_ee_points=np.zeros((3,6,3),np.float32),right_ee_points=np.zeros((3,6,3),np.float32),
            ee_link_ids=np.array([0,0,1,1,2,2]),joint_q=np.zeros((3,19)),openness=np.ones((3,2)),
            left_ee_pose=pose.copy(),right_ee_pose=pose.copy(),fps=np.array(60.),table_top_z=np.array(.8))

    def test_valid_schema_is_not_physics_acceptance(self):
        r=check_export(self.a,6);self.assertTrue(r['schema_pass']);self.assertFalse(r['accepted_for_training'])
        self.assertIn('unverified',r['time_status'])

    def test_malformed_link_ids_caught(self):
        self.a['ee_link_ids']=self.a['ee_link_ids'][:-1]
        with self.assertRaises(ValueError):check_export(self.a,6)

    def test_scrambled_link_histogram_is_insufficient(self):
        self.a['ee_link_ids']=np.array([0,1,2,0,1,2])
        with self.assertRaises(ValueError):check_export(self.a,6)

    def test_mismatched_state_lengths_caught(self):
        self.a['joint_q']=self.a['joint_q'][:-1]
        with self.assertRaises(ValueError):check_export(self.a,6)

    def test_velocity_or_delta_cannot_silently_replace_state_displacement(self):
        self.a['left_ee_pose'][:,0]=[0,.1,.2]
        self.a['delta_left_ee_pos']=np.array([[6,0,0],[6,0,0],[0,0,0.]])
        with self.assertRaises(ValueError):check_export(self.a,6)
        self.a['delta_left_ee_pos']/=60
        self.assertTrue(check_export(self.a,6)['schema_pass'])

    def test_partial_file_failure_and_second_file_examined(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);good=root/'good.npz';bad=root/'bad.npz'
            np.savez_compressed(good,**self.a)
            bad.write_bytes(b'not a completed zip')
            r=qualify_files([good,bad],expected_ee_points=6)
            self.assertEqual((r['requested'],r['examined'],r['schema_pass'],r['rejected']),(2,2,1,1))
            self.assertEqual(r['accepted_for_training'],0)

    def test_missing_inputs_and_empty_request_fail(self):
        with self.assertRaises(ValueError):qualify_files([])
        r=qualify_files(['/no/such/sim1/episode.npz'],expected_ee_points=6)
        self.assertEqual(r['rejected'],1)

    def test_explicit_times_must_agree_with_fps(self):
        self.a['state_time_s']=np.array([0,.1,.2]);self.a['source_state_index']=np.arange(3)
        with self.assertRaises(ValueError):check_export(self.a,6)
        self.a['state_time_s']=np.arange(3)/60
        self.assertTrue(check_export(self.a,6)['schema_pass'])

    def test_nan_and_zero_pose_quaternion_rejected(self):
        self.a['left_ee_pose'][0,6]=0
        with self.assertRaises(ValueError):check_export(self.a,6)
        self.a['left_ee_pose'][0,6]=1;self.a['fps']=np.array(np.nan)
        with self.assertRaises(ValueError):check_export(self.a,6)

    def test_trim_then_rederive_terminal_deltas(self):
        self.a['joint_q'][:,3]=[0,.1,.3]
        self.a['left_ee_pose'][:,0]=[0,.02,.03]
        original=outgoing_state_deltas(self.a)
        trimmed={k:(v[:2].copy() if np.ndim(v)>0 and k!='ee_link_ids' else v)
                 for k,v in self.a.items()}
        trimmed.update({k:v[:2].copy() for k,v in original.items()})
        with self.assertRaises(ValueError):check_export(trimmed,6)
        fixed=outgoing_state_deltas(trimmed)
        self.assertNotEqual(trimmed['delta_joint_q'][-1,3],0)
        trimmed.update(fixed)
        self.assertTrue(check_export(trimmed,6)['schema_pass'])
        self.assertFalse(trimmed['transition_valid'][-1])
        np.testing.assert_array_equal(self.a['joint_q'][:,3],[0,.1,.3])

    def test_quaternion_delta_world_order_and_sign_equivalence(self):
        # Independent SciPy-generated reference for xyz Euler degrees
        # [[0,0,0],[30,50,0],[80,-10,20]]. Keep numeric tests simulator/data-extra free.
        self.a['left_ee_pose'][:,3:]=[[0.,0.,0.,1.],
            [.23456971600980447,.40821789367673483,-.109381654946615,.875426098065593],
            [.642207031413782,.04544329401881429,.18768755374173832,.7418075343388333]]
        self.a.update(outgoing_state_deltas(self.a))
        expected=np.array([[.23456971600980447,.40821789367673483,-.109381654946615,.875426098065593],
            [.46978729356085647,-.37730834770935934,-.006054062510123638,.7980611868050735]])
        np.testing.assert_allclose(np.abs(np.sum(expected*self.a['delta_left_ee_quat'][:-1],axis=1)),1)
        self.a['delta_left_ee_quat']*=-1
        self.assertTrue(check_export(self.a,6)['schema_pass'])
        self.a['delta_left_ee_quat'][-1]=[np.sin(np.deg2rad(10)),0,0,np.cos(np.deg2rad(10))]
        with self.assertRaises(ValueError):check_export(self.a,6)

    def test_terminal_padding_not_an_observed_transition(self):
        self.a.update(outgoing_state_deltas(self.a))
        self.a['transition_valid'][-1]=True
        with self.assertRaises(ValueError):check_export(self.a,6)


if __name__=='__main__':unittest.main()
