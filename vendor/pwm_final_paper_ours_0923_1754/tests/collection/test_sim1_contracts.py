import unittest
import numpy as np

from collection.sim1.contracts import (geometry_diagnostics, integer_state_times, points_from_recorded_links,
                       principal_stretches, require_complete_usd_timeline,
                       require_link_membership, resample_both_arms, snap_material_vertex)


class ContractTests(unittest.TestCase):
    def poses(self):
        p = np.zeros((2,7)); p[:,6] = 1
        return p

    def test_right_only_translation_capped(self):
        l, r = self.poses(), self.poses(); r[1,0] = .101
        L,R,O,I = resample_both_arms(l,r,np.zeros((2,2)))
        self.assertLessEqual(np.linalg.norm(np.diff(R[:,:3],axis=0),axis=1).max(), .004+1e-12)
        np.testing.assert_allclose(L[:,:3],0)
        np.testing.assert_allclose(R[-1],r[-1]); self.assertEqual(I[-1],1)

    def test_openness_is_not_a_jump(self):
        p = self.poses(); p[1,0] = .01
        L,R,O,I = resample_both_arms(p,p,[[0,1],[1,0]])
        self.assertLessEqual(np.abs(np.diff(O,axis=0)).max(),.025+1e-12)
        np.testing.assert_allclose(O[0],[0,1]); np.testing.assert_allclose(O[-1],[1,0])

    def test_antipodal_quaternions_remain_finite(self):
        p = self.poses(); p[1,6] = -1; p[1,0] = .004
        L,R,_,_ = resample_both_arms(p,p,np.zeros((2,2)))
        self.assertTrue(np.isfinite(L).all())
        np.testing.assert_allclose(np.linalg.norm(R[:,3:],axis=1),1)

    def test_rotation_only_motion_capped(self):
        p = self.poses(); p[1,3:] = [0,0,1,0]
        L,_,_,_ = resample_both_arms(p,self.poses(),np.zeros((2,2)))
        angles = 2*np.arccos(np.clip(np.abs(np.sum(L[1:,3:]*L[:-1,3:],axis=1)),0,1))
        self.assertLessEqual(angles.max(),.05+1e-10)

    def test_resample_rejects_bad_inputs(self):
        p = self.poses()
        for limit in [0,-1,np.nan,np.inf]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                resample_both_arms(p,p,np.zeros((2,2)),max_step_m=limit)
        with self.assertRaises(ValueError): resample_both_arms(p,p,np.full((2,2),1.1))
        with self.assertRaises(ValueError): resample_both_arms(p[:0],p[:0],np.zeros((0,2)))
        with self.assertRaises(ValueError): resample_both_arms(p,p[:1],np.zeros((2,2)))
        p[0,3:] = 0
        with self.assertRaises(ValueError): resample_both_arms(p,p,np.zeros((2,2)))

    def test_frame_budget_is_checked_before_allocation(self):
        p = self.poses(); p[1,0] = 10000
        with self.assertRaises(ValueError):
            resample_both_arms(p,p,np.zeros((2,2)),max_frames=20)

    def test_frame_budget_cannot_be_infinite(self):
        p=self.poses()
        for limit in (np.inf,np.nan,1.5,True,0):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                resample_both_arms(p,p,np.zeros((2,2)),max_frames=limit)

    def test_snap_after_jitter_returns_real_particle(self):
        q = np.array([[0,0,.81],[.01,0,.81],[.02,0,.81]])
        r = snap_material_vertex(q,[0,0],jitter_xy=[.012,0],min_support=2)
        self.assertEqual(r['particle_id'],1); np.testing.assert_array_equal(r['point_m'],q[1])
        self.assertFalse(r['grasp_success_verified'])

    def test_grasp_missing_material_or_support_fails_closed(self):
        q = np.array([[0,0,.81],[1,0,.81],[2,0,.81]])
        with self.assertRaises(ValueError): snap_material_vertex(q,[.5,.5])
        with self.assertRaises(ValueError): snap_material_vertex(q,[0,0])
        with self.assertRaises(ValueError): snap_material_vertex(np.zeros((0,3)),[0,0])

    def test_grasp_limits_cannot_be_nonfinite(self):
        q=np.array([[0,0,.8],[.001,0,.8],[.002,0,.8]])
        for options in ({'max_snap_m':np.nan},{'support_radius_m':np.inf},{'min_support':1.5}):
            with self.subTest(options=options),self.assertRaises(ValueError):snap_material_vertex(q,[0,0],**options)

    def test_integer_clock_no_accumulation(self):
        t = integer_state_times(np.arange(200001),60)
        np.testing.assert_allclose(t*60,np.arange(200001),rtol=0,atol=3e-11)
        self.assertEqual(t[-1],200000/60)
        self.assertTrue(np.all(np.diff(t)>0))
        for indices in ([1,2],[0,2],[0,1,1]):
            with self.assertRaises(ValueError): integer_state_times(indices,60)

    def test_integer_clock_rejects_float_indices(self):
        with self.assertRaises(ValueError): integer_state_times(np.array([0.,1.]),60)

    def test_usd_timeline_cannot_invent_zero_or_gap(self):
        require_complete_usd_timeline([0,1,2],np.arange(3),60,60)
        for times in ([1,2],[0,1,3],[0,1,1],[0,2]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                require_complete_usd_timeline(times,np.arange(3),60,60)

    def test_usd_separate_timecode_rate(self):
        expected=require_complete_usd_timeline([0,2,4],np.arange(3),120,60)
        np.testing.assert_array_equal(expected,[0,2,4])

    def test_link_membership_requires_order_not_histogram(self):
        np.testing.assert_array_equal(require_link_membership([0,0,1,1,2,2],6),[2,2,2])
        with self.assertRaises(ValueError): require_link_membership([0,1,2,0,1,2],6)

    def test_link_membership_rejects_malformed(self):
        for ids,n in [([0,1,2],1000),([0,1,3],3),([0.,1.,2.],3),([0,0,1],3)]:
            with self.subTest(ids=ids),self.assertRaises(ValueError): require_link_membership(ids,n)

    def test_points_use_each_recorded_link(self):
        p = np.array([[1,0,0],[1,0,0],[1,0,0.]])
        poses = np.zeros((2,3,7)); poses[:,:,6]=1; poses[1,:,0]=[2,3,4]
        poses[1,1,3:]=[0,0,1,0]
        out=points_from_recorded_links(p,[0,1,2],poses)
        np.testing.assert_allclose(out[0,:,0],[1,1,1])
        np.testing.assert_allclose(out[1,:,0],[3,2,5])

    def test_materialization_rejects_float32_overflow(self):
        poses=np.zeros((2,3,7));poses[:,:,6]=1;poses[:,:,0]=1e100
        with self.assertRaises(ValueError):points_from_recorded_links(np.zeros((3,3)),[0,1,2],poses)

    def test_mesh_metric_does_not_connect_nearby_layers(self):
        q=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,.001],[1,0,.001],[0,1,.001]])
        f=np.array([[0,1,2],[3,4,5]])
        inv=np.repeat(np.eye(2)[None],2,axis=0)
        q[3:,2]+=2
        np.testing.assert_allclose(principal_stretches(q,f,inv),1)

    def test_mesh_metric_detects_in_plane_stretch(self):
        q=np.array([[0.,0,0],[1.5,0,0],[0,1,0]])
        np.testing.assert_allclose(principal_stretches(q,[[0,1,2]],np.eye(2)[None]),[[1.5,1]])

    def test_quality_every_frame_and_table_footprint(self):
        q=np.array([[[0.,0,.81],[2,0,.1]],[[0,0,.79],[2,0,.1]],[[0,0,.81],[2,0,.1]]])
        r=geometry_diagnostics(q,[0,.1,.2],table_top_z=.8,table_xy_bounds=[[-1,1],[-1,1]])
        self.assertAlmostEqual(r['max_table_penetration_m'],.01)
        self.assertEqual(r['examined_frames'],3)
        self.assertEqual(r['per_frame_outside_table_vertices'],[1,1,1])
        self.assertFalse(r['accepted_for_training'])

    def test_quality_uses_seconds(self):
        q=np.array([[[0.,0,.81]],[[.01,0,.81]]])
        r=geometry_diagnostics(q,[0,.1],table_top_z=.8,table_xy_bounds=[[-1,1],[-1,1]])
        self.assertAlmostEqual(r['max_particle_speed_m_s'],.1)

    def test_quality_nonfinite_short_or_bad_time_rejected(self):
        q=np.zeros((2,1,3))
        for t in ([0,0],[1,0],[0,np.nan]):
            with self.assertRaises(ValueError):
                geometry_diagnostics(q,t,table_top_z=.8,table_xy_bounds=[[-1,1],[-1,1]])
        q[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            geometry_diagnostics(q,[0,1],table_top_z=.8,table_xy_bounds=[[-1,1],[-1,1]])


if __name__ == '__main__': unittest.main()
