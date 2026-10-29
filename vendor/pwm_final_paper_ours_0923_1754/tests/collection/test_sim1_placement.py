import unittest
import numpy as np
from collection.sim1.placement import quarter_turn_input, swept_drop_aabb_clear
from test_sim1_inputs import material


class PlacementTests(unittest.TestCase):
    def test_rigid_rotation_preserves_all_material_fields_and_height(self):
        source=material();original={k:v.copy() for k,v in source.items()}
        for turns in range(4):
            placed,receipt=quarter_turn_input(source,turns=turns,center_xy=[.7,.1])
            for key in source:
                np.testing.assert_array_equal(source[key],original[key])
                if key!='vertices_world_m':np.testing.assert_array_equal(placed[key],source[key])
            q,r=source['vertices_world_m'],placed['vertices_world_m']
            np.testing.assert_array_equal(q[:,2],r[:,2])
            np.testing.assert_allclose(np.linalg.norm(np.diff(q,axis=0),axis=-1),
                                       np.linalg.norm(np.diff(r,axis=0),axis=-1))
            self.assertFalse(receipt['collision_qualified'])

    def test_invalid_transform_is_not_a_silent_noop(self):
        for turns in (-1,4,1.5,True):
            with self.assertRaises(ValueError):quarter_turn_input(material(),turns=turns,center_xy=[0,0])
        with self.assertRaises(ValueError):quarter_turn_input(material(),turns=0,center_xy=[np.nan,0])

    def test_swept_volume_catches_intermediate_overlap(self):
        q=np.array([[0.,0.,1.],[.1,.1,1.]])
        mesh=dict(shape_id=8,vertices=np.array([[.03,.03,.90],[.06,.06,.92]]))
        self.assertFalse(swept_drop_aabb_clear(q,[mesh])['clear'])
        mesh['vertices']+=np.array([1.,0,0])
        self.assertTrue(swept_drop_aabb_clear(q,[mesh])['clear'])

    def test_empty_robot_and_upward_drop_are_not_certified_clear(self):
        q=np.ones((3,3))
        with self.assertRaises(ValueError):swept_drop_aabb_clear(q,[])
        with self.assertRaises(ValueError):swept_drop_aabb_clear(q,[dict(shape_id=0,vertices=q)],floor_center_z=2.)
