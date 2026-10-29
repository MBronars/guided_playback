import unittest
import importlib.util
import os
import numpy as np
from collection.sim1.geometry import broadphase, cross_mesh_pairs, strict_crossing


class ClearanceBroadphaseTests(unittest.TestCase):
    def test_crossing_a_face_without_approaching_vertices(self):
        cloth=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]])
        body=np.array([[-.001,0.,-.01],[.001,0.,-.01],[0.,0.,.01]])
        ids,surface=broadphase(cloth,body)
        self.assertEqual(len(ids),0);self.assertTrue(surface)

    def test_sphere_proximity_without_surface_overlap(self):
        cloth=np.array([[0.,0.,.005]])
        body=np.array([[-1.,-1.,0.],[1.,-1.,0.],[0.,1.,0.]])
        ids,surface=broadphase(cloth,body)
        self.assertEqual(ids.tolist(),[0]);self.assertFalse(surface)

    def test_separation_and_touch(self):
        cloth=np.array([[0.,0.,1.],[1.,0.,1.],[0.,1.,1.]])
        ids,surface=broadphase(cloth,cloth+[0,0,1])
        self.assertFalse(surface);self.assertEqual(len(ids),0)
        ids,surface=broadphase(cloth,cloth)
        self.assertTrue(surface);self.assertEqual(len(ids),3)

    def test_invalid_radius_and_empty_input(self):
        cloth=np.zeros((3,3))
        for radius in (0.,-1.,np.nan):
            with self.assertRaises(ValueError):broadphase(cloth,cloth,radius)
        with self.assertRaises(ValueError):broadphase(cloth[:0],cloth)


@unittest.skipUnless(os.environ.get('PWM_SIM1_GEOMETRY_TESTS') == '1' and importlib.util.find_spec('open3d'),
                     'Run with PWM_SIM1_GEOMETRY_TESTS=1 in the SIM1 geometry environment')
class CrossMeshPruningTests(unittest.TestCase):
    def test_pruning_preserves_whole_mesh_candidates_and_face_ids(self):
        import open3d as o3d
        for seed in range(30):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                a, b = rng.normal(size=(30, 3)), rng.normal(size=(30, 3))
                a[:9] += 10.
                b[-9:] -= 10.
                faces = np.arange(30).reshape(-1, 3)
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.concatenate((a, b))),
                    o3d.utility.Vector3iVector(np.concatenate((faces, faces + len(a)))))
                reference = []
                for i, j in np.asarray(mesh.get_self_intersecting_triangles()).reshape(-1, 2):
                    if (i < len(faces)) == (j < len(faces)):
                        continue
                    ia, ib = (i, j - len(faces)) if i < len(faces) else (j, i - len(faces))
                    reference.append(dict(robot_face=int(ia), cloth_face=int(ib),
                        classification=strict_crossing(a[faces[ia]], b[faces[ib]])))
                reference.sort(key=lambda row: (row['robot_face'], row['cloth_face']))
                self.assertEqual(cross_mesh_pairs(a, faces, b, faces), reference)

    def test_small_body_crosses_face_far_from_all_vertices(self):
        cloth = np.array([[-1., -1., 0.], [1., -1., 0.], [0., 1., 0.]])
        body = np.array([[-.001, 0., -.01], [.001, 0., -.01], [0., 0., .01]])
        faces = np.array([[0, 1, 2]])
        result = cross_mesh_pairs(body, faces, cloth, faces)
        self.assertEqual(result, [dict(robot_face=0, cloth_face=0,
                                      classification='proper_noncoplanar_crossing')])
        self.assertEqual(cross_mesh_pairs(body + [0., 0., 1.], faces, cloth, faces), [])
