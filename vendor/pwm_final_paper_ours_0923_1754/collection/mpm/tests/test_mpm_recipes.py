import os
import unittest
import numpy as np

from collection.mpm.recipes import load_bank, scene, specifications


@unittest.skipUnless(os.environ.get("PWM_MPM_REFERENCE_ROOT"), "requires explicit frozen reference assets")
class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata, cls.objects = load_bank()

    def test_complete_predeclared_matrix(self):
        specs = specifications()
        self.assertEqual(len(specs), 30)
        self.assertEqual(len({(s['scenario'], s['count'], s['seed']) for s in specs}), 30)
        self.assertEqual([s['index'] for s in specs], list(range(30)))

    def test_all_inputs_exact_source_points_weights_and_rigid_maps(self):
        for spec in specifications():
            with self.subTest(spec=spec):
                initial, geometry = scene(spec, self.metadata, self.objects)
                repeated, again = scene(spec, self.metadata, self.objects)
                self.assertEqual(geometry, again)
                for key in initial: np.testing.assert_array_equal(initial[key], repeated[key])
                np.testing.assert_array_equal(np.bincount(initial['object_ids']), [1024] * spec['count'])
                self.assertEqual(len(set(geometry['asset_ids'])), spec['count'])
                for oid, transform in enumerate(geometry['transforms']):
                    source = self.objects[transform['bank_index']]
                    rotation = np.array(transform['rotation'])
                    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
                    self.assertAlmostEqual(np.linalg.det(rotation), 1)
                    expected = ((source['source_x'].astype(float) - transform['source_midpoint_m']) @ rotation.T
                                + transform['translation_after_centering_m']).astype(np.float32)
                    selected = initial['object_ids'] == oid
                    np.testing.assert_array_equal(initial['x'][selected], expected)
                    np.testing.assert_array_equal(initial['particle_volume'][selected], source['particle_volume'])
                    np.testing.assert_array_equal(initial['source_particle_index'][selected], source['source_particle_index'])
                self.assertTrue(np.isfinite(initial['x']).all())

    def test_wall_failure_is_retained_not_repositioned(self):
        metadata, objects = self.metadata, [dict(o) for o in self.objects]
        for obj in objects:
            obj['source_x'] = obj['source_x'] * 10
        _, geometry = scene(specifications()[0], metadata, objects)
        self.assertIn('initial_physical_wall_bounds', geometry['placement_reasons'])

    def test_invalid_family_fails(self):
        with self.assertRaises(ValueError): scene(dict(count=7, scenario='unknown'), self.metadata, self.objects)


if __name__ == '__main__': unittest.main()
