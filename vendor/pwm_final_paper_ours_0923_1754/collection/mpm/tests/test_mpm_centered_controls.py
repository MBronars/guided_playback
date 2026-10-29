import os
import unittest
import numpy as np
from collection.mpm.recipes import load_bank, scene, specifications


@unittest.skipUnless(os.environ.get('PWM_MPM_REFERENCE_ROOT'),'explicit frozen source assets required')
class CenteredRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.meta,cls.objects=load_bank()

    def test_fixed_matrix_and_no_initial_rejections(self):
        specs=specifications('centered-v2')
        self.assertEqual(len(specs),90)
        for spec in specs:
            _,p=scene(spec,self.meta,self.objects)
            self.assertEqual(p['recipe_version'],'centered-v2')
            self.assertEqual(p['placement_reasons'],[],msg=str(spec))

    def test_materials_weights_and_point_ids_unchanged(self):
        for old,new in zip(specifications('interaction-v1'),specifications('centered-v2')):
            a,_=scene(old,self.meta,self.objects);b,_=scene(new,self.meta,self.objects)
            for key in a:
                if key not in ('x','v'):np.testing.assert_array_equal(a[key],b[key])

    def test_stiffness_phases_only_change_E(self):
        specs=specifications('centered-v2')
        for i in range(30):
            a,_=scene(specs[i],self.meta,self.objects)
            for phase in (1,2):
                b,_=scene(specs[i+phase*30],self.meta,self.objects)
                for key in a:
                    if key!='E':np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__':unittest.main()
