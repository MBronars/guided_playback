import unittest
import numpy as np
from collection.sim1.fold_plan import select_targets, quintic, FAMILIES


class FoldTargetsTests(unittest.TestCase):
    def setUp(self):
        x,y=np.meshgrid(np.linspace(.43,.95,41),np.linspace(-.3,.3,41))
        self.points=np.stack([x.ravel(),y.ravel(),np.full(x.size,.815)],axis=-1)

    def test_targets_are_observed_vertices(self):
        for family in FAMILIES:
            targets=select_targets(self.points,family)
            for target in targets.values():
                np.testing.assert_array_equal(target['grasp'],self.points[target['vertex']])

    def test_families_are_materially_distinct(self):
        a,b,c=[select_targets(self.points,f) for f in FAMILIES]
        self.assertEqual(list(a),[0]);self.assertEqual(list(b),[1]);self.assertEqual(list(c),[0,1])
        self.assertLess(a[0]['destination'][1],a[0]['grasp'][1])
        self.assertGreater(b[1]['destination'][0],b[1]['grasp'][0])
        self.assertGreater(b[1]['destination'][1],b[1]['grasp'][1])
        for side in (0,1):
            self.assertAlmostEqual(c[side]['destination'][1],c[side]['grasp'][1])
            self.assertAlmostEqual(c[side]['destination'][0]-c[side]['grasp'][0],.22)

    def test_reject_bad_input(self):
        for points in ([],[[np.nan,0,0]],[[1,2]]):
            with self.assertRaises(ValueError):select_targets(points,'left-inward')
        with self.assertRaises(ValueError):select_targets(self.points,'unknown')
        with self.assertRaises(ValueError):select_targets([[.5,0,.81],[.6,.01,.81]],'two-hand-half')

    def test_quintic_monotone_and_rest_boundaries(self):
        u=np.linspace(0,1,10001);q=quintic(u)
        self.assertEqual(q[0],0);self.assertEqual(q[-1],1)
        self.assertTrue(np.all(np.diff(q)>=-1e-14))
        self.assertLess(abs(q[1]-q[0])*10000,1e-6)
        self.assertLess(abs(q[-1]-q[-2])*10000,1e-6)


if __name__=='__main__':unittest.main()
