from types import SimpleNamespace
import unittest
import importlib.util
if importlib.util.find_spec('warp') is None:
    raise unittest.SkipTest('requires collection-mpm environment (warp-lang)')

import numpy as np
from collection.mpm.grouped import Group, check_values, no_timer_sync
from collection.mpm.solver import MPM_Simulator_WARP
from collection.mpm.wall_materials import POLICY
from collection.common import digest
from pathlib import Path
from unittest.mock import patch


def backend():
    return SimpleNamespace(calls_checked=2,steps=7,failed=False,
        config={'floor_m':.1,'observation_emergency_depth_m':.06})


class GroupTests(unittest.TestCase):
    def test_source_matches_wall_certificate(self):
        p=Path(__file__).resolve().parents[3]/'collection/mpm/grouped.py'
        self.assertEqual(digest(p),POLICY['grouped_source_sha256'])

    def test_scoped_timer_and_particle_transforms_compile(self):
        for block in (256,64):no_timer_sync(object.__new__(MPM_Simulator_WARP),block)

    def test_invalid_block_rejected(self):
        with self.assertRaises(ValueError):no_timer_sync(object.__new__(MPM_Simulator_WARP),128)

    def test_all_guard_failures_stop(self):
        for index,value in ((0,1),(1,1),(2,1),(5,.01),(6,8.1)):
            b=backend();v=np.array([0,0,0,0,1,.1,0.]);v[index]=value
            with self.assertRaises(FloatingPointError):check_values(b,v)
            self.assertTrue(b.failed)

    def test_floor_alert_not_silently_promoted_to_hard_failure(self):
        b=backend();check_values(b,np.array([0,0,0,1,1,.097,0.]))
        self.assertFalse(b.failed);self.assertEqual(b.first_floor_alert_step,7)

    def test_all_worlds_checked_after_one_failure(self):
        g=object.__new__(Group);bs=[backend(),backend()]
        g.worlds=[SimpleNamespace(backend=b) for b in bs]
        g.status=SimpleNamespace(numpy=lambda:np.array([[1,0,0,0,1,.1,0.],[0,0,0,0,1,.1,0.]]))
        with self.assertRaises(FloatingPointError):g.check()
        self.assertEqual([b.calls_checked for b in bs],[3,3])

    def test_empty_group_rejected(self):
        with self.assertRaises(ValueError):Group([])

    def test_fused_grid_matches_original_on_cpu(self):
        import warp as wp
        from test_wall_filter import CompiledObserverTests
        from collection.mpm.observer import WallObservedBackend
        from collection.mpm.contracts import Clock
        from collection.mpm import grouped
        wp.config.kernel_cache_dir='/tmp/pwm-pre-scale-backend-test-cache';wp.init()
        b,_,_=CompiledObserverTests().backend()
        w=WallObservedBackend(b,Clock(5e-5,1,2))
        with patch.object(grouped,'load',return_value=(None,None,w)):
            result=grouped.grid_test(None,None)
        self.assertTrue(result['velocity_and_events_bit_exact'])


if __name__=='__main__':unittest.main()
