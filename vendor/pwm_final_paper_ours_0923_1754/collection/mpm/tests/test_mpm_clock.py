"""Material-aware fixed clocks are stability budgets, not accuracy certificates."""
import unittest
import numpy as np
from collection.mpm.clock import choose_clock, wave_speeds
from collection.mpm.contracts import Clock


def static(E,nu=.2,density=1000.):
    return dict(E=np.atleast_1d(E).astype(np.float32),nu=np.atleast_1d(nu).astype(np.float32),
                density=np.atleast_1d(density).astype(np.float32))


class ClockTests(unittest.TestCase):
    def test_refined_grid_uses_actual_spacing_and_preserves_saved_times(self):
        base,_=choose_clock(static(1e6))
        fine,budget=choose_clock(static(1e6),dx_m=(100/64)/256)
        self.assertEqual(fine.substep_dt_s,50e-6)
        self.assertAlmostEqual(budget['initial_budgeted_courant'],.33860266717529297,places=6)
        np.testing.assert_array_equal(base.state_time_s,fine.state_time_s)
        stiff,_=choose_clock(static(1e7),dx_m=(100/64)/256)
        self.assertLess(stiff.substep_dt_s,base.substep_dt_s)
        for dx in (0,-1,float('nan')):
            with self.assertRaises(ValueError):choose_clock(static(1e6),dx_m=dx)

    def test_extreme_material_buckets_and_identical_saved_times(self):
        cases=[(1e4,.2,1000.,50e-6),(1e6,.2,1000.,50e-6),(1e7,.2,1000.,50e-6),
               (5e7,.2,1000.,25e-6),(1e8,.2,1000.,12.5e-6),
               (1e6,.499,1000.,12.5e-6),(1e6,.4995,1000.,6.25e-6),(1e6,.2,10.,12.5e-6)]
        for E,nu,rho,dt in cases:
            with self.subTest(E=E,nu=nu,rho=rho):
                clock,budget=choose_clock(static(E,nu,rho))
                self.assertEqual(clock.substep_dt_s,dt)
                self.assertLessEqual(budget['initial_budgeted_courant'],.4)
                self.assertIs(budget['accuracy_certificate'],False)
                np.testing.assert_array_equal(clock.state_time_s,Clock(1e-4,417,48).state_time_s)

    def test_stiffest_object_sets_scene_clock(self):
        clock,_=choose_clock(static([1e4,1e8]))
        self.assertEqual(clock.substep_dt_s,12.5e-6)

    def test_unrepresentable_clock_and_invalid_materials_fail(self):
        for E,nu,rho in [(1e10,.2,1000.),(1e6,.4999,1000.),(-1,.2,1000.),
                         (1e6,.5,1000.),(1e6,-1,1000.),(1e6,.2,0.),(np.nan,.2,1000.)]:
            with self.subTest(E=E,nu=nu,rho=rho),self.assertRaises(ValueError):choose_clock(static(E,nu,rho))

    def test_linear_wave_speed_independent_formula(self):
        E,nu,rho=2e6,.3,800.
        expected=np.sqrt(E*(1-nu)/((1+nu)*(1-2*nu)*rho))
        self.assertAlmostEqual(float(wave_speeds(E,nu,rho)),expected)

    def test_output_interval_is_archived_clock_not_exact_24hz(self):
        clock,_=choose_clock(static(1e6))
        self.assertEqual(len(clock.state_time_s),49)
        self.assertAlmostEqual(clock.state_time_s[-1],2.0016)
        self.assertNotEqual(clock.state_time_s[1],1/24)

if __name__=='__main__':unittest.main()
