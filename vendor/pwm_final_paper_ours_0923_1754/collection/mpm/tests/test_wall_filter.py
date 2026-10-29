"""Synthetic fault tests plus real compiled CPU observer/solver equivalence."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import importlib.util

if any(importlib.util.find_spec(name) is None for name in ('warp', 'h5py')):
    raise unittest.SkipTest('requires collection-mpm environment')

import numpy as np
import warp as wp

from collection.mpm.walls import FIELDS, POINTS, NORMALS, metadata, validate_evidence
from collection.mpm.export import collect_passive, export_wall_free, validate_wall_free, POLICY
from collection.mpm.contracts import Clock, identity
from collection.mpm.recording import atomic_raw
from collection.mpm.validate import read_raw, check_tar
from collection.common import digest as sha256
from collection.mpm.observer import WallObservedBackend, Observation, observe_grid, observe_particles, NEVER
from collection.mpm.physics import CollectionBackend
from collection.mpm.physics import install_original_walls_in_padded_grid


def synthetic(index=0, wall=None):
    clock = Clock(5e-5, 5, 2)
    case = dict(release_id='wall-unit-tests', case_index=index, seed_namespace='synthetic',
                asset_ids=['synthetic-cube'], config_sha256='f' * 64, task='passive')
    source = Path(__file__).resolve()
    m = dict(case=case, episode_id=identity(case)[0], task='passive', n_objects=1,
             clock=asdict(clock), floor_m=.1, grid_lim_m=100/64, n_grid=100,
             volume_policy='synthetic-test-only', source_sha256={str(source): sha256(source)},
             units=dict(position='m', velocity='m/s', time='s', volume='m^3', density='kg/m^3', E='Pa'),
             population='unit-test', split='diagnostic', scenario='drop', material_names=['elastic'],
             wall_observation=metadata(sha256(source)))
    a = dict(x=np.tile(np.array([.75, .75, .3], np.float32), (3, 8, 1)),
             v=np.zeros((3, 8, 3), np.float32), F_trial=np.tile(np.eye(3, dtype=np.float32), (3, 8, 1, 1)),
             C=np.zeros((3, 8, 3, 3), np.float32), object_ids=np.zeros(8, np.int32),
             particle_volume=np.full(8, 1e-6, np.float32), E=np.array([0], np.float32),
             nu=np.array([.2], np.float32), density=np.array([1000], np.float32),
             mat_type=np.array([0], np.int32), yield_stress=np.array([0], np.float32),
             state_time_s=clock.state_time_s, selected=np.zeros(8, np.int32),
             command_velocity=np.zeros((8, 3), np.float32), transition_active=np.zeros(2, np.int32),
             active_substeps=np.zeros(2, np.int32), wall_simulation_steps=np.array([0, 5, 10], np.int64),
             wall_covered_substeps=np.repeat(np.array([[0], [5], [10]], np.int64), 5, axis=1),
             wall_first_grid_step=np.full((3, 5), -1, np.int64), wall_first_particle_step=np.full((3, 5), -1, np.int64))
    if wall is not None:
        a['wall_first_grid_step'][1:, wall] = 2
    return m, a


class ContractTests(unittest.TestCase):
    def test_clean_evidence(self):
        self.assertTrue(validate_evidence(*synthetic())['wall_free'])

    def test_every_wall_persists_between_output_states(self):
        for wall in range(5):
            m, a = synthetic(wall=wall)
            self.assertFalse(validate_evidence(m, a)['wall_free'])
            # No saved position ever crosses a plane, but a step-2 impulse counts.
            self.assertEqual(validate_evidence(m, a)['covered_substeps'], 10)

    def test_missing_legacy_evidence_is_unknown(self):
        m, a = synthetic(); del m['wall_observation']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'raw.h5'; atomic_raw(path, m, a)
            r = validate_wall_free(path)
            self.assertTrue(r['integrity_pass']); self.assertEqual(r['wall_status'], 'unknown')
            self.assertFalse(r['eligible_for_wall_free_export'])

    def test_corrupt_evidence_fails_closed(self):
        for name in FIELDS:
            with self.subTest(field=name):
                m, a = synthetic(); del a[name]
                with self.assertRaisesRegex(ValueError, 'shape_or_dtype'): validate_evidence(m, a)
                m, a = synthetic(); a[name] = a[name].astype(np.float32)
                with self.assertRaisesRegex(ValueError, 'shape_or_dtype'): validate_evidence(m, a)

    def test_missing_one_observer_step_rejected(self):
        m, a = synthetic(); a['wall_covered_substeps'][2, 4] -= 1
        with self.assertRaisesRegex(ValueError, 'coverage'): validate_evidence(m, a)

    def test_reset_or_rewritten_first_event_rejected(self):
        for event in (-1, 3, 0, 11):
            m, a = synthetic(wall=1); a['wall_first_grid_step'][2, 1] = event
            with self.assertRaises(ValueError): validate_evidence(m, a)

    def test_clock_and_planes_cannot_change(self):
        m, a = synthetic(); a['wall_simulation_steps'][1] = 6
        with self.assertRaisesRegex(ValueError, 'clock'): validate_evidence(m, a)
        m, a = synthetic(); m['wall_observation']['points_m'][2][0] = 1.5325
        with self.assertRaisesRegex(ValueError, 'policy'): validate_evidence(m, a)

    def test_saved_contact_cannot_claim_no_contact(self):
        for wall, (point, normal) in enumerate(zip(POINTS, NORMALS)):
            m, a = synthetic(); axis = np.flatnonzero(normal)[0]
            a['x'][0, 0, axis] = point[axis]
            with self.assertRaisesRegex(ValueError, 'saved_contact'): validate_evidence(m, a)
            a['wall_first_particle_step'][:, wall] = 0
            self.assertFalse(validate_evidence(m, a)['wall_free'])

    def test_floor_is_not_a_wall(self):
        m, a = synthetic(); a['x'][:, :, 2] = .095
        self.assertTrue(validate_evidence(m, a)['wall_free'])

    def test_invalid_physics_never_exports_even_if_wall_free(self):
        m, a = synthetic(); a['x'][:, :, 2] = .05
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/'raw.h5'; atomic_raw(raw, m, a)
            result = validate_wall_free(raw)
            self.assertFalse(result['integrity_pass'])
            self.assertFalse(result['eligible_for_wall_free_export'])

    def test_export_membership_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paths = []
            for i, wall in enumerate((None, 1)):
                m, a = synthetic(i, wall); path = root/f'{i}.h5'
                atomic_raw(path, m, a); paths.append(path)
            out = root/'out'; result = export_wall_free(paths, out)
            self.assertEqual((result['retained'], result['excluded']), (1, 1))
            check_tar(out/'diagnostic-export/00000.tar')
            receipt = json.loads((out/'FILTER.json').read_text())
            self.assertEqual(receipt['retention']['scenario:drop']['retained'], 1)
            inner = json.loads((out/'diagnostic-export/COMPLETE.json').read_text())
            self.assertEqual([r['raw_sha256'] for r in inner['episodes']], [sha256(paths[0])])
            self.assertTrue(export_wall_free(paths, out)['resumed'])
            # A damaged selection receipt must not silently resume.
            with (out/'FILTER.json').open('a') as f: f.write(' ')
            with self.assertRaisesRegex(ValueError, 'artifact changed'): export_wall_free(paths, out)

    def test_zero_survivors_creates_receipt_not_tar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root/'raw.h5'; atomic_raw(raw, *synthetic(wall=4))
            self.assertEqual(export_wall_free([raw], root/'out')['retained'], 0)
            self.assertFalse(list((root/'out').rglob('*.tar')))
            self.assertTrue((root/'out/FILTER.json').is_file())

    def test_duplicate_identity_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); a = root/'a.h5'; b = root/'b.h5'
            atomic_raw(a, *synthetic()); atomic_raw(b, *synthetic())
            with self.assertRaisesRegex(ValueError, 'Duplicate episode'): export_wall_free([a, b], root/'out')
            link = root/'link.h5'; link.symlink_to(a)
            with self.assertRaisesRegex(ValueError, 'symlink'): export_wall_free([link], root/'link-out')


class CompiledObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise RuntimeError('CPU tests only')
        wp.config.kernel_cache_dir = '/tmp/pwm-pre-scale-backend-test-cache'
        wp.init()

    def backend(self, *, x=None, velocity=None):
        m, arrays = synthetic(); static = {k: arrays[k] for k in ('object_ids', 'particle_volume', 'E', 'nu', 'density', 'mat_type', 'yield_stress')}
        if x is None:
            x = np.array([[.74+i*.02, .74+j*.02, .3+k*.02] for i in range(2) for j in range(2) for k in range(2)], np.float32)
        config = dict(n_grid=100, grid_lim_m=100/64, floor_m=.1, floor_contact=True, friction=.3,
                      floor_tolerance_m=.002, observation_emergency_depth_m=.06, gravity_m_s2=[0., 0., 0.],
                      dt_s=5e-5, precision='float64-master', softening=.1, plasticine_policy='archived-fcr-log-return')
        b = CollectionBackend(x, static, config, np.zeros_like(x) if velocity is None else velocity, device='cpu')
        install_original_walls_in_padded_grid(b)
        return b, m, static

    def test_observers_do_not_change_real_solver_state(self):
        # Moving, near-wall case exercises a real collider impulse too.
        x = np.tile([.035, .75, .3], (8, 1)).astype(np.float32)
        v = np.tile([-1., .2, 0.], (8, 1)).astype(np.float32)
        plain, _, _ = self.backend(x=x, velocity=v)
        wrapped, _, _ = self.backend(x=x, velocity=v)
        observed = WallObservedBackend(wrapped, Clock(5e-5, 2, 2))
        for _ in range(4):
            plain.step(5e-5, None, np.zeros(8, np.int32))
            observed.step(5e-5, None, np.zeros(8, np.int32))
        for key in plain.read(): np.testing.assert_array_equal(plain.read()[key], observed.read()[key])
        self.assertEqual(observed.wall_snapshot()['wall_first_grid_step'][1], 1)
        self.assertEqual(observed.wall_snapshot()['wall_first_particle_step'][1], -1)
        self.assertGreater(observed.read()['x'][:, 0].min(), .03)

    def test_all_five_grid_directions_and_no_force_for_empty_or_outgoing(self):
        b, _, _ = self.backend(); observed = WallObservedBackend(b, Clock(5e-5, 1, 2))
        s, model = b.solver.mpm_state, b.solver.mpm_model
        observed.counter.fill_(1)
        for wall, param in enumerate(b.solver.collider_params[1::2]):
            axis = int(np.flatnonzero(NORMALS[wall])[0]); node = [48, 48, 32]
            node[axis] = 1 if NORMALS[wall, axis] > 0 else 95
            mass = np.zeros((100, 100, 100), np.float32)
            velocity = np.zeros((100, 100, 100, 3), np.float32)
            for occupied, sign, expected in [(False, -1., NEVER), (True, 1., NEVER), (True, -1., 1)]:
                mass[tuple(node)] = float(occupied); velocity[tuple(node)] = sign*NORMALS[wall]
                s.grid_m = wp.array(mass, dtype=float, device='cpu')
                s.grid_v_out = wp.array(velocity, dtype=wp.vec3, device='cpu')
                wp.launch(observe_grid, dim=(100, 100, 100), inputs=[0., 5e-5, s, model, param], device='cpu')
                self.assertEqual(observed.first_grid.numpy()[wall], expected)
                np.testing.assert_array_equal(s.grid_v_out.numpy(), velocity)
            # The flag survives an empty grid before the next saved frame.
            s.grid_m.zero_()
            wp.launch(observe_grid, dim=(100, 100, 100), inputs=[0., 5e-5, s, model, param], device='cpu')
            self.assertEqual(observed.first_grid.numpy()[wall], 1)

    def test_geometric_touch_each_plane_and_floor_excluded(self):
        b, _, _ = self.backend(); observed = WallObservedBackend(b, Clock(5e-5, 1, 2))
        points = np.tile([.75, .75, .1], (6, 1)).astype(np.float32)
        for wall in range(5):
            axis = np.flatnonzero(NORMALS[wall])[0]; points[wall, axis] = POINTS[wall, axis]
        wp.launch(observe_particles, dim=(6, 5), inputs=[wp.array(points, dtype=wp.vec3, device='cpu'),
                  observed.points, observed.normals, observed.counter, observed.first_particle], device='cpu')
        np.testing.assert_array_equal(observed.first_particle.numpy(), np.zeros(5, np.int32))
        observed.first_particle.fill_(NEVER)
        wp.launch(observe_particles, dim=(1, 5), inputs=[wp.array(points[5:], dtype=wp.vec3, device='cpu'),
                  observed.points, observed.normals, observed.counter, observed.first_particle], device='cpu')
        np.testing.assert_array_equal(observed.first_particle.numpy(), np.full(5, NEVER, np.int32))

    def test_removed_detector_or_mutated_wall_rejected(self):
        b, _, _ = self.backend(); observed = WallObservedBackend(b, Clock(5e-5, 1, 2))
        original = b.solver.collider_params[2]; original.point = wp.vec3(.01, 0., 0.)
        with self.assertRaisesRegex(ValueError, 'changed'): observed.step(5e-5, None, np.zeros(8, np.int32))
        original.point = wp.vec3(0., 0., 1.47)
        b.solver.grid_postprocess.pop()
        with self.assertRaisesRegex(ValueError, 'changed'): observed.wall_snapshot()

    def test_unsupported_wall_and_midrun_install_rejected(self):
        b, _, _ = self.backend(); b.solver.collider_params[2].friction = .1
        with self.assertRaisesRegex(ValueError, 'Unsupported'): WallObservedBackend(b, Clock(5e-5, 1, 2))
        b.solver.collider_params[2].friction = 0.; b.steps = 1
        with self.assertRaisesRegex(ValueError, 'fresh'): WallObservedBackend(b, Clock(5e-5, 1, 2))

    def test_real_atomic_collection_keeps_floor_contact_and_exact_clock(self):
        x = np.tile([.75, .75, .102], (8, 1)).astype(np.float32)
        b, m, static = self.backend(x=x, velocity=np.tile([0., 0., -.1], (8, 1)).astype(np.float32))
        clock = Clock(5e-5, 1, 2); m['clock'] = asdict(clock); m['source_sha256'] = b.source_sha256
        observed = WallObservedBackend(b, clock)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/'raw.h5'; collect_passive(observed, observed, m, static, raw)
            r = validate_wall_free(raw)
            self.assertTrue(r['eligible_for_wall_free_export'], r)
            manifest, a = read_raw(raw, POLICY)
            np.testing.assert_array_equal(a['wall_covered_substeps'][-1], [2]*5)
            np.testing.assert_array_equal(a['state_time_s'], clock.state_time_s)
            np.testing.assert_array_equal(a['x'][0], x)
            with self.assertRaises(ValueError): collect_passive(observed, observed, m, static, raw)


if __name__ == '__main__':
    unittest.main()
