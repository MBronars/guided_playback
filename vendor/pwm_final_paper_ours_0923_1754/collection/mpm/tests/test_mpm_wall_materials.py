import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from collection.mpm.wall_materials import (classify, interval_lower_bound, SAND_ONLY,
                                         check_guard_provenance, SUPPORTED_SOURCES)
from collection.mpm.export import validate_candidate, export_candidates, validate_wall_free
from collection.mpm.recording import atomic_raw


def fixture(sand=True, mixed=False, wall=1):
    from test_wall_filter import synthetic
    m, a = synthetic(wall=wall)
    a['mat_type'][:] = 2 if sand else 0
    m['material_names'] = ['sand' if sand else 'elastic']
    if mixed:
        m['n_objects'] = 2
        m['case']['asset_ids'].append('second-fixture')
        from collection.mpm.contracts import identity
        m['episode_id'] = identity(m['case'])[0]
        a['object_ids'][4:] = 1
        for key in ('E', 'nu', 'density', 'mat_type', 'yield_stress'):
            a[key] = np.repeat(a[key], 2)
        a['mat_type'][1] = 0
        m['material_names'] = ['sand', 'elastic']
    return m, a


class MaterialWallTests(unittest.TestCase):
    def test_wall_free_is_unchanged_for_non_sand(self):
        m, a = fixture(sand=False, wall=None)
        self.assertEqual(classify(m, a)['status'], 'wall_free')

    def test_all_sand_can_touch_wall(self):
        m, a = fixture()
        self.assertTrue(classify(m, a)['eligible'])

    def test_no_sand_cannot_explain_native_wall_event(self):
        m, a = fixture(sand=False)
        self.assertFalse(classify(m, a)['eligible'])
        self.assertIn('wall_contact_without_sand', classify(m, a)['reasons'])

    def test_missing_native_guard_provenance_is_unknown(self):
        m, a = fixture(mixed=True)
        self.assertFalse(classify(m, a)['eligible'])
        self.assertEqual(classify(m, a)['status'], 'unknown')

    def test_guard_provenance_requires_every_pinned_source(self):
        m = dict(solver_config=dict(precision='float64-master', n_grid=100, grid_lim_m=100/64),
                 geometry_provenance=dict(recipe_version='fresh-shapes-v1'),
                 source_sha256={'/fixture/collection/mpm/' + k: v for k, v in SUPPORTED_SOURCES.items()})
        with patch('collection.mpm.wall_materials.digest', side_effect=lambda p: SUPPORTED_SOURCES[Path(p).name]):
            check_guard_provenance(m)
            for name in SUPPORTED_SOURCES:
                bad = copy.deepcopy(m)
                bad['source_sha256']['/fixture/collection/mpm/' + name] = '0' * 64
                with self.assertRaises(ValueError): check_guard_provenance(bad)

    def test_guard_provenance_detects_changed_source_on_disk(self):
        m = dict(solver_config=dict(precision='float64-master', n_grid=100, grid_lim_m=100/64),
                 geometry_provenance=dict(recipe_version='fresh-shapes-v1'),
                 source_sha256={'/fixture/collection/mpm/' + k: v for k, v in SUPPORTED_SOURCES.items()})
        with patch('collection.mpm.wall_materials.digest', return_value='0' * 64):
            with self.assertRaises(ValueError): check_guard_provenance(m)

    def test_mixed_materials_far_from_wall_can_be_certified(self):
        m, a = fixture(mixed=True)
        with patch('collection.mpm.wall_materials.check_guard_provenance'):
            self.assertEqual(classify(m, a)['status'], 'sand_only')

    def test_non_sand_saved_contact_is_rejected(self):
        m, a = fixture(mixed=True)
        a['x'][1:, 4:, 0] = np.float32(.03)
        a['wall_first_particle_step'][1:, 1] = 2
        with patch('collection.mpm.wall_materials.check_guard_provenance'):
            self.assertEqual(classify(m, a)['status'], 'non_sand_contact')

    def test_non_sand_grid_support_is_not_called_sand(self):
        m, a = fixture(mixed=True)
        a['x'][:, 4:, 0] = .05  # inside plane, but within its grid support
        with patch('collection.mpm.wall_materials.check_guard_provenance'):
            r = classify(m, a)
            self.assertFalse(r['eligible'])
            self.assertEqual(r['status'], 'unknown')

    def test_between_frame_excursion_not_hidden_by_safe_endpoints(self):
        d = np.array([[.1], [.1]])
        self.assertLess(interval_lower_bound(d, [0, .0417])[0, 0], 0)
        # Nearest-endpoint speed bound must cover an out-and-back excursion.
        self.assertGreater(interval_lower_bound(np.array([[.4], [.4]]), [0, .0417])[0, 0], 2/64)

    def test_lipschitz_bound_covers_random_bounded_paths(self):
        rng = np.random.default_rng(8)
        dt = .00005
        for _ in range(100):
            increments = rng.uniform(-8, 8, 834) * dt
            path = .3 + np.r_[0, np.cumsum(increments)]
            lower = interval_lower_bound(np.array([[path[0]], [path[-1]]]), [0, 834 * dt])[0, 0]
            self.assertLessEqual(lower, path.min())

    def test_all_five_planes_allow_sand_but_not_elastic(self):
        for wall in range(5):
            self.assertTrue(classify(*fixture(wall=wall))['eligible'])
            self.assertFalse(classify(*fixture(sand=False, wall=wall))['eligible'])

    def test_missing_wall_evidence_cannot_be_relaxed(self):
        m, a = fixture(); del a['wall_first_grid_step']
        with self.assertRaises(ValueError): classify(m, a)

    def test_sand_only_exports_are_tagged_not_called_wall_free(self):
        m, a = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); raw = root / 'raw.h5'; atomic_raw(raw, m, a)
            self.assertFalse(validate_wall_free(raw)['eligible_for_wall_free_export'])
            q = validate_candidate(raw, wall_policy=SAND_ONLY)
            self.assertTrue(q['eligible_for_export'])
            self.assertFalse(q['eligible_for_wall_free_export'])
            done = export_candidates([raw], root / 'export', wall_policy=SAND_ONLY)
            self.assertEqual(done['retained'], 1)
            inner = json.loads((root / 'export/diagnostic-export/COMPLETE.json').read_text())
            self.assertEqual(inner['episodes'][0]['selection_policy']['version'], SAND_ONLY)
            self.assertFalse(inner['episodes'][0]['accepted_for_training'])
            self.assertTrue(export_candidates([raw], root / 'export', wall_policy=SAND_ONLY)['resumed'])

    def test_numerical_failures_still_block_sand(self):
        m, a = fixture(); a['F_trial'][-1] *= -1
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw.h5'; atomic_raw(raw, m, a)
            self.assertFalse(validate_candidate(raw, wall_policy=SAND_ONLY)['eligible_for_export'])

    def test_unknown_policy_fails(self):
        with self.assertRaises(ValueError): validate_candidate('not-read', wall_policy='allow-everything')

    def test_other_validation_errors_cannot_be_relaxed(self):
        m, a = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw.h5'; atomic_raw(raw, m, a)
            failed = validate_wall_free(raw)
            failed['reasons'].append('interaction:failed to analyze')
            with patch('collection.mpm.export.validate_wall_free', return_value=failed):
                self.assertFalse(validate_candidate(raw, wall_policy=SAND_ONLY)['eligible_for_export'])


if __name__ == '__main__':
    unittest.main()
