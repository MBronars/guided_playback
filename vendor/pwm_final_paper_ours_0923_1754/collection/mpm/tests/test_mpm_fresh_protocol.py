from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from collection.mpm.collapse import CollapsePolicy, analyze
from collection.mpm.fresh_banks import physical, VOLUME_POLICY
from collection.mpm.protocol import cohort, coverage, specifications, VERSION
from collection.mpm.recipes import scene


def bank(n=19, role='train'):
    points = np.random.default_rng(17).uniform(-.5, .5, (1024, 3)).astype(np.float32)
    x, weights = physical(points, .16)
    objects = [dict(points_unit=points, source_x=x, particle_volume=weights,
                    source_particle_index=np.arange(1024)) for _ in range(n)]
    metadata = dict(schema='fresh-candidate-bank-v1', bank_role=role, volume_policy=VOLUME_POLICY,
                    objects=[dict(asset_uid=f'uid-{i:05}', object_size_m=.08) for i in range(n)])
    return metadata, objects


class CollapseTests(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(3).uniform(-.05, .05, (1024, 3))
        self.ids = np.zeros(1024, np.int32)

    def test_translation_and_rotation_do_not_contract(self):
        p = self.points
        q = p[:, [1, 2, 0]] + [1, 2, 3]
        report = analyze(np.stack([p, q]), self.ids)
        self.assertTrue(report['eligible'])
        np.testing.assert_allclose(report['objects'][0]['rms_radius_ratio'], 1)

    def test_flat_sheet_is_not_point_collapse(self):
        report = analyze(np.stack([self.points, self.points * [1, 1, 0]]), self.ids)
        self.assertTrue(report['eligible'])

    def test_exact_and_extreme_positive_contraction(self):
        for scale in (0, .001, .09):
            report = analyze(np.stack([self.points, self.points * scale]), self.ids)
            self.assertFalse(report['eligible'])
            self.assertIn('object_0:extreme_contraction', report['reasons'])

    def test_escaping_outliers_do_not_hide_core_collapse(self):
        q = self.points * .001
        q[:20] *= 100000
        report = analyze(np.stack([self.points, q]), self.ids)
        self.assertFalse(report['eligible'])
        self.assertGreater(report['objects'][0]['rms_radius_ratio'][-1], 1)

    def test_other_object_cannot_hide_collapse(self):
        x = np.stack([np.r_[self.points, self.points + 1], np.r_[self.points * .001, self.points + 1]])
        report = analyze(x, np.repeat([0, 1], 1024))
        self.assertEqual(report['reasons'], ['object_0:extreme_contraction'])

    def test_interior_frame_checked_even_after_recovery(self):
        report = analyze(np.stack([self.points, self.points * .01, self.points]), self.ids)
        self.assertEqual(report['objects'][0]['severe_contraction_states'], [1])

    def test_positive_but_degenerate_F_is_rejected(self):
        f = np.tile(np.eye(3), (2, 1024, 1, 1))
        f[1] *= .01
        report = analyze(np.stack([self.points] * 2), self.ids, f)
        self.assertIn('object_0:widespread_degenerate_deformation', report['reasons'])

    def test_moderate_contraction_warns_only(self):
        report = analyze(np.stack([self.points, self.points * .4]), self.ids)
        self.assertTrue(report['eligible'])
        self.assertTrue(report['warnings'])

    def test_invalid_policy_and_nonfinite_fail(self):
        with self.assertRaises(ValueError): CollapsePolicy(minimum_radius_ratio=-1)
        p = self.points.copy(); p[0] = np.nan
        with self.assertRaises(ValueError): analyze(np.stack([self.points, p]), self.ids)

    def test_raw_export_gate_rejects_collapse_without_negative_determinant(self):
        from test_wall_filter import synthetic
        from collection.mpm.recording import atomic_raw
        from collection.mpm.export import validate_wall_free, export_wall_free
        m, arrays = synthetic()
        m['collapse_policy'] = asdict(CollapsePolicy())
        arrays['x'][:, :, 0] += np.linspace(-.02, .02, 8, dtype=np.float32)
        arrays['x'][-1] = arrays['x'][0].mean(axis=0)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw.h5'
            atomic_raw(raw, m, arrays)
            result = validate_wall_free(raw)
            self.assertFalse(result['eligible_for_wall_free_export'])
            self.assertIn('object_0:point_collapse', result['reasons'])
            complete = export_wall_free([raw], Path(tmp) / 'export')
            self.assertEqual(complete['retained'], 0)
            self.assertFalse(list((Path(tmp) / 'export').rglob('*.tar')))

    def test_fresh_raw_cannot_disable_collapse_policy(self):
        from test_wall_filter import synthetic
        from collection.mpm.recording import atomic_raw
        from collection.mpm.validate import validate_raw
        m, arrays = synthetic()
        m['geometry_provenance'] = dict(recipe_version=VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / 'raw.h5'; atomic_raw(raw, m, arrays)
            result = validate_raw(raw)
            self.assertFalse(result['integrity_pass'])
            self.assertIn('Fresh protocol requires pinned collapse checks', result['reasons'])


class ProtocolTests(unittest.TestCase):
    def test_counts_splits_and_seed_namespaces(self):
        namespaces = set()
        for split in ('train', 'validation', 'test'):
            for evaluation in (('train',) if split == 'train' else ('seen', 'unseen', 'count_seen', 'count_unseen')):
                metadata, _ = bank(role=cohort(split, evaluation))
                specs = specifications(metadata, split=split, evaluation=evaluation)
                self.assertEqual({r['count'] for r in specs}, {5, 6, 7} if evaluation.startswith('count_') else {1, 2, 3, 4})
                for r in specs:
                    self.assertNotIn(r['seed_namespace'], namespaces)
                    namespaces.add(r['seed_namespace'])
                    self.assertEqual(len(set(r['asset_indices'])), r['count'])

    def test_10k_coverage_balanced_counts_and_chunk_independence(self):
        from collections import Counter
        m, _ = bank(10000)
        specs = specifications(m, split='train', evaluation='train', cycles=67)
        c = coverage(specs, m)
        self.assertEqual(c['planned_unique_shapes'], 10000)
        self.assertEqual(len(set(Counter(s['count'] for s in specs).values())), 1)
        block = specifications(m, split='train', evaluation='train', cycles=1, cycle_start=66)
        for a, b in zip(specs[-60:], block):
            self.assertEqual({k: v for k, v in a.items() if k != 'index'}, {k: v for k, v in b.items() if k != 'index'})

    def test_all_counts_families_and_materials_generate(self):
        for evaluation in ('train', 'count_seen'):
            m, objects = bank()
            specs = specifications(m, split='train' if evaluation == 'train' else 'validation', evaluation=evaluation)
            for spec in specs:
                a, p = scene(spec, m, objects)
                b, q = scene(spec, m, objects)
                self.assertEqual(p, q)
                for k in a: np.testing.assert_array_equal(a[k], b[k])
                self.assertEqual(a['x'].shape, (1024 * spec['count'], 3))
                self.assertTrue((a['particle_volume'] > 0).all())
                self.assertTrue(np.isin(a['E'], [1e4, 1e5, 1e6]).all())
                self.assertTrue(set(p['placement_reasons']) <= {'initial_spacing'})
                for oid, transform in enumerate(p['transforms']):
                    ix = transform['bank_index']
                    x, weights = physical(objects[ix]['points_unit'], transform['maximum_full_mesh_extent_m'])
                    expected = ((x.astype(float) - transform['source_midpoint_m']) @ np.array(transform['rotation']).T
                                + transform['translation_after_centering_m']).astype(np.float32)
                    np.testing.assert_array_equal(a['x'][oid * 1024:(oid + 1) * 1024], expected)
                    np.testing.assert_array_equal(a['particle_volume'][oid * 1024:(oid + 1) * 1024], weights)

    def test_forbidden_counts_and_wrong_cohorts_fail(self):
        m, objects = bank()
        spec = specifications(m, split='train', evaluation='train')[0]
        with self.assertRaises(ValueError): scene(dict(spec, count=5), m, objects)
        with self.assertRaises(ValueError): cohort('test', 'train')
        with self.assertRaises(ValueError): specifications(m, split='test', evaluation='unseen')

    def test_proxy_total_matches_occupied_cells(self):
        m, objects = bank()
        x, w = physical(objects[0]['points_unit'], .20)
        cells = np.floor(x / np.float32(1 / 64)).astype(int)
        self.assertAlmostEqual(float(w.astype(float).sum()), len(np.unique(cells, axis=0)) * (1 / 64)**3, places=9)

    def test_policy_roundtrip(self):
        self.assertEqual(CollapsePolicy(**asdict(CollapsePolicy())), CollapsePolicy())

    def test_distinct_evaluation_sets_cannot_share_export(self):
        from test_wall_filter import synthetic
        from collection.mpm.recording import atomic_raw
        from collection.mpm.validate import validate_export_cohorts
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for evaluation in ('seen', 'count_seen'):
                m, a = synthetic()
                m.update(object_cohort='train', split='validation', split_manifest_sha256='a' * 64, evaluation=evaluation)
                path = Path(tmp) / (evaluation + '.h5'); atomic_raw(path, m, a); paths.append(path)
            with self.assertRaisesRegex(ValueError, 'Mixed object cohorts'):
                validate_export_cohorts(paths)


if __name__ == '__main__':
    unittest.main()
