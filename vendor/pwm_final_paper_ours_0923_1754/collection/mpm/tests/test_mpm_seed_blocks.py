import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from collection.common import digest, write_once
from collection.mpm.recipes import load_bank, specifications


class SeedBlockTests(unittest.TestCase):
    def test_default_population_retained_and_blocks_do_not_overlap(self):
        default = specifications('centered-v2')
        self.assertEqual(len(default), 90)
        self.assertEqual({s['seed'] for s in default}, {2026090900, 2026090901})
        later = specifications('centered-v2', seed_start=2026090902, seed_count=2)
        self.assertFalse({s['seed'] for s in default} & {s['seed'] for s in later})
        self.assertEqual([s['index'] for s in later], list(range(90)))

    def test_seed_assignment_is_independent_of_plan_chunk_boundaries(self):
        together = specifications('centered-v2', seed_start=99, seed_count=3)
        separate = [s for seed in (99, 100, 101) for s in
                    specifications('centered-v2', seed_start=seed, seed_count=1)]
        def key(s): return s['seed'], s['count'], s['scenario'], s['stiffness_phase']
        a, b = {key(s): s for s in together}, {key(s): s for s in separate}
        self.assertEqual(set(a), set(b))
        self.assertTrue(all(a[k]['material_offset'] == b[k]['material_offset'] for k in a))

    def test_unbounded_or_ambiguous_seed_request_rejected(self):
        for kwargs in ({'seed_start': -1}, {'seed_start': True}, {'seed_start': 2**32},
                       {'seed_count': 0}, {'seed_count': 33}, {'seed_count': 1.5}):
            with self.assertRaises(ValueError): specifications('centered-v2', **kwargs)

    def bank_fixture(self, root, *, weights=None, indices=None, path=None, leak=False):
        bank, pool = root / 'bank', root / 'pools'
        bank.mkdir(); pool.mkdir()
        write_once(pool / 'objaverse_uid_train.json', [f'asset{i}' for i in range(7)])
        write_once(pool / 'objaverse_uid_holdout.json', ['heldout'])
        rows = []
        for i in range(7):
            p = bank / f'object-{i}.npz'
            np.savez(p, source_x=np.ones((1024, 3), dtype=np.float32),
                particle_volume=np.ones(1024) if weights is None else weights,
                source_particle_index=np.arange(1024) if indices is None else indices)
            rows.append(dict(file=str(path or p.name), sha256=digest(p),
                             asset_uid='heldout' if leak and i == 0 else f'asset{i}', object_size_m=.1))
        write_once(bank / 'MANIFEST.json', dict(objects=rows,
            train_pool_sha256=digest(pool / 'objaverse_uid_train.json'),
            heldout_pool_sha256=digest(pool / 'objaverse_uid_holdout.json')))
        return bank, pool

    def test_explicit_bank_and_split_hashes_are_verified(self):
        with tempfile.TemporaryDirectory() as temp:
            bank, pool = self.bank_fixture(Path(temp))
            self.assertEqual(len(load_bank(bank, pool)[1]), 7)
            (pool / 'objaverse_uid_holdout.json').write_text('["different"]')
            with self.assertRaisesRegex(ValueError, 'Asset split changed'): load_bank(bank, pool)

    def test_invalid_weights_ids_traversal_and_holdout_rejected(self):
        for kwargs in (dict(weights=np.zeros(1024)), dict(weights=np.full(1024, np.nan)),
                       dict(indices=np.zeros(1024, dtype=int)), dict(path='../outside.npz'), dict(leak=True)):
            with self.subTest(kwargs=list(kwargs)), tempfile.TemporaryDirectory() as temp:
                bank, pool = self.bank_fixture(Path(temp), **kwargs)
                with self.assertRaises(ValueError): load_bank(bank, pool)


if __name__ == '__main__': unittest.main()
