from argparse import Namespace
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
import importlib.util
from unittest.mock import patch

if any(importlib.util.find_spec(name) is None for name in ('warp', 'h5py')):
    raise unittest.SkipTest('requires collection-mpm environment')

import numpy as np

from collection.mpm import collect as mpm
from collection.mpm import runtime as ref
from collection.common import digest, write_once


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref.load()
        import test_wall_filter
        cls.fixture = staticmethod(test_wall_filter.synthetic)

    def fixture_root(self, root, *, wall=None):
        from collection.mpm.recording import atomic_raw
        initial_folder = root / 'cases/example'
        attempt = initial_folder / 'attempts/attempt-00000'
        attempt.mkdir(parents=True)
        m, arrays = self.fixture(wall=wall)
        mpm.save_npz(initial_folder / 'initial.npz', {'x': arrays['x'][0]})
        write_once(initial_folder / 'MANIFEST.json', m)
        atomic_raw(attempt / 'raw.h5', m, arrays)
        initial_sha = digest(initial_folder / 'initial.npz')
        case = dict(label='example', kind='coverage', initial_sha256=initial_sha,
                    manifest_sha256=digest(initial_folder / 'MANIFEST.json'))
        write_once(root / 'PLAN.json', dict(cases=[case], planned=1, source_sha256={}))
        write_once(attempt / 'RESULT.json', dict(label='example', kind='coverage', status='complete',
                    raw_sha256=digest(attempt / 'raw.h5'), initial_sha256=initial_sha, accepted_for_training=False))
        return attempt

    def test_export_completion_binds_accounting_and_raw_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            self.fixture_root(root)
            out = Path(tmp) / 'export'
            complete = mpm.export(root, out)
            self.assertEqual(complete['retained'], 1)
            self.assertIn('ATTEMPTS.json', complete['artifacts'])
            self.assertIn('filtered/FILTER.json', complete['artifacts'])
            self.assertTrue(mpm.export(root, out)['resumed'])
            self.assertFalse(complete['accepted_for_training'])

    def test_corruption_blocks_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            self.fixture_root(root)
            out = Path(tmp) / 'export'
            mpm.export(root, out)
            with (out / 'ATTEMPTS.json').open('a') as stream: stream.write(' ')
            with self.assertRaisesRegex(ValueError, 'Artifact changed'): mpm.export(root, out)

    def test_failed_validator_never_publishes_outer_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            self.fixture_root(root)
            out = Path(tmp) / 'export'
            api, *_ = ref.load()
            with patch.object(api, 'export_wall_free', side_effect=RuntimeError('validator crashed')):
                with self.assertRaises(RuntimeError): mpm.export(root, out)
            self.assertFalse(out.exists())
            self.assertTrue(out.with_name(out.name + '.request.json').exists())

    def test_zero_survivors_has_accounting_no_tar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            self.fixture_root(root, wall=1)
            out = Path(tmp) / 'export'
            result = mpm.export(root, out)
            self.assertEqual(result['retained'], 0)
            self.assertFalse(list(out.rglob('*.tar')))

    def test_modified_inputs_block_inspection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            self.fixture_root(root)
            with (root / 'cases/example/initial.npz').open('ab') as stream: stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Prepared case changed'): mpm.inspect(root)

    def test_unrun_is_accounted_not_silently_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'run'
            self.fixture_root(root)
            # Use a different explicit attempt; the first attempt remains intact.
            out = Path(tmp) / 'export'
            result = mpm.export(root, out, attempt_index=1)
            self.assertEqual(result['retained'], 0)
            rows = json.loads((out / 'ATTEMPTS.json').read_text())['cases']
            self.assertEqual(rows[0]['status'], 'unrun')

    def test_wrapped_and_plain_cpu_paths_match(self):
        import warp as wp
        from test_wall_filter import CompiledObserverTests
        api, _, _, _, contracts, _ = ref.load()
        wp.config.kernel_cache_dir = '/tmp/pwm-pre-scale-backend-test-cache'
        wp.init()
        plain, m, static = CompiledObserverTests().backend()
        initial = dict(plain.read(), **static)
        m['solver_config'] = dict(plain.config)
        m['clock'] = asdict(contracts.Clock(5e-5, 2, 2))
        wrapped = ref.backend(initial, m, 'cpu')
        for _ in range(4):
            plain.step(5e-5, None, np.zeros(8, np.int32))
            wrapped.step(5e-5, None, np.zeros(8, np.int32))
        for key in plain.read():
            np.testing.assert_array_equal(plain.read()[key], wrapped.read()[key])

    def test_foreign_gpu_process_is_rejected(self):
        with patch('subprocess.check_output', side_effect=['GPU-test, 100, 0\n', 'GPU-test, 99, 100\n']):
            with self.assertRaisesRegex(RuntimeError, 'Foreign process'): mpm.gpu_status('GPU-test')

    def test_gpu_own_pid_allowed_only_during_run(self):
        outputs = ['GPU-test, 100, 0\n', f'GPU-test, {os.getpid()}, 100\n']
        with patch('subprocess.check_output', side_effect=outputs):
            self.assertEqual(mpm.gpu_status('GPU-test', allow_own=True)['process_ids'], [os.getpid()])


if __name__ == '__main__': unittest.main()
