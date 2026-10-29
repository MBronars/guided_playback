"""Portable package boundaries; no research paths or simulator initialization."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from collection.consumer import import_consumer
from collection.mpm.runtime import reference_root, sources, verify_sources
from collection.storage import copy_shard


class MigrationTests(unittest.TestCase):
    def test_reference_location_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'PWM_MPM_REFERENCE_ROOT'):
                reference_root()

    def test_source_closure_is_package_local(self):
        values=sources()
        self.assertTrue(values)
        self.assertTrue(all(p.startswith('collection/') for p in values))
        self.assertFalse(any('reports/' in p for p in values))
        self.assertFalse(any('/sim1/' in p for p in values))
        verify_sources(values)
        values[next(iter(values))]='0'*64
        with self.assertRaisesRegex(ValueError,'Prepared source changed'): verify_sources(values)

    def test_source_path_traversal_rejected(self):
        with self.assertRaises(ValueError):verify_sources({'../outside.py':'0'*64})

    def test_consumer_rejects_missing_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'Model root'):import_consumer(tmp)

    def test_consumer_rejects_already_loaded_foreign_data(self):
        from types import SimpleNamespace
        root=Path(__file__).resolve().parents[2]
        with patch.dict(sys.modules, {'data':SimpleNamespace(__file__='/unrelated/data/__init__.py')}):
            with self.assertRaisesRegex(ValueError,'Conflicting'):import_consumer(root)

    def test_cli_and_review_do_not_import_simulator_or_scipy(self):
        code='import sys; import collection.mpm.collect; assert "warp" not in sys.modules; assert "scipy" not in sys.modules'
        subprocess.run([sys.executable,'-c',code],check=True,timeout=30)
        if importlib.util.find_spec('h5py'):
            subprocess.run([sys.executable,'-c',code+'; import collection.mpm.review; assert "warp" not in sys.modules; assert "scipy" not in sys.modules'],check=True,timeout=30)

    def test_copy_is_verified_and_does_not_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); source=p/'source'; dest=p/'nested/copy'
            source.write_bytes(b'preserve source')
            row=copy_shard(source,dest)
            self.assertEqual(row['bytes'],15)
            self.assertEqual(source.read_bytes(),dest.read_bytes())
            with self.assertRaises(FileExistsError):copy_shard(source,dest)

    def test_folding_interfaces_do_not_import_optional_backends(self):
        code = ('import sys; import collection.sim1.fold_plan; '
                'import collection.sim1.replay; import collection.sim1.replay_review; '
                'import collection.sim1.actual_export; import collection.sim1.replay_visualize; '
                'assert not ({"warp", "newton", "scipy", "open3d"} & set(sys.modules))')
        subprocess.run([sys.executable, '-c', code], check=True, timeout=30)
