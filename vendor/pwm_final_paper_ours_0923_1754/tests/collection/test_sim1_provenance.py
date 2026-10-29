from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from collection.common import digest
from collection.sim1.environment import check_module_roots
from collection.sim1.provenance import snapshot_sources, robot_assets


class SourceTests(unittest.TestCase):
    def test_numeric_modules_do_not_initialize_simulator(self):
        code = ('import sys; import collection.sim1.environment, collection.sim1.recording, '
                'collection.sim1.export, collection.sim1.qualify; '
                'assert "warp" not in sys.modules and "newton" not in sys.modules')
        subprocess.run([sys.executable,'-c',code],check=True,timeout=20)

    def test_foreign_parent_modules_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(sys.modules, {'components.x': SimpleNamespace(__file__='/foreign/x.py')}):
                with self.assertRaisesRegex(ValueError,'already-loaded'): check_module_roots(tmp)

    def test_archive_must_match_original_digest_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); p=root/'source.py'; p.write_text('original')
            sources={str(p):digest(p)}
            r=snapshot_sources(sources,root/'source.tar.gz')
            self.assertEqual(r['files'],1)
            with self.assertRaises(FileExistsError): snapshot_sources(sources,root/'source.tar.gz')
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError,'no longer'): snapshot_sources(sources,root/'bad.tar.gz')
            self.assertFalse((root/'bad.tar.gz').exists())
            r=snapshot_sources(sources,root/'recovered.tar.gz',recovered={str(p):b'original'})
            self.assertEqual(r['files'],1)

    def test_robot_asset_escape_and_missing_mesh_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'acone').mkdir(); urdf=root/'acone/acone.urdf'
            urdf.write_text('<robot><mesh filename="missing.STL"/></robot>')
            with self.assertRaises(FileNotFoundError): robot_assets(root)
            urdf.write_text('<robot><mesh filename="file:///foreign"/></robot>')
            with self.assertRaises(ValueError): robot_assets(root)
