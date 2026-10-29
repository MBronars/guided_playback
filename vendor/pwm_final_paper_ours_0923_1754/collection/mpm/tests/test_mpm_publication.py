"""CPU-only publication/source-ownership checks; no deployment or Slurm writes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'collection/mpm/launch'))
import mpm_shared_runtime as publish
sys.path.pop(0)


class PublicationTests(unittest.TestCase):
    def test_integrated_package_is_exact_deployed_source(self):
        receipt=json.loads((ROOT/'collection/mpm/docs/SOURCE_PARITY.json').read_text())
        self.assertEqual(len(receipt['exact_package_files']),36)
        for name,sha in receipt['exact_package_files'].items():
            sha=receipt.get('intentional_updates',{}).get(name,{}).get('sha256',sha)
            with self.subTest(name=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),sha)

    def test_closure_excludes_folding_and_model_code(self):
        files=publish.maintained_sources(ROOT)
        self.assertIn('collection/mpm/grouped.py',files)
        self.assertIn('collection/mpm/launch/mpm_training_campaign.py',files)
        self.assertNotIn('collection/mpm/launch/mpm_collect.py',files)
        self.assertNotIn('collection/mpm/evaluation.py',files)
        self.assertNotIn('collection/mpm/evaluation_suite.py',files)
        self.assertNotIn('collection/mpm/evaluation_supervisor.py',files)
        self.assertFalse(any(n.startswith(('collection/sim1/','data/','models/','stats/')) for n in files))
        self.assertFalse(any(n.startswith(('collection/mpm/tests/','collection/mpm/docs/')) for n in files))
        self.assertEqual(len(files),46)
        self.assertEqual(list(sorted((ROOT/'collection/mpm/launch').glob('mpm*.sbatch'))),[ROOT/'collection/mpm/launch/mpm_training_campaign.sbatch'])

    def test_domain_layout_has_no_old_launch_or_test_copies(self):
        self.assertIn('testpaths = ["tests", "collection"]',(ROOT/'pyproject.toml').read_text())
        self.assertFalse(list((ROOT/'launch').glob('mpm*')))
        self.assertFalse(list((ROOT/'tests/collection').glob('test_mpm*.py')))
        self.assertFalse((ROOT/'tests/test_mpm_collect.py').exists())
        self.assertFalse((ROOT/'tests/test_mpm_publication.py').exists())
        for name in ('RUNBOOK.md','ASSETS.md','INTEGRATION.md','SOURCE_PARITY.json'):
            self.assertTrue((ROOT/'collection/mpm/docs'/name).is_file())

    def test_domain_source_maps_to_unchanged_frozen_layout(self):
        self.assertEqual(publish.frozen_path('collection/mpm/launch/mpm_training_campaign.py'),
                         'mpm_training_campaign.py')
        self.assertEqual(publish.frozen_path('collection/mpm/grouped.py'),
                         'source/collection/mpm/grouped.py')
        files=publish.maintained_sources(ROOT)
        paths=[publish.frozen_path(n) for n in files]
        self.assertEqual(len(paths),len(set(paths)))
        self.assertFalse(any('/launch/' in n or '/tests/' in n for n in paths))

    def test_dirty_source_cannot_be_published(self):
        with patch.object(publish.operator,'checked',side_effect=['a'*40,' M collection/mpm/example.py']):
            with self.assertRaisesRegex(ValueError,'Commit the MPM source'):
                publish.repository_receipt(ROOT,{'collection/mpm/example.py':Path('/unused')})

    def test_receipt_binds_committed_bytes_and_rejects_racing_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'example.py';path.write_text('approved\n')
            result=subprocess.CompletedProcess([],0,b'approved\n',b'')
            with patch.object(publish.operator,'checked',side_effect=['a'*40,'']),\
                 patch.object(publish.subprocess,'run',return_value=result):
                receipt=publish.repository_receipt(root,{'example.py':path})
            self.assertEqual(receipt['git_commit'],'a'*40)
            self.assertEqual(receipt['source_files']['example.py'],hashlib.sha256(b'approved\n').hexdigest())
            path.write_text('changed\n')
            with patch.object(publish.operator,'checked',side_effect=['a'*40,'']),\
                 patch.object(publish.subprocess,'run',return_value=result):
                with self.assertRaisesRegex(ValueError,'committed blob'):
                    publish.repository_receipt(root,{'example.py':path})

    def test_stage_requires_canonical_project_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'uncreated'
            with self.assertRaisesRegex(ValueError,'canonical committed checkout'):
                publish.stage(out,None,None,None,None,None)
            self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
