import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'collection/mpm/launch'))
spec=importlib.util.spec_from_file_location('campaign_test',ROOT/'collection/mpm/launch/mpm_campaign.py')
campaign=importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)
sys.path.pop(0)


class CampaignTests(unittest.TestCase):
    def test_complete_schedule_has_no_repeated_training_cycles(self):
        entries=[e for w in range(6252) for e in campaign.wave(w)]
        train=[e for e in entries if e['split']=='train']
        self.assertEqual(len(train)*60,750000)
        self.assertEqual([e['cycle'] for e in train],list(range(1,12501)))
        self.assertEqual(len({e['name'] for e in entries}),len(entries))
        self.assertFalse(any(e['split']=='test' for e in entries))

    def test_first_training_then_four_independent_validation_categories(self):
        self.assertTrue(all(e['split']=='train' for e in campaign.wave(0)))
        self.assertEqual([e['evaluation'] for w in (1,2) for e in campaign.wave(w)],
                         ['seen','unseen','count_seen','count_unseen'])

    def test_out_of_range_schedule_rejected(self):
        for i in (-1,6252,True,1.2):
            with self.assertRaises(ValueError):campaign.wave(i)

    def test_bad_runtime_cannot_probe_gpu_or_create_output(self):
        with patch.object(campaign,'verify') as verify:
            with self.assertRaises(ValueError):campaign.run(Path('/missing'),Path('/missing-output'),False,5)
            verify.assert_not_called()

    def test_worker_requires_explicit_gpu_allocation(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''},clear=True):
            with self.assertRaises(ValueError):campaign.worker(Path('/missing'),'GPU-fixture')

    def test_publication_is_idempotent_and_detects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); unit=root/'chunks/train-00001'
            shards=unit/'filtered/diagnostic-export'; shards.mkdir(parents=True)
            (shards/'00000.tar').write_bytes(b'test shard')
            (unit/'COMPLETE.json').write_text('{}')
            result=dict(entry=campaign.wave(0)[0],retained=1)
            campaign.publish(root,unit,result);campaign.publish(root,unit,result)
            target=root/'datasets/train/train-00001/00000.tar'
            self.assertTrue(os.path.samefile(target,shards/'00000.tar'))
            target.write_bytes(b'changed')
            with self.assertRaises(ValueError):campaign.publish(root,unit,result)

    def test_completion_hash_rejects_changed_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'raw').write_bytes(b'original')
            campaign.support.write(root/'COMPLETE.json',dict(status='complete',files={'raw':campaign.support.sha(root/'raw')}))
            campaign.completed(root)
            (root/'raw').write_bytes(b'changed')
            with self.assertRaises(ValueError):campaign.completed(root)

    def test_supported_batch_shell_requires_explicit_training_paths(self):
        p=ROOT/'collection/mpm/launch/mpm_training_campaign.sbatch'; subprocess.run(['bash','-n',str(p)],check=True)
        text=p.read_text()
        for required in ('--gres=gpu:RTX_PRO_6000:4',
                         '--no-requeue','MPM_TRAINING_BUNDLE:?','MPM_TRAINING_OUTPUT:?'):
            self.assertIn(required,text)


if __name__=='__main__':unittest.main()
