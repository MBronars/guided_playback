"""CPU-only orchestration tests; no GPU allocation or Slurm submission."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('mpm_scale_launcher', ROOT/'collection/mpm/launch/mpm_scale.py')
scale = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scale)


class ScaleLaunchTests(unittest.TestCase):
    def test_schedule_is_bounded_deterministic_disjoint(self):
        blocks = scale.schedule(320, 2)
        self.assertEqual(blocks, scale.schedule(320, 2))
        self.assertEqual(len(blocks)*90, 29160)
        self.assertEqual([b['role'] for b in blocks[:5]], ['train','validation','test','validation','test'])
        seeds = [seed for b in blocks for seed in b['seeds']]
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(0 <= seed < 2**32-32 for seed in seeds))
        for train, evaluation in [(0,2),(1001,2),(1,0),(1,101),(True,2)]:
            with self.assertRaises(ValueError): scale.schedule(train,evaluation)

    def test_frozen_source_and_external_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); code=root/'worker.py'; bank=root/'bank.json'
            code.write_text('code'); bank.write_text('{}')
            config=dict(files={'worker.py':scale.sha(code)}, external_sha256={str(bank):scale.sha(bank)})
            scale.write(root/'BUNDLE.json',config)
            self.assertEqual(scale.verify_bundle(root),config)
            code.write_text('changed')
            with self.assertRaisesRegex(ValueError,'bundle changed'): scale.verify_bundle(root)
            code.write_text('code'); bank.write_text('changed')
            with self.assertRaisesRegex(ValueError,'inputs changed'): scale.verify_bundle(root)

    def test_resume_requires_complete_unchanged_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/'raw.h5'; raw.write_bytes(b'original')
            with self.assertRaisesRegex(ValueError,'Incomplete block'): scale.completion(root)
            receipt=dict(status='complete',files={'raw.h5':scale.sha(raw)})
            scale.write(root/'COMPLETE.json',receipt)
            self.assertEqual(scale.completion(root),receipt)
            raw.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'payload changed'): scale.completion(root)

    def test_write_once_and_explicit_status_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'STATUS.json'; scale.write(path,{'status':'running'})
            with self.assertRaises(FileExistsError): scale.write(path,{})
            scale.write(path,{'status':'complete'},replace=True)
            self.assertEqual(scale.read(path),{'status':'complete'})

    def test_current_folding_node_forbidden_before_cuda_probe(self):
        for node in ('babel-l9-20','babel-z9-16'):
            with patch.dict(os.environ,{'SLURM_JOB_ID':'fake'}), patch.object(scale.socket,'gethostname',return_value=node), patch.object(scale.subprocess,'check_output') as probe:
                with self.assertRaisesRegex(ValueError,'separate Slurm node'): scale.discover({})
                probe.assert_not_called()

    def test_cuda_discovery_uses_remapped_uuids(self):
        output='Warp initialization\nDEVICES='+json.dumps([dict(uuid='GPU-a',name='NVIDIA RTX PRO 6000 Blackwell'),dict(uuid='GPU-b',name='NVIDIA RTX PRO 6000 Blackwell')])
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fake','CUDA_VISIBLE_DEVICES':'3,5'}), patch.object(scale.socket,'gethostname',return_value='separate-node'), patch.object(scale.subprocess,'check_output',return_value=output):
            self.assertEqual(scale.discover({'genesis_python':sys.executable}),['GPU-a','GPU-b'])

    def test_wrong_gpu_type_rejected(self):
        output='DEVICES='+json.dumps([dict(uuid='GPU-a',name='A100'),dict(uuid='GPU-b',name='A100')])
        with patch.dict(os.environ,{'SLURM_JOB_ID':'fake','CUDA_VISIBLE_DEVICES':'0,1'}), patch.object(scale.socket,'gethostname',return_value='separate-node'), patch.object(scale.subprocess,'check_output',return_value=output):
            with self.assertRaisesRegex(ValueError,'RTX PRO 6000'): scale.discover({'genesis_python':sys.executable})

    def test_subprocess_failure_stops_owned_peer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaisesRegex(RuntimeError,'Child failed'):
                scale.children([[sys.executable,'-c','raise SystemExit(7)'],[sys.executable,'-c','import time; time.sleep(30)']],
                               [dict(os.environ)]*2,[root/'fail.log',root/'peer.log'],5)

    def test_sbatch_syntax_and_resource_contract(self):
        script=ROOT/'collection/mpm/launch/mpm_training_campaign.sbatch'
        subprocess.run(['bash','-n',str(script)],check=True)
        text=script.read_text()
        for directive in ('--gres=gpu:RTX_PRO_6000:4','--nodes=1','--time=2-00:00:00','--no-requeue'):
            self.assertIn('#SBATCH '+directive,text)
        self.assertNotIn('sbatch ',text)


if __name__ == '__main__':
    unittest.main()
