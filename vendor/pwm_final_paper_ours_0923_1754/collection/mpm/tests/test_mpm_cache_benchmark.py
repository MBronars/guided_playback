"""A completed benchmark process is not automatically a passing experiment."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'launch'))
import mpm_cache_benchmark as benchmark
import mpm_training_campaign as campaign
import mpm_four as four
import mpm_scale as support
sys.path.pop(0)
from collection.mpm import collect


class BenchmarkTests(unittest.TestCase):
    def test_failed_comparison_preserves_receipt_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bundle=root/'bundle';bundle.mkdir();plan=root/'plan';plan.mkdir();out=root/'result'
            cases=[dict(kind='coverage',label='case',provenance={},template=dict(split='train',n_objects=1))]
            argv=['benchmark','run','--bundle',str(bundle),'--plan',str(plan),'--out',str(out),'--uuid','GPU-test','--pair-only']
            with patch.object(sys,'argv',argv),patch.object(sys,'path',list(sys.path)),\
                 patch.dict(os.environ,SLURM_JOB_ID='test',CUDA_VISIBLE_DEVICES='GPU-test'),\
                 patch.object(campaign,'verify',return_value={'reference':str(root)}),\
                 patch.object(collect,'gpu_status'),patch.object(collect,'inspect',return_value={'cases':cases}),\
                 patch.object(collect,'verify_attempt',return_value={'status':'complete'}),\
                 patch.object(benchmark.subprocess,'run'),patch.object(four,'compare',return_value={'passed':False}),\
                 patch.object(support,'sha',return_value='a'*64),patch.object(support,'read',return_value={}):
                with self.assertRaisesRegex(ValueError,'Numerical comparison failed'):benchmark.main()
            result=json.loads((out/'RESULT.json').read_text())
            self.assertFalse(result['passed']);self.assertFalse(result['training_release'])
            self.assertTrue((out/'FAILURE.json').exists())


if __name__=='__main__':unittest.main()
