import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from collection.common import digest, write_once
from collection.mpm.banks import assignment, verify_assignment
from collection.mpm.recipes import load_bank


class BankTests(unittest.TestCase):
    def fixture(self, root, role='validation'):
        pool = root/'pools'; pool.mkdir()
        write_once(pool/'objaverse_uid_train.json', [f'train-{i}' for i in range(10)])
        write_once(pool/'objaverse_uid_holdout.json', [f'eval-{i}' for i in range(30)])
        value = assignment(pool, 'test-v1'); split = root/'SPLITS.json'; write_once(split, value)
        bank = root/'bank'; bank.mkdir(); rows=[]
        for i, uid in enumerate(value['cohorts'][role][:7]):
            path=bank/f'{i}.npz'
            np.savez(path,source_x=np.ones((1024,3),np.float32),particle_volume=np.ones(1024,np.float32),source_particle_index=np.arange(1024))
            rows.append(dict(asset_uid=uid,file=path.name,sha256=digest(path),object_size_m=.1))
        metadata=dict(objects=rows,bank_role=role,split_manifest_sha256=digest(split),
            train_pool_sha256=digest(pool/'objaverse_uid_train.json'),heldout_pool_sha256=digest(pool/'objaverse_uid_holdout.json'))
        write_once(bank/'MANIFEST.json', metadata)
        return bank,pool,split,value,metadata

    def test_deterministic_disjoint_complete_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,pool,split,value,_=self.fixture(Path(tmp))
            self.assertEqual(value,assignment(pool,'test-v1'))
            cohorts={k:set(v) for k,v in value['cohorts'].items()}
            for a,b in [('train','validation'),('train','test'),('validation','test')]:
                self.assertFalse(cohorts[a]&cohorts[b])
            self.assertEqual(len(cohorts['validation']|cohorts['test']),30)
            self.assertEqual(verify_assignment(split,pool),value)
            self.assertEqual(len(load_bank(bank,pool,role='validation',split_manifest=split)[1]),7)

    def test_explicit_role_required_and_wrong_role_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,pool,split,_,_=self.fixture(Path(tmp))
            for kwargs in ({},{'role':'validation'},{'role':'test','split_manifest':split},{'role':'train','split_manifest':split}):
                with self.assertRaises(ValueError):load_bank(bank,pool,**kwargs)

    def test_cross_role_uid_rejected_even_with_valid_payload_hash(self):
        for foreign in ('train','test'):
            with tempfile.TemporaryDirectory() as tmp:
                bank,pool,split,value,metadata=self.fixture(Path(tmp))
                metadata['objects'][0]['asset_uid']=value['cohorts'][foreign][0]
                (bank/'MANIFEST.json').write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError,'outside declared bank role'):
                    load_bank(bank,pool,role='validation',split_manifest=split)

    def test_modified_split_rejected_even_if_bank_hash_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank,pool,split,value,metadata=self.fixture(Path(tmp))
            value['cohorts']['validation'][0],value['cohorts']['test'][0]=value['cohorts']['test'][0],value['cohorts']['validation'][0]
            split.write_text(json.dumps(value));metadata['split_manifest_sha256']=digest(split)
            (bank/'MANIFEST.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError,'assignment changed'):
                load_bank(bank,pool,role='validation',split_manifest=split)

    def test_all_three_roles_load(self):
        for role in ('train','validation','test'):
            with tempfile.TemporaryDirectory() as tmp:
                bank,pool,split,_,_=self.fixture(Path(tmp),role)
                self.assertEqual(len(load_bank(bank,pool,role=role,split_manifest=split)[1]),7)

    def test_prepared_bank_mutation_blocks_inspection(self):
        from unittest.mock import patch
        from collection.mpm.collect import inspect
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bank,pool,split,_,_=self.fixture(root)
            planroot=root/'prepared';planroot.mkdir()
            write_once(planroot/'PLAN.json',dict(source_sha256={},cases=[],planned=0,
                bank_source_sha256={str(bank/'MANIFEST.json'):digest(bank/'MANIFEST.json'),str(split):digest(split)}))
            with patch('collection.mpm.collect.ref.verify_sources'):
                inspect(planroot)
                (bank/'MANIFEST.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'bank or object split changed'):inspect(planroot)

    def test_cohort_survives_export_and_mixed_roles_are_rejected(self):
        import tarfile
        from test_wall_filter import synthetic
        from collection.mpm.recording import atomic_raw
        from collection.mpm.export import export_wall_free
        from collection.mpm.validate import export_diagnostic
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paths=[]
            for index,role in enumerate(('validation','test')):
                m,a=synthetic(index=index);m.update(object_cohort=role,split_manifest_sha256='f'*64)
                path=root/f'{role}.h5';atomic_raw(path,m,a);paths.append(path)
            with self.assertRaisesRegex(ValueError,'Mixed object cohorts'):
                export_wall_free(paths,root/'mixed')
            with self.assertRaisesRegex(ValueError,'Mixed object cohorts'):
                export_diagnostic(paths,root/'mixed-direct')
            export_wall_free(paths[:1],root/'validation')
            tarpath=next((root/'validation/diagnostic-export').glob('*.tar'))
            with tarfile.open(tarpath) as f:
                member=next(m for m in f.getmembers() if m.name.endswith('.provenance.json'))
                lineage=json.load(f.extractfile(member))
                self.assertEqual(lineage[0]['object_cohort'],'validation')
                self.assertEqual(lineage[0]['split_manifest_sha256'],'f'*64)

    def test_eval_export_without_split_hash_is_rejected(self):
        from test_wall_filter import synthetic
        from collection.mpm.recording import atomic_raw
        from collection.mpm.export import export_wall_free
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m,a=synthetic();m['object_cohort']='test'
            raw=root/'raw.h5';atomic_raw(raw,m,a)
            with self.assertRaisesRegex(ValueError,'frozen split hash'):export_wall_free([raw],root/'out')

if __name__=='__main__':unittest.main()
