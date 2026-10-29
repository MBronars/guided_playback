"""Cache identity and ownership tests; no GPU or production writes."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'launch'))
import mpm_four as four
sys.path.pop(0)


class KernelCacheTests(unittest.TestCase):
    def test_legacy_worker_does_not_silently_enable_unseeded_cache(self):
        import inspect
        self.assertIs(inspect.signature(four.worker).parameters['reuse_cache'].default,False)

    def test_seed_copies_exact_qualified_artifacts_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lane=root/'lane';source=lane/'kernel-cache/mod';source.mkdir(parents=True)
            (source/'kernel.ptx').write_bytes(b'qualified binary')
            mode=dict(size=4,block=64,fused=True)
            four.record_cache_seed(lane,uuid='GPU-a',bundle_sha256='bundle',warp_version='1',device_arch=120,mode=mode)
            seed=lane/'CACHE_SEED.json';sha=four.support.sha(seed)
            identity=dict(device_uuid='GPU-a',bundle_sha256='bundle',warp='1',device_arch=120,mode=mode)
            cache=four.reusable_kernel_cache(identity,root/'private')
            four.copy_cache_seed(cache,seed,sha,identity)
            four.copy_cache_seed(cache,seed,sha,identity,require_existing=True)
            (cache/'unexpected.ptx').write_bytes(b'extra')
            with self.assertRaisesRegex(ValueError,'Unqualified'):four.copy_cache_seed(cache,seed,sha,identity,require_existing=True)
            (cache/'unexpected.ptx').unlink()
            self.assertEqual((cache/'mod/kernel.ptx').read_bytes(),b'qualified binary')
            four.copy_cache_seed(cache,seed,sha,identity)
            with self.assertRaisesRegex(ValueError,'another runtime'):four.copy_cache_seed(cache,seed,sha,dict(identity,device_uuid='GPU-b'))
            (cache/'mod/kernel.ptx').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):four.copy_cache_seed(cache,seed,sha,identity)
            self.assertEqual((cache/'mod/kernel.ptx').read_bytes(),b'changed','Never overwrite a mismatched cache')

    def test_seed_manifest_cannot_escape_its_cache_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);lane=root/'lane';(lane/'kernel-cache').mkdir(parents=True)
            identity=dict(device_uuid='GPU-a',bundle_sha256='bundle',warp='1',device_arch=120,mode={})
            seed=lane/'CACHE_SEED.json';seed.write_text(json.dumps(dict(schema='mpm-qualified-cache-seed-v1',uuid='GPU-a',
                **{k:identity[k] for k in ('bundle_sha256','warp','device_arch','mode')},files={'../escape':'hash'})))
            cache=four.reusable_kernel_cache(identity,root/'private')
            with self.assertRaisesRegex(ValueError,'Unsafe cache entry'):four.copy_cache_seed(cache,seed,four.support.sha(seed),identity)

    def test_reuse_is_stable_but_runtime_device_and_mode_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cache';identity=dict(bundle='a',device='GPU-a',mode={'size':4},warp='1')
            first=four.reusable_kernel_cache(identity,root)
            (first/'sentinel').write_text('compiled')
            self.assertEqual(four.reusable_kernel_cache(dict(reversed(list(identity.items()))),root),first)
            self.assertEqual((first/'sentinel').read_text(),'compiled')
            for key,value in [('bundle','b'),('device','GPU-b'),('mode',{'size':8}),('warp','2')]:
                self.assertNotEqual(four.reusable_kernel_cache(dict(identity,**{key:value}),root),first)
            self.assertEqual(first.stat().st_mode&0o777,0o700)

    def test_foreign_writable_or_symlink_root_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cache';root.mkdir(mode=0o777);root.chmod(0o777)
            with self.assertRaisesRegex(ValueError,'private'):four.reusable_kernel_cache({},root)
            link=Path(tmp)/'link';link.symlink_to(root,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'private'):four.reusable_kernel_cache({},link)

    def test_changed_or_symlink_receipt_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'cache';cache=four.reusable_kernel_cache({'bundle':'a'},root)
            receipt=cache/'IDENTITY.json';receipt.write_text(json.dumps({'bundle':'b'}))
            with self.assertRaisesRegex(ValueError,'identity'):four.reusable_kernel_cache({'bundle':'a'},root)
            receipt.unlink();receipt.symlink_to(Path(tmp)/'missing')
            with self.assertRaisesRegex(ValueError,'identity'):four.reusable_kernel_cache({'bundle':'a'},root)


if __name__=='__main__':unittest.main()
