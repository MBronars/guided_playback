"""CPU-only operator regression tests; never submit real jobs."""
import concurrent.futures
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

LAUNCH=Path(__file__).resolve().parents[3]/'collection/mpm/launch'
sys.path.insert(0,str(LAUNCH))
spec=importlib.util.spec_from_file_location('mpm_operator',LAUNCH/'mpm_collect.py')
operator=importlib.util.module_from_spec(spec);spec.loader.exec_module(operator)
sys.path.pop(0)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.c=dict(schema='mpm-operator-v1',template_bundle=str(self.root/'template'),template_sha256='a'*64,
            campaigns_root=str(self.root/'campaigns'),registry_root=str(self.root/'registry'),
            first_cycle=84000,last_cycle_exclusive=100000,cycles_per_run=2000,
            partition='katef',account='kfragki2',qos='katef_qos')

    def tearDown(self):self.temp.cleanup()

    def seed_ready(self,name='night-one'):
        r=operator.plan(self.c,name);r['proposed_only']=False;r['created_epoch']=1.
        Path(self.c['registry_root']).mkdir(exist_ok=True)
        operator.write_once(Path(self.c['registry_root'])/(name+'.json'),r)
        root=Path(r['root']);root.mkdir(parents=True);bundle=root/'bundle';bundle.mkdir()
        operator.write_once(bundle/'BUNDLE.json',dict(genesis_python='/python',cycle_start=r['cycle_start'],training_cycles=r['cycles']))
        r=dict(r,bundle_sha256=operator.digest(bundle/'BUNDLE.json'))
        operator.write_once(root/'READY.json',r)
        return root,r

    def test_plan_is_read_only_and_excludes_existing_campaign_ranges(self):
        r=operator.plan(self.c,'night-one')
        self.assertEqual((r['cycle_start'],r['cycle_stop']),(84000,86000))
        self.assertFalse(Path(self.c['registry_root']).exists())
        self.assertFalse(Path(self.c['campaigns_root']).exists())
        self.assertTrue(r['proposed_only'])

    def test_old_template_rejects_two_gpu_request_before_reserving_seeds(self):
        self.c['collection']=dict(hardware=dict(gpus=2,gpu_type='RTX_PRO_6000',walltime_hours=48))
        with patch.object(operator,'digest',return_value='a'*64),\
             patch.object(operator,'read',return_value=dict(schema='mpm-four-training-campaign-v2')):
            with self.assertRaisesRegex(ValueError,'Two-GPU collection requires'):
                operator.preflight(self.c,scheduler=False)
        self.assertFalse(Path(self.c['registry_root']).exists())

    def test_status_distinguishes_archived_failure_and_active_recovery(self):
        root,r=self.seed_ready()
        recovery=root/'recovery-001';(recovery/'capsule').mkdir(parents=True)
        operator.write_once(recovery/'capsule/RECOVERY.json',{})
        operator.write_once(recovery/'RECOVERED.json',dict(retained=160))
        operator.write_once(root/'ACTIVE_RECOVERY.json',dict(root=str(recovery),
            capsule_sha256=operator.digest(recovery/'capsule/RECOVERY.json')))
        result=operator.status(self.c,r['name'])
        self.assertEqual(result['total_retained_with_recovery'],160)
        self.assertFalse(result['active_recovery']['collection']['exists'])
        self.assertTrue(result['active_recovery']['original_failure_preserved'])

    def test_bounded_allocations_never_wrap(self):
        rows=[]
        for expected in range(84000,100000,2000):
            start=operator.next_cycle(self.c,rows);self.assertEqual(start,expected)
            rows.append(dict(cycle_start=start,cycle_stop=start+2000))
        with self.assertRaises(ValueError):operator.next_cycle(self.c,rows)

    def test_chained_recovery_counts_previous_data_once(self):
        root,r=self.seed_ready();recovery=root/'recovery-002'
        (recovery/'capsule').mkdir(parents=True)
        prior=root/'recovery-001/collection'
        chunk=prior/'chunks/train-1';chunk.mkdir(parents=True)
        operator.write_once(chunk/'COMPLETE.json',dict(status='complete',entry=dict(evaluation='train'),retained=53,attempts=60))
        operator.write_once(recovery/'capsule/RECOVERY.json',dict(previous_collections=[str(prior)]))
        operator.write_once(recovery/'RECOVERED.json',dict(retained=160))
        operator.write_once(root/'ACTIVE_RECOVERY.json',dict(root=str(recovery),capsule_sha256=operator.digest(recovery/'capsule/RECOVERY.json')))
        result=operator.status(self.c,r['name'])
        self.assertEqual(result['total_retained_with_recovery'],213)
        self.assertEqual(result['active_recovery']['previous_collections'][0]['retained'],{'train':53})

    def test_names_cannot_escape_roots(self):
        for name in ('../bad','a/b','a,b','UPPER','a\n','a'*49,'','-name'):
            with self.assertRaises(ValueError):operator.validate_name(name)

    def test_atomic_receipt_never_overwrites(self):
        f=self.root/'r.json';operator.write_once(f,{'a':1})
        with self.assertRaises(FileExistsError):operator.write_once(f,{'a':2})
        self.assertEqual(operator.read(f),{'a':1})
        self.assertEqual(stat.S_IMODE(f.stat().st_mode),0o640)
        self.assertEqual([p.name for p in self.root.iterdir()],['r.json'])

    def test_shared_registry_lock_assigns_disjoint_ranges(self):
        def reserve(i):
            with operator.registry_lock(self.c):
                r=operator.plan(self.c,f'run-{i}')
                operator.write_once(Path(self.c['registry_root'])/(r['name']+'.json'),r)
                return r['cycle_start']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            values=list(pool.map(reserve,range(4)))
        self.assertEqual(sorted(values),[84000,86000,88000,90000])
        self.assertEqual(stat.S_IMODE((Path(self.c['registry_root'])/'.lock').stat().st_mode),0o660)

    def test_restrictive_umask_keeps_shared_receipts_readable(self):
        old=os.umask(0o077)
        try:
            with operator.registry_lock(self.c):
                receipt=Path(self.c['registry_root'])/'one.json'
                operator.write_once(receipt,dict(owner_uid=os.getuid()))
        finally:os.umask(old)
        self.assertEqual(stat.S_IMODE(receipt.stat().st_mode),0o640)
        self.assertEqual(stat.S_IMODE((receipt.parent/'.lock').stat().st_mode),0o660)

    def test_submit_is_idempotent_and_has_exact_bundle_and_output(self):
        root,r=self.seed_ready();calls=[]
        def command(cmd,**kwargs):
            calls.append(cmd)
            return '4321' if cmd[0]=='sbatch' else 'verified'
        with patch.object(operator,'checked',side_effect=command):
            a=operator.submit(self.c,r['name']);b=operator.submit(self.c,r['name'])
        self.assertEqual(a['job_id'],'4321');self.assertTrue(b['already_submitted'])
        self.assertEqual(sum(cmd[0]=='sbatch' for cmd in calls),1)
        command=next(cmd for cmd in calls if cmd[0]=='sbatch')
        self.assertIn('--chdir='+str(root),command)
        self.assertIn('--export=ALL,MPM_TRAINING_BUNDLE='+r['bundle']+',MPM_TRAINING_OUTPUT='+r['output']+',MPM_TRAINING_PYTHON=/python',command)
        self.assertIn('mpm_training_campaign.sbatch',command[-1])
        self.assertIn('--gres=gpu:RTX_PRO_6000:4',command)

    def test_submission_requests_configured_two_gpus(self):
        root,r=self.seed_ready()
        r['collection']=dict(hardware=dict(gpus=2,gpu_type='RTX_PRO_6000',walltime_hours=48))
        command=operator.submission_command(r,dict(genesis_python='/python'))
        self.assertIn('--gres=gpu:RTX_PRO_6000:2',command)
        self.assertNotIn('--gres=gpu:RTX_PRO_6000:4',command)

    def test_ambiguous_submission_preserves_claim_and_cannot_retry(self):
        root,r=self.seed_ready()
        def command(cmd,**kwargs):
            if cmd[0]=='sbatch':raise subprocess.TimeoutExpired(cmd,60)
            return 'verified'
        with patch.object(operator,'checked',side_effect=command):
            with self.assertRaises(subprocess.TimeoutExpired):operator.submit(self.c,r['name'])
        self.assertTrue((root/'SUBMISSION_UNCERTAIN.json').exists())
        with patch.object(operator,'checked') as call:
            with self.assertRaises(RuntimeError):operator.submit(self.c,r['name'])
            call.assert_not_called()

    def test_changed_bundle_or_existing_data_cannot_submit(self):
        root,r=self.seed_ready();Path(r['output']).mkdir()
        with patch.object(operator,'checked') as call:
            with self.assertRaises(FileExistsError):operator.submit(self.c,r['name'])
            call.assert_not_called()
        (Path(r['bundle'])/'BUNDLE.json').write_text('{}')
        with self.assertRaises(ValueError):operator.ready(self.c,r['name'])

    def test_scheduler_test_creates_no_submission_claim(self):
        root,r=self.seed_ready()
        response=subprocess.CompletedProcess([],0,'','forecast')
        with patch.object(operator,'checked',return_value='verified'),patch.object(operator.subprocess,'run',return_value=response) as call:
            result=operator.submit(self.c,r['name'],test_only=True)
        self.assertFalse(result['submitted']);self.assertIn('--test-only',call.call_args.args[0])
        self.assertFalse((root/'SUBMISSION_CLAIM.json').exists())

    def test_continuation_uses_new_cycles_and_preserves_previous_root(self):
        root,r=self.seed_ready()
        p=operator.plan(self.c,'night-two',continue_from='night-one')
        self.assertEqual(p['cycle_start'],r['cycle_stop'])
        self.assertEqual(p['continue_from'],'night-one')
        self.assertNotEqual(p['root'],r['root']);self.assertTrue(root.exists())

    def test_status_counts_only_complete_receipts_not_partial_directories(self):
        root=self.root/'data';chunk=root/'chunks/train-84000';chunk.mkdir(parents=True)
        operator.write_once(chunk/'COMPLETE.json',dict(status='complete',entry=dict(evaluation='train'),retained=55,attempts=60))
        (root/'chunks/train-84001').mkdir()
        result=operator.collection_status(root)
        self.assertEqual(result['completed_chunks'],1);self.assertEqual(result['retained'],{'train':55})
        self.assertFalse(result['accepted_for_training'])

    def test_config_rejects_unsafe_exports_and_unbounded_seeds(self):
        f=self.root/'config.json'
        for config in (dict(self.c,campaigns_root='/tmp/a,b'),dict(self.c,first_cycle=0),dict(self.c,cycles_per_run=4000)):
            f.write_text(json.dumps(config))
            with self.assertRaises(ValueError):operator.load_config(f)


if __name__=='__main__':unittest.main()
