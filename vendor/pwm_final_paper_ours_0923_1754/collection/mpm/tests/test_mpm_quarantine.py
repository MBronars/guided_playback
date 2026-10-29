import copy,json,sys,unittest,os,signal,tempfile,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'launch'))
import mpm_campaign as campaign
import mpm_four as four

class QuarantineTests(unittest.TestCase):
    def rows(self):
        return [dict(integrity_pass=True) for _ in range(59)]+[dict(
            integrity_pass=False,eligible_for_export=False,eligible_for_wall_free_export=False,
            collapse=dict(eligible=False,reasons=['object_0:extreme_contraction']),
            reasons=['object_0:extreme_contraction'],episode_id='bad',raw_path='/bad',raw_sha256='a'*64)]
    def test_isolated_contraction_is_excluded_without_modifying_verdict(self):
        rows=self.rows();original=copy.deepcopy(rows)
        r=campaign.quarantine_policy(rows)
        self.assertEqual(len(r['excluded']),1);self.assertEqual(rows,original)
        self.assertFalse(r['thresholds_changed'])
    def test_unknown_integrity_and_deformation_failures_remain_fatal(self):
        for reason in ('nan','hash_changed','object_0:point_collapse','object_0:widespread_degenerate_deformation'):
            rows=self.rows();rows[-1]['reasons']=[reason];rows[-1]['collapse']['reasons']=[reason]
            with self.assertRaises(ValueError):campaign.quarantine_policy(rows)
    def test_multiple_small_batch_or_wrong_eligibility_is_fatal(self):
        rows=self.rows()
        for r in (rows+[rows[-1]],rows[-2:]):
            with self.assertRaises(ValueError):campaign.quarantine_policy(r)
        rows[-1]['eligible_for_export']=True
        with self.assertRaises(ValueError):campaign.quarantine_policy(rows)
    def test_repeated_exclusions_stop_across_chunks(self):
        clean=dict(quarantine=dict(excluded=[]));bad=dict(quarantine=dict(excluded=[{}]))
        four.check_quarantine_window([bad]+[clean]*9)
        with self.assertRaises(ValueError):four.check_quarantine_window([bad]+[clean]*8+[bad])
        four.check_quarantine_window([bad]+[clean]*9+[bad])

    def test_recovery_can_route_worker_through_telemetry_capsule(self):
        self.test_recovery_offset_keeps_gpu_driver_and_gates_first_wave(Path('/capsule/mpm_recovery.py'))

    def test_recovery_offset_keeps_gpu_driver_and_gates_first_wave(self,worker_driver=None):
        uuids=['GPU-0','GPU-1'];mode=dict(name='fused4',size=4,block=64,fused=True)
        config=dict(gpu_count=2,training_only=True,cycle_start=10,training_cycles=6,
            shutdown_reserve_seconds=18000,maximum_admission_seconds=162000,
            genesis_python='/python',retained_training_target=240,maximum_training_attempts=240)
        def entries(index,count,training):
            self.assertTrue(training);self.assertEqual(count,2)
            return [dict(name=f'train-{10+2*index+i}',cycle=10+2*index+i,split='train',evaluation='train',lane=i) for i in range(2)]
        def approve(root,bundle,results):
            self.assertEqual([r['entry']['cycle'] for r in results],[12,13])
            four.support.write(root/'PILOT_APPROVED.json',dict(pilot_sha256=four.support.sha(root/'PILOT_READY.json')))
        controller=SimpleNamespace(__file__='/frozen/mpm_training_campaign.py',verify=lambda _:config,
            qualified=lambda *_:dict(mode=mode,lanes=[dict(uuid=u) for u in uuids],files={f'lane-{i}/CACHE_SEED.json':'sha' for i in range(2)}),
            discover=lambda _:uuids,entries=entries,approve_pilot=approve)
        calls=[]
        def command(cmd,**kw):
            calls.append(cmd)
            unit=Path(cmd[cmd.index('--unit')+1])
            if cmd[2]=='prepare':unit.mkdir(parents=True)
            elif cmd[2]=='review':self.assertEqual(cmd[1],'/capsule/mpm_recovery.py')
        def workers(commands,*a,**kw):
            self.assertEqual(len(commands),2)
            self.assertTrue(all(c[1]==str(worker_driver) if worker_driver else c[1].endswith('/mpm_training_campaign.py') for c in commands))
            self.assertTrue(all('--cache-seed' in c for c in commands))
            signal.raise_signal(signal.SIGUSR1)
        result={e['name']:dict(entry=e,status='complete',attempts=60,retained=55) for e in entries(1,2,True)}
        handlers={s:signal.getsignal(s) for s in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM)}
        try:
            with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,SLURM_JOB_ID='123'):
                root=Path(tmp);bundle=root/'bundle';bundle.mkdir();(bundle/'BUNDLE.json').write_text('{}')
                qual=root/'qual';qual.mkdir();(qual/'TRAINING_PASS.json').write_text('{}')
                with patch.object(four.base,'environment',return_value={}),patch.object(four.subprocess,'run',side_effect=command),\
                     patch.object(four.support,'children',side_effect=workers),patch.object(four.base,'completed',side_effect=lambda p:result[p.name]),\
                     patch.object(four.base,'publish'),patch.object(four.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)):
                    four.run(bundle,root/'out',qual,time.time()+172000,controller=controller,start_wave=1,review_driver=Path('/capsule/mpm_recovery.py'),worker_driver=worker_driver)
                self.assertTrue((root/'out/PILOT_APPROVED.json').exists())
                self.assertEqual(json.loads((root/'out/STOP.json').read_text())['training_retained'],110)
                requested=[json.loads(c[c.index('--entry')+1])['cycle'] for c in calls if c[2]=='prepare']
                self.assertEqual(sorted(requested),[12,13,14,15])
        finally:
            for s,h in handlers.items():signal.signal(s,h)

if __name__=='__main__':unittest.main()
