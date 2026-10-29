import copy
import os
import json
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'collection/mpm/launch'))
import mpm_training_campaign as campaign
sys.path.pop(0)


class TrainingCampaignTests(unittest.TestCase):
    def test_schedule_is_bounded_disjoint_and_training_only(self):
        rows=[r for i in range(3125) for r in campaign.entries(i)]
        self.assertEqual([r['cycle'] for r in rows],list(range(70000,82500)))
        self.assertEqual(len(rows)*60,750000)
        self.assertTrue(all(r['split']==r['evaluation']=='train' for r in rows))
        for i in (-1,3125,False,1.5):
            with self.assertRaises(ValueError):campaign.entries(i)
        for count in (1,3,True,4.0):
            with self.assertRaises(ValueError):campaign.entries(0,count)
        with self.assertRaises(ValueError):campaign.entries(0,4,False)
        self.assertEqual(campaign.entries(0,cycle_start=84000,cycles=2000)[0]['cycle'],84000)
        self.assertEqual(campaign.entries(499,cycle_start=84000,cycles=2000)[-1]['cycle'],85999)
        for start,count in ((99000,2000),(84000,3),(-1,2000),(84000,12504)):
            with self.assertRaises(ValueError):campaign.contract(start,count)

    def test_two_gpu_schedule_and_writer_contract(self):
        rows=[r for i in range(1000) for r in campaign.entries(i,2,cycle_start=84000,cycles=2000)]
        self.assertEqual([r['cycle'] for r in rows],list(range(84000,86000)))
        self.assertEqual({r['lane'] for r in rows},{0,1})
        self.assertEqual(campaign.contract(gpu_count=2)['background_writer'],
                         dict(workers=4,max_bytes=64*1024**2,max_pending=32))
        with self.assertRaises(ValueError):campaign.entries(1000,2,cycle_start=84000,cycles=2000)

    def test_two_gpu_first_wave_requires_both_consumers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'PILOT_READY.json').write_text('{}')
            bundle=root/'bundle';bundle.mkdir()
            (bundle/'BUNDLE.json').write_text('{"cycle_start":70000,"gpu_count":2}')
            rows=self.pilot_results()[:2]
            with self.assertRaises(ValueError):campaign.approve_pilot(root,bundle,rows[:1])
            campaign.approve_pilot(root,bundle,rows)
            self.assertTrue((root/'PILOT_APPROVED.json').exists())

    def test_48_hour_single_allocation_without_nested_submission(self):
        script=ROOT/'collection/mpm/launch/mpm_training_campaign.sbatch'
        subprocess.run(['bash','-n',str(script)],check=True)
        text=script.read_text()
        for expected in ('--time=2-00:00:00','--gres=gpu:RTX_PRO_6000:4','--mem=128G','--signal=B:USR1@18000'):
            self.assertIn(expected,text)
        self.assertNotIn('sbatch ',text)
        self.assertTrue(campaign.contract()['startup_then_collection_same_allocation'])
        self.assertFalse(campaign.contract()['evaluation_approved'])

    def test_deadline_uses_scheduler_epoch_and_rejects_wrong_scope(self):
        job=dict(job_id=123,job_state=['RUNNING'],end_time=dict(set=True,infinite=False,number=172000))
        with patch.dict(os.environ,SLURM_JOB_ID='123'):
            self.assertEqual(campaign.deadline(job,1000),172000)
            for bad in (dict(job,job_id=124),dict(job,job_state=['PENDING']),
                        dict(job,end_time=dict(set=True,infinite=True,number=172000)),
                        dict(job,end_time=dict(set=True,infinite=False,number=18000)),
                        dict(job,end_time=dict(set=True,infinite=False,number=200000))):
                with self.assertRaises(ValueError):campaign.deadline(bad,1000)

    def pilot_results(self):
        return [dict(entry=e,status='complete',attempts=60,retained=55,
                     model=dict(profiles=2,episodes_per_profile=55),
                     attempt_status=dict(complete=59,placement_rejected=1)) for e in campaign.entries(0)]

    def test_automatic_first_wave_requires_all_four_reviewed_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'PILOT_READY.json').write_text('{}')
            bundle=root/'bundle';bundle.mkdir();(bundle/'BUNDLE.json').write_text('{"cycle_start":70000}')
            campaign.approve_pilot(root,bundle,self.pilot_results())
            r=campaign.support.read(root/'PILOT_APPROVED.json')
            self.assertFalse(r['manual_visual_review'])
            self.assertEqual(r['pilot_sha256'],campaign.support.sha(root/'PILOT_READY.json'))

    def test_failed_or_incomplete_first_wave_cannot_authorize_continuation(self):
        examples=[]
        for key,value in [('retained',0),('status','failed'),('attempts',59),
                          ('model',dict(profiles=1,episodes_per_profile=55)),
                          ('attempt_status',dict(complete=59,failed=1))]:
            rows=self.pilot_results();rows[0][key]=value;examples.append(rows)
        examples.append(self.pilot_results()[:3])
        rows=self.pilot_results();rows[0]['entry']['evaluation']='count_seen';examples.append(rows)
        for rows in examples:
            with patch.object(campaign.support,'write') as write,patch.object(campaign.support,'read',return_value={'cycle_start':70000}):
                with self.assertRaises(ValueError):campaign.approve_pilot(Path('/not-created'),Path('/bundle'),rows)
                write.assert_not_called()

    def test_qualification_requires_all_four_lanes_and_grid_checks(self):
        c=dict(passed=True,integrity_pass=True,selection_equal=True,position_rms_m=1e-5,
               wall_events=dict(contact_presence_equal=True,maximum_step_difference=2))
        lane=dict(mode=campaign.MODE,passed=True,speedup=5.,comparisons=[c]*20,repeats=[c]*2,
                  candidate_repeats=[c]*2,cross_architecture=[dict(case=i,reference_lane=j,serial=c,candidate=c)
                                                            for i in range(20) for j in range(2)])
        lanes=[dict(lane,uuid=f'GPU-{i}') for i in range(4)]
        files={'/q/TRAINING_PASS.json':dict(scope=campaign.contract(),mode=campaign.MODE,evaluation_approved=False,
            bundle_sha256='hash',lanes=lanes,files={f'lane-{i}/CACHE_SEED.json':'hash' for i in range(4)},consumer=[dict(profiles=2,episodes_per_profile=20)]*4),
            '/q/REQUEST.json':dict(uuids=[r['uuid'] for r in lanes]),'/b/BUNDLE.json':dict(training_campaign=campaign.contract())}
        for i,l in enumerate(lanes):
            files[f'/q/lane-{i}/RESULT.json']=l
            files[f'/q/lane-{i}/ISOLATION.json']=dict(grid=dict(grid_cells=1000000,mutation_rejected=True,velocity_and_events_bit_exact=True),
                failure_injection=dict(healthy_world_unchanged=True,nonfinite_stopped=True,physics_steps=0))
        def check(fs):
            with patch.object(campaign.support,'read',side_effect=lambda p:fs[str(p)]),\
                 patch.object(campaign.support,'sha',return_value='hash'),patch('collection.common.check_hashes'):
                return campaign.qualified(Path('/q'),Path('/b'))
        self.assertEqual(len(check(files)['lanes']),4)
        two=copy.deepcopy(files)
        two['/b/BUNDLE.json']=dict(gpu_count=2,training_campaign=campaign.contract(gpu_count=2))
        two['/q/TRAINING_PASS.json'].update(scope=campaign.contract(gpu_count=2),lanes=lanes[:2],
            consumer=[dict(profiles=2,episodes_per_profile=20)]*2)
        two['/q/REQUEST.json']=dict(uuids=[r['uuid'] for r in lanes[:2]])
        self.assertEqual(len(check(two)['lanes']),2)
        incomplete=copy.deepcopy(two);incomplete['/q/TRAINING_PASS.json']['consumer'].pop()
        with self.assertRaises(ValueError):check(incomplete)
        incomplete=copy.deepcopy(two);incomplete['/q/TRAINING_PASS.json']['lanes'].pop()
        with self.assertRaises(ValueError):check(incomplete)
        bad=copy.deepcopy(files);bad['/q/TRAINING_PASS.json']['lanes'].pop()
        with self.assertRaises(ValueError):check(bad)
        bad=copy.deepcopy(files);bad['/q/lane-3/ISOLATION.json']['grid']['velocity_and_events_bit_exact']=False
        with self.assertRaises(ValueError):check(bad)

    def test_bounded_runner_uses_controller_and_drains_review_on_stop(self):
        import mpm_four as four
        uuids=[f'GPU-{i}' for i in range(4)]
        config=dict(gpu_count=4,training_only=True,cycle_start=70000,shutdown_reserve_seconds=18000,
                    maximum_admission_seconds=162000,genesis_python='/python',retained_training_target=500000,
                    maximum_training_attempts=750000,training_cycles=12500)
        controller=SimpleNamespace(__file__=campaign.__file__,verify=lambda _:config,
            qualified=lambda *_:dict(mode=campaign.MODE,lanes=[dict(uuid=u) for u in uuids],
                                    files={f'lane-{i}/CACHE_SEED.json':'hash' for i in range(4)}),
            discover=lambda _:uuids,entries=campaign.entries,approve_pilot=campaign.approve_pilot)
        calls=[]
        def subprocess_run(command,**kwargs):
            calls.append(command)
            self.assertTrue(command[1].endswith('/mpm_training_campaign.py'))
            if command[2]=='prepare':
                unit=Path(command[command.index('--unit')+1]);unit.mkdir(parents=True)
            return SimpleNamespace(returncode=0)
        def children(commands,*args,**kwargs):
            self.assertEqual(len(commands),4)
            self.assertTrue(all('--cache-seed' in cmd and '--cache-seed-sha' in cmd for cmd in commands))
            calls.extend(commands)
            signal.raise_signal(signal.SIGUSR1)
        handlers={sig:signal.getsignal(sig) for sig in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM)}
        try:
            with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,SLURM_JOB_ID='123'):
                root=Path(tmp);bundle=root/'bundle';bundle.mkdir();(bundle/'BUNDLE.json').write_text('{"cycle_start":70000}')
                qual=root/'qual';qual.mkdir();(qual/'TRAINING_PASS.json').write_text('{}')
                results={r['entry']['name']:r for r in self.pilot_results()}
                with patch.object(four.base,'environment',return_value={}),\
                     patch.object(four.subprocess,'run',side_effect=subprocess_run),\
                     patch.object(four.support,'children',side_effect=children),\
                     patch.object(four.base,'completed',side_effect=lambda u:results[u.name]),\
                     patch.object(four.base,'publish') as publish,\
                     patch.object(four.shutil,'disk_usage',return_value=SimpleNamespace(free=1024**4)):
                    four.run(bundle,root/'output',qual,time.time()+172000,controller=controller)
                self.assertEqual(publish.call_count,4)
                receipt=json.loads((root/'output/STOP.json').read_text())
                self.assertEqual(receipt['training_retained'],220)
                self.assertEqual(receipt['reason'],'allocation_budget_stop')
                self.assertTrue((root/'output/PILOT_APPROVED.json').exists())
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)


if __name__=='__main__':unittest.main()
