import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import copy
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'collection/mpm/launch'))
spec=importlib.util.spec_from_file_location('four_test',ROOT/'collection/mpm/launch/mpm_four.py')
four=importlib.util.module_from_spec(spec);spec.loader.exec_module(four)
sys.path.pop(0)


class FourTests(unittest.TestCase):
    def test_disjoint_bounded_training_schedule(self):
        train=[e for i in range(3126) for e in four.entries(i) if e['split']=='train']
        self.assertEqual(len(train)*60,750000)
        self.assertEqual([e['cycle'] for e in train],list(range(20000,32500)))
        self.assertTrue(all(e['cycle']>12500 for e in train))

    def test_eval_splits_never_include_test(self):
        es=four.entries(1)
        self.assertEqual([e['evaluation'] for e in es],['seen','unseen','count_seen','count_unseen'])
        self.assertTrue(all(e['split']=='validation' for e in es))

    def test_two_gpu_schedule_is_disjoint_and_complete(self):
        es=[e for i in range(6252) for e in four.entries(i,2)]
        train=[e for e in es if e['split']=='train']
        self.assertEqual([e['cycle'] for e in train],list(range(40000,52500)))
        self.assertEqual(len(train)*60,750000)
        self.assertEqual([e['evaluation'] for e in es if e['split']=='validation'],
                         ['seen','unseen','count_seen','count_unseen'])
        self.assertEqual({e['lane'] for e in es},{0,1})
        self.assertTrue(all(e['cycle']>=40000 for e in es))
        self.assertEqual(four.qualification_contract(2)['gpu_count'],2)
        for count in (0,1,3,8,True):
            with self.assertRaises(ValueError):four.entries(0,count)
        with self.assertRaises(ValueError):four.entries(6252,2)

    def test_training_only_schedule_cannot_collect_evaluation(self):
        es=[e for i in range(6250) for e in four.entries(i,2,True)]
        self.assertEqual([e['cycle'] for e in es],list(range(40000,52500)))
        self.assertTrue(all(e['split']=='train' and e['evaluation']=='train' for e in es))
        with self.assertRaises(ValueError):four.entries(6250,2,True)
        with self.assertRaises(ValueError):four.entries(0,4,True)

    def test_training_worker_rejects_out_of_scope_plans(self):
        entry={'split':'train','evaluation':'train'}
        case={'kind':'coverage','template':{'split':'train','n_objects':4}}
        four.require_training_plan(entry,[case])
        with self.assertRaises(ValueError):four.require_training_plan(dict(entry,evaluation='seen'),[case])
        with self.assertRaises(ValueError):four.require_training_plan(entry,[dict(case,template={'split':'train','n_objects':5})])
        with self.assertRaises(ValueError):four.require_training_plan(entry,[dict(case,template={'split':'validation','n_objects':4})])

    def test_reused_allocation_requires_shutdown_deadline(self):
        with patch.object(four,'verify',return_value={'gpu_count':2}),\
             patch.object(four,'qualified',return_value={'mode':four.MODES[2]}),\
             patch.object(four,'discover',return_value=['GPU-a','GPU-b']):
            with self.assertRaisesRegex(ValueError,'absolute deadline'):
                four.run(Path('/bundle'),Path('/never-created'),Path('/qual'))
            with self.assertRaisesRegex(ValueError,'shutdown reserve'):
                four.run(Path('/bundle'),Path('/never-created'),Path('/qual'),deadline_epoch=1)

    def test_schedule_rejects_unbounded_values(self):
        for value in (-1,3126,False,1.5):
            with self.assertRaises(ValueError):four.entries(value)

    def test_no_submission_without_passing_qualification(self):
        with patch.object(four,'qualified',side_effect=ValueError('no pass')),patch.object(four.subprocess,'check_output') as call:
            with self.assertRaises(ValueError):four.submit(Path('/bundle'),Path('/qual'),Path('/out'))
            call.assert_not_called()

    def test_four_gpu_request_and_conditional_launch(self):
        for name in ('mpm_training_campaign.sbatch',):
            path=ROOT/'collection/mpm/launch'/name;subprocess.run(['bash','-n',str(path)],check=True)
            self.assertIn('--gres=gpu:RTX_PRO_6000:4',path.read_text())
            self.assertIn('set -euo pipefail',path.read_text())
        self.assertIn('--time=2-00:00:00',(ROOT/'collection/mpm/launch/mpm_training_campaign.sbatch').read_text())

    def test_predeclared_gate_and_concurrency(self):
        self.assertEqual(four.POSITION_RMS_M,1e-4)
        self.assertEqual([m['size'] for m in four.MODES if m['fused']],[4,8,16,32])
        self.assertEqual(four.QUALIFICATION_MODES,[dict(name='fused4',size=4,block=64,fused=True)])
        self.assertEqual(four.WALL_EVENT_STEP_TOLERANCE,2)
        self.assertEqual(four.SERIAL_REPEAT_CASES,(3,25,26))
        self.assertEqual(four.SERIAL_REPEATS,2)

    def test_wall_event_timing_is_bounded_and_exactness_preserved(self):
        a=np.full((3,5),-1,dtype=np.int32);a[1:,3]=100
        for shift in (0,1,2,-1,-2):
            b=a.copy();b[1:,3]+=shift
            r=four.compare_events(a,b)
            self.assertTrue(r['passed']);self.assertEqual(r['exact'],shift==0)
            self.assertEqual(r['maximum_step_difference'],abs(shift))
        for shift in (3,-3,1000):
            b=a.copy();b[1:,3]+=shift
            self.assertFalse(four.compare_events(a,b)['passed'])

    def test_event_sentinel_is_never_treated_as_small_timing_error(self):
        a=np.full((3,5),-1,dtype=np.int32);b=a.copy();b[1:,3]=0
        for x,y in ((a,b),(b,a)):
            self.assertFalse(four.compare_events(x,y)['passed'])

    def test_same_final_contact_is_insufficient(self):
        a=np.full((3,5),-1,dtype=np.int32);a[1:,3]=100
        b=a.copy();b[1,3]=-1  # Delayed saved-frame detection still fails.
        self.assertFalse(four.compare_events(a,b)['passed'])
        b=a.copy();b[1:,2]=b[1:,3];b[1:,3]=-1  # Different wall fails.
        self.assertFalse(four.compare_events(a,b)['passed'])

    def test_invalid_event_arrays_rejected(self):
        a=np.full((3,5),-1,dtype=np.int32)
        for b in (a[:2],a[:,:4],a.astype(float),a-1):
            with self.assertRaises(ValueError):four.compare_events(a,b)

    def test_event_tolerance_cannot_waive_other_gates(self):
        row=dict(integrity_pass=True,selection_equal=True,position_rms_m=1e-5,
                 wall_events=dict(contact_presence_equal=True,maximum_step_difference=2))
        self.assertTrue(four.comparison_passes(row))
        for field,value in [('integrity_pass',False),('selection_equal',False),
                            ('position_rms_m',.00010001),('position_rms_m',float('nan')),
                            ('position_rms_m',float('inf')),('position_rms_m',-1)]:
            self.assertFalse(four.comparison_passes(dict(row,**{field:value})))
        for field,value in [('contact_presence_equal',False),('maximum_step_difference',3)]:
            changed=dict(row,wall_events=dict(row['wall_events'],**{field:value}))
            self.assertFalse(four.comparison_passes(changed))

    def test_qualification_rechecks_every_lane_and_repeat(self):
        mode=four.QUALIFICATION_MODES[0]
        c=dict(passed=True,integrity_pass=True,selection_equal=True,position_rms_m=1e-5,
               wall_events=dict(contact_presence_equal=True,maximum_step_difference=2))
        repeats=[dict(c,case=i,repeat=j) for i in four.SERIAL_REPEAT_CASES for j in range(four.SERIAL_REPEATS)]
        fixtures={
            '/bundle/BUNDLE.json':dict(gpu_count=2),
            '/qual/PASS.json':dict(gpu_count=2,mode=mode,bundle_sha256='hash',files={}),
        }
        for lane in range(2):
            fixtures[f'/qual/lane-{lane}/RESULT.json']=dict(results=[dict(mode=mode,passed=True,speedup=5.,
                comparisons=[copy.deepcopy(c) for _ in range(32)],repeat_comparisons=copy.deepcopy(repeats))])
            fixtures[f'/qual/lane-{lane}/REPEATABILITY.json']=dict(passed=True,comparisons=copy.deepcopy(repeats))
        def check(fs):
            with patch.object(four.support,'read',side_effect=lambda p:fs[str(p)]),\
                 patch.object(four.support,'sha',return_value='hash'),\
                 patch('collection.common.check_hashes'):
                return four.qualified(Path('/qual'),Path('/bundle'))
        self.assertEqual(check(fixtures)['gpu_count'],2)
        changed=copy.deepcopy(fixtures);changed['/qual/PASS.json']['gpu_count']=4
        with self.assertRaises(ValueError):check(changed)
        changed=copy.deepcopy(fixtures)
        changed['/qual/lane-1/RESULT.json']['results'][0]['comparisons'][31]['wall_events']['maximum_step_difference']=3
        with self.assertRaises(ValueError):check(changed)
        changed=copy.deepcopy(fixtures)
        changed['/qual/lane-1/REPEATABILITY.json']['comparisons'].pop()
        with self.assertRaises(ValueError):check(changed)
        changed=copy.deepcopy(fixtures)
        changed['/qual/lane-0/RESULT.json']['results'][0]['repeat_comparisons'][0]['position_rms_m']=.000101
        with self.assertRaises(ValueError):check(changed)


if __name__=='__main__':unittest.main()
