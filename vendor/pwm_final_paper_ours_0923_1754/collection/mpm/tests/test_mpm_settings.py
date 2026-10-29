import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'launch'))
import mpm_settings as settings
import mpm_collect as operator
import mpm_training_campaign as campaign
sys.path.pop(0)


class SettingsTests(unittest.TestCase):
    def metadata(self,role='train'):
        return dict(schema='fresh-candidate-bank-v1',bank_role=role,
                    objects=[dict(asset_uid=f'object-{i}') for i in range(100)])

    def test_defaults_reproduce_original_specifications_exactly(self):
        from collection.mpm.protocol import specifications
        for cycle in (0,84000,86000):
            self.assertEqual(settings.specifications(self.metadata(),recipe={},cycle=cycle,limit=60),
                             specifications(self.metadata(),split='train',evaluation='train',cycle_start=cycle))

    def test_budget_is_exact_and_bounded_for_partial_waves(self):
        for n in (4,5,239,240,241,500,1000,120000):
            r=settings.normalize(dict(max_attempts=n));count=settings.cycles(r)
            self.assertEqual(count%4,0);self.assertLessEqual(count,2000)
            values=[settings.attempts(r,i) for i in range(count)]
            self.assertEqual(sum(values),n);self.assertTrue(all(1<=v<=60 for v in values))

    def test_split_category_selects_bank_without_leakage(self):
        for split in ('validation','test'):
            for evaluation in ('seen','unseen','count_seen','count_unseen'):
                r=settings.normalize(dict(split=split,evaluation=evaluation))
                role=split if evaluation.endswith('unseen') else 'train'
                self.assertEqual(settings.bank_role(r),role)
                self.assertFalse(settings.capability(r)['launch_supported'])
                rows=settings.specifications(self.metadata(role),recipe=r,cycle=0,limit=6)
                allowed=[5,6,7] if evaluation.startswith('count_') else [1,2,3,4]
                self.assertTrue(all(s['count'] in allowed and s['split']==split for s in rows))
                with self.assertRaises(ValueError):settings.specifications(self.metadata('wrong'),recipe=r,cycle=0,limit=6)

    def test_invalid_options_and_unknown_keys_fail_closed(self):
        for value in (dict(split='train',evaluation='unseen'),dict(object_counts=[5]),
                      dict(object_counts=[1,1]),dict(object_counts=[True]),dict(object_weights=[1]),
                      dict(object_weights=[1,0,1,1]),dict(object_weights=[1,float('nan'),1,1]),
                      dict(action_weights={'typo':1}),dict(object_counts=[1],action_weights={'stack':1}),
                      dict(max_attempts=3),dict(max_attempts=120001),dict(max_attempts=True),
                      dict(target_retained=120001),dict(typo=True),
                      dict(hardware={'gpus':3,'gpu_type':'RTX_PRO_6000','walltime_hours':48})):
            with self.subTest(value=value),self.assertRaises(ValueError):settings.normalize(value)

    def test_two_gpu_budget_is_bounded_and_exact(self):
        for budget in (4,61,121,120000):
            r=settings.normalize(dict(max_attempts=budget,hardware=dict(gpus=2,gpu_type='RTX_PRO_6000',walltime_hours=48)))
            n=settings.cycles(r)
            self.assertEqual(n%2,0)
            self.assertEqual(sum(settings.attempts(r,i) for i in range(n)),budget)
            self.assertTrue(all(1<=settings.attempts(r,i)<=60 for i in range(n)))

    def test_custom_mixture_is_deterministic_and_has_unique_keys(self):
        r=dict(object_counts=[1,3],object_weights=[100,1],action_weights={'translate':1,'stack':1})
        a=settings.specifications(self.metadata(),recipe=r,cycle=42,limit=60)
        self.assertEqual(a,settings.specifications(self.metadata(),recipe=r,cycle=42,limit=60))
        b=settings.specifications(self.metadata(),recipe=r,cycle=43,limit=60)
        self.assertFalse({s['seed_namespace'] for s in a}&{s['seed_namespace'] for s in b})
        self.assertGreater(sum(s['count']==1 for s in a),50)
        self.assertTrue(all(s['scenario']==('translate' if s['count']==1 else 'stack') for s in a))
        self.assertTrue(all(len(set(s['asset_indices']))==s['count'] for s in a))

    def test_configuration_change_changes_sampling_identity(self):
        a=settings.specifications(self.metadata(),recipe={'object_counts':[2]},cycle=42,limit=12)
        b=settings.specifications(self.metadata(),recipe={'object_counts':[2],'action_weights':{'stack':1}},cycle=42,limit=12)
        self.assertFalse({s['seed_namespace'] for s in a}&{s['seed_namespace'] for s in b})

    def test_custom_materials_do_not_lock_to_one_stiffness_each(self):
        rows=settings.specifications(self.metadata(),recipe={'object_counts':[1]},cycle=42,limit=9)
        pairs={(s['material_offset'],(s['material_offset']+s['stiffness_phase'])%3) for s in rows}
        self.assertEqual(pairs,{(material,stiffness) for material in range(3) for stiffness in range(3)})

    def test_hardware_and_budget_do_not_change_custom_sample_prefix(self):
        a=dict(object_counts=[2],max_attempts=240)
        b=dict(a,max_attempts=1000,hardware=dict(gpus=4,gpu_type='RTX_PRO_6000',walltime_hours=24))
        self.assertEqual(settings.specifications(self.metadata(),recipe=a,cycle=42,limit=12),
                         settings.specifications(self.metadata(),recipe=b,cycle=42,limit=12))

    def test_evaluation_preflight_rejects_before_io_or_submission(self):
        c=dict(collection=dict(split='validation',evaluation='unseen'))
        with patch.object(operator,'digest') as read:
            with self.assertRaisesRegex(ValueError,'Evaluation generation'):operator.preflight(c)
            read.assert_not_called()

    def test_partial_first_wave_still_requires_every_review(self):
        r=settings.normalize(dict(max_attempts=9));config=dict(cycle_start=84000,recipe_profile=r)
        rows=[]
        for i in range(4):
            n=settings.attempts(r,i)
            rows.append(dict(entry=dict(cycle=84000+i,split='train',evaluation='train'),status='complete',
                attempts=n,retained=n,model=dict(profiles=2,episodes_per_profile=n),attempt_status={'complete':n}))
        with patch.object(campaign.support,'read',return_value=config),patch.object(campaign.support,'sha',return_value='x'),patch.object(campaign.support,'write'):
            campaign.approve_pilot(Path('/r'),Path('/b'),rows)
            rows[0]['attempts']+=1
            with self.assertRaises(ValueError):campaign.approve_pilot(Path('/r'),Path('/b'),rows)


if __name__=='__main__':unittest.main()
