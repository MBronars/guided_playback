import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'collection/mpm/launch'))
import mpm_ada as ada
sys.path.pop(0)


class AdaTests(unittest.TestCase):
    def test_devices_are_distinct_ada_only(self):
        devices = [dict(uuid=f'GPU-{i}', name='NVIDIA RTX 6000 Ada Generation') for i in range(2)]
        self.assertEqual(ada.validate_devices(devices), ['GPU-0', 'GPU-1'])
        for bad in (devices[:1], [devices[0], devices[0]],
                    [devices[0], dict(uuid='GPU-1', name='NVIDIA RTX PRO 6000')]):
            with self.assertRaises(ValueError): ada.validate_devices(bad)

    def test_bounded_disjoint_training_only(self):
        self.assertEqual([r['cycle'] for r in ada.pilot_entries()], [60000, 60001])
        self.assertTrue(all(r['split']==r['evaluation']=='train' for r in ada.pilot_entries()))
        self.assertFalse(ada.contract()['automatic_production'])
        self.assertEqual(ada.contract()['maximum_pilot_attempts'], 120)
        self.assertFalse(ada.MODE['fused'])
        self.assertFalse(ada.contract()['wall_fusion_approved'])

    def test_all_comparisons_and_both_references_required(self):
        c = dict(passed=True, integrity_pass=True, selection_equal=True, position_rms_m=0.,
                 wall_events=dict(contact_presence_equal=True, maximum_step_difference=2))
        r = dict(speedup=2., comparisons=[c]*20, repeats=[c]*2, candidate_repeats=[c]*2,
                 cross_architecture=[dict(case=i, reference_lane=j, serial=c, candidate=c)
                                     for i in range(20) for j in range(2)])
        self.assertTrue(ada.lane_passes(r))
        for key in ('comparisons', 'repeats', 'candidate_repeats', 'cross_architecture'):
            bad=copy.deepcopy(r); bad[key].pop()
            self.assertFalse(ada.lane_passes(bad))
        for field, value in [('position_rms_m', .00010001), ('selection_equal', False), ('integrity_pass', False)]:
            bad=copy.deepcopy(r); bad['cross_architecture'][0]['candidate'][field]=value
            self.assertFalse(ada.lane_passes(bad))
        for speed in (1., float('nan'), float('inf')):
            self.assertFalse(ada.lane_passes(dict(r, speedup=speed)))


if __name__ == '__main__': unittest.main()
