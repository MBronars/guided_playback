import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'collection/mpm/launch'))
spec=importlib.util.spec_from_file_location('training_test',ROOT/'collection/mpm/launch/mpm_training.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
sys.path.pop(0)


class TrainingTests(unittest.TestCase):
    def test_scope_is_all_training_strata_not_selected_good_cases(self):
        rows=[{'case':{'template':{'split':'train','n_objects':n},
                       'provenance':{'stiffness_phase':p%3}}} for n in range(1,5) for p in range(5)]
        rows.append({'case':{'template':{'split':'validation','n_objects':6}}})
        self.assertEqual(training.training_indices(rows),list(range(20)))
        with self.assertRaises(ValueError):training.training_indices(rows[1:])
        rows[0]['case']['template']['n_objects']=5
        with self.assertRaises(ValueError):training.training_indices(rows)

    def test_changed_simulation_or_consumers_invalidate_reuse(self):
        source=dict(files={'source/physics.py':'x','source/loader.py':'y'},banks='bank',bank_complete_sha256='hash',reference='ref')
        training.same_sources(source,source)
        for changed in (dict(source,files={'source/physics.py':'z','source/loader.py':'y'}),
                        dict(source,files={'source/physics.py':'x'}),dict(source,bank_complete_sha256='new')):
            with self.assertRaises(ValueError):training.same_sources(source,changed)


if __name__=='__main__':unittest.main()
