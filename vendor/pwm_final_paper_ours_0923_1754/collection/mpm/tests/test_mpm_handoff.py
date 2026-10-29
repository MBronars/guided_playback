import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('handoff_test',ROOT/'collection/mpm/launch/mpm_handoff.py')
handoff=importlib.util.module_from_spec(spec);spec.loader.exec_module(handoff)


class HandoffTests(unittest.TestCase):
    def test_process_identity_cannot_follow_reused_pid(self):
        original=handoff.proc(os.getpid())
        self.assertEqual(handoff.same_process(original)['pid'],os.getpid())
        for field,value in [('start','different'),('cmd',['another-command']),('uid',-1)]:
            with patch.object(handoff,'proc',return_value=dict(original,**{field:value})):
                with self.assertRaises(RuntimeError):handoff.same_process(original)

    def test_only_last_review_unit_is_a_parking_candidate(self):
        root=Path('/serial');entry={'name':'train-44'}
        cmd=['python','driver.py','review','--unit','/serial/chunks/train-44']
        self.assertTrue(handoff.last_review({'cmd':cmd},root,entry))
        self.assertFalse(handoff.last_review({'cmd':[s.replace('review','worker') for s in cmd]},root,entry))
        self.assertFalse(handoff.last_review({'cmd':cmd[:-1]+['/serial/chunks/train-43']},root,entry))


if __name__=='__main__':unittest.main()
