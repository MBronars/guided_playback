"""Allocation affinity tests for the diagnostic-only capacity sweep."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'launch'))
from mpm_capacity_benchmark import affinity, ARMS, diagnostic_writer
sys.path.pop(0)


class CapacityTests(unittest.TestCase):
    def test_adapter_flushes_before_verification_and_restores_functions(self):
        from collection.mpm.journal import FrameJournal, verify_journal
        from collection.mpm.contracts import Clock
        from test_mpm_async_io import frame
        grouped = SimpleNamespace(FrameJournal=FrameJournal, verify_journal=verify_journal)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'journal'
            with diagnostic_writer(grouped):
                journal = grouped.FrameJournal(root, initial_sha256='a'*64,
                    manifest_sha256='b'*64, clock=Clock(.0001, 2, 2))
                for step in (0, 2, 4):
                    journal.append(step, frame(step))
                self.assertTrue(grouped.verify_journal(root, initial_sha256='a'*64,
                    manifest_sha256='b'*64, complete=True)['complete'])
            self.assertIs(grouped.FrameJournal, FrameJournal)
            self.assertIs(grouped.verify_journal, verify_journal)
            with self.assertRaisesRegex(ValueError, 'producer'):
                with diagnostic_writer(grouped):
                    raise ValueError('producer')
            self.assertIs(grouped.FrameJournal, FrameJournal)
            self.assertIs(grouped.verify_journal, verify_journal)

    def test_lanes_have_disjoint_bounded_cpu_sets(self):
        available = list(range(100, 228))
        for total, _ in ARMS.values():
            a, b = [affinity(available, total, lane) for lane in (0, 1)]
            self.assertFalse(set(a) & set(b))
            self.assertEqual(len(a)+len(b), total)
            self.assertLessEqual(set(a+b), set(available))

    def test_unallocated_or_invalid_requests_fail(self):
        for available, total, lane in [(range(8), 32, 0), (range(128), 128, 0), (range(128), 8, 2)]:
            with self.assertRaises(ValueError):
                affinity(available, total, lane)


if __name__ == '__main__':
    unittest.main()
