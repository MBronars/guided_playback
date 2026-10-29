from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'launch'))
from mpm_async_io import BoundedFrameWriter, simulate
sys.path.pop(0)
from collection.mpm.journal import FrameJournal,verify_journal
from collection.mpm.contracts import Clock
from types import SimpleNamespace


def fake_simulate(root, *, fail=False):
    """Exercise the private binding adapter without importing CUDA/Warp."""
    journal=FrameJournal(root,initial_sha256='a'*64,manifest_sha256='b'*64,clock=Clock(.0001,2,2))
    for step in (0,2,4):journal.append(step,frame(step))
    if fail:raise ValueError('simulation failure')
    return verify_journal(root,initial_sha256='a'*64,manifest_sha256='b'*64,complete=True)


def frame(step):
    return dict(x=np.full((2,3),step,np.float32),v=np.zeros((2,3),np.float32),
        F_trial=np.tile(np.eye(3,dtype=np.float32),(2,1,1)),C=np.zeros((2,3,3),np.float32),
        wall_first_grid_step=np.full(5,-1,np.int64),wall_first_particle_step=np.full(5,-1,np.int64),
        wall_covered_substeps=np.full(5,step,np.int64),wall_simulation_steps=np.array(step,np.int64))


class AsyncWriterTests(unittest.TestCase):
    def test_production_adapter_flushes_without_mutating_module_globals(self):
        grouped=SimpleNamespace(simulate=fake_simulate,FrameJournal=FrameJournal,verify_journal=verify_journal)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertTrue(simulate(grouped,root/'async')['complete'])
            self.assertTrue(fake_simulate(root/'sync')['complete'])
            for f in (root/'sync').iterdir():self.assertEqual(f.read_bytes(),(root/'async'/f.name).read_bytes())
            self.assertIs(fake_simulate.__globals__['FrameJournal'],FrameJournal)
            with self.assertRaisesRegex(ValueError,'simulation failure'):simulate(grouped,root/'fail',fail=True)
            with patch.object(FrameJournal,'append',side_effect=OSError('write failed')):
                with self.assertRaisesRegex(OSError,'write failed'):simulate(grouped,root/'io-fail')
            self.assertFalse(self.verify(root/'io-fail')['complete'])

    def journal(self,root):return FrameJournal(root,initial_sha256='a'*64,manifest_sha256='b'*64,clock=Clock(.0001,2,2))

    def verify(self,root,**kw):return verify_journal(root,initial_sha256='a'*64,manifest_sha256='b'*64,**kw)

    def test_byte_exact_serialization_and_order_for_interleaved_journals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);serial=[self.journal(root/f's{i}') for i in range(2)]
            parallel=[self.journal(root/f'p{i}') for i in range(2)]
            with BoundedFrameWriter(max_bytes=4096,workers=2,max_pending=3) as writer:
                for step in (0,2,4):
                    for a,b in zip(serial,parallel):
                        arrays=frame(step);a.append(step,arrays);writer.append(b,step,arrays)
                writer.flush();self.assertLessEqual(writer.peak_pending_bytes,4096)
            for a,b in zip(serial,parallel):
                self.assertTrue(self.verify(b.root,complete=True)['complete'])
                for f in a.root.iterdir():self.assertEqual(f.read_bytes(),(b.root/f.name).read_bytes())

    def test_owned_copy_and_byte_budget_apply_before_next_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=self.journal(Path(tmp)/'frames');arrays=frame(0);size=sum(a.nbytes for a in arrays.values())
            release=threading.Event();entered=threading.Event();append=journal.append
            def delayed(step,data):
                entered.set()
                if not release.wait(3):raise TimeoutError('test release missing')
                append(step,data)
            attempted=threading.Event();returned=threading.Event();escaped=[]
            def unblock():
                attempted.wait(2);escaped.append(returned.wait(.05));release.set()
            observer=threading.Thread(target=unblock)
            with BoundedFrameWriter(max_bytes=size,workers=2,max_pending=16) as writer,patch.object(journal,'append',side_effect=delayed):
                writer.append(journal,0,arrays);self.assertTrue(entered.wait(2));arrays['x'].fill(99)
                observer.start();attempted.set()
                writer.append(journal,2,frame(2));returned.set();observer.join(timeout=3);writer.flush()
                self.assertEqual(writer.peak_pending_bytes,size)
                self.assertEqual(escaped,[False],'second append must wait for byte capacity')
            with np.load(journal.root/'frame-00000.npz') as data:np.testing.assert_array_equal(data['x'],0)

    def test_io_failure_is_propagated_and_no_false_complete_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=self.journal(Path(tmp)/'frames');writer=BoundedFrameWriter(max_bytes=4096)
            with patch.object(journal,'append',side_effect=OSError('injected write failure')):
                writer.append(journal,0,frame(0))
                with self.assertRaisesRegex(OSError,'injected'):writer.flush()
                with self.assertRaisesRegex(OSError,'injected'):writer.close()
            self.assertFalse(self.verify(journal.root)['complete'])
            with self.assertRaises(RuntimeError):writer.append(journal,2,frame(2))

    def test_bad_clock_and_oversize_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=self.journal(Path(tmp)/'frames')
            with BoundedFrameWriter(max_bytes=1) as writer:
                with self.assertRaisesRegex(ValueError,'clock'):writer.append(journal,2,frame(2))
                with self.assertRaisesRegex(ValueError,'byte budget'):writer.append(journal,0,frame(0))
            self.assertEqual(self.verify(journal.root)['states'],0)

    def test_producer_failure_is_not_masked_by_background_io_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            journal=self.journal(Path(tmp)/'frames')
            with patch.object(journal,'append',side_effect=OSError('disk')):
                with self.assertRaisesRegex(ValueError,'simulation'):
                    with BoundedFrameWriter(max_bytes=4096) as writer:
                        writer.append(journal,0,frame(0));raise ValueError('simulation')


if __name__=='__main__':unittest.main()
