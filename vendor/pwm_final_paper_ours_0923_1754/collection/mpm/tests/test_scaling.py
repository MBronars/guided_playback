import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
import selectors
import importlib.util
from unittest.mock import patch

import numpy as np

from collection.mpm.contracts import Clock
from collection.mpm.journal import FrameJournal, verify_journal
from collection.mpm.recipes import load_bank, scene, specifications
from collection.mpm.interactions import analyze, positive_stencil_cells
from collection.mpm.collect import gpu_status


def frame(step):
    return dict(x=np.zeros((2, 3), np.float32), v=np.zeros((2, 3), np.float32),
        F_trial=np.tile(np.eye(3, dtype=np.float32), (2, 1, 1)), C=np.zeros((2, 3, 3), np.float32),
        wall_first_grid_step=np.full(5, -1, np.int64), wall_first_particle_step=np.full(5, -1, np.int64),
        wall_covered_substeps=np.full(5, step, np.int64), wall_simulation_steps=np.array(step, np.int64))


class JournalTests(unittest.TestCase):
    def test_killed_writer_keeps_frame_prefix_and_releases_lease(self):
        from collection.lease import gpu_lease
        with tempfile.TemporaryDirectory() as tmp:
            code = ('import sys, signal\nfrom pathlib import Path\n'
                'from collection.lease import gpu_lease\n'
                'from collection.mpm.journal import FrameJournal\n'
                'from collection.mpm.contracts import Clock\n'
                'from test_scaling import frame\n'
                'with gpu_lease("GPU-kill-test", sys.argv[1]):\n'
                ' j=FrameJournal(Path(sys.argv[1])/"frames", initial_sha256="a"*64, manifest_sha256="b"*64, clock=Clock(.0001,2,2))\n'
                ' j.append(0,frame(0))\n print("ready",flush=True)\n signal.pause()\n')
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent)+os.pathsep+os.getcwd())
            child = subprocess.Popen([sys.executable, '-c', code, tmp], stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, env=env)
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(child.stdout, selectors.EVENT_READ)
                    self.assertTrue(selector.select(timeout=20), 'owned child did not become ready')
                self.assertEqual(child.stdout.readline().strip(), 'ready')
                child.kill()  # Only this test's newly launched child PID.
                child.wait(timeout=10)
                self.assertEqual(child.returncode, -9)
                result = self.verify(Path(tmp)/'frames')
                self.assertEqual(result['states'], 1)
                self.assertFalse(result['complete'])
                with gpu_lease('GPU-kill-test', tmp): pass
            finally:
                if child.poll() is None: child.kill(); child.wait(timeout=10)
                child.stdout.close(); child.stderr.close()

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'requires optional HDF5 collection dependency')
    def test_all_journal_fields_match_completed_raw(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'frames'; journal=self.make(root)
            records=[frame(i) for i in (0,2,4)]
            for i, record in enumerate(records): journal.append(i*2,record)
            raw=Path(tmp)/'raw.h5'
            with h5py.File(raw,'w') as f:
                for key in records[0]: f[key]=np.stack([r[key] for r in records])
            self.assertTrue(self.verify(root,complete=True,raw_path=raw)['complete'])
            with h5py.File(raw,'r+') as f: f['v'][1,0,0]=1
            with self.assertRaises(AssertionError): self.verify(root,complete=True,raw_path=raw)

    def test_gpu_lease_excludes_other_process_and_releases(self):
        from collection.lease import gpu_lease
        with tempfile.TemporaryDirectory() as tmp:
            code='from collection.lease import gpu_lease\nwith gpu_lease("GPU-test", '+repr(tmp)+'):\n print("owned")'
            with gpu_lease('GPU-test', tmp):
                result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=20)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('holds this GPU lease',result.stderr)
            subprocess.run([sys.executable,'-c',code],check=True,capture_output=True,timeout=20)

    def make(self, root):
        return FrameJournal(root, initial_sha256='a'*64, manifest_sha256='b'*64, clock=Clock(.0001, 2, 2))

    def verify(self, root, **kw):
        return verify_journal(root, initial_sha256='a'*64, manifest_sha256='b'*64, **kw)

    def test_prefix_is_readable_but_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'frames';j=self.make(root);j.append(0,frame(0));j.append(2,frame(2))
            self.assertEqual(self.verify(root)['states'],2)
            with self.assertRaisesRegex(ValueError,'Incomplete'):self.verify(root,complete=True)
            j.append(4,frame(4));self.assertTrue(self.verify(root,complete=True)['complete'])

    def test_single_writer_and_no_skipped_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'frames';j=self.make(root)
            with self.assertRaises(FileExistsError):self.make(root)
            with self.assertRaisesRegex(ValueError,'clock'):j.append(2,frame(2))

    def test_corruption_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'frames';j=self.make(root);j.append(0,frame(0))
            with (root/'frame-00000.npz').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'changed'):self.verify(root)

    def test_publish_interruption_leaves_only_committed_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'frames';j=self.make(root);j.append(0,frame(0))
            with patch('collection.mpm.journal.write_once',side_effect=InterruptedError('injected')):
                with self.assertRaises(InterruptedError):j.append(2,frame(2))
            r=self.verify(root);self.assertEqual(r['states'],1);self.assertEqual(r['orphan_payloads'],1)
            with self.assertRaises(FileExistsError):j.append(2,frame(2))

    def test_receipt_chain_rejects_removed_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'frames';j=self.make(root);j.append(0,frame(0));j.append(2,frame(2))
            (root/'frame-00000.json').unlink()
            with self.assertRaisesRegex(ValueError,'gap'):self.verify(root)


class WitnessTests(unittest.TestCase):
    def test_shared_support_is_not_assumed_from_close_centroids(self):
        a=positive_stencil_cells(np.array([[.5,.5,.5]]))
        b=positive_stencil_cells(np.array([[1.,1.,1.]]))
        self.assertEqual(len(np.intersect1d(a,b)),0)
        self.assertGreater(len(np.intersect1d(a,positive_stencil_cells([[.51,.5,.5]]))),0)
        with self.assertRaises(ValueError):positive_stencil_cells([[-1,0,0]])

    def test_context_only_contact_does_not_qualify(self):
        x=np.tile(np.array([[.5,.5,.5],[1.,.5,.5]],np.float32),(49,1,1));x[:4,1,0]=.51
        m=dict(n_objects=2,scenario='collision',geometry_provenance=dict(recipe_version='interaction-v1',requested_pairs=[[0,1]]))
        self.assertFalse(analyze(m,dict(x=x,object_ids=np.array([0,1])))['semantic_eligible'])
        x[:4,1,0]=1.;x[4:,1,0]=.51
        self.assertTrue(analyze(m,dict(x=x,object_ids=np.array([0,1])))['semantic_eligible'])

    def test_idle_driver_memory_is_not_a_foreign_process(self):
        with patch('subprocess.check_output',side_effect=['GPU-test, 3, 0\n','']):gpu_status('GPU-test')
        with patch('subprocess.check_output',side_effect=['GPU-test, 3, 0\n','GPU-test, 77, 1\n']):
            with self.assertRaisesRegex(RuntimeError,'Foreign'):gpu_status('GPU-test')
        with patch('subprocess.check_output',side_effect=['GPU-test, 33, 0\n','']):
            with self.assertRaisesRegex(RuntimeError,'guard'):gpu_status('GPU-test')


@unittest.skipUnless(os.environ.get('PWM_MPM_REFERENCE_ROOT'), 'explicit frozen assets required')
class CorrectedSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.metadata,cls.objects=load_bank()

    def test_material_stiffness_crossing_does_not_change_geometry(self):
        specs=specifications('interaction-v1');self.assertEqual(len(specs),90)
        for index in range(30):
            scenes=[scene(specs[index+30*p],self.metadata,self.objects)[0] for p in range(3)]
            for other in scenes[1:]:
                for key in ('x','v','object_ids','particle_volume','mat_type','source_particle_index'):
                    np.testing.assert_array_equal(scenes[0][key],other[key])
            for oid in range(len(scenes[0]['E'])):
                self.assertEqual({float(a['E'][oid]) for a in scenes},{1e4,1e5,1e6})

    def test_two_object_stack_has_one_object_above_the_other(self):
        spec=next(s for s in specifications('interaction-v1') if s['count']==2 and s['scenario']=='stack')
        a,g=scene(spec,self.metadata,self.objects);low=a['x'][a['object_ids']==0];high=a['x'][a['object_ids']==1]
        self.assertGreater(high[:,2].min(),low[:,2].max()+.21)
        self.assertEqual(g['requested_pairs'],[[0,1]])

    def test_collision_facing_planes_and_velocity(self):
        for spec in specifications('interaction-v1'):
            if spec['scenario']!='collision':continue
            a,g=scene(spec,self.metadata,self.objects)
            for left,right in g['requested_pairs']:
                x=a['x'][a['object_ids']==left];y=a['x'][a['object_ids']==right]
                self.assertAlmostEqual(float(y[:,0].min()-x[:,0].max()),.26,places=6)
                self.assertGreater(a['v'][a['object_ids']==left,0].mean(),0)
                self.assertLess(a['v'][a['object_ids']==right,0].mean(),0)
