import tempfile
import json
from pathlib import Path
import unittest
import importlib.util
from unittest.mock import patch

if importlib.util.find_spec('trimesh') is None:
    raise unittest.SkipTest('requires MPM asset-generation environment (trimesh)')

import numpy as np
import trimesh
from collection.mpm.assets import geometry,fps,seed_for,fetch,digest,process


class AssetTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_bounded(self):
        mesh=trimesh.creation.icosphere(subdivisions=2)
        a,qa=geometry(mesh.vertices,mesh.faces,'sphere')
        b,qb=geometry(mesh.vertices,mesh.faces,'sphere')
        self.assertEqual(qa,qb);np.testing.assert_array_equal(a['points_unit'],b['points_unit'])
        self.assertTrue(qa['eligible']);self.assertEqual(a['points_unit'].shape,(1024,3))
        self.assertLessEqual(np.abs(a['points_unit']).max(),.5)

    def test_geometry_hash_ignores_translation_uniform_scale_vertex_order(self):
        mesh=trimesh.creation.icosphere(subdivisions=1)
        _,a=geometry(mesh.vertices,mesh.faces,'a')
        p=np.arange(len(mesh.vertices))[::-1];inverse=np.argsort(p)
        _,b=geometry((mesh.vertices*3+7)[p],inverse[mesh.faces],'b')
        self.assertEqual(a['geometry_sha256'],b['geometry_sha256'])
        self.assertNotEqual(a['point_sha256'],b['point_sha256'])

    def test_disconnected_shapes_are_not_repaired(self):
        a=trimesh.creation.icosphere(subdivisions=1);b=a.copy();b.apply_translation([4,0,0])
        mesh=trimesh.util.concatenate([a,b]);_,quality=geometry(mesh.vertices,mesh.faces,'two')
        self.assertFalse(quality['eligible']);self.assertEqual(quality['significant_components'],2)

    def test_planar_shape_rejected(self):
        vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]])
        _,quality=geometry(vertices,[[0,1,2],[0,2,3]],'plane')
        self.assertIn('near_planar_or_linear_geometry',quality['reasons'])

    def test_nonfinite_and_bad_faces_rejected(self):
        with self.assertRaises(ValueError):geometry([[0,0,0],[1,0,0],[0,np.nan,0]],[[0,1,2]],'bad')
        with self.assertRaises(ValueError):geometry([[0,0,0],[1,0,0],[0,1,0]],[[0,1,3]],'bad')

    def test_fps_fixed_ties(self):
        points=np.array([[0.,0,0],[1.,0,0],[-1.,0,0],[0.,1,0]])
        np.testing.assert_array_equal(fps(points,4),[0,1,2,3])
        self.assertEqual(seed_for('a'),seed_for('a'));self.assertNotEqual(seed_for('a'),seed_for('b'))

    def test_scene_graph_transforms_not_flattened_away(self):
        scene=trimesh.Scene();scene.add_geometry(trimesh.creation.box(),node_name='a')
        transform=np.eye(4);transform[0,3]=4
        scene.add_geometry(trimesh.creation.box(),node_name='b',transform=transform)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'scene.glb';scene.export(path)
            loaded=trimesh.load(path,force='mesh',process=False,skip_materials=True)
            self.assertAlmostEqual(float(np.ptp(loaded.vertices[:,0])),5.)
            _,quality=geometry(loaded.vertices,loaded.faces,'scene')
            self.assertFalse(quality['eligible'])

    def test_protected_fingerprint_does_not_require_training_complexity_or_sampling(self):
        from collection.mpm.assets import POLICY
        mesh=trimesh.creation.icosphere(subdivisions=1)
        with patch.dict(POLICY,maximum_faces=5),patch('collection.mpm.assets.fps') as sampler:
            with self.assertRaisesRegex(ValueError,'excessive_faces'):
                geometry(mesh.vertices,mesh.faces,'protected')
            arrays,quality=geometry(mesh.vertices,mesh.faces,'protected',fingerprint_only=True)
            self.assertIsNone(arrays);self.assertIn('geometry_sha256',quality)
            sampler.assert_not_called()

    def test_existing_mesh_is_hash_checked_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mesh=root/'source.glb';mesh.write_bytes(b'original')
            row=dict(local_path=str(mesh),sha256=digest(mesh))
            self.assertEqual(fetch(row,root),mesh)
            row['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'historical_mesh_hash_mismatch'):fetch(row,root)
            self.assertEqual(mesh.read_bytes(),b'original')

    def test_completed_shape_resume_rejects_payload_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'objects/uid';folder.mkdir(parents=True)
            row=dict(uid='uid');payload=folder/'points.npz';payload.write_bytes(b'original')
            value=dict(candidate=row,status='qualified',payload_sha256=digest(payload))
            (folder/'RESULT.json').write_text(json.dumps(value))
            self.assertEqual(process((row,str(root))),value)
            payload.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Resume payload changed'):process((row,str(root)))

    def test_source_archive_copy_is_exact_and_no_clobber(self):
        from collection.mpm.asset_archive import copy_verified
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.glb';target=root/'archive/source.glb'
            source.write_bytes(b'original');expected=digest(source)
            copy_verified(source,target,expected)
            self.assertEqual(target.read_bytes(),source.read_bytes())
            copy_verified(source,target,expected)
            self.assertEqual(source.read_bytes(),b'original')
            target.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Existing archive differs'):copy_verified(source,target,expected)
            self.assertEqual(target.read_bytes(),b'changed')

    def test_source_archive_hash_failure_does_not_publish(self):
        from collection.mpm.asset_archive import copy_verified
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source.glb';target=root/'archive/source.glb'
            source.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'copy mismatch'):copy_verified(source,target,'0'*64)
            self.assertFalse(target.exists());self.assertEqual(source.read_bytes(),b'wrong')


if __name__=='__main__':unittest.main()
