"""Synthetic schema/loader fixtures ONLY: no simulator or physics certification."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

HAS_CONSUMER = all(importlib.util.find_spec(name) for name in ('torch', 'webdataset', 'h5py'))


@unittest.skipUnless(HAS_CONSUMER, 'requires model consumer environment')
class FreshConsumerTests(unittest.TestCase):
    def test_all_seven_counts_both_stats_full_and_c4_f16(self):
        from collection.common import digest
        from collection.mpm.collapse import CollapsePolicy
        from collection.mpm.contracts import Clock, identity
        from collection.mpm.recording import atomic_raw
        from collection.mpm.walls import metadata
        from collection.mpm.export import export_wall_free
        from collection.mpm.review import check_model
        clock = Clock(5e-5, 834, 48)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paths = {}
            for k in range(1, 8):
                n = k * 1024
                case = dict(release_id='synthetic-consumer-contract-only', case_index=k,
                            seed_namespace='not-simulation', asset_ids=[f'fixture-{i}' for i in range(k)],
                            config_sha256='a' * 64, task='passive')
                eid = identity(case)[0]
                source = Path(__file__).resolve()
                manifest = dict(case=case, episode_id=eid, task='passive', n_objects=k,
                    clock=asdict(clock), floor_m=.1, grid_lim_m=100 / 64, n_grid=100,
                    volume_policy='synthetic-fixture-not-physical', source_sha256={str(source): digest(source)},
                    units=dict(position='m', velocity='m/s', time='s', volume='m^3', density='kg/m^3', E='Pa'),
                    population='synthetic-consumer-unit-test', split='diagnostic', scenario='random',
                    material_names=['elastic'] * k, collapse_policy=asdict(CollapsePolicy()),
                    wall_observation=metadata(digest(source)), accepted_for_training=False)
                initial = np.random.default_rng(k).uniform(-.04, .04, (n, 3)).astype(np.float32)
                initial += np.repeat(np.array([[.4 + .3 * (i % 3), .4 + .3 * (i // 3), .4]
                                                for i in range(k)], np.float32), 1024, axis=0)
                x = np.tile(initial, (49, 1, 1)); x[:, :, 0] += np.arange(49, dtype=np.float32)[:, None] * .001
                a = dict(x=x, v=np.full_like(x, .01),
                         F_trial=np.tile(np.eye(3, dtype=np.float32), (49, n, 1, 1)),
                         C=np.zeros((49, n, 3, 3), np.float32),
                         object_ids=np.repeat(np.arange(k, dtype=np.int32), 1024),
                         particle_volume=np.full(n, 1e-6, np.float32),
                         E=np.full(k, 1e5, np.float32), nu=np.full(k, .2, np.float32),
                         density=np.full(k, 1000, np.float32), mat_type=np.zeros(k, np.int32),
                         yield_stress=np.zeros(k, np.float32), state_time_s=clock.state_time_s,
                         selected=np.zeros(n, np.int32), command_velocity=np.zeros((n, 3), np.float32),
                         transition_active=np.zeros(48, np.int32), active_substeps=np.zeros(48, np.int32),
                         wall_simulation_steps=np.arange(49, dtype=np.int64) * 834,
                         wall_covered_substeps=np.repeat((np.arange(49, dtype=np.int64) * 834)[:, None], 5, axis=1),
                         wall_first_grid_step=np.full((49, 5), -1, np.int64),
                         wall_first_particle_step=np.full((49, 5), -1, np.int64))
                raw = root / f'fixture-{k}.h5'; atomic_raw(raw, manifest, a); paths[eid] = raw
            result = export_wall_free(list(paths.values()), root / 'filtered')
            self.assertEqual(result['retained'], 7)
            report = check_model(root, paths)
            self.assertEqual(report['profiles'], 2)
            self.assertEqual(report['episodes_per_profile'], 7)


if __name__ == '__main__':
    unittest.main()
