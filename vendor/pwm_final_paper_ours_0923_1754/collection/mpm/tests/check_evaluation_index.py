"""CPU integration check of real frozen evaluation indices against simulator raws.

Run explicitly; unlike unit fixtures this needs completed published episodes.
It never fits statistics, collects episodes, or changes model/training files.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def check(root, training_release, output, categories):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Set CUDA_VISIBLE_DEVICES to the empty string for this CPU check')
    root = Path(root).resolve(strict=True); training_release = Path(training_release).resolve(strict=True)
    if Path(output).exists(): raise FileExistsError(output)
    worker = runpy.run_path(str(root / 'driver.py'))
    root, plan = worker['verify_plan'](root)
    # Exact consumer snapshot, with the frozen collection contract helpers.
    source = Path(plan['bundles']['drag']['path']) / 'source'
    sys.path[:0] = [str(root / 'model'), str(source)]
    import h5py
    import numpy as np
    import torch
    from data.indexed_episodes import IndexedEpisodeDataset, validate_index_pair
    from data.unified_dataset import CANONICAL_TRANSLATION
    from collection.mpm.drag.review import normalization_review
    torch.set_num_threads(1)
    artifact = training_release / 'stats/normalization_collections_20260916_v2.json'
    artifact_sha = worker['sha'](artifact)
    sources = {name: worker['sha'](Path(module.__file__)) for name, module in list(sys.modules.items())
               if name.startswith('data.') and getattr(module, '__file__', None)}
    for name, module in list(sys.modules.items()):
        if name.startswith('data.') and getattr(module, '__file__', None):
            if not Path(module.__file__).resolve().is_relative_to(root / 'model'):
                raise ValueError('Consumer escaped its frozen snapshot: ' + name)
    results = []
    for category in categories:
        config = plan['categories'][category]; task = config['task']
        domain = 'mpm_drag' if task == 'drag' else 'mpm_fresh'
        worker['authorize_test'](plan, task)
        manifest = worker['publish'](root, plan, category)
        index = root / 'indices' / plan['split'] / category
        train = training_release / domain / 'train'
        validate_index_pair(train, index, domain)
        raws = {}
        for result in worker['selection'](root, plan, category, deep=True)['chosen']:
            row = config['candidates'][result['index']]
            folder = worker['case_root'](root, category, row) / result['attempt']
            raw = worker['read'](folder / 'SIMULATION.json')
            raws[result['episode_id']] = raw
        for profile in ('xy_p99_per_domain', 'xy_p99_global'):
            for future in (16, -1):
                dataset = IndexedEpisodeDataset(index, domain=domain, training=False, shuffle=False,
                    context_len=4, future_len=future, normalization={'artifact':str(artifact), 'profile':profile})
                if {r['episode_id'] for r in dataset.records} != set(manifest['episode_ids']):
                    raise ValueError('Index omitted or added a published episode')
                for record in dataset.records:
                    raw = raws[record['episode_id']]; path = Path(raw['raw_path'])
                    if worker['sha'](path) != raw['raw_sha256']: raise ValueError('Raw changed')
                    batch = dataset.window(record, 0)
                    with h5py.File(path) as stream:
                        fields = ('x','object_ids','E','nu','density','mat_type')
                        if task == 'drag':
                            fields += ('command_velocity','state_time_s','transition_active','selected')
                        arrays = {name:stream[name][()] for name in fields}
                    positions = torch.from_numpy(arrays['x'][:batch.num_frames]).unsqueeze(0)
                    positions = positions + torch.tensor(CANONICAL_TRANSLATION['mpm'])
                    torch.testing.assert_close(batch.positions, positions, atol=0, rtol=0)
                    torch.testing.assert_close(batch.future_deltas, positions[:,4:] - positions[:,3:-1], atol=0, rtol=0)
                    ids = arrays['object_ids']
                    torch.testing.assert_close(batch.object_ids, torch.from_numpy(ids).long()[None], atol=0, rtol=0)
                    for attr, field in (('E','E'),('nu','nu'),('rho','density'),('mat_type','mat_type')):
                        expected = torch.from_numpy(arrays[field][ids])[None]
                        if attr == 'mat_type': expected = expected.long()
                        torch.testing.assert_close(getattr(batch,attr), expected, atol=0, rtol=0)
                    if task == 'drag':
                        delta = (arrays['command_velocity'].astype(np.float64)
                                 * np.diff(arrays['state_time_s'])[:,None,None]).astype(np.float32)
                        gate = arrays['transition_active'][:,None].astype(bool) & arrays['selected'][None].astype(bool)
                        end = batch.num_frames - 1
                        torch.testing.assert_close(batch.actions, torch.from_numpy(delta[3:end])[None], atol=0, rtol=0)
                        torch.testing.assert_close(batch.action_active, torch.from_numpy(gate[3:end])[None], atol=0, rtol=0)
                        torch.testing.assert_close(batch.context_action_active, torch.from_numpy(gate[:3])[None], atol=0, rtol=0)
                        torch.testing.assert_close(batch.action_mask, torch.from_numpy(arrays['selected'].astype(bool))[None], atol=0, rtol=0)
                        normalization_review(batch.delta_stats, batch.actions)
                    elif batch.actions.any() or batch.action_mask.any():
                        raise ValueError('Passive episode gained actuation')
                    if not torch.isfinite(batch.delta_stats.normalize(batch.future_deltas)).all():
                        raise ValueError('Nonfinite normalized target')
                    if worker['sha'](path) != raw['raw_sha256']: raise ValueError('Raw changed during check')
                results.append(dict(category=category, domain=domain, profile=profile, future_len=future,
                    episodes=len(dataset.records), raw_fields_exact=True, index_sha256=worker['sha'](index/'INDEX.json'),
                    training_index_sha256=worker['sha'](train/'INDEX.json')))
    worker['verify_plan'](root)
    if worker['sha'](artifact) != artifact_sha: raise ValueError('Training stats changed')
    report = dict(suite_plan_sha256=worker['sha'](root/'PLAN.json'), script_sha256=worker['sha'](__file__),
        normalization_artifact=str(artifact), normalization_sha256=artifact_sha, consumer_modules=sources,
        results=results, passed=True, physical_accuracy_claim=False, gpu_simulations_started=False)
    worker['write'](Path(output), report)
    return dict(passed=True, checked_category_profile_horizon_combinations=len(results), output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','training-release','output'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--categories', nargs='+', required=True)
    print(json.dumps(check(**vars(parser.parse_args()))))
