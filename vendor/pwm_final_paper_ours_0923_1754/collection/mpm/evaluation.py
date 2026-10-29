"""Evaluation readiness helpers, not a production gate or a test-set launcher.

Inventory reads JSON metadata only (never pickle payloads). Smoke preparation
executes a verified, frozen training source bundle, not the mutable checkout.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import tarfile

from collection.common import canonical, check_hashes, digest, write_once


def training_inventory(roots, allowed, split_sha256, *, task):
    """Measure actual exported membership; this is not a full tar integrity audit."""
    if task not in ('passive', 'drag') or not roots:
        raise ValueError('Explicit task and nonempty training roots required')
    roots = [Path(p).resolve(strict=True) for p in roots]
    if len(set(roots)) != len(roots) or any(
            a != b and a.is_relative_to(b) for a in roots for b in roots):
        raise ValueError('Duplicate or nested training roots')
    episodes, assets, shards, counts = set(), set(), {}, Counter()
    for root in roots:
        paths = sorted(root.rglob('*.tar'))
        if not paths:
            raise ValueError('No exported training shards: ' + str(root))
        for path in paths:
            resolved = str(path.resolve(strict=True))
            if resolved in shards:
                raise ValueError('Duplicate shard through aliases')
            stat = path.stat()
            metadata = []
            with tarfile.open(path, 'r:') as stream:
                for member in stream:
                    if not member.name.endswith('.provenance.json'):
                        continue
                    if not member.isfile() or not 0 < member.size <= 16 << 20:
                        raise ValueError('Invalid provenance member')
                    payload = stream.extractfile(member).read()
                    rows = json.loads(payload)
                    if not isinstance(rows, list) or not rows:
                        raise ValueError('Nonempty batched provenance required')
                    metadata.append(dict(member=member.name, sha256=hashlib.sha256(payload).hexdigest()))
                    for row in rows:
                        if (row.get('split') != 'train' or row.get('evaluation') != 'train'
                                or row.get('object_cohort') != 'train'
                                or row.get('split_manifest_sha256') != split_sha256):
                            raise ValueError('Training membership/split mismatch')
                        is_drag = row.get('task') == 'drag' or bool(row.get('action_schema'))
                        if is_drag != (task == 'drag'):
                            raise ValueError('Mixed passive/drag inventory')
                        eid, uids = row.get('episode_id'), row.get('asset_ids')
                        if not isinstance(eid, str) or not eid or eid in episodes:
                            raise ValueError('Missing or duplicate episode identity')
                        if (not isinstance(uids, list) or not 1 <= len(uids) <= 4
                                or len(set(uids)) != len(uids) or not set(uids) <= allowed):
                            raise ValueError('Exported asset outside training bank/count contract')
                        episodes.add(eid)
                        assets.update(uids)
                        counts[len(uids)] += 1
            if not metadata:
                raise ValueError('No provenance in shard: ' + str(path))
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Training shard changed during inventory')
            shards[resolved] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, provenance=metadata)
        if paths != sorted(root.rglob('*.tar')):
            raise ValueError('Training roots grew during inventory; use a frozen export snapshot')
    return dict(task=task, roots=[str(p) for p in roots], episodes=len(episodes),
                episode_ids=sorted(episodes), seen_asset_uids=sorted(assets),
                object_counts=dict(sorted(counts.items())), shards=shards,
                metadata_only=True, full_shard_payloads_verified=False,
                note='Exported membership, not proof these episodes were consumed by a trained model')


def inventory(assets, passive_roots, drag_roots):
    from .recipes import load_bank
    assets = Path(assets).resolve(strict=True)
    split = assets / 'SPLITS.json'
    banks, members = {}, {}
    for role in ('train', 'validation', 'test'):
        metadata, objects = load_bank(assets / role, assets / 'pool', role=role, split_manifest=split)
        members[role] = {row['asset_uid'] for row in metadata['objects']}
        banks[role] = dict(shapes=len(objects), manifest_sha256=digest(assets / role / 'MANIFEST.json'))
        del objects
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if members[a] & members[b]:
            raise ValueError('Overlapping shape banks')
    training = {}
    for task, roots in (('passive', passive_roots), ('drag', drag_roots)):
        if roots:
            training[task] = training_inventory(roots, members['train'], digest(split), task=task)
    if not training:
        raise ValueError('At least one explicit exported training source required')
    return dict(schema='mpm-evaluation-inventory-v1', assets=str(assets),
                split_sha256=digest(split), banks=banks, training=training,
                production_approved=False, target_retained_per_category=100)


# Deliberately execute inside the frozen bundle. The current checkout contains
# concurrent numerical work which must not silently enter an evaluation run.
PASSIVE_SMOKE = '''
import json, sys
from pathlib import Path
from collection.mpm.collect import prepare, inspect
from collection.mpm.protocol import specifications
args=json.loads(sys.argv[1]); assets=Path(args['assets']); out=Path(args['output'])
for evaluation, counts in [('unseen', (1,4)), ('count_unseen', (5,6,7))]:
    def factory(metadata):
        specs=specifications(metadata,split='validation',evaluation=evaluation,cycle_start=917)
        selected=[]
        for n in counts:
            row=next(s for s in specs if s['count']==n and s['stiffness_phase']==2 and s['family_index']==args['passive_family_index'])
            selected.append(dict(row,index=len(selected)))
        return selected
    root=out/evaluation
    prepare(root,'fresh-shapes-v1',bank=assets/'validation',pool=assets/'pool',
            bank_role='validation',split_manifest=assets/'SPLITS.json',
            dataset_split='validation',evaluation=evaluation,specification_factory=factory)
    plan=inspect(root)
    print(json.dumps(dict(root=str(root),cases=[c['label'] for c in plan['cases']])),flush=True)
'''

DRAG_SMOKE = '''
import json, sys
from pathlib import Path
from collection.mpm.drag.prepare import prepare
args=json.loads(sys.argv[1]); out=Path(args['output'])
for index in (0,215):
    result=prepare(out/f'index-{index:03d}',args['assets'],name='evaluation-validation-smoke-v1',
        count=1,start=index,split='validation',evaluation='unseen',n_grid=256,
        object_counts=[1,2,3,4],event_counts=[1,2,3],material_layout='homogeneous',speed_range=[.05,.25])
    print(json.dumps(result),flush=True)
'''


REVIEW_SMOKE = '''
import json, sys
from pathlib import Path
from collection.common import digest, write_once
from collection.mpm.export import validate_candidate, eligible, export_candidates
from collection.mpm.wall_materials import SAND_ONLY, STRICT
args=json.loads(sys.argv[1]); root=Path(args['root']); out=Path(args['output'])
task=args['task']; policy=SAND_ONLY if task=='passive' else STRICT
raws=[]; rows=[]; controls=[]
if task=='passive':
    for name in ('unseen','count_unseen'):
        prepared=root/name
        plan=json.loads((prepared/'PLAN.json').read_text())
        for case in plan['cases']:
            folder=prepared/'cases'/case['label']/'attempts/attempt-00000'
            if case['kind']=='matched_control' and name=='count_unseen': continue
            result=json.loads((folder/'RESULT.json').read_text())
            if result['status'] not in ('complete','placement_rejected'): raise ValueError('Failed smoke attempt')
            if case['kind']=='matched_control':
                q=result['quality']; controls.append(dict(label=case['label'],quality=q))
                if not q['integrity_pass']: raise ValueError('Control integrity failed')
                continue
            row=dict(label=case['label'],status=result['status'],count=case['template']['n_objects'],evaluation=name)
            if result['status']=='complete':
                path=folder/'raw.h5'
                if digest(path)!=result['raw_sha256']: raise ValueError('Changed smoke raw')
                raws.append(path); row['quality']=validate_candidate(path,wall_policy=policy)
            rows.append(row)
    if {c['label']:c['quality']['eligible_for_wall_free_export'] for c in controls}!={'control-00':True,'control-02':False}:
        raise ValueError('Expected one positive and one negative wall control')
else:
    for index in (0,215):
        prepared=root/f'index-{index:03d}'
        plan=json.loads((prepared/'PLAN.json').read_text())
        label=plan['cases'][0]['label']; folder=prepared/args['drag_execution']/label
        result=json.loads((folder/'RESULT.json').read_text())
        row=dict(label=label,status=result['status'],evaluation='unseen')
        if result['status']=='complete':
            path=folder/'raw.h5'
            if digest(path)!=result['raw_sha256']: raise ValueError('Changed smoke raw')
            raws.append(path); row['quality']=validate_candidate(path,wall_policy=policy)
        elif result['status'] not in ('settling_rejected',): raise ValueError('Unexpected drag failure')
        rows.append(row)
if not raws: raise ValueError('No completed raws for export/consumer validation')
consumer={}; retained=0
for category in sorted({r['evaluation'] for r in rows}):
    proposed=[r['quality'] for r in rows if r['evaluation']==category and 'quality' in r]
    kept={r['episode_id']:Path(r['raw_path']) for r in proposed if eligible(r)}
    target=out/category
    if proposed:
        selected=export_candidates([Path(r['raw_path']) for r in proposed],target/'filtered',wall_policy=policy)
        if selected['retained']!=len(kept): raise ValueError('Export retention mismatch')
    if not kept:
        consumer[category]=dict(episodes=0,checked=False,reason='No retained smoke episode in this category')
        continue
    if task=='passive':
        from collection.mpm.review import check_model
        consumer[category]=check_model(target,kept)
    else:
        from collection.consumer import import_consumer
        import_consumer()
        from collection.mpm.drag.review import loader_review
        consumer[category]=loader_review(target/'filtered/diagnostic-export',list(kept.values()))
        if len(consumer[category])!=2 or any(r['episodes']!=len(kept) or not r['exact_raw_window_and_command_match'] for r in consumer[category]):
            raise ValueError('Incomplete drag loader review')
    retained+=len(kept)
if not retained: raise ValueError('No retained episodes for real loader validation')
report=dict(task=task,validation_only=True,controls=controls,outcomes=rows,
            retained=retained,consumer=consumer,production_approved=False,
            limitation='Small pipeline smoke; not full category qualification or convergence evidence')
write_once(out/'REVIEW.json',report)
print(json.dumps(dict(task=task,retained=retained,review=str(out/'REVIEW.json'))),flush=True)
'''


def _execute_frozen(bundle, manifest, output, python, code, request, env, success_name):
    """Publish success only after the child and the post-run source check pass."""
    try:
        write_once(output / 'CHILD.json', dict(code=code, sha256=hashlib.sha256(code.encode()).hexdigest(),
                                              python=str(python), cwd=str(bundle / 'source')))
        result = subprocess.run([str(Path(python).resolve(strict=True)), '-c', code, canonical(request)],
                                cwd=bundle / 'source', env=env, capture_output=True, text=True, timeout=900)
        receipt = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr,
                       production_approved=False)
        if result.returncode:
            write_once(output / 'FAILURE.json', receipt)
            raise RuntimeError(result.stderr[-4000:])
        check_hashes(bundle, manifest['files'])
        if 'model_source_sha256' in request:
            check_hashes(Path(request['model_root']), request['model_source_sha256'])
        write_once(output / success_name, receipt)
        return receipt
    except Exception as error:
        if not (output / 'FAILURE.json').exists():
            write_once(output / 'FAILURE.json', dict(error=type(error).__name__ + ': ' + str(error),
                                                    production_approved=False))
        raise


def review_smoke(bundle, root, output, python, model_root, *, task, drag_execution='serial'):
    if task not in ('passive', 'drag'):
        raise ValueError('Unknown task')
    if not drag_execution or not drag_execution.replace('-', '').isalnum():
        raise ValueError('Explicit simple drag execution directory required')
    bundle = Path(bundle).resolve(strict=True)
    manifest = json.loads((bundle / 'BUNDLE.json').read_text())
    check_hashes(bundle, manifest['files'])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = dict(task=task, root=str(Path(root).resolve(strict=True)), output=str(output),
                   bundle=str(bundle), bundle_sha256=digest(bundle / 'BUNDLE.json'),
                   helper_sha256=digest(__file__), model_root=str(Path(model_root).resolve(strict=True)),
                   drag_execution=drag_execution)
    model = Path(request['model_root'])
    model_files = sorted((model / 'data').glob('*.py')) + [model / 'stats/normalization.json']
    request['model_source_sha256'] = {str(p.relative_to(model)): digest(p) for p in model_files}
    write_once(output / 'REQUEST.json', request)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PWM_MODEL_ROOT=request['model_root'], PYTHONPATH=str(bundle / 'source'), PYTHONNOUSERSITE='1')
    return _execute_frozen(bundle, manifest, output, python, REVIEW_SMOKE, request, env, 'COMPLETE.json')


def prepare_smoke(bundle, assets, output, python, reference, *, task, passive_family_index=4):
    if task not in ('passive', 'drag'):
        raise ValueError('Unknown task')
    if type(passive_family_index) is not int or passive_family_index not in range(5):
        raise ValueError('Passive family index must be 0..4')
    bundle, assets = Path(bundle).resolve(strict=True), Path(assets).resolve(strict=True)
    manifest = json.loads((bundle / 'BUNDLE.json').read_text())
    check_hashes(bundle, manifest['files'])
    expected = manifest.get('bank_complete_sha256', manifest.get('bank_sha256'))
    if not expected or digest(assets / 'COMPLETE.json') != expected:
        raise ValueError('Frozen bundle asset release mismatch')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = dict(task=task, bundle=str(bundle), bundle_sha256=digest(bundle / 'BUNDLE.json'),
                   assets=str(assets), output=str(output), validation_only=True,
                   test_data_generated=False, production_approved=False,
                   helper_sha256=digest(__file__), passive_family_index=passive_family_index)
    write_once(output / 'REQUEST.json', request)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               PWM_MPM_REFERENCE_ROOT=str(Path(reference).resolve(strict=True)),
               PYTHONPATH=str(bundle / 'source'), PYTHONNOUSERSITE='1')
    code = PASSIVE_SMOKE if task == 'passive' else DRAG_SMOKE
    return _execute_frozen(bundle, manifest, output, python, code, request, env, 'PREPARED.json')


def render_review(review, output):
    """Render saved particles with the actual selection verdict, not a stricter one."""
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    review = Path(review).resolve(strict=True)
    review_sha256 = digest(review)
    report = json.loads(review.read_text())
    if report['task'] != 'passive':
        raise ValueError('Use the frozen drag reviewer for selected-patch visualizations')
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    colors = np.array(['#277fba', '#cb4a93', '#d39612']); media = []
    for category in sorted({row['evaluation'] for row in report['outcomes']}):
        rows = [r for r in report['outcomes'] if r['evaluation'] == category and 'quality' in r]
        if not rows:
            continue
        fig, axes = plt.subplots(2, len(rows), figsize=(6 * len(rows), 7.5), squeeze=False, dpi=100)
        values = []; artists = []
        for column, row in enumerate(rows):
            quality = row['quality']; raw = Path(quality['raw_path'])
            if digest(raw) != quality['raw_sha256']:
                raise ValueError('Raw changed since review')
            with h5py.File(raw) as stream:
                values.append({key: stream[key][()] for key in ('x', 'object_ids', 'mat_type', 'state_time_s')})
            a = values[-1]
            accepted = quality.get('eligible_for_export', quality['eligible_for_wall_free_export'])
            label = 'RETAINED under declared filter' if accepted else 'REJECTED by declared filter'
            for projection, (u, v) in enumerate(((0, 1), (0, 2))):
                ax = axes[projection, column]
                ax.set(xlim=(0, 1.5), ylim=(0, 1.5), aspect='equal', xlabel='x (m)',
                       ylabel='y (m)' if projection == 0 else 'z (m)')
                ax.set_title(row['label'] + '\n' + label, fontsize=10,
                             color='#176b36' if accepted else '#aa3333')
                if projection == 1: ax.axhline(.1, color='black', lw=.8)
                dots = ax.scatter(a['x'][0, :, u], a['x'][0, :, v], s=2,
                                  c=colors[a['mat_type'][a['object_ids']]])
                artists.append((column, u, v, dots))
        heading = fig.suptitle(''); fig.tight_layout(rect=(0, .09, 1, .94))
        fig.text(.5, .025, 'Actual saved particles; fixed metric axes. XY above, XZ below.\n'
                 'Blue elastic / pink plasticine / gold sand. Passive policy permits sand-only wall contact.\n'
                 'Validation smoke, not a released test set. Every third saved state; approximately real-time playback.',
                 ha='center', fontsize=10)
        indices = list(range(0, max(len(a['x']) for a in values), 3)); frames = []
        for index in indices:
            for column, u, v, dots in artists:
                a = values[column]; frame = min(index, len(a['x']) - 1)
                dots.set_offsets(a['x'][frame][:, [u, v]])
            heading.set_text(f'{category}: saved time {values[0]["state_time_s"][min(index, len(values[0]["x"])-1)]:.4f} s')
            fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()))
        path = output / (category + '.gif')
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=125, loop=0)
        frames[-1].save(output / (category + '-last.png')); plt.close(fig)
        with Image.open(path) as gif:
            if gif.n_frames != len(indices):
                raise ValueError('Incomplete GIF')
        for row in rows:
            if digest(Path(row['quality']['raw_path'])) != row['quality']['raw_sha256']:
                raise ValueError('Raw changed during rendering')
        media.append(dict(path=str(path), sha256=digest(path), source_frame_indices=indices,
                          raw_sha256=[row['quality']['raw_sha256'] for row in rows]))
    if digest(review) != review_sha256:
        raise ValueError('Review changed during rendering')
    write_once(output / 'MEDIA.json', dict(review_sha256=review_sha256, helper_sha256=digest(__file__), media=media))
    return dict(output=str(output), galleries=len(media))


def index_records(publication, manifest, training):
    """Index exactly one approved category; never repartition evaluation episodes."""
    import math
    from collection.mpm.evaluation_suite import CATEGORIES
    publication = Path(publication).resolve(strict=True)
    category = manifest.get('category')
    if category not in CATEGORIES or manifest.get('schema') != 'mpm-evaluation-suite-v1':
        raise ValueError('Expected a frozen evaluation category publication')
    task, evaluation, counts = CATEGORIES[category]
    domain = 'mpm_drag' if task == 'drag' else 'mpm_fresh'
    if (manifest.get('task') != task or manifest.get('split') not in ('validation', 'test')
            or manifest.get('accepted_for_training') is not False):
        raise ValueError('Evaluation publication role mismatch')
    if (training.get('schema') != 'pwm-indexed-episodes-v1' or training.get('domain') != domain
            or training.get('split') != 'train' or not training.get('records')):
        raise ValueError('Matching explicit training index required')
    train_ids = {r['episode_id'] for r in training['records']}
    train_groups = {r['group'] for r in training['records']}
    train_assets = {u for r in training['records'] for u in r['asset_ids']}
    expected = manifest['episode_ids']
    if len(expected) != len(set(expected)) or len(expected) != manifest['retained']:
        raise ValueError('Duplicate/incomplete published episode membership')
    if {p.name for p in publication.glob('*.tar')} != set(manifest['shards']):
        raise ValueError('Published shard membership changed')
    rows = []; seen = set(); actual_counts = Counter()
    for name, checksum in sorted(manifest['shards'].items()):
        if Path(name).name != name or not name.endswith('.tar'):
            raise ValueError('Unsafe shard path')
        path = publication / name
        if path.is_symlink() or digest(path) != checksum:
            raise ValueError('Changed published shard')
        stat = path.stat(); found = False
        with tarfile.open(path, 'r:') as archive:
            for member in archive:
                if not member.name.endswith('.provenance.json'): continue
                if not member.isfile() or not 0 < member.size <= 16 << 20:
                    raise ValueError('Invalid provenance member')
                provenance = json.load(archive.extractfile(member))
                if not isinstance(provenance, list) or not provenance:
                    raise ValueError('Missing published provenance')
                key = member.name[:-len('.provenance.json')]
                for index, p in enumerate(provenance):
                    eid = p['episode_id']; assets = p['asset_ids']; times = p['state_time_s']
                    if eid in seen or eid not in expected or eid in train_ids or eid in train_groups:
                        raise ValueError('Duplicate, missing or training-overlapping episode')
                    if (p.get('task') != task or p.get('split') != manifest['split']
                            or p.get('evaluation') != evaluation
                            or p.get('object_cohort') != (manifest['split'] if evaluation.endswith('unseen') else 'train')
                            or len(assets) not in counts or len(assets) != len(set(assets))):
                        raise ValueError('Provenance category/count mismatch')
                    if evaluation.endswith('unseen'):
                        if set(assets) & train_assets: raise ValueError('Held-out shape appears in actual training index')
                    elif not set(assets) <= train_assets:
                        raise ValueError('Seen shape absent from actual training index')
                    if (len(times) < 20 or not all(math.isfinite(t) for t in times)
                            or not all(math.isclose(b-a, .0417, rel_tol=0, abs_tol=1e-12) for a,b in zip(times,times[1:]))):
                        raise ValueError('Incorrect saved clock')
                    if (task == 'drag' and p.get('action_schema') != 'mpm-drag-command-sequence-v4'
                            or task == 'passive' and p.get('action_schema')):
                        raise ValueError('Wrong action schema')
                    rows.append(dict(episode_id=eid, group=eid, file=str(path), size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns, device=stat.st_dev, inode=stat.st_ino, key=key, row=index,
                        frames=len(times), sha256=checksum, raw_sha256=p['raw_sha256'],
                        asset_ids=assets, dt=.0417, action_schema=p.get('action_schema'), evaluation_category=category))
                    seen.add(eid); actual_counts[str(len(assets))] += 1; found = True
        if not found: raise ValueError('Shard has no episode provenance')
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Shard changed while indexing')
    if seen != set(expected) or dict(actual_counts) != manifest['quotas']:
        raise ValueError('Indexed episodes do not exactly fill published quotas')
    return domain, sorted(rows, key=lambda row: row['episode_id'])


def index_export(suite_root, category, training_index, output):
    """Owner-side handoff to the current indexed domains, not a collection gate."""
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    root = Path(suite_root).resolve(strict=True)
    seal = json.loads((root / 'SEAL.json').read_text())
    if digest(root / 'driver.py') != seal['driver_sha256']:
        raise ValueError('Frozen evaluation driver changed')
    worker = runpy.run_path(str(root / 'driver.py'))
    root, plan = worker['verify_plan'](root)
    if category not in plan['categories']: raise ValueError('Unknown frozen category')
    worker['authorize_test'](plan, plan['categories'][category]['task'])
    manifest = worker['publish'](root, plan, category)
    publication = root / 'datasets' / plan['split'] / category
    training_path = Path(training_index).resolve(strict=True) / 'INDEX.json'
    training_sha = digest(training_path); training = json.loads(training_path.read_text())
    domain, rows = index_records(publication, manifest, training)
    if digest(training_path) != training_sha: raise ValueError('Training index changed')
    result = dict(schema='pwm-indexed-episodes-v1', domain=domain,
        split='val' if plan['split'] == 'validation' else 'test', records=rows,
        source_content_hashes_verified=True, source_files_verified=len(manifest['shards']),
        split_policy=dict(selection='Exact published category membership; no random split or reassignment',
                          engineering_validation_only=plan['split'] == 'validation'),
        provenance=dict(category=category, publication=str(publication),
            publication_manifest_sha256=digest(publication / 'MANIFEST.json'),
            suite_plan_sha256=digest(root / 'PLAN.json'), helper_sha256=digest(__file__),
            training_index=str(training_path), training_index_sha256=training_sha,
            note='No raw/tar mutation, statistics fitting, or merge into a training release'))
    output.mkdir(parents=True, exist_ok=False); os.chmod(output, 0o2750)
    worker['write'](output / 'INDEX.json', result)
    return dict(output=str(output), domain=domain, split=result['split'], episodes=len(rows),
                index_sha256=digest(output / 'INDEX.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory')
    inv.add_argument('--assets', required=True)
    inv.add_argument('--passive-roots', nargs='+', default=[])
    inv.add_argument('--drag-roots', nargs='+', default=[])
    inv.add_argument('--output', required=True)
    smoke = sub.add_parser('prepare-smoke')
    for name in ('bundle', 'assets', 'output', 'python', 'reference'):
        smoke.add_argument('--' + name, required=True)
    smoke.add_argument('--task', required=True, choices=('passive', 'drag'))
    smoke.add_argument('--passive-family-index', type=int, default=4)
    review = sub.add_parser('review-smoke')
    for name in ('bundle', 'root', 'output', 'python', 'model-root'):
        review.add_argument('--' + name, required=True)
    review.add_argument('--task', required=True, choices=('passive', 'drag'))
    review.add_argument('--drag-execution', default='serial')
    render = sub.add_parser('render')
    render.add_argument('--review', required=True)
    render.add_argument('--output', required=True)
    indexed = sub.add_parser('index-export')
    for name in ('suite-root', 'category', 'training-index', 'output'):
        indexed.add_argument('--' + name, required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'inventory':
        output = args.pop('output')
        if Path(output).exists():
            raise FileExistsError(output)
        result = inventory(**args)
        write_once(output, result)
        print(canonical(dict(banks=result['banks'], training={k:dict(episodes=v['episodes'],
            seen_shapes=len(v['seen_asset_uids']), shards=len(v['shards'])) for k,v in result['training'].items()})))
    elif command == 'prepare-smoke':
        print(canonical(prepare_smoke(**args)))
    elif command == 'review-smoke':
        print(canonical(review_smoke(**args)))
    elif command == 'index-export':
        print(canonical(index_export(**args)))
    else:
        print(canonical(render_review(**args)))


if __name__ == '__main__':
    main()
