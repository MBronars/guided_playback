"""Frozen, serial-reference MPM evaluation collection.

This orchestrates existing simulators; it does not extend a fast training gate.
Test execution requires a completed validation run of this exact driver and
numerical/consumer sources. No import of the mutable checkout in child workers.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import shutil
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace


VERSION = 'mpm-evaluation-suite-v1'
DRAG_VERSION = 'mpm-drag-command-sequence-v4'
CATEGORIES = {
    'passive_seen': ('passive', 'seen', (1, 2, 3, 4)),
    'passive_unseen': ('passive', 'unseen', (1, 2, 3, 4)),
    'passive_count_seen': ('passive', 'count_seen', (5, 6, 7)),
    'passive_count_unseen': ('passive', 'count_unseen', (5, 6, 7)),
    'drag_seen': ('drag', 'seen', (1, 2, 3, 4)),
    'drag_unseen': ('drag', 'unseen', (1, 2, 3, 4)),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Atomic exclusive JSON; identical receipts may be recovered after a crash."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Refusing to replace receipt: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            # mkstemp defaults to owner-only even inside a lab-shared release.
            # Publish immutable receipts readable (not writable) by that group.
            os.fchmod(stream.fileno(), 0o640)
            stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(name)  # Only this call's private temporary file.


def check_files(root, files):
    root = Path(root)
    for name, expected in files.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or (root / p).is_symlink():
            raise ValueError('Unsafe manifest path')
        if sha(root / p) != expected:
            raise ValueError('Changed bound artifact: ' + str(root / p))


def quotas(counts, target):
    if type(target) is not int or not len(counts) <= target <= 100:
        raise ValueError('Target must cover every object count and be at most 100')
    return {str(n): target // len(counts) + int(i < target % len(counts)) for i, n in enumerate(counts)}


def candidate_plan(name, split, category, allowed_uids, *, target=100, reserves=8):
    if split not in ('validation', 'test') or category not in CATEGORIES:
        raise ValueError('Explicit evaluation split/category required')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9-]{0,63}', name):
        raise ValueError('Simple unique suite name required')
    if type(reserves) is not int or not 1 <= reserves <= 20:
        raise ValueError('Reserve multiplier must be 1..20')
    task, evaluation, counts = CATEGORIES[category]
    allowed = sorted(allowed_uids)
    if len(allowed) != len(set(allowed)) or len(allowed) < max(counts):
        raise ValueError('Insufficient/duplicate eligible shapes')
    q = quotas(counts, target)
    namespace = f'{VERSION}-{name}-{split}-{category}'
    factors = {}
    for n in counts:
        if task == 'passive':
            actions = ('drop', 'translate') if n == 1 else ('collision', 'drop_on', 'stack', 'scatter', 'random')
            factors[n] = list(itertools.product(range(3), range(3), actions))
        else:
            factors[n] = list(range(n - 1, 216, 4))
        factors[n].sort(key=lambda value: signature([namespace, n, value]))
    drag_order = sorted(allowed, key=lambda uid: hashlib.sha256(
        f'{DRAG_VERSION}:{split}:{evaluation}:{namespace}:{uid}'.encode()).hexdigest())
    rows = []
    for draw in range(max(q.values()) * reserves):
        for n in counts:
            if draw >= q[str(n)] * reserves:
                continue
            index = len(rows)
            row = dict(index=index, count=n, task=task, split=split, evaluation=evaluation)
            factor = factors[n][draw % len(factors[n])]
            if task == 'passive':
                key = f'{namespace}:{index}'
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
                row.update(seed_namespace=key, seed=seed, material_offset=factor[0],
                           stiffness_phase=factor[1], scenario=factor[2],
                           asset_uids=random.Random(seed).sample(allowed, n))
            else:
                native = factor + 216 * (draw // len(factors[n]))
                key = f'{DRAG_VERSION}:{split}:{evaluation}:{namespace}:{native}'
                cursor = (native // 4) * 10 + sum((1, 2, 3, 4)[:native % 4])
                row.update(native_index=native, prepare_name=namespace, seed_namespace=key,
                           seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little'),
                           material=native // 4 % 2, stiffness_phase=native // 8 % 3,
                           events=native // 24 % 3 + 1, first_family=native // 72 % 3,
                           asset_uids=[drag_order[(cursor + j) % len(drag_order)] for j in range(n)])
            row['id'] = signature(row)
            rows.append(row)
    if len({r['seed_namespace'] for r in rows}) != len(rows):
        raise ValueError('Duplicate candidate namespace')
    return dict(task=task, evaluation=evaluation, counts=list(counts), quotas=q,
                target=target, allowed_uids=allowed, candidates=rows,
                selection='First passing candidates in declared order within each count quota; never completion order')


def restrict_bank(metadata, objects, allowed_uids):
    """Restriction only: preserve original points, weights, row order and cohort."""
    allowed = set(allowed_uids)
    indices = [i for i, row in enumerate(metadata['objects']) if row['asset_uid'] in allowed]
    if {metadata['objects'][i]['asset_uid'] for i in indices} != allowed:
        raise ValueError('Eligibility roster includes an absent bank shape')
    return dict(metadata, objects=[metadata['objects'][i] for i in indices]), [objects[i] for i in indices]


class LazyBank:
    """Load unchanged, hash-pinned NPZs only when a prepared scene uses them.

    The inventory already validated every payload against this exact manifest.
    This avoids decompressing all 10K shapes for each individual test scene.
    """
    def __init__(self, root, rows):
        self.root, self.rows, self.cache = Path(root), rows, {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if index not in self.cache:
            import numpy as np
            from collection.common import relative_path
            row = self.rows[index]; path = relative_path(self.root, row['file'])
            if sha(path) != row['sha256']: raise ValueError('Bank payload changed')
            with np.load(path, allow_pickle=False) as data:
                self.cache[index] = {key:data[key].copy() for key in data.files}
        return self.cache[index]


def make_plan(root, inventory, passive_bundle, drag_bundle, python, model_root, reference,
              *, name, split, target=100, reserves=8, validation_root=None):
    if split == 'test' and target != 100:
        raise ValueError('Final test suite requires 100 retained episodes per category')
    if split == 'test' and validation_root is None:
        raise ValueError('Test plans must name their validation suite')
    inv = read(inventory); assets = Path(inv['assets']).resolve(strict=True)
    if sha(assets / 'SPLITS.json') != inv['split_sha256']:
        raise ValueError('Inventory split changed')
    banks = {role: read(assets / role / 'MANIFEST.json') for role in ('train', 'validation', 'test')}
    for role in banks:
        if sha(assets / role / 'MANIFEST.json') != inv['banks'][role]['manifest_sha256']:
            raise ValueError('Inventory bank changed')
    members = {role: {r['asset_uid'] for r in value['objects']} for role, value in banks.items()}
    if any(members[a] & members[b] for a, b in itertools.combinations(members, 2)):
        raise ValueError('Overlapping shape cohorts')
    bound = {}
    for task, bundle in (('passive', passive_bundle), ('drag', drag_bundle)):
        bundle = Path(bundle).resolve(strict=True); b = read(bundle / 'BUNDLE.json')
        check_files(bundle, b['files'])
        if sha(assets / 'COMPLETE.json') != b.get('bank_complete_sha256', b.get('bank_sha256')):
            raise ValueError('Numerical bundle uses another asset release')
        bound[task] = dict(path=str(bundle), sha256=sha(bundle / 'BUNDLE.json'))
    categories = {}
    for key, (task, evaluation, counts) in CATEGORIES.items():
        seen = set(inv['training'][task]['seen_asset_uids'])
        if not seen <= members['train']:
            raise ValueError('Seen inventory contains held-out/non-bank UIDs')
        allowed = members[split] if evaluation.endswith('unseen') else seen
        categories[key] = candidate_plan(name, split, key, allowed, target=target, reserves=reserves)
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=False)
    model_root = Path(model_root).resolve(strict=True)
    model_files = sorted((model_root / 'data').glob('*.py')) + [model_root / 'stats/normalization.json']
    files = {}
    for p in model_files:
        relative = Path('model') / p.relative_to(model_root)
        target_path = root / relative; target_path.parent.mkdir(parents=True, exist_ok=True)
        before = sha(p); shutil.copy2(p, target_path)
        if sha(p) != before or sha(target_path) != before:
            raise ValueError('Model source changed during snapshot')
        files[str(relative)] = before
    shutil.copy2(__file__, root / 'driver.py'); files['driver.py'] = sha(__file__)
    shutil.copy2(inventory, root / 'INVENTORY.json'); files['INVENTORY.json'] = sha(inventory)
    plan = dict(schema=VERSION, name=name, split=split, assets=str(assets),
                asset_complete_sha256=sha(assets / 'COMPLETE.json'), split_sha256=inv['split_sha256'],
                inventory_sha256=files['INVENTORY.json'], bundles=bound, files=files,
                python=str(Path(python).resolve(strict=True)), reference=str(Path(reference).resolve(strict=True)),
                categories=categories, target_per_category=target, reserve_multiplier=reserves,
                validation_root=str(Path(validation_root).resolve(strict=True)) if validation_root else None,
                numerical_policy='Frozen serial reference only; no training fast-path extension',
                physical_accuracy_claim=False, accepted_for_training=False)
    write(root / 'PLAN.json', plan)
    write(root / 'SEAL.json', dict(plan_sha256=sha(root / 'PLAN.json'), driver_sha256=files['driver.py']))
    return dict(root=str(root), split=split, candidates=sum(len(c['candidates']) for c in categories.values()),
                target=6 * target, submitted=False)


def verify_plan(root):
    root = Path(root).resolve(strict=True); plan = read(root / 'PLAN.json'); seal = read(root / 'SEAL.json')
    if plan['schema'] != VERSION or sha(root / 'PLAN.json') != seal['plan_sha256']:
        raise ValueError('Plan changed')
    check_files(root, plan['files'])
    if sha(__file__) != seal['driver_sha256']:
        raise ValueError('Use the immutable driver.py stored in this suite')
    for item in plan['bundles'].values():
        bundle = Path(item['path'])
        if sha(bundle / 'BUNDLE.json') != item['sha256']:
            raise ValueError('Frozen numerical bundle changed')
        check_files(bundle, read(bundle / 'BUNDLE.json')['files'])
    if sha(Path(plan['assets']) / 'COMPLETE.json') != plan['asset_complete_sha256']:
        raise ValueError('Asset release changed')
    if sha(Path(plan['assets']) / 'SPLITS.json') != plan['split_sha256']:
        raise ValueError('Shape split changed')
    return root, plan


def case_root(root, category, row):
    return root / 'work' / category / f"candidate-{row['index']:05d}"


def check_outcome(root, plan, category, row, *, deep=False):
    folder = case_root(root, category, row); record = read(folder / 'OUTCOME.json')
    if record['candidate_id'] != row['id'] or record['plan_sha256'] != sha(root / 'PLAN.json'):
        raise ValueError('Outcome belongs to another candidate/plan')
    attempt = folder / record['attempt']
    if sha(attempt / 'REVIEW.json') != record['review_sha256']:
        raise ValueError('Changed candidate review')
    review = read(attempt / 'REVIEW.json')
    if review['candidate_id'] != row['id'] or record['retained'] != review['retained']:
        raise ValueError('Outcome/review mismatch')
    if deep:
        check_files(attempt, review['artifacts'])
    return record


def selection(root, plan, category, *, deep=False):
    """Only a contiguous declared prefix may influence final membership."""
    config = plan['categories'][category]; counts = Counter(); chosen = []; outcomes = []
    pending = None; skipped = []
    for row in config['candidates']:
        if counts[row['count']] >= config['quotas'][str(row['count'])]:
            skipped.append(row['index']); continue
        path = case_root(root, category, row) / 'OUTCOME.json'
        if not path.exists():
            pending = row; break
        result = check_outcome(root, plan, category, row, deep=deep); outcomes.append(result)
        if result['retained']:
            if result['episode_id'] in {r['episode_id'] for r in chosen}:
                raise ValueError('Duplicate retained episode')
            counts[row['count']] += 1; chosen.append(result)
    complete = all(counts[n] == config['quotas'][str(n)] for n in config['counts'])
    return dict(complete=complete, counts=dict(counts), chosen=chosen, outcomes=outcomes,
                pending=pending, exhausted=not complete and pending is None, skipped=skipped)


def _bound_manifest(root, plan, category, row, inputs):
    prepared = read(inputs / 'PLAN.json')
    entries = [c for c in prepared['cases'] if c.get('kind', 'coverage') != 'matched_control']
    if len(entries) != 1:
        raise ValueError('One planned candidate per input required')
    entry = entries[0]; directory = inputs / 'cases' / entry['label']
    if sha(directory / 'initial.npz') != entry['initial_sha256'] or sha(directory / 'MANIFEST.json') != entry['manifest_sha256']:
        raise ValueError('Prepared input changed')
    m = read(directory / 'MANIFEST.json')
    expected_role = plan['split'] if row['evaluation'].endswith('unseen') else 'train'
    if (m['case']['asset_ids'] != row['asset_uids'] or m['case']['seed_namespace'] != row['seed_namespace']
            or m['split'] != plan['split'] or m['evaluation'] != row['evaluation']
            or m['object_cohort'] != expected_role or m['n_objects'] != row['count']
            or m['split_manifest_sha256'] != plan['split_sha256']):
        raise ValueError('Prepared scene deviates from frozen candidate')
    train_ids = set(read(root / 'INVENTORY.json')['training'][row['task']]['episode_ids'])
    if m['episode_id'] in train_ids:
        raise ValueError('Training episode leaked into evaluation')
    return entry, m


def child(root, category, index, attempt, phase, gpu=None):
    root, plan = verify_plan(root); config = plan['categories'][category]; row = config['candidates'][index]
    if not re.fullmatch(r'attempt-[0-9]{4}', attempt) or row['index'] != index:
        raise ValueError('Invalid candidate/attempt identity')
    directory = case_root(root, category, row) / attempt; inputs = directory / 'input'
    from collection.mpm import runtime
    source = Path(plan['bundles'][row['task']]['path']) / 'source'
    if runtime.ROOT.resolve() != source.resolve():
        raise ValueError('Child imported the wrong numerical source')
    if phase == 'prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU-only preparation required')
        from collection.mpm import recipes
        from collection.mpm.banks import allowed_uids as cohort_uids
        inventory = read(root / 'INVENTORY.json')
        def restricted(bank, pool, *, role, split_manifest):
            bank = Path(bank)
            if sha(bank / 'MANIFEST.json') != inventory['banks'][role]['manifest_sha256']:
                raise ValueError('Audited bank manifest changed')
            meta = read(bank / 'MANIFEST.json')
            assigned = cohort_uids(meta, pool, role=role, split_manifest=split_manifest)
            if not set(config['allowed_uids']) <= assigned:
                raise ValueError('Eligibility roster escaped its shape cohort')
            meta, _ = restrict_bank(meta, list(range(len(meta['objects']))), config['allowed_uids'])
            return meta, LazyBank(bank, meta['objects'])
        recipes.load_bank = restricted
        assets = Path(plan['assets']); role = plan['split'] if row['evaluation'].endswith('unseen') else 'train'
        try:
            if row['task'] == 'passive':
                from collection.mpm.collect import prepare
                def factory(metadata):
                    indices = {r['asset_uid']: i for i, r in enumerate(metadata['objects'])}
                    return [dict(index=index, count=row['count'], seed=row['seed'], seed_namespace=row['seed_namespace'],
                        recipe_version='fresh-shapes-v1', split=plan['split'], evaluation=row['evaluation'],
                        material_offset=row['material_offset'], stiffness_phase=row['stiffness_phase'],
                        scenario=row['scenario'], asset_indices=[indices[u] for u in row['asset_uids']])]
                prepare(inputs, 'fresh-shapes-v1', bank=assets / role, pool=assets / 'pool', bank_role=role,
                        split_manifest=assets / 'SPLITS.json', dataset_split=plan['split'],
                        evaluation=row['evaluation'], specification_factory=factory)
                cache = root / 'kernel-cache' / row['task']; cache.mkdir(parents=True, exist_ok=True)
                (inputs / 'kernel-cache').symlink_to(cache, target_is_directory=True)
            else:
                from collection.mpm.drag.prepare import prepare
                prepare(inputs, assets, name=row['prepare_name'], count=1, start=row['native_index'],
                        split=plan['split'], evaluation=row['evaluation'], n_grid=256, object_counts=[1,2,3,4],
                        event_counts=[1,2,3], material_layout='homogeneous', speed_range=[.05,.25])
        except ValueError as error:
            reasons = error.args[0] if error.args else None
            if (row['task'] == 'drag' and isinstance(reasons, list) and reasons
                    and all(isinstance(r, str) and r.startswith(('drag_initial_wall_margin', 'drag_initial_spacing:')) for r in reasons)):
                write(directory / 'PREPARED.json', dict(candidate_id=row['id'], placement_rejected=True, reasons=reasons))
                return
            raise
        entry, m = _bound_manifest(root, plan, category, row, inputs)
        write(directory / 'PREPARED.json', dict(candidate_id=row['id'], placement_rejected=False,
              plan_sha256=sha(inputs / 'PLAN.json'), initial_sha256=entry['initial_sha256'], episode_id=m['episode_id']))
    elif phase == 'simulate':
        authorize_test(plan, row['task'])
        if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
            raise ValueError('Exactly the selected GPU UUID must be visible')
        entry, m = _bound_manifest(root, plan, category, row, inputs)
        from collection.mpm.collect import gpu_status
        def wait_idle(uuid):
            deadline = time.monotonic() + 30
            while True:
                try: return gpu_status(uuid)
                except RuntimeError as error:
                    if str(error) != 'GPU idle/memory guard failed' or time.monotonic() >= deadline:
                        raise
                    time.sleep(1)
        wait_idle(gpu)
        if row['task'] == 'passive':
            from collection.mpm.collect import collect, verify_attempt
            collect(SimpleNamespace(root=inputs, uuid=gpu, execution='candidate', start=2, limit=1,
                                    attempt=0, wall_seconds=1200, case_seconds=1100))
            result = verify_attempt(inputs / 'cases' / entry['label'] / 'attempts/attempt-00000')
            if result['status'] not in ('complete', 'placement_rejected'):
                raise RuntimeError('Simulation failed; preserve and diagnose: ' + canonical(result.get('reasons')))
        else:
            from collection.lease import gpu_lease
            import warp as wp
            wp.config.kernel_cache_dir = str(root / 'kernel-cache' / 'drag')
            from collection.mpm.drag.simulate import simulate
            with gpu_lease(gpu):
                # Recheck after acquiring ownership; never waive foreign processes.
                wait_idle(gpu)
                simulate(inputs, [entry['label']], directory / 'simulation', serial=True, uuid=gpu)
            result = read(directory / 'simulation' / entry['label'] / 'RESULT.json')
            if result['status'] not in ('complete', 'settling_rejected'):
                raise RuntimeError('Unexpected drag simulation status')
        write(directory / 'SIMULATION.json', result)
    elif phase == 'review':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU-only review required')
        prepared = read(directory / 'PREPARED.json')
        base = dict(candidate_id=row['id'], retained=False, count=row['count'], episode_id=None, task=row['task'])
        if prepared['placement_rejected']:
            base.update(status='placement_rejected', reasons=prepared['reasons'])
        else:
            entry, m = _bound_manifest(root, plan, category, row, inputs)
            result = read(directory / 'SIMULATION.json'); base.update(status=result['status'], episode_id=m['episode_id'])
            if result['status'] == 'complete':
                from collection.mpm.export import validate_candidate, eligible, export_candidates
                from collection.mpm.wall_materials import SAND_ONLY, STRICT
                policy = SAND_ONLY if row['task'] == 'passive' else STRICT
                raw = Path(result['raw_path'])
                if not raw.resolve().is_relative_to(directory.resolve()) or sha(raw) != result['raw_sha256']:
                    raise ValueError('Simulation raw path/hash mismatch')
                quality = validate_candidate(raw, wall_policy=policy)
                retained = bool(eligible(quality)); base.update(quality=quality, retained=retained)
                if retained:
                    exported = export_candidates([raw], directory / 'export' / 'filtered', wall_policy=policy)
                    if exported['retained'] != 1:
                        raise ValueError('Expected exactly one retained export')
                    from collection.consumer import import_consumer
                    import_consumer(root / 'model')
                    if row['task'] == 'passive':
                        from collection.mpm.review import check_model
                        consumer = check_model(directory / 'export', {m['episode_id']: raw})
                        if consumer['profiles'] != 2 or consumer['episodes_per_profile'] != 1:
                            raise ValueError('Incomplete passive consumer check')
                    else:
                        from collection.mpm.drag.review import loader_review
                        consumer = loader_review(directory / 'export/filtered/diagnostic-export', [raw])
                        if len(consumer) != 2 or any(r['episodes'] != 1 or not r['exact_raw_window_and_command_match'] for r in consumer):
                            raise ValueError('Incomplete drag consumer check')
                        base.update(events=len(m['action']['events']), material_names=m['material_names'],
                                    actual_families=[e['family'] for e in m['action']['events']])
                    base['consumer'] = consumer
            elif result['status'] not in ('placement_rejected', 'settling_rejected'):
                raise ValueError('Infrastructure failure cannot become a rejected candidate')
        artifacts = {}
        for p in directory.rglob('*'):
            if p.is_file() and (p.name in ('PREPARED.json', 'SIMULATION.json', 'raw.h5', 'initial.npz',
                                           'MANIFEST.json', 'PLAN.json', 'RESULT.json', 'SETTLING.json')
                                or p.suffix == '.tar'):
                artifacts[str(p.relative_to(directory))] = sha(p)
        base['artifacts'] = artifacts
        verify_plan(root)
        write(directory / 'REVIEW.json', base)
    else:
        raise ValueError('Unknown child phase')


def run_child(root, plan, category, row, attempt, phase, gpu):
    folder = case_root(root, category, row) / attempt; folder.mkdir(parents=True, exist_ok=True)
    source = Path(plan['bundles'][row['task']]['path']) / 'source'
    env = dict(os.environ, PYTHONPATH=str(source), PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1',
               CUDA_VISIBLE_DEVICES=gpu if phase == 'simulate' else '', OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PWM_MPM_REFERENCE_ROOT=plan['reference'],
               PWM_MODEL_ROOT=str(root / 'model'))
    command = [plan['python'], str(root / 'driver.py'), '_child', '--root', str(root), '--category', category,
               '--index', str(row['index']), '--attempt', attempt, '--phase', phase, '--gpu', gpu]
    try:
        with (folder / (phase + '.log')).open('ab') as log:
            result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=3900 if phase == 'simulate' else 1200)
        if result.returncode:
            raise RuntimeError(f'{category} candidate {row["index"]} {phase} failed; inspect {folder / (phase + ".log")}')
    except Exception as error:
        receipt = folder / ('FAILURE-' + phase + '.json')
        if not receipt.exists():
            write(receipt, dict(candidate_id=row['id'], phase=phase, error=str(error), terminal_candidate=False))
        raise


@contextmanager
def lock_category(root, category):
    directory = root / 'locks'; directory.mkdir(exist_ok=True)
    with (directory / (category + '.lock')).open('a') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError('Category already has a worker') from error
        yield


def finish_candidate(root, plan, category, row, gpu, *, retry_interrupted=False):
    folder = case_root(root, category, row); folder.mkdir(parents=True, exist_ok=True)
    attempts = sorted(folder.glob('attempt-*'))
    if attempts:
        current = attempts[-1]
        if not (current / 'REVIEW.json').exists() and not (current / 'SIMULATION.json').exists():
            if not retry_interrupted:
                raise RuntimeError('Interrupted attempt preserved; inspect before --retry-interrupted')
            current = folder / f'attempt-{len(attempts):04d}'
    else:
        current = folder / 'attempt-0000'
    current.mkdir(exist_ok=True)
    write(current / 'ATTEMPT.json', dict(candidate_id=row['id'], plan_sha256=sha(root / 'PLAN.json'),
                                        attempt=current.name, prior_attempts=[p.name for p in attempts if p != current]))
    if not (current / 'PREPARED.json').exists():
        run_child(root, plan, category, row, current.name, 'prepare', gpu)
    prepared = read(current / 'PREPARED.json')
    if prepared['candidate_id'] != row['id']:
        raise ValueError('Prepared receipt mismatch')
    for prior in attempts:
        receipt = prior / 'PREPARED.json'
        if prior != current and receipt.exists():
            previous = read(receipt)
            if (previous['candidate_id'] != row['id'] or
                    previous.get('initial_sha256') != prepared.get('initial_sha256') or
                    previous['placement_rejected'] != prepared['placement_rejected']):
                raise ValueError('Infrastructure retry changed its prepared scene')
    if not prepared['placement_rejected'] and not (current / 'SIMULATION.json').exists():
        run_child(root, plan, category, row, current.name, 'simulate', gpu)
    if not (current / 'REVIEW.json').exists():
        run_child(root, plan, category, row, current.name, 'review', gpu)
    review = read(current / 'REVIEW.json'); check_files(current, review['artifacts'])
    record = dict(candidate_id=row['id'], plan_sha256=sha(root / 'PLAN.json'), attempt=current.name,
                  review_sha256=sha(current / 'REVIEW.json'), retained=review['retained'],
                  episode_id=review['episode_id'], count=row['count'], index=row['index'], status=review['status'])
    if record['candidate_id'] != review['candidate_id']:
        raise ValueError('Candidate/review mismatch')
    write(folder / 'OUTCOME.json', record)
    return record


def publish(root, plan, category):
    state = selection(root, plan, category, deep=True)
    if not state['complete']:
        raise ValueError('Cannot publish incomplete quotas')
    destination = root / 'datasets' / plan['split'] / category
    sources = {}; shards = {}
    for result in state['chosen']:
        row = plan['categories'][category]['candidates'][result['index']]
        export = case_root(root, category, row) / result['attempt'] / 'export/filtered/diagnostic-export'
        paths = sorted(export.glob('*.tar'))
        if not paths: raise ValueError('Retained episode has no shards')
        for p in paths:
            name = f'{result["index"]:05d}-{p.name}'; sources[name] = p; shards[name] = sha(p)
    manifest = dict(schema=VERSION, split=plan['split'], category=category, task=CATEGORIES[category][0],
                    plan_sha256=sha(root / 'PLAN.json'), quotas=plan['categories'][category]['quotas'],
                    episode_ids=[r['episode_id'] for r in state['chosen']], retained=len(state['chosen']),
                    shards=shards, accepted_for_training=False)
    if destination.exists():
        if read(destination / 'MANIFEST.json') != manifest:
            raise ValueError('Published manifest/membership changed')
        if {p.name for p in destination.glob('*.tar')} != set(shards):
            raise ValueError('Published shard membership changed')
        check_files(destination, shards)
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + category + '-', dir=destination.parent))
    for name, source in sources.items(): os.link(source, stage / name)
    write(stage / 'MANIFEST.json', manifest)
    # mkdtemp is 0700 regardless of the collection umask. Release the completed
    # category for lab-group traversal without granting world or group writes.
    os.chmod(stage, 0o2750)
    os.rename(stage, destination)
    return manifest


def compatibility(plan):
    return dict(schema=plan['schema'], bundles=plan['bundles'], asset_complete_sha256=plan['asset_complete_sha256'],
                inventory_sha256=plan['inventory_sha256'], files=plan['files'], numerical_policy=plan['numerical_policy'])


def qualify(root, task):
    root, plan = verify_plan(root)
    if plan['split'] != 'validation' or task not in ('passive', 'drag'):
        raise ValueError('Qualification uses validation only')
    reviews = []; published = {}
    for key, config in plan['categories'].items():
        if config['task'] != task: continue
        state = selection(root, plan, key, deep=True)
        if not state['complete']: raise ValueError('Validation quotas incomplete: ' + key)
        publish(root, plan, key)
        published[key] = sha(root / 'datasets/validation' / key / 'MANIFEST.json')
        for result in state['chosen']:
            row = config['candidates'][result['index']]
            reviews.append(read(case_root(root, key, row) / result['attempt'] / 'REVIEW.json'))
    if task == 'drag' and not any(r['count'] >= 2 and r.get('events', 0) >= 2 for r in reviews):
        raise ValueError('Need a retained multi-object multi-event validation example')
    result = dict(schema=VERSION, task=task, validation_plan_sha256=sha(root / 'PLAN.json'),
                  compatibility=compatibility(plan), manifests=published, reference_execution_verified=True,
                  physical_accuracy_claim=False, acceleration_approved=False)
    write(root / ('QUALIFIED-' + task + '.json'), result)
    return result


def authorize_test(plan, task):
    if plan['split'] == 'validation': return
    result = qualify(plan['validation_root'], task)
    if result['compatibility'] != compatibility(plan):
        raise ValueError('Validation does not cover these exact sources, consumer and training inventory')


def run(root, category, gpu, *, hours=2, max_candidates=10000, retry_interrupted=False):
    root, plan = verify_plan(root)
    if category not in plan['categories'] or not re.fullmatch(r'GPU-[a-zA-Z0-9-]+', gpu):
        raise ValueError('Explicit category/GPU required')
    if not 0 < hours <= 48 or not 1 <= max_candidates <= 10000:
        raise ValueError('Bounded time/candidate budget required')
    authorize_test(plan, CATEGORIES[category][0])
    deadline = time.monotonic() + hours * 3600
    with lock_category(root, category):
        for _ in range(max_candidates):
            state = selection(root, plan, category)
            if state['complete']:
                return dict(complete=True, publication=publish(root, plan, category))
            if state['exhausted']: raise RuntimeError('Predeclared candidate budget exhausted; quotas remain incomplete')
            if time.monotonic() + 4200 > deadline:
                return dict(complete=False, reason='shutdown_reserve', counts=state['counts'])
            result = finish_candidate(root, plan, category, state['pending'], gpu, retry_interrupted=retry_interrupted)
            print(canonical(dict(category=category, result=result)), flush=True)
        state = selection(root, plan, category)
        if state['complete']: publish(root, plan, category)
        return dict(complete=state['complete'], counts=state['counts'], reason='bounded_candidate_limit')


def status(root):
    root, plan = verify_plan(root)
    return {key: {k:v for k,v in selection(root, plan, key).items() if k in ('complete', 'counts', 'exhausted')}
            for key in plan['categories']}


def lane(root, categories, gpu, *, hours):
    deadline = time.monotonic() + hours * 3600; results = {}
    for category in categories:
        remaining = (deadline - time.monotonic()) / 3600
        if remaining <= 4200 / 3600:
            return dict(complete=False, reason='shutdown_reserve', results=results)
        results[category] = run(root, category, gpu, hours=remaining)
        if not results[category]['complete']:
            return dict(complete=False, results=results)
    return dict(complete=True, results=results)


def allocated_gpus():
    """Resolve Slurm's explicit device list without creating CUDA contexts."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True, timeout=15)
    rows = [tuple(part.strip() for part in line.split(',')) for line in text.splitlines() if line.strip()]
    indices = dict(rows); all_uuids = set(indices.values()); resolved = []
    for item in visible:
        value = item if item in all_uuids else indices.get(item)
        if value is None: raise ValueError('Cannot resolve explicitly allocated CUDA device')
        resolved.append(value)
    if len(resolved) != 2 or len(set(resolved)) != 2:
        raise ValueError('Exactly two explicitly allocated GPUs required')
    return resolved


def allocation(root, task, *, hours=12):
    root, plan = verify_plan(root)
    if not os.environ.get('SLURM_JOB_ID'): raise ValueError('Slurm allocation required')
    if not 2 <= hours <= 48: raise ValueError('Bounded allocation duration required')
    authorize_test(plan, task)
    gpus = allocated_gpus(); categories = [k for k,v in CATEGORIES.items() if v[0] == task]
    processes = []
    try:
        for i, gpu in enumerate(gpus):
            command = [plan['python'], str(root/'driver.py'), 'lane', '--root', str(root), '--gpu', gpu,
                       '--hours', str(hours - .25), '--categories', *categories[i::2]]
            processes.append(subprocess.Popen(command))
        exits = [p.wait() for p in processes]
    finally:
        # Only children created by this launcher are stopped on launcher failure.
        for p in processes:
            if p.poll() is None: p.terminate()
    completed = all(selection(root,plan,key)['complete'] for key in categories)
    return dict(complete=completed and not any(exits), worker_exit_codes=exits, task=task)


def sbatch(root, task, *, hours=12):
    root, plan = verify_plan(root)
    if task not in ('passive','drag') or type(hours) is not int or not 2 <= hours <= 48:
        raise ValueError('Explicit task and 2..48-hour request required')
    logs = root/'logs'; logs.mkdir(exist_ok=True)
    if any(c in str(root) for c in ('\n','\r',' ')):
        raise ValueError('Use a suite path without whitespace for Slurm headers')
    script = f'''#!/bin/bash
#SBATCH --job-name=mpm-eval-{task}
#SBATCH --partition=katef
#SBATCH --account=kfragki2
#SBATCH --qos=katef_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:RTX_PRO_6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time={hours}:00:00
#SBATCH --no-requeue
#SBATCH --output={logs}/%x-%j.out
set -euo pipefail
umask 0007
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
exec {shlex.quote(plan['python'])} {shlex.quote(str(root/'driver.py'))} allocation \\
  --root {shlex.quote(str(root))} --task {task} --hours {hours}
'''
    path = root / f'collect-{task}.sbatch'
    with path.open('x') as stream: stream.write(script)
    write(root/f'SBATCH-{task}.json',dict(path=str(path),sha256=sha(path),submitted=False,hours=hours,gpus=2))
    return dict(script=str(path),submitted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan')
    for key in ('root','inventory','passive-bundle','drag-bundle','python','model-root','reference','name','split'):
        p.add_argument('--' + key, required=True)
    p.add_argument('--target', type=int, default=100); p.add_argument('--reserves', type=int, default=8)
    p.add_argument('--validation-root')
    p = sub.add_parser('run'); p.add_argument('--root', required=True); p.add_argument('--category', choices=CATEGORIES, required=True)
    p.add_argument('--gpu', required=True); p.add_argument('--hours', type=float, default=2)
    p.add_argument('--max-candidates', type=int, default=10000); p.add_argument('--retry-interrupted', action='store_true')
    p = sub.add_parser('status'); p.add_argument('--root', required=True)
    p = sub.add_parser('qualify'); p.add_argument('--root', required=True); p.add_argument('--task', choices=('passive','drag'), required=True)
    for command in ('allocation','sbatch'):
        p=sub.add_parser(command); p.add_argument('--root',required=True)
        p.add_argument('--task',choices=('passive','drag'),required=True); p.add_argument('--hours',type=int,default=12)
    p=sub.add_parser('lane'); p.add_argument('--root',required=True); p.add_argument('--gpu',required=True)
    p.add_argument('--categories',nargs='+',choices=CATEGORIES,required=True); p.add_argument('--hours',type=float,default=12)
    p = sub.add_parser('_child')
    for key in ('root','category','attempt','phase'): p.add_argument('--' + key, required=True)
    p.add_argument('--index', type=int, required=True); p.add_argument('--gpu')
    args = vars(parser.parse_args()); command = args.pop('command')
    if command == 'plan': result = make_plan(**args)
    elif command == 'run': result = run(**args)
    elif command == 'status': result = status(**args)
    elif command == 'qualify': result = qualify(**args)
    elif command == 'lane': result = lane(**args)
    elif command == 'allocation': result = allocation(**args)
    elif command == 'sbatch': result = sbatch(**args)
    else: result = child(**args)
    print(canonical(result), flush=True)
    if command in ('lane','allocation') and not result['complete']: raise SystemExit(2)


if __name__ == '__main__':
    main()
