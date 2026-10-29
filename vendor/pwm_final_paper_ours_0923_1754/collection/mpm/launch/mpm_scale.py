"""Freeze, preflight, and supervise bounded, split-aware passive MPM collection.

CPU-only preflight is safe outside Slurm. `run` requires a separate allocation.
No submission, automatic retries, numerical changes, or training promotion.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROLES = ('train', 'validation', 'test')
EXCLUDED_NODES = ('babel-l9-20', 'babel-z9-16')
GENESIS = '/home/robertre/miniconda3/envs/genesis/bin/python'
MODEL = '/data/user_data/robertre/conda_envs/pwm_blackwell/bin/python'
REFERENCE = Path('/data/user_data/robertre/particle_world_model/reports/mpm_correctness_20260907')
STAGING = Path('/data/group_data/katefgroup-ssd/world_model/particle_world_model/staging')
BLOCK_RESERVE = 2400  # Do not start a block within 40 minutes of soft deadline.


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    path = Path(path)
    if not replace and path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def schedule(train_blocks, eval_blocks):
    if type(train_blocks) is not int or not 1 <= train_blocks <= 1000:
        raise ValueError('train_blocks must be 1..1000')
    if type(eval_blocks) is not int or not 1 <= eval_blocks <= 100:
        raise ValueError('eval_blocks must be 1..100')
    order = [('train', 0)]
    order += [(role, i) for i in range(eval_blocks) for role in ROLES[1:]]
    order += [('train', i) for i in range(1, train_blocks)]
    return [dict(name=f'{role}-{i:04d}', role=role, index=i,
                 seeds=[2026100000 + ROLES.index(role) * 100000 + 2 * i + lane for lane in range(2)])
            for role, i in order]


def freeze(bundle, research):
    bundle.mkdir(parents=True, exist_ok=False)
    source = bundle / 'source'
    old = research / 'runs/overnight_mpm_20260912/source-v1'
    for name in ('data', 'stats'):
        shutil.copytree(old / name, source / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    (source / 'collection/mpm').mkdir(parents=True)
    for folder in ('collection', 'collection/mpm'):
        for path in (research / folder).glob('*.py'):
            shutil.copy2(path, source / folder / path.name)
    (bundle / 'helpers').mkdir()
    for name in ('bench.py', 'batch_group.py'):
        shutil.copy2(research / 'runs/mpm_parallel_20260912' / name, bundle / 'helpers' / name)
    for name in ('mpm_scale.py', 'mpm_scale_worker.py', 'mpm_scale_review.py'):
        shutil.copy2(research / 'collection/mpm/launch' / name, bundle / name)
    banks = STAGING / 'mpm-scale-banks-20260912-v1'
    splits = STAGING / 'mpm-role-banks-20260912-v1/SPLITS.json'
    pool = REFERENCE / 'source-snapshot/source/configs'
    external = [splits, *[banks / role / 'MANIFEST.json' for role in ROLES],
                *[pool / f'objaverse_uid_{name}.json' for name in ('train', 'holdout')]]
    write(bundle / 'BUNDLE.json', dict(version=1, banks=str(banks), split_manifest=str(splits),
          pool=str(pool), reference=str(REFERENCE), genesis_python=GENESIS, model_python=MODEL,
          files={str(p.relative_to(bundle)): sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()},
          external_sha256={str(p): sha(p) for p in external},
          physics='Original solver/launch sizes; batched guard transfers; E=1e4,1e5,1e6 Pa, nu=.2, rho=1000',
          accepted_for_training=False))


def verify_bundle(bundle):
    config = read(bundle / 'BUNDLE.json')
    for name, expected in config['files'].items():
        if sha(bundle / name) != expected:
            raise ValueError('Frozen bundle changed: ' + name)
    for name, expected in config['external_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Bank/split inputs changed: ' + name)
    return config


def bank_sets(bundle, config):
    sys.path.insert(0, str(bundle / 'source'))
    from collection.mpm.recipes import load_bank
    result = {}
    for role in ROLES:
        metadata, _ = load_bank(Path(config['banks']) / role, config['pool'], role=role,
                                split_manifest=config['split_manifest'])
        result[role] = sorted(r['asset_uid'] for r in metadata['objects'])
    for i, role in enumerate(ROLES):
        for other in ROLES[i+1:]:
            if set(result[role]) & set(result[other]):
                raise ValueError('Cross-role object overlap')
    return result


def preflight(bundle, train_blocks, eval_blocks):
    config = verify_bundle(bundle)
    sets = bank_sets(bundle, config)
    blocks = schedule(train_blocks, eval_blocks)
    for interpreter in ('genesis_python', 'model_python'):
        if not os.access(config[interpreter], os.X_OK):
            raise ValueError('Missing interpreter: ' + config[interpreter])
    if not Path(config['reference']).is_dir():
        raise ValueError('Missing historical controls')
    return config, sets, blocks


def completion(unit):
    """Resume only fully reviewed blocks whose durable payloads are unchanged."""
    marker = unit / 'COMPLETE.json'
    if not marker.exists():
        raise ValueError(f'Incomplete block requires review; refusing automatic rerun: {unit}')
    result = read(marker)
    if result['status'] != 'complete':
        raise ValueError('Invalid completion marker')
    for name, expected in result['files'].items():
        if sha(unit / name) != expected:
            raise ValueError('Completed payload changed: ' + str(unit / name))
    return result


def discover(config):
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().split('.')[0] in EXCLUDED_NODES:
        raise ValueError('A separate Slurm node is required; folding node is forbidden')
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Slurm must bind the two allocated GPUs')
    # Disposable child resolves CUDA ordinals after Slurm remapping. Parent never
    # creates a CUDA context; nvidia-smi indices are NOT assumed to be CUDA indices.
    code = ('import json,warp as wp; wp.init(); '
            'print("DEVICES="+json.dumps([dict(uuid=d.uuid,name=d.name) for d in wp.get_cuda_devices()]))')
    output = subprocess.check_output([config['genesis_python'], '-c', code], text=True, timeout=60)
    devices = json.loads(next(line[8:] for line in output.splitlines() if line.startswith('DEVICES=')))
    if len(devices) != 2 or len({d['uuid'] for d in devices}) != 2:
        raise ValueError('Exactly two distinct allocated GPUs required')
    if any('RTX PRO 6000' not in d['name'] for d in devices):
        raise ValueError('Both allocated devices must be RTX PRO 6000')
    return [d['uuid'] for d in devices]


def children(commands, envs, logs, timeout):
    """Only manage processes created by this supervisor; never foreign workers."""
    procs, streams = [], []
    try:
        for command, env, log in zip(commands, envs, logs):
            stream = log.open('x'); streams.append(stream)
            procs.append(subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                          start_new_session=True))
        deadline = time.monotonic() + timeout
        while True:
            codes = [p.poll() for p in procs]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'Child failed ({codes}); inspect {logs}')
            if all(code == 0 for code in codes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Block subprocess deadline; inspect {logs}')
            time.sleep(1)
    finally:
        for child in procs:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=15)
        for stream in streams:
            stream.close()


def run(args):
    bundle = args.bundle.resolve(strict=True)
    config, sets, blocks = preflight(bundle, args.train_blocks, args.eval_blocks)
    if sha(__file__) != config['files']['mpm_scale.py']:
        raise ValueError('Execute the frozen bundle launcher, not mutable launch source')
    uuids = discover(config)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    lock = (root / '.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    request = dict(bundle=str(bundle), bundle_sha256=sha(bundle / 'BUNDLE.json'), blocks=blocks,
                   accepted_for_training=False)
    if args.resume:
        if read(root / 'REQUEST.json') != request:
            raise ValueError('Resume request differs from original; refusing')
    else:
        write(root / 'REQUEST.json', request)
    job = os.environ['SLURM_JOB_ID']
    write(root / f'EXECUTION-{job}.json', dict(job=job, node=socket.gethostname(), gpu_uuids=uuids,
          started=datetime.now(timezone.utc).isoformat(), max_seconds=args.max_seconds))
    stop = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop.__setitem__(0, True))
    def terminate(*_):
        raise InterruptedError('Slurm/user terminated supervisor')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, terminate)
    cpu = dict(os.environ, CUDA_VISIBLE_DEVICES='', PWM_MPM_REFERENCE_ROOT=config['reference'],
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ.update(cpu)
    from collection.mpm import collect
    began = time.monotonic(); done = []; status = 'running'; current = None
    def report():
        totals = {role: dict(attempts=0, retained=0, retained_state_frames=0) for role in ROLES}
        for row in done:
            for key in totals[row['role']]:
                totals[row['role']][key] += row[key]
        write(root / 'STATUS.json', dict(status=status, current_block=current, completed_blocks=len(done),
              requested_blocks=len(blocks), totals=totals, seconds=time.monotonic()-began,
              updated=datetime.now(timezone.utc).isoformat(), accepted_for_training=False), replace=True)
    try:
        report()
        for block in blocks:
            current = block['name']; unit = root / current
            if unit.exists():
                done.append(completion(unit)); report(); continue
            if stop[0] or (root / 'STOP').exists() or time.monotonic()-began >= args.max_seconds-BLOCK_RESERVE:
                status = 'stopped_at_block_boundary'; break
            if shutil.disk_usage(root).free < 100 * 1024**3:
                raise RuntimeError('Less than 100 GiB free; stopping before next block')
            verify_bundle(bundle)
            unit.mkdir(); report(); plans = []
            for lane, seed in enumerate(block['seeds']):
                target = unit / f'input-{lane}'
                collect.prepare(target, 'centered-v2', seed_start=seed, seed_count=1,
                                bank=Path(config['banks'])/block['role'], pool=config['pool'],
                                bank_role=block['role'], split_manifest=config['split_manifest'])
                collect.inspect(target)
                plans.append(dict(lane=lane, role=block['role'], input=str(target), seed=seed,
                                  sha256=sha(target/'PLAN.json')))
            write(unit/'PLAN.json', dict(plans=plans, gpu_uuids=uuids, slurm_job_id=job, object_ids=sets,
                  driver_sha256=sha(bundle/'mpm_scale_worker.py'), block=block, accepted_for_training=False))
            commands = [[config['genesis_python'], str(bundle/'mpm_scale_worker.py'), '--bundle', str(bundle),
                         '--root', str(unit), '--lane', str(lane), '--parent', str(os.getpid())] for lane in range(2)]
            children(commands, [dict(cpu, CUDA_VISIBLE_DEVICES=uuid) for uuid in uuids],
                     [unit/f'lane-{lane}.log' for lane in range(2)], 1800)
            command = [config['model_python'], str(bundle/'mpm_scale_review.py'), '--bundle', str(bundle),
                       '--root', str(unit), '--out', str(unit/'review'), '--role', block['role']]
            if block['index'] != 0:
                command.append('--no-media')
            children([command], [cpu], [unit/'review.log'], 600)
            verify_bundle(bundle)
            reviewed = read(unit/'review/RESULT.json')
            if reviewed['status'] != 'complete' or reviewed['attempts'] != 90:
                raise ValueError('Block review did not finish correctly')
            # Hash raw/journal/export/input payloads, not just a marker. No logs or
            # cache files are required for numerical replay, but retain them too.
            receipt = dict(status='complete', role=block['role'], attempts=90,
                           retained=reviewed['retained'], retained_state_frames=reviewed['retained_state_frames'],
                           files={str(p.relative_to(unit)): sha(p) for p in sorted(unit.rglob('*'))
                                  if p.is_file() and p.suffix != '.log'}, accepted_for_training=False)
            write(unit/'COMPLETE.json', receipt); done.append(receipt); report()
            print(json.dumps(dict(block=current, retained=receipt['retained'], completed_blocks=len(done))), flush=True)
        else:
            status = 'complete'
    except BaseException as error:
        status = f'failed: {type(error).__name__}: {error}'
        raise
    finally:
        report(); lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'preflight', 'run'))
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--research', type=Path)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--train-blocks', type=int, default=320)
    parser.add_argument('--eval-blocks', type=int, default=2)
    parser.add_argument('--max-seconds', type=int, default=46*3600)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'freeze':
        if args.research is None:
            parser.error('--research is required for freeze')
        freeze(args.bundle.resolve(), args.research.resolve(strict=True))
    elif args.command == 'preflight':
        config, sets, blocks = preflight(args.bundle.resolve(strict=True), args.train_blocks, args.eval_blocks)
        print(json.dumps(dict(status='preflight_pass', blocks=len(blocks), attempt_ceiling=90*len(blocks),
              bank_objects={r: len(v) for r,v in sets.items()}, first_blocks=blocks[:5],
              excluded_nodes=EXCLUDED_NODES, requested_gpus='gpu:RTX_PRO_6000:2', submitted=False), indent=2))
    else:
        if args.root is None or not BLOCK_RESERVE < args.max_seconds <= 46*3600:
            parser.error('run requires --root and 2400 < --max-seconds <= 165600')
        run(args)


if __name__ == '__main__':
    main()
