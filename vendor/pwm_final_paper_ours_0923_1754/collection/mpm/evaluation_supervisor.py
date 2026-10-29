"""Bounded recovery around an unchanged, qualified evaluation driver.

Only explicit GPU-contention failures can be retried automatically. Numerical,
source-integrity and data-review failures are preserved and block that category.
No GPU is reset, no foreign process is stopped, and no physics filter is relaxed.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import time
import traceback


RESOURCE_ERRORS = (
    'RuntimeError: Foreign process on selected GPU; owned worker stops',
    'RuntimeError: GPU idle/memory guard failed',
)


def approved_resource_failure(line, current):
    if line in RESOURCE_ERRORS:
        return True
    prefix = 'RuntimeError: Simulation failed; preserve and diagnose: '
    if not line.startswith(prefix):
        return False
    try:
        reasons = json.loads(line[len(prefix):])
    except (ValueError, TypeError):
        return False
    allowed = {message.replace('RuntimeError: ', 'RuntimeError:', 1) for message in RESOURCE_ERRORS}
    if (not isinstance(reasons, list) or not reasons
            or not all(isinstance(reason, str) and reason in allowed for reason in reasons)):
        return False
    # Passive collection records the underlying error before its driver wraps it.
    # Require the on-disk result to corroborate the exact resource-only reasons.
    results = list(Path(current).glob('input/cases/*/attempts/attempt-*/RESULT.json'))
    if len(results) != 1:
        return False
    result = json.loads(results[0].read_text())
    return result.get('status') == 'failed' and result.get('reasons') == reasons


def retry_decision(folder, *, max_attempts=12):
    """Fail closed unless an unfinished simulation has explicit resource evidence."""
    attempts = sorted(Path(folder).glob('attempt-*'))
    if not attempts:
        return False
    current = attempts[-1]
    if (current / 'REVIEW.json').exists():
        return False  # Frozen driver checks hashes and recovers its OUTCOME.
    failures = sorted(current.glob('FAILURE-*.json'))
    if any(p.name != 'FAILURE-simulate.json' for p in failures):
        raise RuntimeError('Non-simulation failure requires review: ' + str(current))
    if (current / 'SIMULATION.json').exists():
        return False  # Completed simulation must never be repeated.
    failure = current / 'FAILURE-simulate.json'
    log = current / 'simulate.log'
    if not failure.exists() or not log.exists():
        raise RuntimeError('Unclassified interrupted attempt: ' + str(current))
    receipt = json.loads(failure.read_text())
    if receipt.get('phase') != 'simulate' or receipt.get('terminal_candidate') is not False:
        raise RuntimeError('Invalid resource-failure receipt: ' + str(current))
    lines = log.read_text(errors='replace').strip().splitlines()
    if not lines or not approved_resource_failure(lines[-1].strip(), current):
        raise RuntimeError('Not an approved resource failure: ' + str(current))
    if len(attempts) >= max_attempts:
        raise RuntimeError('Resource retry budget exhausted: ' + str(current))
    return True


def gpu_idle(uuid):
    """Mirror the frozen preflight; do not waive its idle threshold."""
    data = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], text=True, timeout=15)
    matches = [[v.strip() for v in line.split(',')] for line in data.splitlines()
               if line.split(',')[0].strip() == uuid]
    if len(matches) != 1:
        raise RuntimeError('Selected GPU missing or ambiguous')
    processes = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
        text=True, timeout=15)
    pids = [line.split(',')[1].strip() for line in processes.splitlines()
            if line.split(',')[0].strip() == uuid]
    info = dict(uuid=uuid, memory_mib=int(matches[0][1]), utilization=int(matches[0][2]), pids=pids)
    return not pids and info['memory_mib'] <= 32 and info['utilization'] == 0, info


@contextmanager
def claim(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main(root, gpu, slot, hours, categories, operator_root, training_release):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', slot) or not 2 <= hours <= 24:
        raise ValueError('Explicit safe slot and bounded duration required')
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('An allocated Slurm step is required')
    root, operator_root = Path(root).resolve(), Path(operator_root).resolve()
    worker = runpy.run_path(str(root / 'driver.py'))
    root, plan = worker['verify_plan'](root)
    if set(categories) != set(plan['categories']) or len(categories) != len(set(categories)):
        raise ValueError('Priority list must contain every category exactly once')
    seal = worker['read'](operator_root / 'SEAL.json')
    worker['check_files'](operator_root, seal['files'])
    if worker['sha'](__file__) != seal['files']['supervisor.py']:
        raise ValueError('Supervisor does not match its operator snapshot')
    control = root / 'supervision-v1'
    control.mkdir(exist_ok=True)
    slot_root = control / slot
    slot_root.mkdir(exist_ok=True)
    deadline = time.monotonic() + hours * 3600

    def event(kind, **data):
        record = dict(kind=kind, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                      slot=slot, gpu=gpu, job=os.environ['SLURM_JOB_ID'],
                      step=os.environ.get('SLURM_STEP_ID'), **data)
        worker['write'](slot_root / f'{time.time_ns()}-{kind}.json', record)
        print(json.dumps(record, sort_keys=True), flush=True)

    def finish(category):
        worker['publish'](root, plan, category)
        index = root / 'indices' / plan['split'] / category
        report = control / f'INDEXED-CHECK-{category}.json'
        domain = 'mpm_drag' if plan['categories'][category]['task'] == 'drag' else 'mpm_fresh'
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(operator_root),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONDONTWRITEBYTECODE='1')
        if not index.exists():
            command = [plan['python'], '-m', 'collection.mpm.evaluation', 'index-export',
                       '--suite-root', str(root), '--category', category,
                       '--training-index', str(Path(training_release) / domain / 'train'),
                       '--output', str(index)]
            subprocess.run(command, env=env, cwd=operator_root, check=True, timeout=1800)
        if not report.exists():
            subprocess.run([plan['python'], str(operator_root / 'check_index.py'), '--root', str(root),
                            '--training-release', str(training_release), '--output', str(report),
                            '--categories', category], env=env, cwd=operator_root,
                           check=True, timeout=3600)
        receipt = worker['read'](report)
        if not receipt.get('passed') or receipt['suite_plan_sha256'] != worker['sha'](root / 'PLAN.json'):
            raise RuntimeError('Invalid indexed integration receipt')
        if len(receipt['results']) != 4 or any(
                row['category'] != category or row['episodes'] != plan['target_per_category']
                or row['index_sha256'] != worker['sha'](index / 'INDEX.json')
                for row in receipt['results']):
            raise RuntimeError('Indexed integration does not cover exact publication')
        worker['write'](control / f'READY-{category}.json', dict(
            category=category, retained=plan['target_per_category'],
            index_sha256=worker['sha'](index / 'INDEX.json'),
            check_sha256=worker['sha'](report), plan_sha256=worker['sha'](root / 'PLAN.json')))
        event('category_ready', category=category)

    with claim(control / f'slot-{slot}.lock') as owned:
        if not owned:
            raise RuntimeError('Supervisor slot already owned')
        event('started', hours=hours, priorities=categories, operator_seal=worker['sha'](operator_root/'SEAL.json'))
        while time.monotonic() + 4500 < deadline:
            worker['verify_plan'](root)
            worker['check_files'](operator_root, seal['files'])
            done = [c for c in categories if (control / f'READY-{c}.json').exists()]
            blocked = [c for c in categories if (control / f'BLOCKED-{c}.json').exists()]
            if len(set(done + blocked)) == len(categories):
                event('finished', ready=done, blocked=blocked)
                return 0 if not blocked else 1
            idle, info = gpu_idle(gpu)
            dispatched = False
            for category in categories:
                if category in done or category in blocked:
                    continue
                with claim(control / f'dispatch-{category}.lock') as selected:
                    if not selected:
                        continue
                    try:
                        state = worker['selection'](root, plan, category)
                        if state['complete']:
                            finish(category)
                            dispatched = True
                            break
                        if state['exhausted']:
                            raise RuntimeError('Frozen candidate reserve exhausted')
                        if not idle:
                            continue
                        # Avoid colliding with a legacy lane; the actual run also locks.
                        with claim(root / 'locks' / (category + '.lock')) as available:
                            if not available:
                                continue
                        folder = worker['case_root'](root, category, state['pending'])
                        retry = retry_decision(folder)
                        event('dispatch', category=category, candidate=state['pending']['index'], retry=retry)
                        try:
                            result = worker['run'](root, category, gpu,
                                hours=(deadline-time.monotonic())/3600,
                                max_candidates=1, retry_interrupted=retry)
                            event('candidate_finished', category=category, result=result)
                        except Exception as error:
                            if str(error) == 'Category already has a worker':
                                event('category_busy', category=category)
                            elif retry_decision(folder):
                                event('resource_retry_pending', category=category, error=str(error))
                                time.sleep(30)
                            else:
                                raise
                        dispatched = True
                        break
                    except Exception:
                        detail = traceback.format_exc()
                        worker['write'](control / f'BLOCKED-{category}.json', dict(
                            category=category, slot=slot, error=detail,
                            plan_sha256=worker['sha'](root/'PLAN.json')))
                        event('category_blocked', category=category, error=detail)
            if not dispatched:
                event('waiting', gpu_state=info, ready=done, blocked=blocked)
                time.sleep(30)
        event('budget_end', ready=[c for c in categories if (control/f'READY-{c}.json').exists()])
        return 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'gpu', 'slot', 'operator-root', 'training-release'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--hours', type=float, default=16)
    parser.add_argument('--categories', nargs='+', required=True)
    raise SystemExit(main(**vars(parser.parse_args())))
