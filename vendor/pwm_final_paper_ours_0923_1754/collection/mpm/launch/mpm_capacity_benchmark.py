"""Diagnostic capacity sweep on two allocated GPUs; never approves production.

All arms use identical prepared scenes and exact reference compiler artifacts.
CPU affinity is bounded; changing grouped world count is an explicit unqualified
experiment, not a change to the published collector. No reservation or promotion.
"""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ARMS = {
    'cpu8-group4': (8, 4),
    'cpu32-group4': (32, 4),
    'cpu32-group8': (32, 8),
    'cpu32-group16': (32, 16),
    'cpu32-group4-async4': (32, 4),
}


@contextmanager
def diagnostic_writer(grouped):
    """Diagnostic-only adapter: immutable writes; drain before raw/journal checks.

    The frozen numerical files are not edited. Record this adapter's source hash
    as additional execution provenance; this is NOT the published worker path.
    """
    from mpm_async_io import BoundedFrameWriter
    original_journal, original_verify = grouped.FrameJournal, grouped.verify_journal
    with BoundedFrameWriter(max_bytes=64*1024**2, workers=4, max_pending=32) as writer:
        def journal(*args, **kwargs):
            actual = original_journal(*args, **kwargs)
            return SimpleNamespace(root=actual.root,
                append=lambda step, arrays: writer.append(actual, step, arrays))
        def verify(*args, **kwargs):
            writer.flush()
            return original_verify(*args, **kwargs)
        grouped.FrameJournal, grouped.verify_journal = journal, verify
        try:
            yield writer
        finally:
            grouped.FrameJournal, grouped.verify_journal = original_journal, original_verify


def affinity(available, total, lane):
    if total not in (8, 32) or lane not in (0, 1) or len(available) < total:
        raise ValueError('Insufficient allocated CPU affinity or invalid lane')
    half = total // 2
    return sorted(available)[lane * half:(lane + 1) * half]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'worker'))
    for name in ('bundle', 'reference', 'plan', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    parser.add_argument('--arm', choices=list(ARMS))
    parser.add_argument('--lane', type=int, choices=(0, 1))
    parser.add_argument('--parent', type=int)
    parser.add_argument('--cpus', type=json.loads)
    a = parser.parse_args()
    bundle = a.bundle.resolve(strict=True)
    sys.path[:0] = [str(bundle), str(bundle / 'source')]
    import mpm_four as four
    import mpm_scale as support
    import mpm_training_campaign as campaign
    from collection.mpm import collect
    config = campaign.verify(bundle)
    os.environ['PWM_MPM_REFERENCE_ROOT'] = config['reference']
    os.environ['PWM_MODEL_ROOT'] = str(bundle / 'source')
    if a.command == 'worker':
        if not a.cpus or not set(a.cpus) <= os.sched_getaffinity(0):
            raise ValueError('CPU request outside allocated affinity')
        os.sched_setaffinity(0, a.cpus)
        request = support.read(a.out / 'REQUEST.json')
        mode = request['mode']
        seed = a.out / 'diagnostic-seed/CACHE_SEED.json'
        started = time.monotonic()
        from collection.mpm import grouped
        asynchronous = a.arm.endswith('-async4')
        with diagnostic_writer(grouped) if asynchronous else nullcontext() as writer:
            four.worker(a.out / 'worker', mode, request['uuid'], a.parent,
                        training_only=True, cache_seed=seed, cache_seed_sha=support.sha(seed))
        r = resource.getrusage(resource.RUSAGE_SELF)
        support.write(a.out / 'METRICS.json', dict(worker_seconds=time.monotonic()-started,
            cpu_seconds=r.ru_utime+r.ru_stime, peak_rss_kib=r.ru_maxrss,
            async_writer_threads=4 if asynchronous else 0,
            peak_owned_queue_bytes=writer.peak_pending_bytes if asynchronous else 0,
            cpu_affinity=sorted(os.sched_getaffinity(0))))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('Run only within an allocated Slurm job')
    if len(set(a.arms)) != len(a.arms) or a.arms[0] not in ('cpu8-group4', 'cpu32-group4'):
        raise ValueError('Unique arms with the unchanged baseline first required')
    plan = collect.inspect(a.plan)
    four.require_training_plan(dict(split='train', evaluation='train'), plan['cases'])
    available = sorted(os.sched_getaffinity(0))
    for arm in a.arms:
        for lane in (0, 1):
            affinity(available, ARMS[arm][0], lane)
    a.out.mkdir(parents=True, exist_ok=False)
    support.write(a.out / 'REQUEST.json', dict(job=os.environ['SLURM_JOB_ID'],
        bundle_sha256=support.sha(bundle/'BUNDLE.json'), driver_sha256=support.sha(__file__),
        plan_sha256=support.sha(a.plan/'PLAN.json'), arms=a.arms, available_cpus=available,
        reference=str(a.reference.resolve()), diagnostic_only=True, production_approved=False,
        async_writer_sha256=support.sha(Path(__file__).with_name('mpm_async_io.py')),
        production_reservation=False, numerical_tolerances_changed=False))
    results = []
    for arm in a.arms:
        processes = []
        total_cpus, size = ARMS[arm]
        mode = dict(name=f'fused{size}', size=size, block=64, fused=True)
        try:
            for lane in (0, 1):
                old = support.read(a.reference/f'lane-{lane}/reuse-cold/KERNEL_CACHE.json')['identity']
                uuid = old['device_uuid']
                collect.gpu_status(uuid)
                out = a.out/arm/f'lane-{lane}'; out.mkdir(parents=True)
                seed = out/'diagnostic-seed'; seed.mkdir()
                origin = a.reference/f'full-lane-{lane}/cold/kernel-cache'
                shutil.copytree(origin, seed/'kernel-cache')
                four.record_cache_seed(seed, uuid=uuid, bundle_sha256=support.sha(bundle/'BUNDLE.json'),
                    warp_version=old['warp'], device_arch=old['device_arch'], mode=mode)
                unit = out/'worker'; unit.mkdir()
                shutil.copytree(a.plan, unit/'plan')
                support.write(unit/'ENTRY.json', dict(split='train', evaluation='train', name=arm))
                cpus = affinity(available, total_cpus, lane)
                support.write(out/'REQUEST.json', dict(uuid=uuid, mode=mode, cpus=cpus,
                    seed_origin=str(origin), production_qualification=False))
                log = (out/'worker.log').open('x')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, OPENBLAS_NUM_THREADS='1',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
                started = time.monotonic()
                p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'worker',
                    '--bundle', str(bundle), '--reference', str(a.reference), '--plan', str(a.plan),
                    '--out', str(out), '--arm', arm, '--lane', str(lane), '--parent', str(os.getpid()),
                    '--cpus', json.dumps(cpus)], env=env, stdout=log, stderr=subprocess.STDOUT)
                processes.append((p, log, out, lane, started))
            deadline = time.monotonic()+1200
            while any(p.poll() is None for p, *_ in processes):
                if time.monotonic() > deadline:
                    raise TimeoutError('Bounded arm deadline')
                time.sleep(1)
            for p, log, out, lane, started in processes:
                if p.returncode:
                    raise RuntimeError(f'{arm} lane {lane} failed: inspect {out}/worker.log')
                metrics = support.read(out/'METRICS.json')
                # Read every journal/raw and matched-control outcome, not just timing.
                for c in plan['cases']:
                    attempt = collect.verify_attempt(out/'worker/plan/cases'/c['label']/'attempts/attempt-00000')
                    expected = 'placement_rejected' if c['provenance'].get('placement_reasons') else 'complete'
                    if attempt['status'] != expected:
                        raise ValueError('Unexpected attempt status: '+c['label'])
                    if expected == 'complete' and not attempt['quality']['integrity_pass']:
                        raise ValueError('Integrity failure: '+c['label'])
                    if c['kind'] == 'matched_control' and (
                            not attempt['quality']['integrity_pass'] or
                            attempt['quality']['eligible_for_wall_free_export'] != c['provenance']['expected_wall_free']):
                        raise ValueError('Matched-control outcome changed')
                comparisons = []
                if arm != a.arms[0]:
                    for c in plan['cases']:
                        if c['kind'] != 'coverage' or c['provenance'].get('placement_reasons'):
                            continue
                        suffix = Path('worker/plan/cases')/c['label']/'attempts/attempt-00000/raw.h5'
                        comparisons.append(dict(label=c['label'], **four.compare(
                            a.out/a.arms[0]/f'lane-{lane}'/suffix, out/suffix)))
                compiled = [s for s in (out/'worker.log').read_text().splitlines() if '(compiled)' in s]
                row = dict(arm=arm, lane=lane, **metrics, compiled_log_lines=compiled,
                    passed=not compiled and all(c['passed'] for c in comparisons), comparisons=comparisons,
                    production_approved=False)
                support.write(out/'RESULT.json', row); results.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != 'comparisons'}), flush=True)
            support.write(a.out/(arm+'.json'), dict(results=results[-2:]))
        except BaseException as exc:
            support.write(a.out/'FAILURE.json', dict(arm=arm, error=repr(exc)))
            raise
        finally:
            for p, log, *_ in processes:
                if p.poll() is None:
                    p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait()
                log.close()
    passed = all(r['passed'] for r in results)
    support.write(a.out/'RESULT.json', dict(passed=passed, arms=results, production_approved=False))
    if not passed:
        raise ValueError('Capacity sweep contains failed numerical comparisons; not approval')


if __name__ == '__main__':
    main()
