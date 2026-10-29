"""Bounded two-Ada training qualification and 120-attempt fresh-data pilot.

Reuse frozen production physics, never a PRO GPU's architecture approval.
No production submission, evaluation generation, retries, or relaxed gates.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import mpm_campaign as base
import mpm_four as four
import mpm_scale as support
import mpm_training as training

MODE = dict(name='launch64', size=4, block=64, fused=False)


def contract():
    return dict(scope='train-1-to-4-only', gpu_count=2, gpu_name='RTX 6000 Ada',
                cases=20, serial_repeat_case=3, serial_repeats=2,
                position_rms_m=1e-4, wall_event_step_tolerance=2,
                exact_contact_presence=True, minimum_speedup=1.05,
                cross_architecture_reference=True, mode=MODE, wall_fusion_approved=False,
                cycles=[60000, 60001], maximum_pilot_attempts=120,
                automatic_production=False, evaluation_approved=False)


def validate_devices(devices):
    if (len(devices) != 2 or len({d['uuid'] for d in devices}) != 2
            or any('RTX 6000 Ada' not in d['name'] for d in devices)):
        raise ValueError('Exactly two distinct allocated RTX 6000 Ada GPUs required')
    return [d['uuid'] for d in devices]


def pilot_entries():
    return [dict(name=f'train-{cycle}', split='train', evaluation='train',
                 cycle=cycle, lane=i) for i, cycle in enumerate(contract()['cycles'])]


def freeze(bundle, source, evidence):
    from collection.common import check_hashes
    previous = support.read(source/'BUNDLE.json')
    check_hashes(source, previous['files'])
    request = support.read(evidence/'REQUEST.json')
    selected = [request['cases'][i] for i in training.training_indices(request['cases'])]
    # Copy only immutable source artifacts, not the previous approval/config.
    bundle.mkdir(parents=True, exist_ok=False)
    for name in previous['files']:
        dest = bundle/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, dest)
    # The training controller supplies the sole supported sbatch. The old Ada
    # pilot shell is neither needed for qualification nor a production entry.
    for name in ('mpm_ada.py',):
        shutil.copy2(Path(__file__).parent/name, bundle/name)
    bound = {str(source/'BUNDLE.json'): support.sha(source/'BUNDLE.json'),
             str(evidence/'REQUEST.json'): support.sha(evidence/'REQUEST.json')}
    for i, item in enumerate(selected):
        for name, key in [('initial.npz', 'initial_sha256'), ('MANIFEST.json', 'manifest_sha256')]:
            path = Path(item['root'])/'cases'/item['case']['label']/name
            if support.sha(path) != item['case'][key]:
                raise ValueError('Changed qualification input')
            bound[str(path)] = support.sha(path)
        for lane in range(2):
            folder = evidence/f'lane-{lane}/serial/{i:02d}'
            training.verify_attempt(folder, item['case'])
            bound[str(folder/'raw.h5')] = support.sha(folder/'raw.h5')
    config = dict(previous, schema='mpm-ada-bounded-pilot-v1', ada_contract=contract(),
                  source_bundle=str(source), evidence=str(evidence), bound_references=bound,
                  retained_training_target=None, maximum_training_attempts=120)
    config['files'] = {str(p.relative_to(bundle)): support.sha(p)
                       for p in bundle.rglob('*') if p.is_file()}
    support.write(bundle/'BUNDLE.json', config)


def verify(bundle):
    config = base.verify(bundle)
    if (config['ada_contract'] != contract()
            or support.sha(__file__) != config['files']['mpm_ada.py']):
        raise ValueError('Changed Ada qualification contract/driver')
    for name, sha in config['bound_references'].items():
        if support.sha(name) != sha:
            raise ValueError('Changed architecture reference: '+name)
    training.same_sources(config, support.read(Path(config['source_bundle'])/'BUNDLE.json'))
    return config


def lane(root, lane_id, uuid, parent, *, mode=None, async_io=False):
    mode = MODE if mode is None else mode
    if mode not in (MODE, four.QUALIFICATION_MODES[0]):
        raise ValueError('Unreviewed qualification mode')
    four.bind(uuid, parent)
    from collection.lease import gpu_lease
    from collection.mpm import grouped
    simulate=grouped.simulate
    if async_io:
        from mpm_async_io import simulate as background_simulate
        simulate=lambda *a,**kw:background_simulate(grouped,*a,**kw)
    import warp as wp
    request = support.read(root/'REQUEST.json')
    selected = request['cases']
    if training.training_indices(selected) != list(range(20)):
        raise ValueError('Training strata changed')
    out = root/f'lane-{lane_id}'
    out.mkdir()
    wp.config.kernel_cache_dir = str(out/'kernel-cache')
    with gpu_lease(uuid):
        # Ada v1 failed fused/original bit equality. Do not waive that test or
        # use fusion: this mode executes the original separate wall kernels.
        grid = (grouped.grid_test(selected[0]['root'], selected[0]['case']) if mode['fused']
                else dict(status='not_applicable_unfused', wall_fusion_approved=False))
        if mode['fused'] and grid != dict(grid_cells=1000000, mutation_rejected=True, velocity_and_events_bit_exact=True):
            raise ValueError('Fused grid check failed')
        failure = grouped.corruption_test(selected[0]['root'], [s['case'] for s in selected[:2]],
                                          block=mode['block'], fused=mode['fused'])
        isolation = dict(grid=grid, failure_injection=failure)
        support.write(out/'ISOLATION.json', isolation)
        if failure != dict(healthy_world_unchanged=True, nonfinite_stopped=True, physics_steps=0):
            raise ValueError('Grid/failure isolation failed')
        serial_seconds = 0.
        for i, item in enumerate(selected):
            r = grouped.simulate(item['root'], [item['case']], [out/f'serial/{i:02d}'],
                                 serial=True, uuid=uuid)
            serial_seconds += r['seconds']
            gc.collect()
            print(json.dumps(dict(lane=lane_id, mode='serial', completed=i+1, total=20)), flush=True)
        repeats = []
        for j in range(2):
            item = selected[3]
            dest = out/f'serial-repeat/03-{j}'
            grouped.simulate(item['root'], [item['case']], [dest], serial=True, uuid=uuid)
            training.verify_attempt(dest, item['case'])
            repeats.append(four.compare(out/'serial/03/raw.h5', dest/'raw.h5'))
            gc.collect()
        comparisons, cross_architecture, candidate_repeats = [], [], []
        seconds = 0.
        for start in range(0, 20, 4):
            cases = [dict(selected[i]['case'], label=f'qual-{i:02d}') for i in range(start, start+4)]
            outputs = [out/mode['name']/f'{i:02d}' for i in range(start, start+4)]
            r = simulate(root/'merged', cases, outputs, block=mode['block'], fused=mode['fused'], uuid=uuid)
            seconds += r['seconds']
            for i, dest in zip(range(start, start+4), outputs):
                training.verify_attempt(out/f'serial/{i:02d}', selected[i]['case'])
                training.verify_attempt(dest, selected[i]['case'])
                comparisons.append(four.compare(out/f'serial/{i:02d}/raw.h5', dest/'raw.h5'))
                # Both preserved PRO serial references, not a chosen best match.
                for old_lane in range(2):
                    ref = Path(request['evidence'])/f'lane-{old_lane}/serial/{i:02d}/raw.h5'
                    cross_architecture.append(dict(case=i, reference_lane=old_lane,
                        serial=four.compare(ref, out/f'serial/{i:02d}/raw.h5'),
                        candidate=four.compare(ref, dest/'raw.h5')))
                if i == 3:
                    candidate_repeats = [four.compare(out/f'serial-repeat/03-{j}/raw.h5', dest/'raw.h5')
                                         for j in range(2)]
            gc.collect()
            print(json.dumps(dict(lane=lane_id, mode=mode['name'], completed=start+4, total=20)), flush=True)
        result = dict(uuid=uuid, mode=mode, serial_seconds=serial_seconds, grouped_seconds=seconds,
                      speedup=serial_seconds/seconds, comparisons=comparisons,
                      repeats=repeats, candidate_repeats=candidate_repeats,
                      cross_architecture=cross_architecture)
        result['passed'] = lane_passes(result)
        if result['passed']:
            four.record_cache_seed(out,uuid=uuid,bundle_sha256=request['bundle_sha256'],
                warp_version=wp.__version__,device_arch=wp.get_device('cuda:0').arch,mode=mode)
        support.write(out/'RESULT.json', result)
        print(json.dumps(dict(lane=lane_id, passed=result['passed'], speedup=result['speedup'])), flush=True)


def lane_passes(result):
    import math
    cross = result['cross_architecture']
    if (len(result['comparisons']) != 20 or len(result['repeats']) != 2
            or len(result['candidate_repeats']) != 2
            or [(r['case'], r['reference_lane']) for r in cross] != [(i, j) for i in range(20) for j in range(2)]
            or not math.isfinite(result['speedup']) or result['speedup'] < 1.05):
        return False
    rows = result['comparisons']+result['repeats']+result['candidate_repeats']
    rows += [r[k] for r in cross for k in ('serial', 'candidate')]
    return all(c['passed'] and four.comparison_passes(c) for c in rows)


def run(bundle, root, config):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Explicit Slurm allocation required')
    code = 'import json,warp as w;w.init();print("DEVICES="+json.dumps([dict(uuid=d.uuid,name=d.name) for d in w.get_cuda_devices()]))'
    output = subprocess.check_output([config['genesis_python'], '-c', code], text=True, timeout=60)
    devices = json.loads(next(s[8:] for s in output.splitlines() if s.startswith('DEVICES=')))
    uuids = validate_devices(devices)
    if shutil.disk_usage(root.parent).free < 500*1024**3:
        raise ValueError('Shared SSD reserve below 500 GiB')
    root.mkdir(parents=True, exist_ok=False)
    start = time.time()
    evidence = Path(config['evidence'])
    old = support.read(evidence/'REQUEST.json')['cases']
    selected = [old[i] for i in training.training_indices(old)]
    for i, item in enumerate(selected):
        # Only initial/manifest needed: never copy historical attempts into inputs.
        dest = root/f'merged/cases/qual-{i:02d}'
        dest.mkdir(parents=True)
        for name in ('initial.npz', 'MANIFEST.json'):
            shutil.copy2(Path(item['root'])/'cases'/item['case']['label']/name, dest/name)
    support.write(root/'REQUEST.json', dict(bundle_sha256=support.sha(bundle/'BUNDLE.json'),
        contract=contract(), devices=devices, cases=selected, evidence=str(evidence),
        job=os.environ['SLURM_JOB_ID'], started_epoch=start))
    env = base.environment(bundle, config)
    def command(action, *args):
        return [config['genesis_python'], str(bundle/'mpm_ada.py'), action, '--bundle', str(bundle), *args]
    def status(state):
        support.write(root/'STATUS.json', dict(state=state, elapsed_seconds=time.time()-start), replace=True)
    try:
        status('qualifying')
        commands = [command('lane', '--root', str(root), '--lane', str(i), '--uuid', u,
                            '--parent', str(os.getpid())) for i, u in enumerate(uuids)]
        support.children(commands, [dict(env, CUDA_VISIBLE_DEVICES=u) for u in uuids],
                         [root/f'lane-{i}.log' for i in range(2)], timeout=7200)
        results = [support.read(root/f'lane-{i}/RESULT.json') for i in range(2)]
        if not all(r['passed'] and lane_passes(r) for r in results):
            raise ValueError('Ada accuracy/repeatability/speed gate failed; no fresh pilot')
        status('consumer_checks')
        # Sequential CPU reviews bound host memory on the fragmented Ada node.
        subprocess.run(command('consumer', '--root', str(root)), env=dict(env, CUDA_VISIBLE_DEVICES=''),
                       check=True, timeout=1200)
        verify(bundle)
        support.write(root/'QUALIFICATION_PASS.json', dict(bundle_sha256=support.sha(bundle/'BUNDLE.json'),
            contract=contract(), devices=devices, lanes=results, consumer=support.read(root/'CONSUMER.json'),
            accepted_for_training=False, evaluation_approved=False))
        pilot_start = time.time()
        units = [root/'pilot/chunks'/e['name'] for e in pilot_entries()]
        status('preparing_pilot')
        for unit, entry in zip(units, pilot_entries()):
            subprocess.run(command('prepare', '--unit', str(unit), '--entry', json.dumps(entry)),
                           env=dict(env, CUDA_VISIBLE_DEVICES=''), check=True, timeout=1200)
        status('collecting_pilot')
        support.children([command('worker', '--unit', str(unit), '--uuid', u, '--parent', str(os.getpid()))
                          for unit, u in zip(units, uuids)],
                         [dict(env, CUDA_VISIBLE_DEVICES=u) for u in uuids],
                         [root/f'pilot-worker-{i}.log' for i in range(2)], timeout=1800)
        status('reviewing_pilot')
        reviewed = []
        for unit in units:
            subprocess.run(command('review', '--unit', str(unit)), env=dict(env, CUDA_VISIBLE_DEVICES=''),
                           check=True, timeout=1200)
            receipt = base.completed(unit)
            base.publish(root/'pilot', unit, receipt)
            reviewed.append(receipt)
        elapsed = time.time()-pilot_start
        retained = sum(r['retained'] for r in reviewed)
        support.write(root/'PILOT_COMPLETE.json', dict(attempts=sum(r['attempts'] for r in reviewed),
            retained=retained, elapsed_seconds=elapsed, retained_per_hour=retained*3600/elapsed,
            qualification_seconds=pilot_start-start, accepted_for_training=False,
            automatic_production=False, dataset_root=str(root/'pilot/datasets/train'),
            chunks={str(unit):support.sha(unit/'COMPLETE.json') for unit in units},
            gifs=[str(unit/'samples.gif') for unit in units]))
        status('complete_awaiting_visual_review')
    except BaseException as exc:
        support.write(root/'FAILURE.json', dict(error=repr(exc), elapsed_seconds=time.time()-start))
        status('failed_no_expansion')
        raise


def consumer(root, *, gpu_count=2, mode=None):
    mode = MODE if mode is None else mode
    from collection.mpm.export import export_candidates
    from collection.mpm.wall_materials import SAND_ONLY
    from collection.mpm.review import check_model
    reviews = []
    for i in range(gpu_count):
        unit = root/f'consumer-{i}'
        export_candidates([root/f'lane-{i}'/mode['name']/f'{j:02d}/raw.h5' for j in range(20)],
                          unit/'filtered', wall_policy=SAND_ONLY)
        rows = support.read(unit/'filtered/FILTER.json')['episodes']
        kept = {r['episode_id']:Path(r['raw_path']) for r in rows if r['eligible_for_export']}
        if not kept:
            raise ValueError('No consumer survivors')
        reviews.append(check_model(unit, kept))
    support.write(root/'CONSUMER.json', reviews)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze', 'verify', 'run', 'lane', 'consumer', 'prepare', 'worker', 'review'])
    for name in ('bundle', 'source', 'evidence', 'root', 'unit'):
        parser.add_argument('--'+name, type=Path, required=name=='bundle')
    parser.add_argument('--lane', type=int)
    parser.add_argument('--parent', type=int)
    parser.add_argument('--uuid')
    parser.add_argument('--entry')
    args = parser.parse_args()
    sys.path.insert(0, str(args.bundle/'source' if args.action!='freeze' else args.source/'source'))
    if args.action == 'freeze':
        freeze(args.bundle, args.source, args.evidence)
        return
    config = verify(args.bundle)
    os.environ['PWM_MPM_REFERENCE_ROOT'] = config['reference']
    os.environ['PWM_MODEL_ROOT'] = str(args.bundle/'source')
    if args.action == 'verify': print('Frozen Ada bundle verified', flush=True)
    elif args.action == 'run': run(args.bundle, args.root, config)
    elif args.action == 'lane': lane(args.root, args.lane, args.uuid, args.parent)
    elif args.action == 'consumer': consumer(args.root)
    elif args.action == 'prepare': base.prepare_plan(args.unit, json.loads(args.entry), config)
    elif args.action == 'worker': four.worker(args.unit, MODE, args.uuid, args.parent, training_only=True)
    elif args.action == 'review': base.review(args.unit)


if __name__ == '__main__':
    main()
