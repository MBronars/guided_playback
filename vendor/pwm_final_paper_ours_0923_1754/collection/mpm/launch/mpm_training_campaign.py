"""One two/four-PRO allocation: training-only startup checks, then collection.

Reuses frozen production physics and bounded wave orchestration. No evaluation
approval, threshold relaxation, automatic retries, or second scheduler job.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import mpm_ada as qualification
import mpm_campaign as base
import mpm_four as four
import mpm_scale as support
import mpm_training as training
import mpm_settings as settings
import mpm_async_io as async_io

MODE = dict(name='fused4', size=4, block=64, fused=True)


def contract(cycle_start=70000, cycles=12500, gpu_count=4):
    if type(gpu_count) is not int or gpu_count not in (2,4):
        raise ValueError('Two or four PRO GPUs required')
    if (type(cycle_start) is not int or type(cycles) is not int or cycle_start<0
            or cycles<gpu_count or cycles>12500 or cycles%gpu_count or cycle_start+cycles>100001):
        raise ValueError('Bounded lane cycle range must fit the pinned sampler')
    return dict(scope='train-1-to-4-only', gpu_count=gpu_count, mode=MODE,
        background_writer=async_io.CONTRACT,
        cases_per_gpu=20, serial_repeats=2, repeated_case=3,
        position_rms_m=1e-4, wall_event_step_tolerance=2,
        exact_contact_presence=True, minimum_speedup=1.05,
        evaluation_approved=False, cycle_start=cycle_start, cycles=cycles,
        startup_then_collection_same_allocation=True,
        first_wave_gate='all-allocated-lanes-automated-quality-and-consumer-checks',
        walltime_seconds=172800, shutdown_reserve_seconds=18000)


def entries(index, gpu_count=4, training_only=True, *, cycle_start=70000, cycles=12500):
    contract(cycle_start,cycles,gpu_count)
    if training_only is not True:
        raise ValueError('Training scope required')
    if type(index) is not int or not 0 <= index < cycles//gpu_count:
        raise ValueError('Outside bounded training schedule')
    return [dict(name=f'train-{cycle_start+gpu_count*index+i}', split='train', evaluation='train',
                 cycle=cycle_start+gpu_count*index+i, lane=i) for i in range(gpu_count)]


def freeze(bundle, source, evidence, cycle_start=70000, cycles=12500, recipe=None):
    if recipe is not None:
        recipe=settings.normalize(recipe)
        cycles=settings.cycles(recipe)
    gpu_count=4 if recipe is None else recipe['hardware']['gpus']
    scope=contract(cycle_start,cycles,gpu_count)
    # Shared freeze binds all 20 prepared inputs and both preserved PRO serial
    # references. It copies source/ byte-for-byte from the live frozen bundle.
    qualification.freeze(bundle, source, evidence)
    for name in ('mpm_four.py', 'mpm_training_campaign.py', 'mpm_training_campaign.sbatch','mpm_settings.py','mpm_async_io.py'):
        shutil.copy2(Path(__file__).parent/name, bundle/name)
    config = support.read(bundle/'BUNDLE.json')
    config.pop('ada_contract')
    config.pop('qualification',None)  # Superseded full-scope historical metadata.
    config.pop('recipe_profile',None);config.pop('recipe_sha256',None)
    config.update(schema='mpm-pro-training-campaign-v3', training_campaign=scope,
        gpu_count=gpu_count, training_only=True, cycle_start=cycle_start, training_cycles=cycles,
        retained_training_target=min(500000,60*cycles),
        maximum_training_attempts=60*cycles, shutdown_reserve_seconds=18000,
        maximum_admission_seconds=162000)
    config['configuration_schema']=settings.VERSION
    if recipe is not None:
        config.update(recipe_profile=recipe,recipe_sha256=settings.identity(recipe),
            maximum_training_attempts=recipe['max_attempts'],
            retained_training_target=recipe['target_retained'] or recipe['max_attempts'],
            maximum_admission_seconds=recipe['hardware']['walltime_hours']*3600-18000)
    config['files'] = {str(p.relative_to(bundle)):support.sha(p)
                       for p in bundle.rglob('*') if p.is_file() and p.name!='BUNDLE.json'}
    support.write(bundle/'BUNDLE.json', config, replace=True)


def verify(bundle):
    config = base.verify(bundle)
    recipe=config.get('recipe_profile')
    target=min(500000,60*config['training_cycles']) if recipe is None else settings.normalize(recipe)['target_retained'] or recipe['max_attempts']
    maximum=60*config['training_cycles'] if recipe is None else recipe['max_attempts']
    admission=162000 if recipe is None else recipe['hardware']['walltime_hours']*3600-18000
    if (support.sha(__file__) != config['files']['mpm_training_campaign.py']
            or config['training_campaign'] != contract(config['cycle_start'],config['training_cycles'],config['gpu_count'])
            or config['training_only'] is not True
            or config['shutdown_reserve_seconds'] != 18000
            or config['maximum_admission_seconds'] != admission
            or config['maximum_training_attempts'] != maximum
            or config['retained_training_target'] != target):
        raise ValueError('Changed PRO training contract')
    if recipe is not None and (config['configuration_schema']!=settings.VERSION
            or settings.normalize(recipe)!=recipe or config['recipe_sha256']!=settings.identity(recipe)
            or config['training_cycles']!=settings.cycles(recipe)
            or config['gpu_count']!=recipe['hardware']['gpus']):
        raise ValueError('Changed collection configuration')
    for name, sha in config['bound_references'].items():
        if support.sha(name) != sha: raise ValueError('Changed reference: '+name)
    training.same_sources(config, support.read(Path(config['source_bundle'])/'BUNDLE.json'))
    return config


def discover(config):
    return four.discover(config)


def deadline(job, now):
    """Use scheduler epoch timestamps, never timezone-dependent display strings."""
    if str(job['job_id']) != os.environ['SLURM_JOB_ID'] or job['job_state'] != ['RUNNING']:
        raise ValueError('Wrong/non-running Slurm job')
    end = job['end_time']
    if not end['set'] or end['infinite']:
        raise ValueError('Finite Slurm deadline required')
    value = end['number']
    if not now+18000 < value <= now+172800:
        raise ValueError('Deadline outside approved allocation/reserve')
    return value


def qualify(bundle, root, config):
    if config.get('recipe_profile') and not settings.capability(config['recipe_profile'])['launch_supported']:
        raise ValueError('Evaluation preview is not approval to launch collection')
    uuids = discover(config)
    root.mkdir(parents=True, exist_ok=False)
    old = support.read(Path(config['evidence'])/'REQUEST.json')['cases']
    selected = [old[i] for i in training.training_indices(old)]
    for i, item in enumerate(selected):
        dest = root/f'merged/cases/qual-{i:02d}'
        dest.mkdir(parents=True)
        for name in ('initial.npz', 'MANIFEST.json'):
            shutil.copy2(Path(item['root'])/'cases'/item['case']['label']/name, dest/name)
    support.write(root/'REQUEST.json', dict(bundle_sha256=support.sha(bundle/'BUNDLE.json'),
        contract=config['training_campaign'], uuids=uuids, cases=selected, evidence=config['evidence'],
        job=os.environ['SLURM_JOB_ID'], started_epoch=time.time()))
    env = base.environment(bundle, config)
    driver = str(bundle/'mpm_training_campaign.py')
    commands = [[config['genesis_python'], driver, 'lane', '--bundle', str(bundle),
                 '--root', str(root), '--lane', str(i), '--uuid', u, '--parent', str(os.getpid())]
                for i, u in enumerate(uuids)]
    support.children(commands, [dict(env, CUDA_VISIBLE_DEVICES=u) for u in uuids],
                     [root/f'lane-{i}.log' for i in range(config['gpu_count'])], timeout=7200)
    lanes = [support.read(root/f'lane-{i}/RESULT.json') for i in range(config['gpu_count'])]
    if not all(r['passed'] and r['mode']==MODE and qualification.lane_passes(r) for r in lanes):
        raise ValueError('Training qualification failed; no collection')
    subprocess.run([config['genesis_python'], driver, 'consumer', '--bundle', str(bundle), '--root', str(root)],
                   env=dict(env, CUDA_VISIBLE_DEVICES=''), check=True, timeout=1800)
    verify(bundle)
    support.write(root/'TRAINING_PASS.json', dict(scope=config['training_campaign'], mode=MODE, lanes=lanes,
        bundle_sha256=support.sha(bundle/'BUNDLE.json'), evaluation_approved=False,
        consumer=support.read(root/'CONSUMER.json'),
        files={str(p.relative_to(root)):support.sha(p) for p in root.rglob('*')
               if p.is_file() and 'kernel-cache' not in p.parts}))


def qualified(root, bundle):
    from collection.common import check_hashes
    r = support.read(root/'TRAINING_PASS.json')
    config=support.read(bundle/'BUNDLE.json')
    count=config.get('gpu_count',4)
    if (r['scope']!=config['training_campaign'] or r['mode']!=MODE or r['evaluation_approved'] is not False
            or r['bundle_sha256']!=support.sha(bundle/'BUNDLE.json') or len(r['lanes'])!=count):
        raise ValueError('Wrong training approval')
    check_hashes(root, r['files'])
    request = support.read(root/'REQUEST.json')
    if len(set(request['uuids']))!=count or [l['uuid'] for l in r['lanes']]!=request['uuids']:
        raise ValueError('Wrong qualification GPUs')
    for i, lane in enumerate(r['lanes']):
        if f'lane-{i}/CACHE_SEED.json' not in r['files']:
            raise ValueError('Qualification did not bind compiled kernels')
        if (lane!=support.read(root/f'lane-{i}/RESULT.json') or not lane['passed']
                or lane['mode']!=MODE or not qualification.lane_passes(lane)):
            raise ValueError('A training lane did not pass')
        isolation = support.read(root/f'lane-{i}/ISOLATION.json')
        if (isolation['grid']!=dict(grid_cells=1000000, mutation_rejected=True, velocity_and_events_bit_exact=True)
                or isolation['failure_injection']!=dict(healthy_world_unchanged=True, nonfinite_stopped=True, physics_steps=0)):
            raise ValueError('Isolation did not pass')
    if len(r['consumer'])!=count or any(c['profiles']!=2 or c['episodes_per_profile']<1 for c in r['consumer']):
        raise ValueError('Incomplete consumer checks')
    return r


def approve_pilot(root, bundle, results):
    config=support.read(bundle/'BUNDLE.json');first=config['cycle_start']
    count=config.get('gpu_count',4)
    if len(results)!=count or sorted(r['entry']['cycle'] for r in results)!=list(range(first,first+count)):
        raise ValueError('First wave is not all declared training chunks')
    for r in results:
        four.require_training_plan(r['entry'], [])
        expected=60 if 'recipe_profile' not in config else settings.attempts(config['recipe_profile'],r['entry']['cycle']-first)
        if (r['status']!='complete' or r['attempts']!=expected or not 0<r['retained']<=expected
                or r['model']['profiles']!=2 or r['model']['episodes_per_profile']!=r['retained']
                or set(r['attempt_status'])-{'complete','placement_rejected'}):
            raise ValueError('First-wave review failed')
    # User authorized startup checks followed by collection in this allocation.
    # Existing model/physics already visually reviewed on PRO; no claim of a new
    # manual visual inspection. Every chunk still generates GIFs for inspection.
    support.write(root/'PILOT_APPROVED.json', dict(pilot_sha256=support.sha(root/'PILOT_READY.json'),
        bundle_sha256=support.sha(bundle/'BUNDLE.json'), manual_visual_review=False,
        approval='Automated first-wave controls, integrity, filters and both consumers passed on all allocated lanes'))


class Controller:
    """Pass per-bundle scheduling through the shared runner without globals."""
    __file__=__file__
    verify=staticmethod(verify)
    qualified=staticmethod(qualified)
    discover=staticmethod(discover)
    approve_pilot=staticmethod(approve_pilot)

    def __init__(self,config): self.config=config

    def entries(self,index,gpu_count=4,training_only=True):
        return entries(index,gpu_count,training_only,cycle_start=self.config['cycle_start'],cycles=self.config['training_cycles'])


def prepare_configured(unit,entry,config,limit=None):
    recipe=config.get('recipe_profile')
    if recipe is None:return base.prepare_plan(unit,entry,config)
    from collection.mpm import collect
    r=settings.normalize(recipe);banks=Path(config['banks'])
    if entry['split']!=r['split'] or entry['evaluation']!=r['evaluation']:
        raise ValueError('Entry does not match configured split/evaluation')
    limit=settings.attempts(r,entry['cycle']-config['cycle_start']) if limit is None else limit
    collect.prepare(unit/'plan',recipe_version='fresh-shapes-v1',bank=banks/settings.bank_role(r),
        pool=banks/'pool',bank_role=settings.bank_role(r),split_manifest=banks/'SPLITS.json',
        dataset_split=r['split'],evaluation=r['evaluation'],cycle_start=entry['cycle'],cycles=1,
        specification_factory=lambda metadata:settings.specifications(metadata,recipe=r,cycle=entry['cycle'],limit=limit))
    support.write(unit/'ENTRY.json',entry)
    support.write(unit/'RECIPE.json',dict(recipe=r,sha256=settings.identity(r),coverage_attempts=limit))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze','verify','run','lane','consumer','prepare','preview','worker','review','audit'])
    for name in ('bundle','source','evidence','root','unit'):
        parser.add_argument('--'+name, type=Path, required=name=='bundle')
    for name in ('lane','parent'): parser.add_argument('--'+name, type=int)
    for name in ('uuid','entry','mode'): parser.add_argument('--'+name)
    parser.add_argument('--cycle-start',type=int,default=70000)
    parser.add_argument('--cycles',type=int,default=12500)
    parser.add_argument('--recipe-json',type=json.loads)
    parser.add_argument('--preview-count',type=int,default=12)
    parser.add_argument('--cache-seed',type=Path)
    parser.add_argument('--cache-seed-sha')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    sys.path.insert(0,str((args.source if args.command=='freeze' else bundle)/'source'))
    if args.command=='freeze': freeze(bundle,args.source,args.evidence,args.cycle_start,args.cycles,args.recipe_json);return
    config = verify(bundle)
    os.environ['PWM_MPM_REFERENCE_ROOT']=config['reference']
    os.environ['PWM_MODEL_ROOT']=str(bundle/'source')
    if args.command=='verify': print('PRO training bundle verified',flush=True)
    elif args.command=='preview':
        recipe=settings.normalize(args.recipe_json)
        if args.unit is None or not 1<=args.preview_count<=60:raise ValueError('preview requires a new unit directory and 1..60 cases')
        preview=dict(config,recipe_profile=recipe)
        entry=dict(name='preview',split=recipe['split'],evaluation=recipe['evaluation'],cycle=config['cycle_start'],lane=0)
        prepare_configured(args.unit,entry,preview,limit=args.preview_count)
        support.write(args.unit/'PREVIEW.json',dict(preview_only=True,production_reservation=False,
            collection_approved=False,recipe=recipe,capability=settings.capability(recipe)))
    elif args.command=='lane': qualification.lane(args.root,args.lane,args.uuid,args.parent,mode=MODE,async_io=True)
    elif args.command=='consumer': qualification.consumer(args.root,gpu_count=config['gpu_count'],mode=MODE)
    elif args.command=='prepare':
        entry=json.loads(args.entry)
        if entry not in Controller(config).entries((entry['cycle']-config['cycle_start'])//config['gpu_count'],config['gpu_count']): raise ValueError('Undeclared training entry')
        prepare_configured(args.unit,entry,config)
    elif args.command=='worker':
        if json.loads(args.mode)!=MODE: raise ValueError('Unqualified collection mode')
        if args.cache_seed is None or args.cache_seed_sha is None:
            raise ValueError('Production worker requires qualified compiled kernels')
        four.worker(args.unit,MODE,args.uuid,args.parent,training_only=True,
                    cache_seed=args.cache_seed,cache_seed_sha=args.cache_seed_sha,async_io=True)
    elif args.command=='review': base.review(args.unit)
    elif args.command=='audit':
        rows=[base.completed(f.parent) for f in sorted(args.root.glob('chunks/*/COMPLETE.json'))]
        if not rows:raise ValueError('No completed chunks to audit')
        seen=set()
        for r in rows:
            four.require_training_plan(r['entry'],[])
            if r['entry']['cycle'] in seen:raise ValueError('Duplicate cycle')
            seen.add(r['entry']['cycle'])
            if not config['cycle_start']<=r['entry']['cycle']<config['cycle_start']+config['training_cycles']:
                raise ValueError('Chunk outside reserved cycle range')
        print(json.dumps(dict(hash_audit_passed=True,completed_chunks=len(rows),
            retained=sum(r['retained'] for r in rows),attempts=sum(r['attempts'] for r in rows),
            evaluation_approved=False,accepted_for_training=False)),flush=True)
    else:
        if 'recipe_profile' in config and not settings.capability(config['recipe_profile'])['launch_supported']:
            raise ValueError('Evaluation preview cannot be run as production collection')
        job = json.loads(subprocess.check_output(['scontrol','--json','show','job',os.environ['SLURM_JOB_ID']],text=True,timeout=60))['jobs'][0]
        end = deadline(job,time.time())
        args.root.mkdir(parents=True,exist_ok=False)
        support.write(args.root/'ALLOCATION.json',dict(job=job['job_id'],deadline_epoch=end,
            started_epoch=time.time(),contract=config['training_campaign'],bundle_sha256=support.sha(bundle/'BUNDLE.json')))
        try:
            qualify(bundle,args.root/'qualification',config)
            four.run(bundle,args.root/'collection',args.root/'qualification',end,controller=Controller(config))
        except BaseException as exc:
            support.write(args.root/'FAILURE.json',dict(error=repr(exc),time_epoch=time.time()))
            raise


if __name__=='__main__': main()
