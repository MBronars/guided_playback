"""Frozen fresh-MPM campaign: bounded two-GPU waves, explicit continuation.

500K means retained training episodes, not attempts or sliding windows. No test
generation, simulator optimization, implicit retries, or release promotion.
"""
import argparse
from argparse import Namespace
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import mpm_scale as support

RESERVE = 12000  # Two preparations + bounded workers + two reviews + shutdown.


def wave(index):
    if type(index) is not int or not 0 <= index < 6252:
        raise ValueError('Wave index outside the declared 750K-attempt campaign')
    if index in (1, 2):
        kinds = ('seen', 'unseen') if index == 1 else ('count_seen', 'count_unseen')
        return [dict(name=f'validation-{e}-00001', split='validation', evaluation=e,
                     cycle=1, lane=i) for i, e in enumerate(kinds)]
    train_index = index if index == 0 else index - 2
    return [dict(name=f'train-{2 * train_index + i + 1:05d}', split='train', evaluation='train',
                 cycle=2 * train_index + i + 1, lane=i) for i in range(2)]


def freeze(bundle, banks, reference, model_root):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Freeze must hide CUDA')
    here = Path(__file__).resolve().parent
    research = here/'source' if (here/'BUNDLE.json').is_file() else here.parents[2]
    sys.path.insert(0, str(research))
    from collection.common import check_hashes
    banks = banks.resolve(strict=True)
    check_hashes(banks, support.read(banks / 'COMPLETE.json')['artifacts'])
    bundle.mkdir(parents=True, exist_ok=False)
    for folder in ('collection', 'collection/mpm'):
        destination = bundle / 'source' / folder
        destination.mkdir(parents=True, exist_ok=True)
        for p in (research / folder).glob('*.py'):
            shutil.copy2(p, destination / p.name)
    for folder in ('data', 'stats'):
        shutil.copytree(model_root / folder, bundle / 'source' / folder,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('mpm_campaign.py', 'mpm_scale.py'):
        shutil.copy2(here / name, bundle / name)
    support.write(bundle / 'BUNDLE.json', dict(schema='fresh-mpm-campaign-v1',
        banks=str(banks), bank_complete_sha256=support.sha(banks / 'COMPLETE.json'),
        reference=str(reference.resolve(strict=True)), genesis_python=support.GENESIS,
        model_python=support.MODEL, retained_training_target=500000, maximum_training_attempts=750000,
        test_collection=False, accepted_for_training=False,
        files={str(p.relative_to(bundle)): support.sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}))


def verify(bundle):
    from collection.common import check_hashes
    config = support.read(bundle / 'BUNDLE.json')
    check_hashes(bundle, config['files'])
    banks = Path(config['banks'])
    if support.sha(banks / 'COMPLETE.json') != config['bank_complete_sha256']:
        raise ValueError('Bank completion changed')
    check_hashes(banks, support.read(banks / 'COMPLETE.json')['artifacts'])
    if support.sha(__file__) != config['files']['mpm_campaign.py']:
        raise ValueError('Run the frozen campaign launcher')
    from collection.mpm.wall_materials import SUPPORTED_SOURCES
    for name, sha in SUPPORTED_SOURCES.items():
        if support.sha(bundle / 'source/collection/mpm' / name) != sha:
            raise ValueError('Unsupported wall-certificate numerical source: ' + name)
    return config


def environment(bundle, config):
    return dict(os.environ, PYTHONPATH=str(bundle / 'source'),
                PWM_MODEL_ROOT=str(bundle / 'source'), PWM_MPM_REFERENCE_ROOT=config['reference'],
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')


def prepare_plan(unit, entry, config):
    from collection.mpm import collect
    banks = Path(config['banks'])
    role = entry['split'] if entry['evaluation'] in ('unseen', 'count_unseen') else 'train'
    collect.prepare(unit / 'plan', recipe_version='fresh-shapes-v1', bank=banks / role,
                    pool=banks / 'pool', bank_role=role, split_manifest=banks / 'SPLITS.json',
                    dataset_split=entry['split'], evaluation=entry['evaluation'],
                    cycles=1, cycle_start=entry['cycle'])
    support.write(unit / 'ENTRY.json', entry)


def worker(unit, uuid):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != uuid:
        raise ValueError('Worker requires its allocated full GPU UUID')
    from collection.mpm import collect
    # The Slurm cgroup owns these processes; a lost supervisor also stops this worker.
    import ctypes
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() == 1:
        raise RuntimeError('Cannot bind worker lifetime to supervisor')
    plan = collect.inspect(unit / 'plan')
    result = collect.collect(Namespace(root=unit / 'plan', uuid=uuid, execution='campaign',
        start=0, limit=len(plan['cases']), attempt=0, wall_seconds=7200, case_seconds=900,
        reuse_own_context=True))
    if result['finished'] != result['requested']:
        raise ValueError('Incomplete wave; preserve partial attempts for explicit recovery')


def quarantine_policy(rows):
    """Exclude an isolated geometric contraction; never excuse broken raw data.

    Eligibility and thresholds are unchanged. This only distinguishes an
    excluded numerical/physical outlier from a fatal storage/control failure.
    """
    import re
    bad=[r for r in rows if not r['integrity_pass']]
    if len(bad)>1 or (bad and len(bad)/len(rows)>.02):
        raise ValueError('Repeated raw integrity/collapse failures: stop for review')
    for r in bad:
        reasons=r.get('reasons',[]);collapse=r.get('collapse',{})
        if (not reasons or set(reasons)!=set(collapse.get('reasons',[]))
                or any(not re.fullmatch(r'object_[0-9]+:extreme_contraction',x) for x in reasons)
                or collapse.get('eligible') is not False
                or r.get('eligible_for_export') is not False
                or r.get('eligible_for_wall_free_export') is not False):
            raise ValueError('Raw integrity/collapse failure: stop for review')
    return dict(policy='isolated-contraction-exclusion-v1',maximum_per_chunk=1,
        maximum_fraction=.02,thresholds_changed=False,
        excluded=[{k:r[k] for k in ('episode_id','raw_path','raw_sha256','reasons')} for r in bad])


def review(unit):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Review must hide CUDA')
    from collection.common import write_once
    from collection.mpm import collect
    from collection.mpm.export import export_candidates
    from collection.mpm.wall_materials import SAND_ONLY
    from collection.mpm.review import check_model
    plan = collect.inspect(unit / 'plan')
    raws, attempts = [], []
    for case in plan['cases']:
        r = collect.verify_attempt(unit / 'plan/cases' / case['label'] / 'attempts/attempt-00000')
        if case['kind'] == 'matched_control':
            if (r['status'] != 'complete' or not r['quality']['integrity_pass'] or
                    r['quality']['eligible_for_wall_free_export'] != case['provenance']['expected_wall_free']):
                raise ValueError('Functionality control failed')
        else:
            attempts.append(dict(label=case['label'], **{k:r[k] for k in ('status',)}))
            if r['status'] == 'failed':
                raise ValueError('Simulation failure: stop campaign for diagnosis; ' + case['label'])
            if r['status'] == 'complete': raws.append(Path(r['raw_path']))
    if not raws: raise ValueError('No complete coverage raws')
    exported = export_candidates(raws, unit / 'filtered', wall_policy=SAND_ONLY)
    rows = support.read(unit / 'filtered/FILTER.json')['episodes']
    quarantine=quarantine_policy(rows)
    write_once(unit/'QUARANTINE.json',quarantine)
    kept = {r['episode_id']: Path(r['raw_path']) for r in rows if r['eligible_for_export']}
    if not kept: raise ValueError('Zero retained clips: stop for review')
    model = check_model(unit, kept)
    # A compact real-state GIF per chunk; never reuse strict-wall labels for sand.
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    import numpy as np
    selected = [r for r in rows if r['eligible_for_export']][:4]
    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4), squeeze=False)
    scenes = []
    for ax, row in zip(axes.flat, selected):
        with h5py.File(row['raw_path'], 'r') as f:
            x = f['x'][()]; materials = f['mat_type'][()][f['object_ids'][()]]
        ax.set(xlim=(0,1.5), ylim=(0,1.5), xlabel='x (m)', ylabel='z (m)', aspect='equal')
        ax.axhline(.1, c='k'); ax.axvline(.03,c='r'); ax.axvline(1.47,c='r'); ax.axhline(1.47,c='r')
        ax.set_title(row['strata']['scenario'] + '\n' + row['wall_materials']['status'])
        dots = ax.scatter(x[0,:,0],x[0,:,2],s=1,c=np.array(['#277fba','#cb4a93','#d39612'])[materials])
        scenes.append((x,dots))
    fig.tight_layout(rect=(0,0,1,.9)); frames=[]
    for t in range(0,49,3):
        for x,dots in scenes: dots.set_offsets(x[t][:,[0,2]])
        fig.suptitle(f'Actual XZ; y hidden; t={t*.0417:.4f}s; sand wall contact allowed')
        fig.canvas.draw(); frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()))
    frames[0].save(unit/'samples.gif',save_all=True,append_images=frames[1:],duration=380,loop=0)
    plt.close(fig)
    result = dict(status='complete', entry=support.read(unit/'ENTRY.json'), attempts=len(attempts),
        simulated=len(raws), retained=exported['retained'], model=model,quarantine=quarantine,
        attempt_status=dict(Counter(r['status'] for r in attempts)),
        reasons=dict(Counter(reason for r in rows if not r['eligible_for_export'] for reason in set(r['reasons']))),
        accepted_for_training=False,
        files={str(p.relative_to(unit)):support.sha(p) for p in sorted(unit.rglob('*'))
               if p.is_file() and ('kernel-cache' not in p.parts)})
    write_once(unit/'COMPLETE.json',result)


def completed(unit):
    from collection.common import check_hashes
    r = support.read(unit/'COMPLETE.json')
    if r['status'] != 'complete': raise ValueError('Invalid chunk completion')
    check_hashes(unit,r['files'])
    return r


def publish(root, unit, result):
    """Atomically expose a reviewed chunk beneath one loader root per split."""
    entry=result['entry']
    group='train' if entry['split']=='train' else 'validation-'+entry['evaluation']
    parent=root/'datasets'/group; parent.mkdir(parents=True,exist_ok=True)
    destination=parent/entry['name']
    shards=sorted((unit/'filtered/diagnostic-export').glob('*.tar'))
    if not shards: raise ValueError('No reviewed shards to publish')
    receipt=dict(chunk_complete_sha256=support.sha(unit/'COMPLETE.json'),
                 retained=result['retained'],accepted_for_training=False,
                 shards={p.name:support.sha(p) for p in shards})
    if destination.exists():
        if support.read(destination/'MANIFEST.json')!=receipt: raise ValueError('Published chunk differs')
        for name,sha in receipt['shards'].items():
            if support.sha(destination/name)!=sha: raise ValueError('Published shard changed')
        return
    stage=parent/('.'+entry['name']+'.partial')
    stage.mkdir(exist_ok=False)
    for shard in shards: os.link(shard,stage/shard.name)
    support.write(stage/'MANIFEST.json',receipt)
    os.rename(stage,destination)


def run(bundle, root, resume, max_seconds):
    if not 12000 <= max_seconds <= 165600:
        raise ValueError('Runtime must leave shutdown reserve inside a 48h allocation')
    config=verify(bundle); uuids=support.discover(config)
    root.mkdir(parents=True,exist_ok=resume)
    lock=(root/'.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request=dict(bundle=str(bundle),bundle_sha256=support.sha(bundle/'BUNDLE.json'),
                 retained_training_target=500000,maximum_training_attempts=750000,test_collection=False,
                 execution='one isolated scene per GPU; two GPUs',accepted_for_training=False)
    if resume:
        if support.read(root/'REQUEST.json')!=request: raise ValueError('Resume request mismatch')
    else: support.write(root/'REQUEST.json',request)
    job=os.environ['SLURM_JOB_ID']; start=time.monotonic(); stop=[False]
    signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True))
    def terminate(*_):
        raise InterruptedError('Allocation/user termination; preserve completed and partial evidence')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,terminate)
    support.write(root/f'EXECUTION-{job}.json',dict(job=job,uuids=uuids,max_seconds=max_seconds,
                  started=datetime.now(timezone.utc).isoformat()))
    env=environment(bundle,config); results=[]; train_retained=0; reason='attempt_budget_exhausted'
    def status(state, index):
        support.write(root/'STATUS.json',dict(state=state,job=job,next_wave=index,
            elapsed_seconds=time.monotonic()-start,training_retained=train_retained,
            training_attempts=sum(r['attempts'] for r in results if r['entry']['split']=='train'),
            completed_chunks=len(results),target=500000,
            dataset_roots=[str(root/'chunks'/r['entry']['name']/'filtered/diagnostic-export') for r in results],
            accepted_for_training=False),replace=True)
    for index in range(6252):
        entries=wave(index); units=[root/'chunks'/e['name'] for e in entries]
        if all((u/'COMPLETE.json').exists() for u in units):
            for u,e in zip(units,entries):
                r=completed(u)
                if r['entry']!=e: raise ValueError('Chunk schedule mismatch')
                publish(root,u,r)
                results.append(r)
                if e['split']=='train': train_retained+=r['retained']
            continue
        if any(u.exists() for u in units):
            raise ValueError('Partial wave preserved; explicit recovery review required')
        if train_retained>=500000: reason='training_target_reached'; break
        if stop[0] or time.monotonic()-start > max_seconds-RESERVE:
            reason='allocation_budget_stop'; break
        if shutil.disk_usage(root).free < 500 * 1024**3:
            reason='free_space_below_500GiB'; break
        status('preparing',index)
        for u,e in zip(units,entries):
            u.mkdir(parents=True)
            subprocess.run([config['genesis_python'],str(bundle/'mpm_campaign.py'),'prepare-plan',
                '--bundle',str(bundle),'--unit',str(u),'--entry',json.dumps(e)],
                env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True,timeout=900)
        status('simulating',index)
        support.children([[config['genesis_python'],str(bundle/'mpm_campaign.py'),'worker',
            '--bundle',str(bundle),'--unit',str(u),'--uuid',uuid] for u,uuid in zip(units,uuids)],
            [dict(env,CUDA_VISIBLE_DEVICES=uuid) for uuid in uuids],
            [root/f'worker-{index:05d}-{i}.log' for i in range(2)],timeout=7500)
        status('reviewing',index)
        for u,e in zip(units,entries):
            with (root/f'review-{e["name"]}.log').open('x') as log:
                subprocess.run([config['genesis_python'],str(bundle/'mpm_campaign.py'),'review',
                    '--bundle',str(bundle),'--unit',str(u)],env=dict(env,CUDA_VISIBLE_DEVICES=''),
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            r=completed(u); publish(root,u,r); results.append(r)
            if e['split']=='train': train_retained+=r['retained']
        status('wave_complete',index+1)
    status(reason,index)
    support.write(root/f'STOP-{job}.json',dict(reason=reason,training_retained=train_retained,
                  completed_chunks=len(results),accepted_for_training=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('freeze','verify','run','prepare-plan','worker','review'))
    p.add_argument('--bundle',type=Path,required=True); p.add_argument('--banks',type=Path)
    p.add_argument('--reference',type=Path); p.add_argument('--model-root',type=Path)
    p.add_argument('--root',type=Path); p.add_argument('--unit',type=Path); p.add_argument('--entry')
    p.add_argument('--uuid'); p.add_argument('--resume',action='store_true')
    p.add_argument('--max-seconds',type=int,default=165600)
    a=p.parse_args(); bundle=a.bundle.resolve()
    if a.command=='freeze':
        freeze(bundle,a.banks,a.reference,a.model_root); return
    sys.path.insert(0,str(bundle/'source'))
    config=support.read(bundle/'BUNDLE.json')
    os.environ['PWM_MPM_REFERENCE_ROOT']=config['reference']
    if a.command=='verify': verify(bundle); print('Frozen campaign verified',flush=True)
    elif a.command=='prepare-plan': prepare_plan(a.unit,json.loads(a.entry),config)
    elif a.command=='worker': worker(a.unit,a.uuid)
    elif a.command=='review': review(a.unit)
    else:
        try: run(bundle,a.root.resolve(),a.resume,a.max_seconds)
        except Exception as exc:
            if a.root and a.root.exists():
                support.write(a.root/f'FAILURE-{os.environ.get("SLURM_JOB_ID","local")}.json',
                              dict(error=type(exc).__name__+': '+str(exc)))
            raise


if __name__=='__main__': main()
