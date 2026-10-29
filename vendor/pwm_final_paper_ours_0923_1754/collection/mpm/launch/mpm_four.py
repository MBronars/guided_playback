"""Four-GPU fresh-data qualification and gated 48-hour collection.

Existing two-GPU campaign is never modified. Cycles >=20000 are reserved here.
Only a hash-bound passing qualification can submit/start the production job.
"""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import stat
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import mpm_campaign as base
import mpm_scale as support

MODES=[dict(name='group4',size=4,block=256,fused=False),
       dict(name='launch64',size=4,block=64,fused=False)]+[
       dict(name=f'fused{n}',size=n,block=64,fused=True) for n in (4,8,16,32)]
POSITION_RMS_M=1e-4  # Existing 0.1 mm scene RMS gate; do not tune after results.
# v1 serial references themselves disagree by two native steps on scene 03.
# This is a capped timing tolerance, NEVER a contact-presence or physics waiver.
WALL_EVENT_STEP_TOLERANCE=2
QUALIFICATION_MODES=[MODES[2]]  # Requalify the measured fastest mode: fused4.
SERIAL_REPEAT_CASES=(3,25,26)  # All scenes implicated by the v1 qualification.
SERIAL_REPEATS=2


def qualification_contract(gpu_count=4):
    return dict(modes=QUALIFICATION_MODES,scene_position_rms_m=POSITION_RMS_M,
                wall_event_step_tolerance=WALL_EVENT_STEP_TOLERANCE,
                exact_contact_presence=True,serial_repeat_cases=list(SERIAL_REPEAT_CASES),
                serial_repeats=SERIAL_REPEATS,cases=32,
                gpu_count=gpu_count,all_allocated_gpus_must_pass=True,minimum_speedup=1.05)


def entries(index,gpu_count=4,training_only=False):
    if type(gpu_count) is not int or gpu_count not in (2,4):raise ValueError('Two or four GPUs required')
    if training_only and gpu_count!=2:raise ValueError('Scoped training uses the qualified two GPUs')
    count=(12500 if training_only else 12504)//gpu_count;validation_waves=0 if training_only else 4//gpu_count
    cycle_start=40000 if gpu_count==2 else 20000
    if type(index) is not int or not 0<=index<count:raise ValueError('Bounded wave index required')
    if 1<=index<=validation_waves:
        es=('seen','unseen','count_seen','count_unseen')[(index-1)*gpu_count:index*gpu_count]
        return [dict(name=f'validation-{e}-{cycle_start}',split='validation',evaluation=e,
                     cycle=cycle_start,lane=i) for i,e in enumerate(es)]
    j=index if index==0 else index-validation_waves
    return [dict(name=f'train-{cycle_start+gpu_count*j+i:05d}',split='train',evaluation='train',
                 cycle=cycle_start+gpu_count*j+i,lane=i) for i in range(gpu_count)]


def freeze(bundle,banks,reference,model,gpu_count=4,training_only=False):
    if gpu_count not in (2,4):raise ValueError('Two or four GPUs required')
    if training_only and gpu_count!=2:raise ValueError('Training scope requires two GPUs')
    base.freeze(bundle,banks,reference,model)
    here=Path(__file__).resolve().parent
    for name in ('mpm_campaign.py','mpm_four.py','mpm_training.py','mpm_handoff.py','mpm_four_qualify.sbatch','mpm_four_collect.sbatch'):
        shutil.copy2(here/name,bundle/name)
    config=support.read(bundle/'BUNDLE.json')
    config.update(schema='fresh-mpm-grouped-gpu-v4',gpu_count=gpu_count,training_only=training_only,
                  cycle_start=40000 if gpu_count==2 else 20000,
                  qualification=qualification_contract(gpu_count),
                  files={str(p.relative_to(bundle)):support.sha(p) for p in sorted(bundle.rglob('*'))
                         if p.is_file() and p.name!='BUNDLE.json'})
    support.write(bundle/'BUNDLE.json',config,replace=True)


def verify(bundle):
    config=base.verify(bundle)
    if support.sha(__file__)!=config['files']['mpm_four.py']:raise ValueError('Use frozen four-GPU driver')
    if (config['gpu_count'] not in (2,4) or config['qualification']!=qualification_contract(config['gpu_count'])
            or config['cycle_start']!=(40000 if config['gpu_count']==2 else 20000)):
        raise ValueError('Qualification changed')
    if type(config['training_only']) is not bool or (config['training_only'] and config['gpu_count']!=2):
        raise ValueError('Invalid collection scope')
    return config


def discover(config):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Four explicitly allocated Slurm GPUs required')
    code='import json,warp as w;w.init();print("DEVICES="+json.dumps([dict(uuid=d.uuid,name=d.name) for d in w.get_cuda_devices()]))'
    output=subprocess.check_output([config['genesis_python'],'-c',code],text=True,timeout=60)
    devices=json.loads(next(s[8:] for s in output.splitlines() if s.startswith('DEVICES=')))
    count=config['gpu_count']
    if len(devices)!=count or len({d['uuid'] for d in devices})!=count or any('RTX PRO 6000' not in d['name'] for d in devices):
        raise ValueError(f'Exactly {count} allocated RTX PRO 6000 GPUs required')
    return [d['uuid'] for d in devices]


def bind(uuid,parent):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:
        raise ValueError('Wrong allocation/device')
    if ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)!=0 or os.getppid()!=parent:
        raise ValueError('Supervisor lifetime mismatch')
    from collection.mpm.collect import gpu_status
    gpu_status(uuid)
    import warp as wp
    wp.init()
    if wp.get_cuda_device_count()!=1 or wp.get_device('cuda:0').uuid!=uuid:raise ValueError('Wrong Warp device')


def selected_cases(root):
    from collection.mpm.collect import inspect
    pools={}
    for name in ('train','count'):
        planroot=root/'inputs'/name/'plan'
        for c in inspect(planroot)['cases']:
            if c['kind']=='coverage' and not c['provenance']['placement_reasons']:
                pools.setdefault(c['template']['n_objects'],[]).append(dict(root=str(planroot),case=c))
    selected=[]
    for count in range(1,8):
        # Cross stiffness phases first; not a best-looking outcome selection.
        candidates=pools[count];chosen=[]
        for phase in range(3):
            chosen.append(next(c for c in candidates if c['case']['provenance']['stiffness_phase']==phase))
        chosen+= [c for c in candidates if c not in chosen][: (5 if count<=4 else 4)-3]
        if len(chosen)!=(5 if count<=4 else 4):raise ValueError('Insufficient qualification strata')
        selected+=chosen
    return selected


def compare_events(reference,candidate):
    """Require the same wall/contact detection at EVERY saved state.

    -1 means no event; it must never be accepted as 'one step from step zero'.
    Native timestamps are retained unmodified, including exact-equality metrics.
    """
    import numpy as np
    a,b=np.asarray(reference),np.asarray(candidate)
    if (a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=5
            or a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu'
            or np.any(a < -1) or np.any(b < -1)):
        raise ValueError('Invalid first-contact event arrays')
    present=a>=0;other=b>=0
    same=bool(np.array_equal(present,other))
    joint=present & other
    maximum=int(np.abs(a[joint].astype(np.int64)-b[joint].astype(np.int64)).max(initial=0))
    return dict(exact=bool(np.array_equal(a,b)),contact_presence_equal=same,
                maximum_step_difference=maximum,
                passed=same and maximum<=WALL_EVENT_STEP_TOLERANCE)


def comparison_passes(c):
    import math
    return bool(c['integrity_pass'] and c['selection_equal']
        and math.isfinite(c['position_rms_m']) and 0<=c['position_rms_m']<=POSITION_RMS_M
        and c['wall_events']['contact_presence_equal']
        and 0<=c['wall_events']['maximum_step_difference']<=WALL_EVENT_STEP_TOLERANCE)


def compare(reference, candidate):
    import h5py
    import numpy as np
    from collection.mpm.export import validate_candidate
    from collection.mpm.wall_materials import SAND_ONLY
    with h5py.File(reference,'r') as a,h5py.File(candidate,'r') as b:
        for k in ('object_ids','particle_volume','E','nu','density','mat_type','yield_stress','state_time_s',
                  'selected','command_velocity','transition_active','active_substeps'):
            np.testing.assert_array_equal(a[k][()],b[k][()])
        for k in ('x','v','F_trial','C'):np.testing.assert_array_equal(a[k][0],b[k][0])
        delta=a['x'][()].astype(float)-b['x'][()].astype(float)
        rms=float(np.sqrt(np.mean(delta**2)));maximum=float(np.linalg.norm(delta,axis=-1).max())
        for k in ('wall_covered_substeps','wall_simulation_steps'):
            np.testing.assert_array_equal(a[k][()],b[k][()])
        events={k:compare_events(a[k][()],b[k][()]) for k in ('wall_first_grid_step','wall_first_particle_step')}
    old=validate_candidate(reference,wall_policy=SAND_ONLY);new=validate_candidate(candidate,wall_policy=SAND_ONLY)
    same=(old['eligible_for_export']==new['eligible_for_export'] and set(old['reasons'])==set(new['reasons']))
    result=dict(position_rms_m=rms,maximum_particle_difference_m=maximum,
                first_events_equal=all(e['exact'] for e in events.values()),selection_equal=same,
                integrity_pass=bool(old['integrity_pass'] and new['integrity_pass']),
                wall_events=dict(contact_presence_equal=all(e['contact_presence_equal'] for e in events.values()),
                    maximum_step_difference=max(e['maximum_step_difference'] for e in events.values()),details=events))
    result['passed']=comparison_passes(result)
    return result


def qualify_lane(root,lane,uuid,parent):
    bind(uuid,parent)
    from collection.lease import gpu_lease
    from collection.mpm import grouped
    from collection.common import write_once
    import warp as wp
    config=support.read(root/'REQUEST.json');selected=config['cases']
    out=root/f'lane-{lane}';out.mkdir()
    wp.config.kernel_cache_dir=str(out/'kernel-cache')
    with gpu_lease(uuid):
        # Qualify fusion itself before considering its end-to-end trajectories.
        grid=grouped.grid_test(selected[0]['root'],selected[0]['case'])
        failures={}
        for m in MODES[:3]:
            failures[m['name']]=grouped.corruption_test(selected[0]['root'],[s['case'] for s in selected[:2]],
                                                      block=m['block'],fused=m['fused'])
            gc.collect()
        write_once(out/'ISOLATION.json',dict(grid=grid,failure_injections=failures))
        reference=[];serial_seconds=0.
        for i,s in enumerate(selected):
            dest=out/'serial'/f'{i:02d}'
            r=grouped.simulate(s['root'],[s['case']],[dest],serial=True,uuid=uuid)
            reference.append(dest/'raw.h5');serial_seconds+=r['seconds'];gc.collect()
            print(json.dumps(dict(lane=lane,mode='serial',completed=i+1,total=32)),flush=True)
        # New same-GPU unchanged-serial repeats; never silently widen thresholds
        # if the reference fails repeatability. Preserve every repeat as evidence.
        repeats={i:[] for i in SERIAL_REPEAT_CASES};repeat_comparisons=[]
        for i in SERIAL_REPEAT_CASES:
            for repeat in range(SERIAL_REPEATS):
                dest=out/'serial-repeat'/f'{i:02d}-{repeat}'
                s=selected[i]
                grouped.simulate(s['root'],[s['case']],[dest],serial=True,uuid=uuid)
                repeats[i].append(dest/'raw.h5')
                repeat_comparisons.append(dict(case=i,repeat=repeat,**compare(reference[i],dest/'raw.h5')))
                gc.collect()
                print(json.dumps(dict(lane=lane,mode='serial-repeat',case=i,repeat=repeat)),flush=True)
        write_once(out/'REPEATABILITY.json',dict(comparisons=repeat_comparisons,
                    passed=all(c['passed'] for c in repeat_comparisons)))
        results=[]
        for mode in QUALIFICATION_MODES:
            # Keep plan/cohort inputs separate; use repeated same-input load roots
            # through a single temporary plan only for this qualification matrix.
            seconds=0.;comparisons=[];repeat_candidates=[]
            for start in range(0,len(selected),mode['size']):
                subset=selected[start:start+mode['size']]
                # load takes one root; a qualified immutable merged plan is built
                # by the supervisor and every initial/manifest digest is retained.
                cases=[dict(s['case'],label=f'qual-{i:02d}') for i,s in enumerate(selected)][start:start+mode['size']]
                outputs=[out/mode['name']/f'{i:02d}' for i in range(start,start+len(subset))]
                r=grouped.simulate(root/'merged',cases,outputs,block=mode['block'],fused=mode['fused'],uuid=uuid)
                seconds+=r['seconds']
                for i,dest in zip(range(start,start+len(subset)),outputs):
                    comparisons.append(compare(reference[i],dest/'raw.h5'))
                    for repeat,ref in enumerate(repeats.get(i,[])):
                        repeat_candidates.append(dict(case=i,repeat=repeat,**compare(ref,dest/'raw.h5')))
                gc.collect()
            result=dict(mode=mode,seconds=seconds,serial_seconds=serial_seconds,
                        speedup=serial_seconds/seconds,comparisons=comparisons,
                        repeat_comparisons=repeat_candidates,
                        passed=all(c['passed'] for c in comparisons+repeat_candidates+repeat_comparisons)
                               and serial_seconds/seconds>=1.05)
            write_once(out/(mode['name']+'.json'),result);results.append(result)
            print(json.dumps(dict(lane=lane,mode=mode['name'],passed=result['passed'],speedup=result['speedup'])),flush=True)
        write_once(out/'RESULT.json',dict(results=results,uuid=uuid))


def qualified(root,bundle):
    if support.read(bundle/'BUNDLE.json').get('training_only'):
        from mpm_training import qualified as training_qualified
        return training_qualified(root,bundle)
    from collection.common import check_hashes
    result=support.read(root/'PASS.json')
    if result['bundle_sha256']!=support.sha(bundle/'BUNDLE.json'):raise ValueError('Qualification bundle mismatch')
    if result['mode'] not in QUALIFICATION_MODES:raise ValueError('Unknown qualified mode')
    check_hashes(root,result['files'])
    index=QUALIFICATION_MODES.index(result['mode'])
    count=support.read(bundle/'BUNDLE.json')['gpu_count']
    if result['gpu_count']!=count:raise ValueError('Qualification GPU count mismatch')
    for lane in range(count):
        r=support.read(root/f'lane-{lane}/RESULT.json')['results'][index]
        repeat=support.read(root/f'lane-{lane}/REPEATABILITY.json')
        expected=[(i,j) for i in SERIAL_REPEAT_CASES for j in range(SERIAL_REPEATS)]
        for rows in (repeat['comparisons'],r['repeat_comparisons']):
            if ([(c['case'],c['repeat']) for c in rows]!=expected
                    or not all(c['passed'] and comparison_passes(c) for c in rows)):
                raise ValueError('Serial repeatability or candidate repeat comparison failed')
        if (r['mode']!=result['mode'] or not r['passed'] or r['speedup']<1.05
                or not repeat['passed'] or len(r['comparisons'])!=32
                or not all(c['passed'] and comparison_passes(c) for c in r['comparisons'])):
            raise ValueError('Qualification does not pass all scenes on every GPU')
    return result


def qualify(bundle,root):
    config=verify(bundle);uuids=discover(config);root.mkdir(parents=True,exist_ok=False)
    env=base.environment(bundle,config)
    for name,entry in [('train',dict(name='qual-train',split='train',evaluation='train',cycle=19999,lane=0)),
                       ('count',dict(name='qual-count',split='validation',evaluation='count_seen',cycle=19999,lane=0))]:
        unit=root/'inputs'/name
        subprocess.run([config['genesis_python'],str(bundle/'mpm_four.py'),'prepare','--bundle',str(bundle),
                        '--unit',str(unit),'--entry',json.dumps(entry)],env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True,timeout=1200)
    selected=selected_cases(root)
    # Exact byte copies with unique labels; this is a mixed qualification-only
    # input collection, never an export cohort or an independent training set.
    for i,s in enumerate(selected):
        shutil.copytree(Path(s['root'])/'cases'/s['case']['label'],root/'merged/cases'/f'qual-{i:02d}')
    support.write(root/'REQUEST.json',dict(bundle_sha256=support.sha(bundle/'BUNDLE.json'),uuids=uuids,cases=selected,
                                         qualification=qualification_contract(len(uuids)),diagnostic_repeats=True))
    commands=[[config['genesis_python'],str(bundle/'mpm_four.py'),'qualify-lane','--bundle',str(bundle),
               '--root',str(root),'--lane',str(i),'--uuid',u,'--parent',str(os.getpid())] for i,u in enumerate(uuids)]
    support.children(commands,[dict(env,CUDA_VISIBLE_DEVICES=u) for u in uuids],
                     [root/f'worker-{i}.log' for i in range(len(uuids))],timeout=12000)
    lanes=[support.read(root/f'lane-{i}/RESULT.json')['results'] for i in range(len(uuids))]
    passed=[j for j in range(len(QUALIFICATION_MODES)) if all(l[j]['passed'] for l in lanes)]
    if not passed:
        support.write(root/'NO_PASS.json',dict(reason='No mode passed every scene on all allocated GPUs; no production'))
        raise ValueError('Strict qualification failed; no four-GPU campaign will be submitted')
    best=min(passed,key=lambda j:max(l[j]['seconds'] for l in lanes));mode=QUALIFICATION_MODES[best]
    # Both normalization consumers for retained candidate trajectories of the
    # chosen mode, independently per input cohort. Repeats never enter production.
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from collection.mpm.export import export_candidates
    from collection.mpm.wall_materials import SAND_ONLY
    from collection.mpm.review import check_model
    reviews=[]
    for lane in range(len(uuids)):
        for name,split in [('train','train'),('count','validation')]:
            raws=[root/f'lane-{lane}'/mode['name']/f'{i:02d}'/'raw.h5' for i,s in enumerate(selected)
                  if s['case']['template']['split']==split]
            unit=root/'consumer'/f'{lane}-{name}'
            export_candidates(raws,unit/'filtered',wall_policy=SAND_ONLY)
            rows=support.read(unit/'filtered/FILTER.json')['episodes']
            kept={r['episode_id']:Path(r['raw_path']) for r in rows if r['eligible_for_export']}
            if not kept:raise ValueError('No consumer survivors')
            reviews.append(check_model(unit,kept))
    support.write(root/'PASS.json',dict(mode=mode,gpu_count=len(uuids),bundle_sha256=support.sha(bundle/'BUNDLE.json'),
        diagnostic_raw_per_hour=32*len(uuids)*3600/max(l[best]['seconds'] for l in lanes),consumer_reviews=reviews,
        files={str(p.relative_to(root)):support.sha(p) for p in sorted(root.rglob('*'))
               if p.is_file() and 'kernel-cache' not in p.parts},accepted_for_training=False))
    print(json.dumps(dict(qualified_mode=mode,production_submission_allowed=True)),flush=True)


def reusable_kernel_cache(identity, directory=None):
    """Private node-local cache, keyed by the frozen runtime and device.

    Call only while holding the GPU lease. Cache contents are disposable build
    artifacts, never qualification evidence or a substitute for source checks.
    """
    encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False)
    key=hashlib.sha256(encoded.encode()).hexdigest()
    root=Path(directory or Path(tempfile.gettempdir())/f'pwm-mpm-kernel-cache-{os.getuid()}')
    for path in (root,root/key):
        path.mkdir(mode=0o700,exist_ok=True)
        info=path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid!=os.getuid()
                or info.st_mode&0o077):
            raise ValueError('Kernel cache must be a private owned directory: '+str(path))
    receipt=root/key/'IDENTITY.json'
    if receipt.exists() or receipt.is_symlink():
        info=receipt.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid()
                or info.st_mode&0o022 or support.read(receipt)!=identity):
            raise ValueError('Kernel cache identity changed or unsafe')
    else:
        from collection.common import write_once
        write_once(receipt,identity)
    return root/key


def record_cache_seed(lane, *, uuid, bundle_sha256, warp_version, device_arch, mode):
    cache=lane/'kernel-cache'
    files={str(p.relative_to(cache)):support.sha(p) for p in sorted(cache.rglob('*')) if p.is_file()}
    if cache.is_symlink() or not files or any(p.is_symlink() for p in cache.rglob('*')):
        raise ValueError('Missing or symlinked compiled cache')
    support.write(lane/'CACHE_SEED.json',dict(schema='mpm-qualified-cache-seed-v1',uuid=uuid,
        bundle_sha256=bundle_sha256,warp=warp_version,device_arch=device_arch,mode=mode,files=files))


def copy_cache_seed(cache, seed, expected_sha, identity, *, require_existing=False):
    """Copy only hash-bound artifacts from the successful qualification lane."""
    if seed.is_symlink() or support.sha(seed)!=expected_sha:raise ValueError('Cache seed receipt changed')
    record=support.read(seed)
    if (record.get('schema')!='mpm-qualified-cache-seed-v1' or not record.get('files')
            or any(record.get(k)!=identity[k] for k in ('bundle_sha256','warp','device_arch','mode'))
            or record.get('uuid')!=identity['device_uuid']):
        raise ValueError('Cache seed belongs to another runtime or GPU')
    source=seed.parent/'kernel-cache'
    if source.is_symlink():raise ValueError('Symlinked seed cache')
    for name,sha in record['files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts:
            raise ValueError('Unsafe cache entry')
        a=source/relative;b=cache/relative
        for parent in (source,cache):
            for component in relative.parts:
                parent=parent/component
                if parent.is_symlink():raise ValueError('Symlinked cache entry')
        if support.sha(a)!=sha:raise ValueError('Qualified compiler artifact changed')
        if b.exists():
            if support.sha(b)!=sha:raise ValueError('Reusable compiler artifact changed')
        else:
            if require_existing:raise ValueError('Qualified compiler artifact disappeared')
            b.parent.mkdir(parents=True,exist_ok=True)
            with a.open('rb') as src,b.open('xb') as dst:shutil.copyfileobj(src,dst)
            if support.sha(b)!=sha:raise ValueError('Compiler artifact copy mismatch')
    if support.sha(seed)!=expected_sha:raise ValueError('Cache seed changed during copy')
    if require_existing:
        actual={str(p.relative_to(cache)) for p in cache.rglob('*') if p.is_file() and p!=cache/'IDENTITY.json'}
        if actual!=set(record['files']):raise ValueError('Unqualified compiler artifacts appeared')


def worker(unit,mode,uuid,parent,training_only=False, *, reuse_cache=False,cache_seed=None,cache_seed_sha=None,async_io=False):
    bind(uuid,parent)
    from collection.lease import gpu_lease
    from collection.mpm import collect,grouped
    simulate=grouped.simulate
    if async_io:
        from mpm_async_io import simulate as background_simulate
        simulate=lambda *a,**kw:background_simulate(grouped,*a,**kw)
    from collection.common import write_once
    import warp as wp
    plan=collect.inspect(unit/'plan')
    if training_only:
        require_training_plan(support.read(unit/'ENTRY.json'),plan['cases'])
    with gpu_lease(uuid):
        if (cache_seed is None)!=(cache_seed_sha is None):raise ValueError('Cache seed requires its bound hash')
        reuse_cache=reuse_cache or cache_seed is not None
        if reuse_cache:
            identity=dict(schema='mpm-private-kernel-cache-v1',
                bundle_sha256=support.sha(Path(__file__).parent/'BUNDLE.json'),
                driver_sha256=support.sha(__file__),python=sys.version,
                executable=str(Path(sys.executable).resolve()),warp=wp.__version__,
                device_uuid=uuid,device_arch=wp.get_device('cuda:0').arch,mode=mode,
                qualified_seed_sha256=cache_seed_sha)
            cache=reusable_kernel_cache(identity)
            if cache_seed is not None:copy_cache_seed(cache,cache_seed,cache_seed_sha,identity)
        else:cache=unit/'kernel-cache';identity=None
        wp.config.kernel_cache_dir=str(cache)
        write_once(unit/'KERNEL_CACHE.json',dict(path=str(cache),reused_across_chunks=reuse_cache,
            identity=identity,qualification_waived=False))
        cases=[]
        for c in plan['cases']:
            out=unit/'plan/cases'/c['label']/'attempts/attempt-00000'
            if c['provenance'].get('placement_reasons'):
                out.mkdir(parents=True)
                write_once(out/'RESULT.json',dict(status='placement_rejected',kind=c['kind'],label=c['label'],
                    initial_sha256=c['initial_sha256'],reasons=c['provenance']['placement_reasons']))
            elif c['kind']=='matched_control':
                # Controls keep their own clocks and never mix with coverage.
                simulate(unit/'plan',[c],[out],block=mode['block'],fused=mode['fused'],uuid=uuid)
            else:cases.append(c)
        for start in range(0,len(cases),mode['size']):
            chunk=cases[start:start+mode['size']]
            outs=[unit/'plan/cases'/c['label']/'attempts/attempt-00000' for c in chunk]
            simulate(unit/'plan',chunk,outs,block=mode['block'],fused=mode['fused'],uuid=uuid)
            print(json.dumps(dict(completed=min(start+len(chunk),len(cases)),total=len(cases),mode=mode)),flush=True)
            gc.collect()
        if cache_seed is not None:
            copy_cache_seed(cache,cache_seed,cache_seed_sha,identity,require_existing=True)


def require_training_plan(entry,cases):
    if entry['split']!='train' or entry['evaluation']!='train':raise ValueError('Evaluation not approved for fast training runner')
    for c in cases:
        if c['kind']=='coverage' and (c['template']['split']!='train' or not 1<=c['template']['n_objects']<=4):
            raise ValueError('Fast training scope is 1–4 objects only')


def check_quarantine_window(results):
    if sum(len(r.get('quarantine',{}).get('excluded',[])) for r in results[-10:])>1:
        raise ValueError('Repeated contraction quarantine within ten chunks; stop for diagnosis')


def run(bundle,root,qualification,deadline_epoch=None, *, controller=None, start_wave=0, review_driver=None, quarantine_history=(), worker_driver=None):
    import fcntl
    from types import SimpleNamespace
    # A frozen controller can supply a narrower qualification/schedule without
    # duplicating the tested bounded production orchestration. Default unchanged.
    runtime=SimpleNamespace(verify=verify,qualified=qualified,discover=discover,entries=entries) if controller is None else controller
    config=runtime.verify(bundle);passed=runtime.qualified(qualification,bundle);uuids=runtime.discover(config);mode=passed['mode']
    driver=bundle/('mpm_four.py' if controller is None else Path(controller.__file__).name)
    reserve=10800 if controller is None else config['shutdown_reserve_seconds']
    target=500000 if controller is None else config['retained_training_target']
    maximum_attempts=750000 if controller is None else config['maximum_training_attempts']
    training_only=config.get('training_only',False)
    if type(start_wave) is not int or start_wave<0 or (start_wave and controller is None):
        raise ValueError('Explicit bounded recovery controller required for schedule offset')
    if training_only and uuids!=[l['uuid'] for l in passed['lanes']]:raise ValueError('Training certificate belongs to different GPUs')
    if config['gpu_count']==2 and deadline_epoch is None:raise ValueError('Reused allocation requires absolute deadline')
    if deadline_epoch is not None and deadline_epoch<time.time()+reserve:raise ValueError('Insufficient shutdown reserve')
    cache_args=[[] for _ in uuids]
    if controller is not None:
        for i in range(len(uuids)):
            name=f'lane-{i}/CACHE_SEED.json'
            if name not in passed['files']:raise ValueError('Qualification did not bind compiled kernels')
            cache_args[i]=['--cache-seed',str(qualification/name),'--cache-seed-sha',passed['files'][name]]
    root.mkdir(parents=True,exist_ok=False)
    lock=(root/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    support.write(root/'REQUEST.json',dict(bundle=str(bundle),bundle_sha256=support.sha(bundle/'BUNDLE.json'),
        qualification=str(qualification),qualification_sha256=support.sha(qualification/('TRAINING_PASS.json' if training_only else 'PASS.json')),mode=mode,
        training_target=target,maximum_training_attempts=maximum_attempts,cycle_start=config['cycle_start'],
        uuids=uuids,test_collection=False,training_only=training_only,deadline_epoch=deadline_epoch,
        start_wave=start_wave,review_driver=None if review_driver is None else str(review_driver),
        worker_driver=None if worker_driver is None else str(worker_driver)))
    env=base.environment(bundle,config);start=time.monotonic();stop=[False];retained=0;results=[]
    def accept_result(r):
        check_quarantine_window(list(quarantine_history)+results+[r])
        results.append(r)
    signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True))
    def terminate(*_):raise InterruptedError('Allocation terminated; preserve raw prefixes')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,terminate)
    def review_unit(unit):
        with (unit/'review.log').open('x') as log:
            subprocess.run([config['genesis_python'],str(driver if review_driver is None else review_driver),'review','--bundle',str(bundle),
                            '--unit',str(unit)],env=dict(env,CUDA_VISIBLE_DEVICES=''),stdout=log,
                            stderr=subprocess.STDOUT,check=True,timeout=1800)
        r=base.completed(unit);base.publish(root,unit,r);return r
    def prepare_units(index):
        es=runtime.entries(index+start_wave,len(uuids),training_only);us=[root/'chunks'/e['name'] for e in es]
        for u,e in zip(us,es):
            subprocess.run([config['genesis_python'],str(driver),'prepare','--bundle',str(bundle),
                '--unit',str(u),'--entry',json.dumps(e)],env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True,timeout=1200)
        return es,us
    reason='attempt_budget_exhausted'
    # One previous wave (at most four chunks) may await CPU review. A review
    # failure prevents the next wave from starting; current raw evidence stays.
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending=[];prepared=prepare_units(0)
        waves=((12500 if training_only else 12504) if controller is None else config['training_cycles'])//len(uuids)-start_wave
        if waves<1:raise ValueError('Recovery offset exhausts the reserved schedule')
        for index in range(waves):
            if training_only and index==1:
                # Do not start wave 2 before wave 1 has passed review AND the
                # requested manual visual/throughput inspection is acknowledged.
                for future in pending:
                    r=future.result();accept_result(r);retained+=r['retained']
                pending=[]
                pilot=dict(chunks=[r['entry']['name'] for r in results],retained=retained,
                           bundle_sha256=support.sha(bundle/'BUNDLE.json'))
                support.write(root/'PILOT_READY.json',pilot)
                if controller is not None:
                    controller.approve_pilot(root,bundle,results)
                support.write(root/'STATUS.json',dict(state='awaiting_pilot_review',training_retained=retained,
                    completed_chunks=len(results),job=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-start),replace=True)
                until=time.monotonic()+1800
                while not (root/'PILOT_APPROVED.json').exists():
                    if time.monotonic()>until or stop[0]:raise TimeoutError('First-wave review not approved; preserve pilot and return to serial')
                    time.sleep(1)
                if support.read(root/'PILOT_APPROVED.json').get('pilot_sha256')!=support.sha(root/'PILOT_READY.json'):
                    raise ValueError('Pilot approval is for a different wave')
            for future in pending:
                if future.done():future.result()  # Fail before new work when known.
            if retained>=target-240:
                for future in pending:
                    r=future.result();accept_result(r)
                    if r['entry']['split']=='train':retained+=r['retained']
                pending=[]
            if retained>=target:reason='target_reached';break
            admission=150000 if controller is None else config['maximum_admission_seconds']
            if stop[0] or time.monotonic()-start>admission:reason='allocation_budget_stop';break
            if deadline_epoch is not None and time.time()>deadline_epoch-reserve:reason='allocation_budget_stop';break
            if shutil.disk_usage(root).free<500*1024**3:reason='free_space_below_500GiB';break
            es,us=prepared
            support.write(root/'STATUS.json',dict(state='simulating',wave=index,training_retained=retained,
                completed_chunks=len(results),mode=mode,job=os.environ['SLURM_JOB_ID'],
                elapsed_seconds=time.monotonic()-start),replace=True)
            # CPU prefetch stays bounded to exactly one next wave.
            next_input=pool.submit(prepare_units,index+1) if index+1<waves else None
            support.children([[config['genesis_python'],str(driver if worker_driver is None else worker_driver),'worker','--bundle',str(bundle),
                '--unit',str(u),'--mode',json.dumps(mode),'--uuid',uuid,'--parent',str(os.getpid()),*extra]
                for u,uuid,extra in zip(us,uuids,cache_args)],
                [dict(env,CUDA_VISIBLE_DEVICES=uuid) for uuid in uuids],
                [root/f'worker-{index:05d}-{i}.log' for i in range(len(uuids))],timeout=7200)
            for future in pending:
                r=future.result();accept_result(r)
                if r['entry']['split']=='train':retained+=r['retained']
            pending=[pool.submit(review_unit,u) for u in us]
            if next_input:prepared=next_input.result()
        for future in pending:
            r=future.result();accept_result(r)
            if r['entry']['split']=='train':retained+=r['retained']
    support.write(root/'STOP.json',dict(reason=reason,training_retained=retained,completed_chunks=len(results)))


def submit(bundle,qualification,output):
    qualified(qualification,bundle)
    if support.read(bundle/'BUNDLE.json')['gpu_count']!=4:raise ValueError('Only four-GPU bundles can submit a new four-GPU job')
    claim=qualification/'SUBMISSION_CLAIM.json'
    from collection.common import write_once
    write_once(claim,dict(output=str(output),bundle=str(bundle),reason='User-authorized gated four-GPU 48-hour collection'))
    command=['sbatch','--parsable','--export=ALL,MPM_FOUR_BUNDLE='+str(bundle)+',MPM_FOUR_QUALIFICATION='+str(qualification)+
             ',MPM_FOUR_OUTPUT='+str(output),str(bundle/'mpm_four_collect.sbatch')]
    job=subprocess.check_output(command,text=True,timeout=60).strip()
    write_once(qualification/'SUBMITTED.json',dict(job=job,command=command,output=str(output)))
    print('FOUR_GPU_COLLECTION_SUBMITTED='+job,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('freeze','verify','prepare','qualify','qualify-lane','worker','review','run','submit','certify-training','verify-training'))
    for arg in ('bundle','root','unit','qualification','banks','reference','model-root','evidence','evidence-bundle'):
        p.add_argument('--'+arg,type=Path,required=arg=='bundle')
    p.add_argument('--entry');p.add_argument('--mode');p.add_argument('--uuid');p.add_argument('--lane',type=int)
    p.add_argument('--parent',type=int);p.add_argument('--gpus',type=int,choices=(2,4),default=4)
    p.add_argument('--deadline-epoch',type=float)
    p.add_argument('--training-only',action='store_true')
    a=p.parse_args();bundle=a.bundle.resolve()
    if a.command=='freeze':freeze(bundle,a.banks,a.reference,a.model_root,a.gpus,a.training_only);return
    sys.path.insert(0,str(bundle/'source'));config=support.read(bundle/'BUNDLE.json')
    os.environ['PWM_MPM_REFERENCE_ROOT']=config['reference'];os.environ['PWM_MODEL_ROOT']=str(bundle/'source')
    if a.command=='verify':verify(bundle);print('Four-GPU bundle verified')
    elif a.command=='certify-training':
        from mpm_training import certify
        certify(bundle,a.root,a.evidence,a.evidence_bundle)
    elif a.command=='verify-training':qualified(a.root,bundle);print('Scoped training certificate verified')
    elif a.command=='prepare':base.prepare_plan(a.unit,json.loads(a.entry),config)
    elif a.command=='qualify-lane':qualify_lane(a.root,a.lane,a.uuid,a.parent)
    elif a.command=='worker':worker(a.unit,json.loads(a.mode),a.uuid,a.parent,config['training_only'])
    elif a.command=='review':base.review(a.unit)
    elif a.command=='submit':submit(bundle,a.qualification,a.root)
    else:
        try:
            if a.command=='qualify':qualify(bundle,a.root)
            else:run(bundle,a.root,a.qualification,a.deadline_epoch)
        except Exception as exc:
            if a.root.exists():support.write(a.root/'FAILURE.json',dict(error=type(exc).__name__+': '+str(exc)))
            raise


if __name__=='__main__':main()
