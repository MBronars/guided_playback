"""Fail-closed drag bundle/Slurm operator. A real approved gate is mandatory.

Planning a bundle does not submit a job. No override skips a missing certificate.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback
import uuid as uuid_module
from collection.common import canonical,digest,write_once,relative_path
from .. import runtime
from ..protocol import cohort
from .protocol import VARIATION

SCHEMA='mpm-drag-launch-v1'
FIXED_RESOLUTION_POLICY='passive-clock-accepted-timestep-sensitivity-v1'
GRID256_POLICY='shared-grid-g256-reviewed-approximation-v1'
CRITICAL=('collection/common.py',*('collection/mpm/'+name+'.py' for name in
    ('physics','solver','kernels','state','material_control','observer','grouped','clock','contracts',
     'walls','wall_materials','collapse','validate','export','recording','journal','runtime')),
    *('collection/mpm/drag/'+name+'.py' for name in ('protocol','fragmentation','interactions','contact','prepare','simulate','grouped','cli','matrix','probe','review','audit','convergence','replay','launch')),
    *('data/'+name+'.py' for name in ('batch','dataset','mpm_drag','unified_dataset','normalization')),
    'stats/normalization.json','collection/mpm/drag/collect.sbatch')


def read(path):return json.loads(Path(path).read_text())


def settings(*,name,attempts,gpus,hours,split,evaluation,object_counts,material_layout,n_grid=100):
    if type(n_grid) is not int or n_grid not in (100,256):raise ValueError('Unsupported collection grid')
    if not isinstance(name,str) or not re.fullmatch('[a-z][a-z0-9-]{0,47}',name):
        raise ValueError('Simple unique lowercase run name required')
    if (type(attempts) is not int or not 1<=attempts<=500000 or type(gpus) is not int or gpus not in (1,2,4)
            or type(hours) is not int or not 2<=hours<=48):
        raise ValueError('Outside bounded attempt/GPU/walltime settings')
    cohort(split,evaluation)
    allowed=(5,6,7) if evaluation.startswith('count_') else (1,2,3,4)
    if (not isinstance(object_counts,(tuple,list)) or not object_counts
            or any(type(n) is not int or n not in allowed for n in object_counts)
            or tuple(sorted(set(object_counts)))!=tuple(object_counts)):
        raise ValueError('Invalid ordered object counts')
    if material_layout not in ('homogeneous','mixed'):raise ValueError('Invalid material layout')


def identity(config):
    import hashlib
    return hashlib.sha256(canonical({k:v for k,v in config.items() if k!='request_sha256'}).encode()).hexdigest()


def share_created(path,gid):
    """Publish only this operator's new artifacts for the output parent's group.

    Never chmod an asset bank, an existing campaign or an arbitrary symlink.
    Callers supply a freshly created bundle/block/receipt, never a workspace root.
    """
    path=Path(path)
    members=[path]+list(path.rglob('*')) if path.is_dir() else [path]
    for member in members:
        if member.is_symlink() or member.stat().st_uid!=os.getuid():raise ValueError('Not a newly owned output: '+str(member))
        os.chown(member,-1,gid)
        os.chmod(member,0o2750 if member.is_dir() else 0o640)


def check_resolution_review(gate):
    """Explicit method choice, not a forged convergence pass or general override."""
    review=gate.get('resolution_review',{})
    if (review.get('policy')!=FIXED_RESOLUTION_POLICY
            or review.get('accepted_timestep_sensitivity') is not True
            or review.get('clock_source_sha256')!=digest(runtime.ROOT/'collection/mpm/clock.py')):
        raise ValueError('Missing explicit passive-clock resolution review')
    decision_path=review.get('decision_path');decision_sha=review.get('decision_sha256')
    if not decision_path or gate.get('evidence',{}).get(decision_path)!=decision_sha or digest(decision_path)!=decision_sha:
        raise ValueError('Resolution decision must be bound to gate evidence')
    decision=read(decision_path)
    if (decision.get('policy')!=FIXED_RESOLUTION_POLICY or decision.get('authorized_by')!='user'
            or decision.get('action_schema')!=VARIATION['schema']
            or decision.get('accepted_timestep_sensitivity') is not True
            or decision.get('convergence_passed') is not False
            or not isinstance(decision.get('instruction'),str) or not decision['instruction'].strip()):
        raise ValueError('Resolution decision does not acknowledge the actual limitation')
    diagnostics=review.get('diagnostic_results',{})
    if not diagnostics:raise ValueError('Missing measured timestep diagnostics')
    for name,sha in diagnostics.items():
        if gate.get('evidence',{}).get(name)!=sha or digest(name)!=sha:
            raise ValueError('Timestep diagnostics must be bound to gate evidence')
        result=read(name)
        if (result.get('diagnostic')!='timestep-control' or result.get('passed') is not False
                or not result.get('comparisons') or not any(r.get('passed') is False for r in result['comparisons'])):
            raise ValueError('Fixed-resolution review must retain the failed timestep result')


def check_grid256_review(gate):
    review=gate.get('method_review',{});path=review.get('decision_path');sha=review.get('decision_sha256')
    if not path or gate.get('evidence',{}).get(path)!=sha or digest(path)!=sha:
        raise ValueError('Missing immutable g256 method choice')
    decision=read(path)
    expected=dict(policy=GRID256_POLICY,n_grid=256,grid_lim_m=100/64,authorized_by='user',
                  known_contact_artifacts_accepted=True,convergence_established=False)
    if any(decision.get(k)!=v for k,v in expected.items()) or not decision.get('instruction'):
        raise ValueError('g256 choice must explicitly acknowledge contact artifacts and unestablished convergence')
    results=decision.get('contact_results',{})
    if not results:raise ValueError('Missing measured g256 contact evidence')
    for name,sha in results.items():
        if gate.get('evidence',{}).get(name)!=sha or digest(name)!=sha or read(name).get('n_grid')!=256:
            raise ValueError('g256 method evidence changed or has wrong grid')


def check_gate(path,assets,*,split,evaluation,object_counts,material_layout,purpose='production',n_grid=100):
    gate=read(path)
    schema={'production':'mpm-drag-scale-gate-v1','pilot':'mpm-drag-pilot-gate-v1'}.get(purpose)
    fixed_schema={'production':'mpm-drag-scale-fixed-resolution-gate-v1',
                  'pilot':'mpm-drag-pilot-fixed-resolution-gate-v1'}.get(purpose)
    fixed=gate.get('schema')==fixed_schema and fixed_schema is not None
    reviewed_schema={'production':'mpm-drag-scale-reviewed-grid-gate-v1','pilot':'mpm-drag-pilot-reviewed-grid-gate-v1'}.get(purpose)
    reviewed=gate.get('schema')==reviewed_schema and reviewed_schema is not None
    fixed=fixed or reviewed
    if schema is None or gate.get('schema') not in (schema,fixed_schema,reviewed_schema) or gate.get('approved') is not True:
        raise ValueError('Dragging has no approved scale certificate')
    if type(n_grid) is not int or n_grid not in (100,256) or (n_grid==256)!=reviewed:
        raise ValueError('Requested grid requires its own explicit reviewed-method gate')
    if gate.get('variation')!=VARIATION:
        raise ValueError('Gate does not qualify the current sand-free sequence/speed variation')
    if gate.get('bank_sha256')!=digest(Path(assets)/'COMPLETE.json'):
        raise ValueError('Qualified asset bank changed')
    if set(gate.get('critical_source_sha256',{}))!=set(CRITICAL):raise ValueError('Incomplete numerical source gate')
    for name,sha in gate['critical_source_sha256'].items():
        if digest(runtime.ROOT/name)!=sha:raise ValueError('Unqualified numerical source change: '+name)
    required={'parity','convergence','consumer','failure_isolation','interaction_review'}
    if purpose=='production':required.add('sustained_pilot')
    if fixed:required.update(('coverage','physical_quality','visual_review','fixed_resolution_review'))
    checks=gate.get('checks',{})
    if set(checks)!=required or any(value is not (False if fixed and key=='convergence' else True)
                                    for key,value in checks.items()):
        raise ValueError('All measured readiness checks must pass')
    if reviewed:check_grid256_review(gate)
    elif fixed:check_resolution_review(gate)
    if not gate.get('evidence'):raise ValueError('Gate has no measured evidence')
    for name,sha in gate['evidence'].items():
        if digest(name)!=sha:raise ValueError('Qualification evidence changed: '+name)
    wanted=dict(split=split,evaluation=evaluation,material_layout=material_layout)
    if not any(all(cap.get(k)==v for k,v in wanted.items()) and cap.get('n_grid',100)==n_grid and set(object_counts)<=set(cap['object_counts'])
               for cap in gate.get('capabilities',[])):
        raise ValueError('Requested split/count/material regime is not qualified')
    return gate


def bundle(output,assets,gate_path,*,name,python,split='train',evaluation='train',
           object_counts=(1,2,3,4),material_layout='homogeneous',attempts=20000,gpus=2,hours=48,
           purpose='production',registry=None,n_grid=100):
    settings(name=name,attempts=attempts,gpus=gpus,hours=hours,split=split,evaluation=evaluation,
             object_counts=object_counts,material_layout=material_layout,n_grid=n_grid)
    check_purpose(purpose,attempts,gpus,hours)
    assets=Path(assets).resolve(strict=True);python=str(Path(python).resolve(strict=True))
    gate_path=Path(gate_path).resolve(strict=True)
    check_gate(gate_path,assets,split=split,evaluation=evaluation,object_counts=object_counts,
               material_layout=material_layout,purpose=purpose,n_grid=n_grid)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    gid=output.parent.stat().st_gid;share_created(output,gid)
    registry=Path(registry).resolve() if registry is not None else output.parent/'drag-reservations'
    try:
        registry.mkdir(parents=True,exist_ok=False)
        os.chown(registry,-1,gid);os.chmod(registry,0o2770)
    except FileExistsError:
        if not registry.is_dir() or not os.access(registry,os.W_OK|os.X_OK):raise
    from .cli import freeze
    freeze(output/'source',consumer=True)
    shutil.copy2(Path(__file__).with_name('collect.sbatch'),output/'collect.sbatch')
    shutil.copy2(Path(__file__).with_name('collect.sbatch'),output/'source/collection/mpm/drag/collect.sbatch')
    shutil.copy2(gate_path,output/'GATE.json')
    request=dict(schema=SCHEMA,purpose=purpose,name=name,namespace=name+'-'+uuid_module.uuid4().hex,
        assets=str(assets),bank_sha256=digest(assets/'COMPLETE.json'),python=python,registry=str(registry),
        split=split,evaluation=evaluation,object_counts=list(object_counts),material_layout=material_layout,
        attempts=attempts,gpus=gpus,hours=hours,n_grid=n_grid,shutdown_reserve_seconds=3600,
        minimum_free_bytes=1<<40,access_gid=gid,gate_sha256=digest(output/'GATE.json'))
    request['files']={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    request['request_sha256']=identity(request)
    write_once(output/'BUNDLE.json',request)
    reservation=registry/(request['namespace']+'.json')
    write_once(reservation,dict(namespace=request['namespace'],bundle=str(output),
        bundle_sha256=digest(output/'BUNDLE.json'),owner_uid=os.getuid(),attempts=attempts))
    share_created(reservation,gid)
    share_created(output,gid)
    return dict(bundle=str(output),bundle_sha256=digest(output/'BUNDLE.json'),namespace=request['namespace'],submitted=False)


def check_purpose(purpose,attempts,gpus,hours):
    if purpose not in ('pilot','production'):raise ValueError('Unknown collection purpose')
    if purpose=='pilot' and (attempts>108 or gpus!=1 or hours!=2):
        raise ValueError('Operator pilot is capped at 108 attempts, one GPU, two hours')


def verify(root):
    root=Path(root).resolve(strict=True);config=read(root/'BUNDLE.json')
    if config.get('schema')!=SCHEMA:raise ValueError('Unsupported drag bundle')
    settings(**{k:config[k] for k in ('name','attempts','gpus','hours','split','evaluation','object_counts','material_layout')},n_grid=config.get('n_grid',100))
    check_purpose(config['purpose'],config['attempts'],config['gpus'],config['hours'])
    if identity(config)!=config.get('request_sha256'):raise ValueError('Bundle request changed')
    if (config['shutdown_reserve_seconds']!=3600 or config['minimum_free_bytes']!=1<<40
            or not re.fullmatch(re.escape(config['name'])+'-[0-9a-f]{32}',config['namespace'])):
        raise ValueError('Unqualified admission/namespace settings')
    if digest(Path(config['assets'])/'COMPLETE.json')!=config['bank_sha256']:raise ValueError('Bundle bank changed')
    reservation=read(Path(config['registry'])/(config['namespace']+'.json'))
    if (reservation.get('namespace')!=config['namespace'] or reservation.get('bundle')!=str(root)
            or reservation.get('bundle_sha256')!=digest(root/'BUNDLE.json')):
        raise ValueError('Namespace reservation mismatch; never run a copied bundle as a new collection')
    required={'GATE.json','collect.sbatch',*('source/'+name for name in CRITICAL)}
    if not required<=set(config['files']):raise ValueError('Incomplete frozen source/consumer closure')
    for name,sha in config['files'].items():
        if digest(relative_path(root,name))!=sha:raise ValueError('Bundle changed: '+name)
    if digest(root/'GATE.json')!=config['gate_sha256']:raise ValueError('Gate changed')
    gate=check_gate(root/'GATE.json',config['assets'],**{k:config[k] for k in ('split','evaluation','object_counts','material_layout','purpose')},n_grid=config.get('n_grid',100))
    for name,sha in gate['critical_source_sha256'].items():
        if digest(root/'source'/name)!=sha:raise ValueError('Frozen source differs from qualification: '+name)
    return config


def submit(root,*,test_only=False):
    root=Path(root).resolve(strict=True);config=verify(root)
    argv=['sbatch','--test-only' if test_only else '--parsable',f'--gres=gpu:RTX_PRO_6000:{config["gpus"]}',
        f'--cpus-per-task={8*config["gpus"]}','--mem=128G',f'--time={config["hours"]}:00:00',
        '--job-name=drag-'+config['name'],'--output='+str(root/'slurm-%j.log'),
        str(root/'collect.sbatch'),str(root),config['python']]
    if not test_only:write_once(root/'SUBMISSION_CLAIM.json',dict(bundle_sha256=digest(root/'BUNDLE.json'),command=argv))
    try:result=subprocess.run(argv,capture_output=True,text=True,check=True)
    except (subprocess.CalledProcessError,OSError) as error:
        if not test_only:
            write_once(root/'SUBMIT_FAILED.json',dict(error=str(error),reservation_preserved=True))
            share_created(root/'SUBMIT_FAILED.json',config['access_gid'])
        raise
    if not test_only:
        write_once(root/'SUBMITTED.json',dict(job_id=result.stdout.strip(),stderr=result.stderr))
        for name in ('SUBMISSION_CLAIM.json','SUBMITTED.json'):share_created(root/name,config['access_gid'])
    return dict(test_only=test_only,stdout=result.stdout.strip(),stderr=result.stderr)


def status(root):
    """Receipt-only status; a started receipt is not evidence of a live process."""
    root=Path(root).resolve(strict=True);config=read(root/'BUNDLE.json')
    if config.get('schema')!=SCHEMA:raise ValueError('Unsupported drag bundle')
    rows=[read(p) for p in sorted((root/'collection').glob('block-*/raws/grouped/group-*/drag-*/RESULT.json'))]
    blocks=[read(p) for p in sorted((root/'collection').glob('block-*/COMPLETE.json'))]
    state=next((label for name,label in (('COMPLETE.json','complete'),('FAILURE.json','failed'),
        ('SUBMIT_FAILED.json','submit_failed'),('RUN_CLAIM.json','started_liveness_unconfirmed'),
        ('SUBMITTED.json','submitted_liveness_unconfirmed')) if (root/name).exists()),'prepared')
    return dict(name=config['name'],purpose=config['purpose'],state=state,planned_attempts=config['attempts'],
        attempted_with_receipts=len(rows),retained_raw=sum(r.get('retained',False) for r in rows),
        exported_clips=sum(r['retained'] for r in blocks),completed_blocks=len(blocks),
        dataset_root=str(root/'data'/config['split']/config['evaluation']),
        counts_are_receipts_not_live_scheduler_state=True)


def reviewed_export(paths, kept, staging, destination):
    """Never expose TARs to training discovery before consumer checks pass."""
    from ..export import export_candidates
    from ..wall_materials import STRICT
    from .review import loader_review
    staging=Path(staging);destination=Path(destination)
    if destination.exists():raise FileExistsError(destination)
    selection=export_candidates(paths,staging,wall_policy=STRICT) if paths else dict(retained=0)
    if selection['retained']!=len(kept):raise ValueError('Raw/export selection mismatch')
    consumer=loader_review(staging/'diagnostic-export',kept) if kept else []
    if kept and (len(consumer)!=2 or any(r['episodes']!=len(kept)
            or not r['exact_raw_window_and_command_match'] for r in consumer)):
        raise ValueError('Incomplete consumer approval')
    if staging.exists():
        write_once(staging/'LOADER_REVIEW.json',dict(consumer=consumer,retained=len(kept)))
        destination.parent.mkdir(parents=True,exist_ok=True)
        # Export artifact paths are relative; same-filesystem rename preserves
        # their hashes and atomically publishes the reviewed directory.
        os.rename(staging,destination)
    return selection,consumer


def worker(root,lane,gpu,deadline):
    root=Path(root).resolve(strict=True);config=verify(root)
    if (type(lane) is not int or not 0<=lane<config['gpus'] or not math.isfinite(deadline)
            or deadline<=time.time() or deadline>time.time()+config['hours']*3600):
        raise ValueError('Invalid bounded worker lane/deadline')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('Wrong worker device')
    if runtime.ROOT!=root/'source':raise ValueError('Worker must import its frozen bundle, not a working tree')
    cpus=sorted(os.sched_getaffinity(0));assigned=cpus[lane::config['gpus']]
    if not assigned:raise ValueError('Insufficient CPU affinity for lanes')
    os.sched_setaffinity(0,assigned)
    from .prepare import prepare
    from .matrix import run as matrix
    from ..export import export_candidates
    from ..wall_materials import STRICT
    from .review import loader_review
    from .interactions import measure
    from .contact import measure as measure_coupling
    from ..validate import read_raw
    from ..export import POLICY
    result=[];block=lane;quarantined=[];consecutive_export_failures=0
    while block*108<config['attempts']:
        if time.time()+config['shutdown_reserve_seconds']>=deadline:break
        if shutil.disk_usage(root).free<config['minimum_free_bytes']+(32<<30):break
        start=block*108;count=min(108,config['attempts']-start)
        folder=root/'collection'/f'block-{block:06d}';folder.mkdir(parents=True,exist_ok=False)
        prepare(folder/'inputs',config['assets'],name=config['namespace'],count=count,start=start,
                **{k:config[k] for k in ('split','evaluation','object_counts','material_layout')},n_grid=config.get('n_grid',100))
        if block==lane:
            # Startup fault/isolation check on each freshly allocated device.
            label=read(folder/'inputs/PLAN.json')['cases'][0]['label'];case=folder/'inputs/cases'/label
            with (folder/'startup.log').open('x') as log:
                subprocess.run([config['python'],'-m','collection.mpm.drag.probe','--initial',str(case/'initial.npz'),
                    '--manifest',str(case/'MANIFEST.json'),'--output',str(folder/'STARTUP.json'),'--uuid',gpu],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        matrix(folder/'inputs',folder/'raws',gpu,serial=False,budget_seconds=max(3600,int(deadline-time.time())))
        outcomes=[read(p) for p in sorted((folder/'raws/grouped').glob('group-*/drag-*/RESULT.json'))]
        paths=[r['raw_path'] for r in outcomes if r['status']=='complete']
        export=root/'data'/config['split']/config['evaluation']/f'block-{block:06d}'
        kept=[r['raw_path'] for r in outcomes if r['retained']]
        pending=root/'export-review'/f'block-{block:06d}'
        try:
            selection,consumer=reviewed_export(paths,kept,pending,export)
        except Exception:
            # Simulation failures above still stop the run. A postprocessing
            # failure preserves raws and unapproved TARs outside the train root,
            # allowing both GPU lanes to continue useful independent collection.
            failure=dict(block=block,attempted=len(outcomes),retained_raw=len(kept),
                error=traceback.format_exc(),pending_export=str(pending),training_published=False)
            write_once(folder/'EXPORT_FAILURE.json',failure);quarantined.append(failure)
            share_created(folder,config['access_gid'])
            if pending.exists():share_created(pending,config['access_gid'])
            print(json.dumps(dict(block=block,export_quarantined=True,retained_raw=len(kept))),flush=True)
            consecutive_export_failures+=1
            if consecutive_export_failures>=3:raise RuntimeError('Three consecutive export failures; bounded quarantine stop')
            if len(outcomes)!=count:break
            block+=config['gpus'];continue
        consecutive_export_failures=0
        interactions=[dict(raw_path=p,**measure(*read_raw(p,POLICY))) for p in kept]
        write_once(folder/'INTERACTIONS.json',dict(episodes=interactions,
            transfer_candidates=sum(r['any_transfer_candidate'] for r in interactions),
            exact_contact_measurement=False))
        coupling=[dict(raw_path=p,**measure_coupling(*read_raw(p,POLICY))) for p in kept]
        write_once(folder/'GRID_COUPLING.json',dict(episodes=coupling,
            transfer_candidates=sum(r['any_transfer_candidate'] for r in coupling),
            exact_mesh_contact_or_impulse=False))
        row=dict(block=block,planned=count,attempted=len(outcomes),retained=len(kept),
                 selection=selection,consumer=consumer,source_plan_sha256=digest(folder/'inputs/PLAN.json'))
        write_once(folder/'COMPLETE.json',row);result.append(row)
        share_created(folder,config['access_gid'])
        if export.exists():share_created(export,config['access_gid'])
        print(json.dumps({k:row[k] for k in ('block','planned','attempted','retained')}),flush=True)
        if len(outcomes)!=count:break  # graceful deadline stop; do not invent completed attempts
        block+=config['gpus']
    write_once(root/f'LANE-{lane}.json',dict(blocks=len(result),attempted=sum(r['attempted'] for r in result),
        retained=sum(r['retained'] for r in result),quarantined_blocks=quarantined,
        reason='budget_or_disk_admission_finished'))
    share_created(root/f'LANE-{lane}.json',config['access_gid'])


def allocation_deadline(job_id,hours):
    if not re.fullmatch(r'[0-9]+',job_id):raise ValueError('Explicit numeric allocation required')
    result=subprocess.check_output(['scontrol','show','job',job_id,'-o'],text=True,timeout=30)
    fields=dict(token.split('=',1) for token in result.split() if '=' in token)
    if fields.get('JobId')!=job_id or fields.get('JobState')!='RUNNING':raise ValueError('Allocation is not running')
    # scontrol emits local node time; timestamp() uses that same local timezone.
    end=datetime.fromisoformat(fields['EndTime']).timestamp()
    deadline=min(end,time.time()+hours*3600)-300
    if deadline-time.time()<3700:raise ValueError('Insufficient allocated time plus shutdown reserve')
    return deadline


def run(root):
    root=Path(root).resolve(strict=True);config=verify(root)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Run requires an explicit Slurm GPU allocation')
    if runtime.ROOT!=root/'source':raise ValueError('Run must import its frozen bundle')
    if Path(sys.executable).resolve()!=Path(config['python']).resolve():raise ValueError('Unqualified Python interpreter')
    deadline=allocation_deadline(os.environ['SLURM_JOB_ID'],config['hours'])
    write_once(root/'RUN_CLAIM.json',dict(job_id=os.environ['SLURM_JOB_ID'],bundle_sha256=digest(root/'BUNDLE.json')))
    share_created(root/'RUN_CLAIM.json',config['access_gid'])
    code='import json,warp as w;assert w.__version__=="1.10.1";w.init();print("DEVICES="+json.dumps([dict(uuid=d.uuid,name=d.name) for d in w.get_cuda_devices()]))'
    output=subprocess.check_output([config['python'],'-c',code],text=True,timeout=60)
    devices=json.loads(next(s[8:] for s in output.splitlines() if s.startswith('DEVICES=')))
    if len(devices)!=config['gpus'] or len({d['uuid'] for d in devices})!=len(devices) or any('RTX PRO 6000' not in d['name'] for d in devices):
        raise ValueError('Allocation must contain the requested number of distinct PRO 6000 GPUs')
    children=[];logs=[]
    def stopped(signum,frame):raise SystemExit('Drag supervisor received signal '+str(signum))
    handlers={s:signal.signal(s,stopped) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        for lane,device in enumerate(devices):
            log=(root/f'lane-{lane}.log').open('x');logs.append(log)
            argv=[config['python'],'-u','-m','collection.mpm.drag.launch','worker','--bundle',str(root),
                '--lane',str(lane),'--uuid',device['uuid'],'--deadline',str(deadline)]
            children.append(subprocess.Popen(argv,env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid']),
                cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None,0) for child in children):raise RuntimeError('A drag lane failed; inspect retained lane logs')
            if time.time()>deadline:raise TimeoutError('Drag job deadline')
            time.sleep(2)
        if any(child.returncode for child in children):raise RuntimeError('Drag lane failed')
        write_once(root/'COMPLETE.json',dict(lanes=[read(root/f'LANE-{i}.json') for i in range(len(children))],
            job_id=os.environ['SLURM_JOB_ID']))
        share_created(root/'COMPLETE.json',config['access_gid'])
    except BaseException:
        write_once(root/'FAILURE.json',dict(error=traceback.format_exc(),outputs_preserved=True))
        share_created(root/'FAILURE.json',config['access_gid'])
        raise
    finally:
        # Every worker owns a new process group, inherited by its simulator
        # descendants. Stop only groups created here, including orphaned children.
        for child in children:
            try:os.killpg(child.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        for child in children:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait()
        for log in logs:log.close()
        for sig,previous in handlers.items():signal.signal(sig,previous)


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    b=s.add_parser('bundle');b.add_argument('--output',required=True);b.add_argument('--assets',required=True)
    b.add_argument('--gate',required=True);b.add_argument('--name',required=True);b.add_argument('--python',required=True)
    b.add_argument('--split',default='train');b.add_argument('--evaluation',default='train')
    b.add_argument('--object-counts',nargs='+',type=int,default=[1,2,3,4]);b.add_argument('--material-layout',default='homogeneous')
    b.add_argument('--attempts',type=int,default=20000);b.add_argument('--gpus',type=int,default=2);b.add_argument('--hours',type=int,default=48)
    b.add_argument('--purpose',choices=('pilot','production'),default='production')
    b.add_argument('--n-grid',type=int,choices=(100,256),default=100)
    b.add_argument('--registry',help='Common group-writable namespace registry; defaults beside output bundles')
    for name in ('submit','verify','status','run','worker'):
        q=s.add_parser(name);q.add_argument('--bundle',required=True)
        if name=='submit':q.add_argument('--test-only',action='store_true')
        if name=='worker':q.add_argument('--lane',type=int,required=True);q.add_argument('--uuid',required=True);q.add_argument('--deadline',type=float,required=True)
    a=p.parse_args()
    if a.command=='bundle':print(json.dumps(bundle(a.output,a.assets,a.gate,**{k:v for k,v in vars(a).items() if k not in ('command','output','assets','gate')})))
    elif a.command=='submit':print(json.dumps(submit(a.bundle,test_only=a.test_only)))
    elif a.command=='verify':print(json.dumps(verify(a.bundle)))
    elif a.command=='status':print(json.dumps(status(a.bundle)))
    elif a.command=='run':run(a.bundle)
    else:
        try:worker(a.bundle,a.lane,a.uuid,a.deadline)
        except BaseException:
            write_once(Path(a.bundle)/f'LANE-{a.lane}-FAILURE.json',dict(error=traceback.format_exc(),outputs_preserved=True))
            raise


if __name__=='__main__':main()
