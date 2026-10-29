#!/usr/bin/env python3
"""Operator entry point for qualified four-PRO6000 MPM training collection.

Use preflight/plan without creating a job; launch prepares and submits once.
No simulator dependencies are imported into this standard-library front end.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import mpm_settings as settings

DEFAULT_CONFIG=Path(__file__).with_name('mpm_collect.json')
NAME=re.compile(r'[a-z][a-z0-9-]{0,47}\Z')


def read(path): return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_once(path,value):
    path=Path(path)
    # Link a fully flushed temporary file into place atomically, without
    # overwriting an existing receipt (including concurrent submit claims).
    import tempfile
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            # mkstemp defaults to 0600; registry readers in the lab group must
            # see committed reservations made by a different UID.
            os.fchmod(stream.fileno(),0o640)
            json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.link(tmp,path)
    finally:
        os.unlink(tmp)


def load_config(path):
    c=read(path)
    allowed={'schema','template_bundle','template_sha256','campaigns_root','registry_root','first_cycle',
             'last_cycle_exclusive','cycles_per_run','partition','account','qos','existing_runs',
             'shared_assets','access_group','collection'}
    if set(c)-allowed:raise ValueError('Unknown operator setting: '+', '.join(sorted(set(c)-allowed)))
    if c['schema']!='mpm-operator-v1':raise ValueError('Unsupported operator configuration')
    if c['cycles_per_run']!=2000 or c['first_cycle']!=84000 or c['last_cycle_exclusive']!=100000:
        raise ValueError('Unreviewed reservation bounds; do not bypass the sampler limit')
    for key in ('template_bundle','campaigns_root','registry_root'):
        if not Path(c[key]).is_absolute():raise ValueError('Absolute path required: '+key)
        if any(x in c[key] for x in (',','\n','\r')):raise ValueError('Path is unsafe for Slurm export: '+key)
    if not re.fullmatch('[0-9a-f]{64}',c['template_sha256']):raise ValueError('Pinned template SHA required')
    for key in ('partition','account','qos'):
        if not re.fullmatch('[A-Za-z0-9_-]+',c[key]):raise ValueError('Invalid scheduler setting: '+key)
    c['collection']=settings.normalize(c.get('collection'))
    return c


def environment(bundle=None):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if bundle is not None:env['PYTHONPATH']=str(Path(bundle)/'source')
    return env


def checked(command, *, env=None, timeout=1800):
    r=subprocess.run([str(x) for x in command],env=env,text=True,capture_output=True,timeout=timeout)
    if r.returncode:
        raise RuntimeError('Command failed: '+str(command[0])+'\n'+r.stdout[-4000:]+'\n'+r.stderr[-4000:])
    return r.stdout.strip()


def validate_name(name):
    if not NAME.fullmatch(name or ''):raise ValueError('Run name: lowercase letters/digits/hyphens, starting with a letter; max 48 characters')
    return name


def records(c):
    root=Path(c['registry_root'])
    return [read(p) for p in sorted(root.glob('*.json'))]


def next_cycle(c, reservations):
    size=c['cycles_per_run']
    for start in range(c['first_cycle'],c['last_cycle_exclusive'],size):
        if all(start+size<=r['cycle_start'] or start>=r['cycle_stop'] for r in reservations):return start
    raise ValueError('Reserved sampler space exhausted. Stop and extend/requalify the sampler; never reuse old cycles.')


def preflight(c, *, scheduler=True, require_launch=True):
    recipe=settings.normalize(c.get('collection'));capability=settings.capability(recipe)
    if require_launch and not capability['launch_supported']:raise ValueError(capability['reason'])
    bundle=Path(c['template_bundle'])
    if digest(bundle/'BUNDLE.json')!=c['template_sha256']:raise ValueError('Template identity changed')
    config=read(bundle/'BUNDLE.json')
    if recipe['hardware']['gpus']==2 and config.get('schema')!='mpm-pro-training-campaign-v3':
        raise ValueError('Two-GPU collection requires a v4-or-later compatible template, not the historical four-only runtime')
    if config.get('configuration_schema')!=settings.VERSION:
        if recipe!=settings.DEFAULT or not require_launch:
            raise ValueError('This template predates configurable collection; select a published settings-v1 template')
    if c.get('shared_assets') and Path(c['shared_assets']).resolve()!=Path(config['banks']).resolve():
        raise ValueError('Shared asset path must match the pinned template bank')
    python=Path(config['genesis_python'])
    if not os.access(python,os.X_OK):raise PermissionError('Cannot execute collection Python: '+str(python))
    for key in ('campaigns_root','registry_root'):
        p=Path(c[key]);ancestor=p
        while not ancestor.exists():ancestor=ancestor.parent
        if not os.access(ancestor,os.W_OK|os.X_OK):raise PermissionError('Output/registry parent is not writable: '+str(ancestor))
        if shutil.disk_usage(ancestor).free<500*1024**3:raise ValueError('Less than 500 GiB free: '+str(ancestor))
    if scheduler:
        for command in ('sbatch','scontrol','squeue','sacct'):
            if shutil.which(command) is None:raise RuntimeError('Missing Slurm command: '+command)
    checked([python,bundle/'mpm_training_campaign.py','verify','--bundle',bundle],env=environment(bundle))
    return dict(passed=True,scope='MPM passive training, 1–4 objects only',gpus=str(recipe['hardware']['gpus'])+' x RTX_PRO_6000',
        walltime=str(recipe['hardware']['walltime_hours'])+' hours',cycle_capacity_per_run=c['cycles_per_run'],maximum_attempts=recipe['max_attempts'],
        collection=recipe,capability=capability,
        template_sha256=c['template_sha256'],python=str(python),bank=config['banks'],
        registry=c['registry_root'],note='No job submitted; no seed reservation created. Cross-user permissions must pass under the operator account.')


def assets(c):
    root=Path(c.get('shared_assets') or read(Path(c['template_bundle'])/'BUNDLE.json')['banks'])
    complete=read(root/'COMPLETE.json');split=read(root/'SPLITS.json')
    rows={role:read(root/role/'MANIFEST.json')['objects'] for role in ('train','validation','test')}
    sets={role:{r['asset_uid'] for r in values} for role,values in rows.items()}
    for role,values in rows.items():
        if len(values)!=complete['cohorts'][role] or len(sets[role])!=len(values) or not sets[role]<=set(split['cohorts'][role]):
            raise ValueError('Asset membership differs from frozen split')
    if any(sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Asset splits overlap')
    return dict(root=str(root),group=c.get('access_group'),cohorts=complete['cohorts'],
        manifests={r:str(root/r/'MANIFEST.json') for r in rows},split_manifest=str(root/'SPLITS.json'),
        prepared_assets_only=True,evaluation_generation_approved=False,
        note='Same shared assets, distinct sequences via the common locked cycle registry. Assets are not included in training merely because they are readable.')


def plan(c,name,continue_from=None):
    validate_name(name)
    recipe=settings.normalize(c.get('collection'))
    previous=None
    if continue_from:
        validate_name(continue_from)
        previous=read(Path(c['registry_root'])/(continue_from+'.json'))
    rs=records(c)
    if any(r['name']==name for r in rs):raise FileExistsError('Run name already reserved; use status/submit, not a new prepare')
    cycle=next_cycle(c,rs)
    root=Path(c['campaigns_root'])/name
    if root.exists():raise FileExistsError('Output already exists: '+str(root))
    return dict(schema='mpm-run-v1',name=name,cycle_start=cycle,cycle_stop=cycle+c['cycles_per_run'],
        cycles=settings.cycles(recipe),root=str(root),bundle=str(root/'bundle'),output=str(root/'data'),
        collection=recipe,recipe_sha256=settings.identity(recipe),capability=settings.capability(recipe),
        template_sha256=c['template_sha256'],template_bundle=c['template_bundle'],
        continue_from=None if previous is None else previous['name'],
        continuation_semantics='New disjoint trajectories; previous reviewed data stays in its original root.',
        partition=c['partition'],account=c['account'],qos=c['qos'],owner_uid=os.getuid(),
        proposed_only=True)


@contextmanager
def registry_lock(c):
    root=Path(c['registry_root']);root.mkdir(parents=True,exist_ok=True)
    path=root/'.lock'
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o660)
        os.fchmod(fd,0o660)
    except FileExistsError:
        # NFS may implement flock through POSIX locks, requiring a writable
        # descriptor for LOCK_EX. The shared publisher grants group rw here.
        fd=os.open(path,os.O_RDWR)
    with os.fdopen(fd,'r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        yield


def prepare(c,name,continue_from=None):
    preflight(c)
    with registry_lock(c):
        r=plan(c,name,continue_from);r['proposed_only']=False;r['created_epoch']=time.time()
        write_once(Path(c['registry_root'])/(name+'.json'),r)
        root=Path(r['root']);root.mkdir(parents=True,exist_ok=False)
        os.chmod(root,0o2750)
        write_once(root/'RESERVED.json',r)
    # On any subsequent failure retain the reservation. Never silently reclaim
    # seeds or overwrite a partial frozen bundle.
    try:
        template=Path(c['template_bundle']);config=read(template/'BUNDLE.json')
        command=[config['genesis_python'],template/'mpm_training_campaign.py','freeze',
            '--bundle',r['bundle'],'--source',config['source_bundle'],'--evidence',config['evidence'],
            '--cycle-start',str(r['cycle_start']),'--cycles',str(r['cycles'])]
        if config.get('configuration_schema')==settings.VERSION:
            command+=['--recipe-json',json.dumps(r['collection'])]
        checked(command,env=environment(template))
        bundle=Path(r['bundle'])
        checked([config['genesis_python'],bundle/'mpm_training_campaign.py','verify','--bundle',bundle],env=environment(bundle))
        # Read-only team inspection must not depend on the creator's umask.
        # These are newly frozen run files, never an existing collector output.
        for path in [bundle]+list(bundle.rglob('*')):
            if path.is_symlink():raise ValueError('Unexpected frozen bundle symlink: '+str(path))
            os.chmod(path,0o2750 if path.is_dir() else 0o640)
        r=dict(r,bundle_sha256=digest(bundle/'BUNDLE.json'))
        write_once(root/'READY.json',r)
        return r
    except BaseException as exc:
        write_once(root/'PREPARE_FAILED.json',dict(error=repr(exc),reservation_preserved=True))
        raise


def ready(c,name):
    validate_name(name)
    reservation=read(Path(c['registry_root'])/(name+'.json'))
    root=Path(reservation['root']);r=read(root/'READY.json')
    if any(r.get(k)!=v for k,v in reservation.items()):raise ValueError('Ready receipt does not match reservation')
    if r['owner_uid']!=os.getuid():raise PermissionError('This run belongs to another UID; ask its owner to operate it')
    if r['template_sha256']!=c['template_sha256']:raise ValueError('Run belongs to another template version')
    if root.resolve()!=Path(c['campaigns_root']).resolve()/name:raise ValueError('Registry root mismatch')
    bundle=Path(r['bundle'])
    if digest(bundle/'BUNDLE.json')!=r['bundle_sha256']:raise ValueError('Prepared bundle changed')
    config=read(bundle/'BUNDLE.json')
    requested=settings.normalize(c.get('collection'))
    if r.get('collection',settings.DEFAULT)!=requested or settings.normalize(config.get('recipe_profile'))!=requested:
        raise ValueError('Run configuration differs; use its original config or a new run name')
    if not settings.capability(requested)['launch_supported']:raise ValueError('Evaluation collection is not approved')
    if config['cycle_start']!=r['cycle_start'] or config['training_cycles']!=r['cycles']:
        raise ValueError('Prepared cycle range differs from reservation')
    return root,r,config


def submission_command(r,config, *, test_only=False):
    root=Path(r['root']);bundle=Path(r['bundle'])
    command=['sbatch','--parsable','--job-name=mpm-'+r['name'],'--chdir='+str(root),
        '--gres=gpu:RTX_PRO_6000:'+str(settings.normalize(r.get('collection'))['hardware']['gpus']),
        '--time='+str(settings.normalize(r.get('collection'))['hardware']['walltime_hours'])+':00:00',
        '--output='+str(root/'slurm-%j.log'),'--partition='+r['partition'],
        '--account='+r['account'],'--qos='+r['qos'],
        '--comment=mpm-operator:'+r['name']+':'+r['bundle_sha256'][:12],
        '--export=ALL,MPM_TRAINING_BUNDLE='+str(bundle)+',MPM_TRAINING_OUTPUT='+r['output']+',MPM_TRAINING_PYTHON='+config['genesis_python'],
        str(bundle/'mpm_training_campaign.sbatch')]
    if test_only:command.insert(1,'--test-only')
    return command


def submit(c,name, *, test_only=False):
    root,r,config=ready(c,name)
    if (root/'SUBMITTED.json').exists():return dict(read(root/'SUBMITTED.json'),already_submitted=True)
    if (root/'SUBMISSION_CLAIM.json').exists():
        raise RuntimeError('Submission already claimed; inspect SUBMISSION_UNCERTAIN.json and Slurm before any retry. No duplicate submitted.')
    if Path(r['output']).exists():raise FileExistsError('Runtime output already exists; never overwrite it')
    checked([config['genesis_python'],Path(r['bundle'])/'mpm_training_campaign.py','verify','--bundle',r['bundle']],
            env=environment(r['bundle']))
    command=submission_command(r,config,test_only=test_only)
    if test_only:
        # Slurm writes its forecast to stderr on success.
        result=subprocess.run(command,text=True,capture_output=True,timeout=60)
        if result.returncode:raise RuntimeError(result.stderr)
        return dict(submitted=False,scheduler_check=result.stdout+result.stderr,command=command)
    write_once(root/'SUBMISSION_CLAIM.json',dict(command=command,time_epoch=time.time(),bundle_sha256=r['bundle_sha256']))
    try:
        output=checked(command,timeout=60)
        if not re.fullmatch(r'\d+(;[A-Za-z0-9_.-]+)?',output):raise ValueError('Unexpected sbatch receipt: '+output)
        result=dict(job_id=output.split(';')[0],slurm_receipt=output,command=command,
                    root=str(root),output=r['output'],submitted_epoch=time.time(),bundle_sha256=r['bundle_sha256'])
        write_once(root/'SUBMITTED.json',result)
        return result
    except BaseException as exc:
        write_once(root/'SUBMISSION_UNCERTAIN.json',dict(error=repr(exc),automatic_retry=False,
            instruction='The scheduler may have accepted the job. Find its mpm-operator comment before resolving this claim.'))
        raise


def launch(c,name,continue_from=None):
    validate_name(name)
    if not (Path(c['registry_root'])/(name+'.json')).exists():prepare(c,name,continue_from)
    elif continue_from is not None:
        if read(Path(c['registry_root'])/(name+'.json'))['continue_from']!=continue_from:
            raise ValueError('Existing run has a different continuation parent')
    return submit(c,name)


def preview(c,out,count=12):
    if type(count) is not int or not 1<=count<=60:raise ValueError('Preview count must be 1..60')
    if out.exists():raise FileExistsError('Preview output already exists')
    preflight(c,scheduler=False,require_launch=False)
    bundle=Path(c['template_bundle']);config=read(bundle/'BUNDLE.json')
    checked([config['genesis_python'],bundle/'mpm_training_campaign.py','preview','--bundle',bundle,
        '--unit',out.resolve(),'--recipe-json',json.dumps(settings.normalize(c.get('collection'))),
        '--preview-count',str(count)],env=environment(bundle))
    return dict(root=str(out.resolve()),preview_only=True,production_reservation=False,
        collection_approved=False,capability=settings.capability(c.get('collection')))


def collection_status(root):
    root=Path(root);results=[];totals=Counter()
    for f in sorted((root/'chunks').glob('*/COMPLETE.json')):
        r=read(f)
        if r.get('status')!='complete':raise ValueError('Invalid completed receipt: '+str(f))
        totals[r['entry']['evaluation']]+=r['retained'];results.append(r)
    out=dict(root=str(root),exists=root.exists(),completed_chunks=len(results),retained=dict(totals),
        attempts=sum(r['attempts'] for r in results),counts_from='COMPLETE receipts; not a full hash audit',
        dataset_root=str(root/'datasets/train'),accepted_for_training=False)
    for name in ('STATUS.json','STOP.json','FAILURE.json'):
        f=root/name
        if f.exists():
            r=read(f);out[name]={k:v for k,v in r.items() if k!='dataset_roots'}
    return out


def status(c,name=None):
    if name is None:
        return dict(registered=[status(c,r['name']) for r in records(c)],
            existing=[dict(name=r['name'],**collection_status(r['collection_root'])) for r in c.get('existing_runs',[])])
    validate_name(name);r=read(Path(c['registry_root'])/(name+'.json'));root=Path(r['root'])
    result=dict(name=name,root=str(root),cycle_start=r['cycle_start'],cycle_stop=r['cycle_stop'],
                recipe=r.get('collection',settings.DEFAULT),scheduled_cycles=r['cycles'],
                ready=(root/'READY.json').exists(),collection=collection_status(root/'data/collection'))
    for n in ('PREPARE_FAILED.json','SUBMISSION_UNCERTAIN.json','SUBMITTED.json'):
        f=root/n
        if f.exists():result[n]=read(f)
    for n in ('FAILURE.json','qualification/TRAINING_PASS.json'):
        f=root/'data'/n
        if f.exists():result[n]=read(f) if n=='FAILURE.json' else 'passed'
    recovery=root/'ACTIVE_RECOVERY.json'
    if recovery.exists():
        pointer=read(recovery);recovery_root=Path(pointer['root']).resolve()
        if not recovery_root.is_relative_to(root.resolve()):raise ValueError('Recovery root escapes registered campaign')
        capsule=recovery_root/'capsule/RECOVERY.json'
        if digest(capsule)!=pointer['capsule_sha256']:raise ValueError('Recovery capsule identity changed')
        active=collection_status(recovery_root/'collection')
        recovered=read(recovery_root/'RECOVERED.json')
        previous=[]
        for path in read(capsule).get('previous_collections',[]):
            prior=Path(path).resolve()
            if not prior.is_relative_to(root.resolve()) or prior in (recovery_root/'collection',root/'data/collection'):
                raise ValueError('Invalid previous recovery collection')
            if str(prior) in [r['root'] for r in previous]:raise ValueError('Duplicate previous collection')
            previous.append(collection_status(prior))
        result['active_recovery']=dict(root=str(recovery_root),collection=active,
            recovered_saved_clips=recovered['retained'],original_failure_preserved=True,
            previous_collections=previous)
        result['total_retained_with_recovery']=(sum(result['collection']['retained'].values())
            +sum(active['retained'].values())+recovered['retained']
            +sum(sum(r['retained'].values()) for r in previous))
        result['status_note']='Original FAILURE/STATUS receipts are historical; inspect active_recovery for the running continuation.'
    if 'SUBMITTED.json' in result and shutil.which('squeue'):
        job=result['SUBMITTED.json']['job_id']
        try:
            result['scheduler']=checked(['squeue','-h','-j',job,'-o','%i %T %M %R'],timeout=30)
            if not result['scheduler']:result['scheduler']=checked(['sacct','-n','-X','-j',job,'--format=JobID,State,Elapsed,ExitCode'],timeout=30)
        except RuntimeError as exc:result['scheduler_unavailable']=str(exc)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('preflight')
    sub.add_parser('assets')
    q=sub.add_parser('preview');q.add_argument('--out',type=Path,required=True);q.add_argument('--count',type=int,default=12)
    for action in ('plan','prepare','launch','submit','status','audit'):
        q=sub.add_parser(action);q.add_argument('--name',required=action!='status')
        if action in ('plan','prepare','launch'):q.add_argument('--continue-from')
        if action=='submit':q.add_argument('--test-only',action='store_true')
    a=p.parse_args();c=load_config(a.config)
    if a.command=='preflight':result=preflight(c)
    elif a.command=='assets':result=assets(c)
    elif a.command=='preview':result=preview(c,a.out,a.count)
    elif a.command=='plan':result=plan(c,a.name,a.continue_from)
    elif a.command=='prepare':result=prepare(c,a.name,a.continue_from)
    elif a.command=='launch':result=launch(c,a.name,a.continue_from)
    elif a.command=='submit':result=submit(c,a.name,test_only=a.test_only)
    elif a.command=='status':result=status(c,a.name)
    else:
        root,r,config=ready(c,a.name)
        result=checked([config['genesis_python'],Path(r['bundle'])/'mpm_training_campaign.py','audit',
                        '--bundle',r['bundle'],'--root',str(root/'data/collection')],env=environment(r['bundle']))
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as exc:
        print('MPM operator error: '+str(exc),file=sys.stderr);sys.exit(1)
