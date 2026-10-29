"""Stage a lab-readable launch/reference closure without exposing private homes."""
import argparse
import grp
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import mpm_collect as operator

DRIVERS = ('mpm_ada.py', 'mpm_campaign.py', 'mpm_four.py', 'mpm_handoff.py',
           'mpm_scale.py', 'mpm_training.py', 'mpm_training_campaign.py',
           'mpm_training_campaign.sbatch','mpm_settings.py','mpm_async_io.py')
LAUNCH = Path('collection/mpm/launch')
DOCS = Path('collection/mpm/docs')


def frozen_path(name):
    """Preserve the existing deployed layout while source stays domain-local."""
    return Path(name).name if Path(name).parent==LAUNCH else 'source/'+name


def maintained_sources(project):
    """Training runtime only; evaluation operators freeze their own closure."""
    project=project.resolve(strict=True)
    evaluation_operators={'collection/mpm/evaluation.py','collection/mpm/evaluation_suite.py',
                         'collection/mpm/evaluation_supervisor.py'}
    files={str(p.relative_to(project)):p for folder in ('collection','collection/mpm')
           for p in (project/folder).glob('*.py') if str(p.relative_to(project)) not in evaluation_operators}
    files.update({str(LAUNCH/name):project/LAUNCH/name for name in DRIVERS})
    if not files or any(not p.is_file() for p in files.values()):
        raise ValueError('Incomplete maintained MPM source')
    return files


def repository_receipt(project,files):
    """A deployment must identify committed source, not private worktree edits."""
    revision=operator.checked(['git','-C',project,'rev-parse','HEAD'])
    status=operator.checked(['git','-C',project,'status','--porcelain','--untracked-files=all','--',*sorted(files)])
    if status:raise ValueError('Commit the MPM source before publishing a deployment:\n'+status)
    # Status alone misses ignored/untracked files, so compare every input with
    # the committed blob as well. Refuse a racing edit or sparse closure.
    for name,path in files.items():
        blob=subprocess.run(['git','-C',str(project),'show',revision+':'+name],capture_output=True,check=True).stdout
        if hashlib.sha256(blob).hexdigest()!=operator.digest(path):
            raise ValueError('Source differs from committed blob: '+name)
    return dict(git_commit=revision,source_files={name:operator.digest(p) for name,p in files.items()},
                model_consumer='Preserved pinned deployment data/stats closure; not copied over the current training checkout.')


def readable_tree(root,gid,executables=False):
    for p in [root]+list(root.rglob('*')):
        if p.is_symlink():
            os.lchown(p,-1,gid)
            target=p.resolve(strict=True)
            if not target.is_relative_to(root.resolve()) and str(target).startswith(('/home/','/data/user_data/')):
                raise ValueError('Private symlink dependency: '+str(p))
            continue
        mode=0o2750 if p.is_dir() else (0o750 if executables and p.stat().st_mode&0o111 else 0o640)
        os.chown(p,-1,gid);os.chmod(p,mode)


def stage(root,template,evidence,assets,python,project):
    if project is None:raise ValueError('--project must select the canonical committed checkout')
    files=maintained_sources(project)
    repository=repository_receipt(project,files)
    root.mkdir(parents=True,exist_ok=False)
    config=operator.read(template/'BUNDLE.json')
    copied={}
    def exact(source,target):
        if source.is_symlink():raise ValueError('Reference symlink not allowed')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        sha=operator.digest(source)
        if operator.digest(target)!=sha:raise ValueError('Copy mismatch')
        copied[str(target.relative_to(root))]=sha
    # The baseline preserves exactly the simulator/consumer source bytes; only
    # deployment paths/interpreter change in this NEW manifest.
    baseline=root/'baseline'
    for name,sha in config['files'].items():
        if operator.digest(template/name)!=sha:raise ValueError('Template changed')
        if name=='REPOSITORY.json':continue  # Replaced by this release's commit.
        exact(template/name,baseline/name)
    # The archived template supplies consumer/reference compatibility only.
    # All maintained MPM Python and the production wrapper come from Git.
    deployed_files=dict(config['files'])
    for name,source in files.items():
        relative=frozen_path(name)
        target=baseline/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        expected=repository['source_files'][name]
        if operator.digest(target)!=expected:raise ValueError('Source changed during publication: '+name)
        deployed_files[relative]=expected
        copied.pop(str(target.relative_to(root)),None)
    operator.write_once(baseline/'REPOSITORY.json',repository)
    deployed_files['REPOSITORY.json']=operator.digest(baseline/'REPOSITORY.json')
    ref=root/'reference-assets'
    original_ref=Path(config['reference'])
    plan=operator.read(original_ref/'wall_filter/run-02/PLAN.json')
    exact(original_ref/'wall_filter/run-02/PLAN.json',ref/'wall_filter/run-02/PLAN.json')
    for c in plan['cases']:
        if c['provenance']['kind']!='matched_control':continue
        a=original_ref/'wall_filter/run-02'/c['label'];b=ref/'wall_filter/run-02'/c['label']
        exact(a/'initial.npz',b/'initial.npz')
        result=operator.read(a/'RESULT.json');raw=Path(result['raw_path'])
        if operator.digest(raw)!=result['raw_sha256']:raise ValueError('Historical control raw changed')
        exact(raw,b/'raw.h5')
        operator.write_once(b/'RESULT.json',dict(result,raw_path=str(b/'raw.h5')))
    old=operator.read(evidence/'REQUEST.json')
    selected=[s for s in old['cases'] if s['case']['template']['split']=='train']
    if len(selected)!=20:raise ValueError('Require all twenty training references')
    updated=[];shared_evidence=root/'qualification-reference'
    for i,item in enumerate(selected):
        case=item['case'];target=shared_evidence/'inputs/train/plan/cases'/case['label']
        for name in ('initial.npz','MANIFEST.json'):
            exact(Path(item['root'])/'cases'/case['label']/name,target/name)
        updated.append(dict(item,root=str(shared_evidence/'inputs/train/plan')))
        for lane in range(2):
            original=evidence/f'lane-{lane}/serial/{i:02d}'
            for f in original.rglob('*'):
                if f.is_file():exact(f,shared_evidence/f'lane-{lane}/serial/{i:02d}'/f.relative_to(original))
    operator.write_once(shared_evidence/'REQUEST.json',dict(old,cases=updated,
        relocation=dict(original_request_sha256=operator.digest(evidence/'REQUEST.json'),
                        note='Only storage roots changed; prepared inputs and serial raw/journal payloads are exact copies.')))
    operator.write_once(baseline/'BUNDLE.json',dict(schema='shared-mpm-baseline-v1',banks=str(assets),
        bank_complete_sha256=operator.digest(assets/'COMPLETE.json'),reference=str(ref),
        genesis_python=str(python),model_python=str(python),retained_training_target=500000,
        maximum_training_attempts=750000,test_collection=False,accepted_for_training=False,
        files=deployed_files))
    operator.write_once(root/'STAGED.json',dict(source_template=str(template),
        source_template_sha256=operator.digest(template/'BUNDLE.json'),exact_copies=copied,
        assets=str(assets),python=str(python),baseline=str(baseline),evidence=str(shared_evidence),
        repository=repository))
    return dict(root=str(root),exact_copies=len(copied),stage_complete=True)


def finish(root,project,environment_root,group):
    gid=grp.getgrnam(group).gr_gid
    s=operator.read(root/'STAGED.json')
    python=Path(s['python']);baseline=Path(s['baseline']);template=root/'template-v1'
    if not python.is_file():raise ValueError('Shared environment not ready')
    # Stage already captured the committed wrapper. Never recopy a mutable
    # checkout's runtime files between staging and freezing the release.
    env=operator.environment(baseline)
    operator.checked([python,baseline/'mpm_training_campaign.py','freeze','--bundle',template,
        '--source',baseline,'--evidence',s['evidence'],'--cycle-start','84000','--cycles','2000'],env=env)
    operator.checked([python,template/'mpm_training_campaign.py','verify','--bundle',template],env=operator.environment(template))
    # This handoff contains no execute-time dependency on a private Python path.
    shutil.copyfile(project/LAUNCH/'mpm_collect.py',root/'mpm_collect.py')
    c=operator.read(project/LAUNCH/'mpm_collect.json')
    c.update(template_bundle=str(template),template_sha256=operator.digest(template/'BUNDLE.json'),
             shared_assets=s['assets'],access_group=group)
    # Historical live roots have their original permissions. New users should
    # inspect shared operator runs, not assume permission to old staging runs.
    c['existing_runs']=[]
    operator.write_once(root/'mpm_collect.json',c)
    shutil.copyfile(project/DOCS/'RUNBOOK.md',root/'RUNBOOK.md')
    readable_tree(environment_root,gid,executables=True)
    os.chown(environment_root.parent,-1,gid);os.chmod(environment_root.parent,0o2750)
    readable_tree(root,gid)
    os.chown(root.parent,-1,gid);os.chmod(root.parent,0o2750)
    return refresh_frontend(root,project,group)


def refresh_frontend(root,project,group):
    """Finish/recover publication; update frontend, not frozen simulation.

    This NFS filesystem supports POSIX modes but not default ACLs. Explicit
    file modes in the operator preserve cross-user access under any umask.
    """
    gid=grp.getgrnam(group).gr_gid
    old=operator.read(root/'READY.json') if (root/'READY.json').exists() else {}
    c=operator.read(root/'mpm_collect.json')
    for key in ('registry_root','campaigns_root'):
        p=Path(c[key]);p.mkdir(parents=True,exist_ok=True)
        os.chown(p,-1,gid);os.chmod(p,0o3770)
    lock=Path(c['registry_root'])/'.lock'
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o660);os.close(fd)
    except FileExistsError:pass
    os.chown(lock,-1,gid);os.chmod(lock,0o660)
    for source,target in [(project/LAUNCH/'mpm_collect.py',root/'mpm_collect.py'),
                          (project/LAUNCH/'mpm_settings.py',root/'mpm_settings.py'),
                          (project/DOCS/'RUNBOOK.md',root/'RUNBOOK.md'),
                          (project/DOCS/'CONFIG.md',root/'CONFIG.md'),
                          (project/DOCS/'PERFORMANCE.md',root/'PERFORMANCE.md'),
                          (project/DOCS/'RECOVERY_10432785.md',root/'RECOVERY_10432785.md'),
                          (project/DOCS/'ASSETS.md',root/'ASSETS.md')]:
        shutil.copyfile(source,target);os.chown(target,-1,gid);os.chmod(target,0o640)
        if target.suffix=='.md':target.write_text(target.read_text().replace('(../launch/mpm_collect.json)','(mpm_collect.json)'))
    with operator.registry_lock(c):pass
    result=operator.preflight(c)
    if old and not (root/'READY-INITIAL.json').exists():operator.write_once(root/'READY-INITIAL.json',old)
    updated=dict(old,preflight=result,group=group,template_sha256=c['template_sha256'],
        operator_sha256=operator.digest(root/'mpm_collect.py'),nfs_group_writable_lock_verified=True,
        account_access='Lab group filesystem modes verified; each UID still needs its own Slurm association.',
        gpu_execution='New environment/GPUs must pass normal startup qualification before collection.')
    operator.write_once(root/'READY-NEXT.json',updated);os.replace(root/'READY-NEXT.json',root/'READY.json')
    os.chown(root/'READY.json',-1,gid);os.chmod(root/'READY.json',0o640)
    return updated


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['stage','finish','refresh'])
    p.add_argument('--root',type=Path,required=True)
    for name in ('template','evidence','assets','python','project','environment-root'):p.add_argument('--'+name,type=Path)
    p.add_argument('--group',default='kfragki2_group');a=p.parse_args()
    if a.command=='stage':r=stage(a.root,a.template,a.evidence,a.assets,a.python,a.project)
    elif a.command=='finish':r=finish(a.root,a.project,a.environment_root,a.group)
    else:r=refresh_frontend(a.root,a.project,a.group)
    print(json.dumps(r,indent=2))
