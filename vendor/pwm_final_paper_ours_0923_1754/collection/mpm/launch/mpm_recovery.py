"""Hash-bound review-only recovery using unchanged qualified GPU workers.

Run from an immutable capsule with RECOVERY.json beside this script. Never
modifies the original failed run or its qualification. The capsule records all
new orchestration/review code; simulation commands use the original bundle.
"""
import argparse,importlib.util,json,os,sys,time
from pathlib import Path

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','review','verify','worker'])
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--unit',type=Path)
    a,extra=p.parse_known_args();capsule=Path(__file__).resolve().parent
    if extra and a.command!='worker':p.error('Unexpected arguments: '+str(extra))
    meta=json.loads((capsule/'RECOVERY.json').read_text())
    if a.bundle.resolve()!=Path(meta['bundle']):raise ValueError('Wrong frozen bundle')
    sys.path[:0]=[str(a.bundle),str(a.bundle/'source')]
    import mpm_training_campaign as original
    import mpm_scale as support
    from collection.common import check_hashes
    check_hashes(capsule,meta['files'])
    if support.sha(a.bundle/'BUNDLE.json')!=meta['bundle_sha256']:raise ValueError('Changed original bundle')
    config=original.verify(a.bundle)
    os.environ['PWM_MPM_REFERENCE_ROOT']=config['reference'];os.environ['PWM_MODEL_ROOT']=str(a.bundle/'source')
    if meta.get('gpu_monitor_retries'):
        monitor=load('recovery_gpu_monitor',capsule/'mpm_gpu_monitor.py')
        monitor.install()
    if a.command=='worker':
        # Delegate all simulation arguments and numerical work to the exact
        # qualified driver. Only collect.gpu_status gets timeout resilience.
        original.main();return
    review=load('recovery_review',capsule/'mpm_campaign.py')
    if a.command=='verify':print('Recovery capsule verified; original GPU runtime unchanged');return
    if a.command=='review':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only review required')
        review.review(a.unit);return
    import subprocess
    job=json.loads(subprocess.check_output(['scontrol','--json','show','job',os.environ['SLURM_JOB_ID']],text=True))['jobs'][0]
    deadline=original.deadline(job,time.time())
    if str(job['job_id'])!=meta['job_id']:raise ValueError('Wrong recovery allocation')
    qualification=Path(meta['qualification'])
    if support.sha(qualification/'TRAINING_PASS.json')!=meta['qualification_sha256']:raise ValueError('Changed qualification')
    approval=original.qualified(qualification,a.bundle)
    if original.discover(config)!=[r['uuid'] for r in approval['lanes']]:raise ValueError('Different GPUs')
    parent=Path(meta['original_collection'])
    start=meta['start_wave'];count=config['gpu_count'];first=config['cycle_start']+start*count
    if not 0<start<config['training_cycles']//count:raise ValueError('Unbounded recovery start')
    if not (parent/'PILOT_APPROVED.json').is_file():raise ValueError('Original pilot never approved')
    # No cycle with saved attempts may be silently rerun.
    for previous in [parent]+[Path(x) for x in meta.get('previous_collections',[])]:
        for unit in (previous/'chunks').glob('train-*'):
            entry=support.read(unit/'ENTRY.json')
            if entry['cycle']>=first:
                raise ValueError('Recovery would overlap a previously prepared cycle')
    four=load('recovery_four',capsule/'mpm_four.py')
    class Controller(original.Controller):
        def verify(self,bundle):
            c=original.verify(bundle)
            # Charge skipped/interrupted cycles conservatively against the cap.
            remaining=c['maximum_training_attempts']-start*count*60
            if remaining<=0:raise ValueError('Attempt budget exhausted')
            return dict(c,maximum_training_attempts=remaining,retained_training_target=remaining)
        def approve_pilot(self,root,bundle,results):
            if len(results)!=count or sorted(r['entry']['cycle'] for r in results)!=list(range(first,first+count)):
                raise ValueError('Wrong recovery first wave')
            for r in results:
                four.require_training_plan(r['entry'],[])
                if (r['status']!='complete' or r['attempts']!=60 or not 0<r['retained']<=60
                        or r['model']['profiles']!=2 or r['model']['episodes_per_profile']!=r['retained']
                        or set(r['attempt_status'])-{'complete','placement_rejected'}):
                    raise ValueError('Recovery first-wave checks failed')
            support.write(root/'PILOT_APPROVED.json',dict(pilot_sha256=support.sha(root/'PILOT_READY.json'),
                bundle_sha256=support.sha(bundle/'BUNDLE.json'),manual_visual_review=False,
                approval='All recovery first-wave quality, control and loader checks passed'))
    root=Path(meta['output'])
    try:
        four.run(a.bundle,root,qualification,deadline,controller=Controller(config),
                 start_wave=start,review_driver=capsule/'mpm_recovery.py',
                 quarantine_history=meta.get('quarantine_history',[]),
                 worker_driver=capsule/'mpm_recovery.py' if meta.get('gpu_monitor_retries') else None)
    except BaseException as exc:
        root.mkdir(parents=True,exist_ok=True)
        support.write(root/'FAILURE.json',dict(error=repr(exc),time_epoch=time.time()))
        raise

if __name__=='__main__':main()
