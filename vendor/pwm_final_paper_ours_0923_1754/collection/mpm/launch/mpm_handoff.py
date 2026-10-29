"""Park one identified serial supervisor at its final review, preserving Slurm.

Run as a separate step INSIDE the existing allocation. Never cancel the batch
job, kill a GPU worker, edit the old bundle, or overwrite a completed episode.
START.json authorizes a verified new bundle; absent it, resume after 30 minutes.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def write(path,data):
    with path.open('x') as f:
        json.dump(data,f,indent=2);f.write('\n')


def proc(pid):
    p=Path('/proc')/str(pid)
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,ppid=int(fields[1]),state=fields[0],start=fields[19],
                cmd=(p/'cmdline').read_bytes().decode().split('\0')[:-1],
                uid=p.stat().st_uid)


def same_process(expected):
    now=proc(expected['pid'])
    if (now['start']!=expected['start'] or now['cmd']!=expected['cmd'] or now['uid']!=os.getuid()):
        raise RuntimeError('Supervisor identity changed; do not signal another process')
    return now


def children(pid):
    path=Path(f'/proc/{pid}/task/{pid}/children')
    return [int(x) for x in path.read_text().split()]


def last_review(child,root,entry):
    cmd=child['cmd']
    return ('review' in cmd and '--unit' in cmd
            and cmd[cmd.index('--unit')+1]==str(root/'chunks'/entry['name']))


def park(serial,root,receipt,timeout=4000):
    sys.path.insert(0,str(Path(serial['cmd'][1]).parent))
    import mpm_campaign as base
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        same_process(serial)
        status=json.loads((root/'STATUS.json').read_text())
        if status['state']=='reviewing':
            entries=base.wave(status['next_wave'])
            candidates=[]
            for pid in children(serial['pid']):
                try:
                    child=proc(pid)
                    if last_review(child,root,entries[-1]):candidates.append(child)
                except FileNotFoundError:pass
            if len(candidates)==1:
                child=candidates[0]
                os.kill(serial['pid'],signal.SIGSTOP)
                # Caller owns recovery from this point, even if verification fails.
                write(receipt/'SIGNALLED.json',dict(supervisor=serial,review=child,status=status))
                for _ in range(100):
                    if same_process(serial)['state']=='T':break
                    time.sleep(.01)
                else:raise RuntimeError('Supervisor did not stop')
                current=json.loads((root/'STATUS.json').read_text())
                kids=children(serial['pid'])
                if current!=status or kids!=[child['pid']]:
                    raise RuntimeError('Review boundary raced; resume instead of risking overlap')
                deadline_review=time.monotonic()+1200
                while time.monotonic()<deadline_review:
                    review=proc(child['pid'])
                    if review['start']!=child['start']:raise RuntimeError('Review PID reused')
                    if review['state']=='Z':break
                    time.sleep(1)
                else:raise TimeoutError('Final serial review did not finish')
                # Completion includes hashes of all raw/export/consumer/GIF evidence.
                # publish is idempotent, so serial resumption remains safe.
                sys.path.insert(0,str(Path(serial['cmd'][1]).parent/'source'))
                completed=[]
                for entry in entries:
                    unit=root/'chunks'/entry['name'];r=base.completed(unit)
                    base.publish(root,unit,r);completed.append(dict(name=entry['name'],retained=r['retained']))
                write(receipt/'PARKED.json',dict(supervisor=serial,completed_wave=completed,
                    old_status=current,job=os.environ['SLURM_JOB_ID'],time=time.time()))
                return
        time.sleep(.1)
    raise TimeoutError('No safe final-review boundary found')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial-pid',type=int,required=True)
    p.add_argument('--serial-root',type=Path,required=True)
    p.add_argument('--serial-bundle',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    a=p.parse_args()
    if os.environ.get('SLURM_JOB_ID')!=a.job:raise ValueError('Wrong allocation')
    serial=proc(a.serial_pid)
    expected=[sys.executable,str(a.serial_bundle/'mpm_campaign.py'),'run','--bundle',str(a.serial_bundle),
              '--root',str(a.serial_root)]
    if serial['cmd']!=expected or serial['uid']!=os.getuid():raise ValueError('Unexpected serial supervisor')
    if time.time()+10800>=a.deadline_epoch:raise ValueError('Insufficient allocation reserve')
    a.receipt.mkdir(parents=True,exist_ok=False)
    write(a.receipt/'REQUEST.json',dict(serial=serial,job=a.job,deadline_epoch=a.deadline_epoch))
    active=None;production=False
    def interrupt(*_):raise InterruptedError('Handoff interrupted; recover original collector')
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,interrupt)
    try:
        park(serial,a.serial_root,a.receipt)
        print('SERIAL_PARKED_AND_WAVE_EXPORTED',flush=True)
        start=a.receipt/'START.json';deadline=time.monotonic()+1800
        while not start.exists():
            if time.monotonic()>deadline:raise TimeoutError('No tested replacement supplied')
            time.sleep(1)
        request=json.loads(start.read_text());bundle=Path(request['bundle'])
        import hashlib
        if hashlib.sha256((bundle/'BUNDLE.json').read_bytes()).hexdigest()!=request['bundle_sha256']:
            raise ValueError('Replacement bundle digest mismatch')
        config=json.loads((bundle/'BUNDLE.json').read_text())
        if config.get('gpu_count')!=2 or config.get('cycle_start')!=40000:
            raise ValueError('Require two-GPU disjoint-cycle replacement')
        command=[sys.executable,str(bundle/'mpm_four.py')]
        for phase in ('verify','verify-training' if config.get('training_only') else 'qualify','run'):
            args=command+[phase,'--bundle',str(bundle)]
            if phase in ('qualify','verify-training'):args+=['--root',request['qualification']]
            if phase=='run':
                production=True
                args+=['--root',request['output'],'--qualification',request['qualification'],
                       '--deadline-epoch',str(a.deadline_epoch)]
            with (a.receipt/(phase+'.log')).open('x') as log:
                active=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                write(a.receipt/(phase.upper()+'-STARTED.json'),dict(pid=active.pid,command=args,time=time.time()))
                active.wait(timeout=max(1,a.deadline_epoch-time.time()-1800) if phase=='run' else 7200)
                if active.returncode:raise RuntimeError(f'{phase} failed: exit {active.returncode}')
                active=None
        write(a.receipt/'COMPLETE.json',dict(time=time.time()))
    except BaseException as e:
        write(a.receipt/'FAILURE.json',dict(error=type(e).__name__+': '+str(e),time=time.time()))
        raise
    finally:
        if active is not None and active.poll() is None:
            os.killpg(active.pid,signal.SIGTERM)
            try:active.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid,signal.SIGKILL);active.wait(timeout=15)
        if (a.receipt/'SIGNALLED.json').exists():
            same_process(serial)
            # After production finishes, stop the old collector at its boundary.
            # On qualification failure, leave its stop flag alone and resume work.
            if production and time.time()+10800>=a.deadline_epoch:os.kill(serial['pid'],signal.SIGUSR1)
            os.kill(serial['pid'],signal.SIGCONT)
            write(a.receipt/'RESUMED.json',dict(time=time.time(),production_started=production))


if __name__=='__main__':main()
