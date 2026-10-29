"""Execute one immutable GPU experiment packet with deadlines and durable status."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);args=p.parse_args()
    manifest=Path(args.manifest).resolve();plan=json.loads(manifest.read_text())
    root=manifest.parent;status_path=manifest.with_name(manifest.stem+'_status.json')
    if status_path.exists():raise FileExistsError(status_path)
    status=dict(state='starting',manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),entries=[])
    def save():
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=status_path.with_suffix('.pending');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(status_path)
    save()
    qualification=json.loads(Path(plan['qualification']).read_text())
    if qualification['status']!='passed':
        status['state']='not started: qualification failed';save();return
    environment=dict(os.environ,PYTHONPATH=plan['source_root'],CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled')
    deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
    reserve=datetime.fromisoformat(plan['stop_launch_utc']).timestamp()
    failures=0
    for entry in plan['entries']:
        row=dict(name=entry['name'],state='pending',scene=entry['scene'],arm=entry['arm']);status['entries'].append(row)
        if time.time()>=reserve:
            row['state']='not run: report reserve';save();continue
        if time.time()+entry.get('estimated_seconds',0)>deadline:
            row['state']='not run: insufficient deadline budget';save();continue
        row['state']='running';row['started_utc']=datetime.now(timezone.utc).isoformat();status['state']='running';save()
        start=time.monotonic()
        try:
            # Python's -m and -c imports search cwd before PYTHONPATH. Both must
            # select the entry's snapshot when a packet mixes source revisions.
            source=Path(entry.get('source_override',plan['source_root'])).resolve()
            if not source.is_dir():raise FileNotFoundError(source)
            row['source_root']=str(source)
            entry_environment=dict(environment,PYTHONPATH=str(source))
            if 'qualification' in entry:
                gate=json.loads(Path(entry['qualification']).read_text())
                if gate['status']!='passed':raise ValueError('Entry qualification failed')
            with (root/(entry['name']+'.log')).open('x') as log:
                process=subprocess.Popen(entry['command'],cwd=source,env=entry_environment,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=process.pid;save()
                try: code=process.wait(timeout=min(entry['timeout_seconds'],max(1,deadline-time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    code=124
            row.update(state='completed' if code==0 else 'failed',returncode=code)
        except Exception as error:row.update(state='failed',error=repr(error))
        row['seconds']=time.monotonic()-start;save()
        failures=failures+1 if row['state']=='failed' else 0
        if failures>=2:
            status['state']='stopped: two consecutive infrastructure failures';save();return
    status['state']='completed';save()


if __name__=='__main__':main()
