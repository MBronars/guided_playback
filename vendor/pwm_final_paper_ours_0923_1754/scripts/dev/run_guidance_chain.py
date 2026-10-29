"""Run immutable experiment stages after explicit completion/qualification gates."""
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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True,type=Path)
    file=parser.parse_args().manifest.resolve();plan=json.loads(file.read_text());root=file.parent
    status_file=file.with_name(file.stem+'_status.json')
    if status_file.exists():raise FileExistsError(status_file)
    status=dict(state='starting',manifest_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),stages=[])
    deadline=datetime.fromisoformat(plan['deadline_utc']).timestamp()
    def save():
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        temporary=status_file.with_suffix('.pending');temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(status_file)
    save()
    try:
        for stage in plan['stages']:
            row=dict(name=stage['name'],state='waiting');status['stages'].append(row);save()
            for gate in stage.get('gates',[]):
                while True:
                    if time.time()>=deadline:raise TimeoutError('Campaign deadline while waiting for gate')
                    path=Path(gate['path'])
                    if path.exists():
                        try:receipt=json.loads(path.read_text())
                        except json.JSONDecodeError:receipt={}
                        value=receipt.get(gate.get('field','status'))
                        if value==gate['expected']:
                            if gate.get('all_entries_completed') and any(v['state']!='completed' for v in receipt.get('entries',[])):
                                raise RuntimeError('Preceding experiment packet did not complete all entries')
                            break
                        if value in ('failed','blocked') or (isinstance(value,str) and value.startswith('stopped')):
                            raise RuntimeError(f'Gate failed: {path}: {value}')
                    time.sleep(15)
            row['state']='running';row['started_utc']=datetime.now(timezone.utc).isoformat();status['state']='running';save()
            env=dict(os.environ,PYTHONPATH=plan['source_root'],CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],
                OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled')
            with (root/(stage['name']+'.log')).open('x') as log:
                process=subprocess.Popen(stage['command'],cwd=plan['source_root'],env=env,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=process.pid;save()
                try:code=process.wait(timeout=min(stage.get('timeout_seconds',18000),max(1,deadline-time.time())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise TimeoutError('Stage timeout')
            row.update(state='completed' if code==0 else 'failed',returncode=code);save()
            if code:raise RuntimeError('Stage subprocess failed')
        status['state']='completed'
    except Exception as error:
        status.update(state='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':main()
