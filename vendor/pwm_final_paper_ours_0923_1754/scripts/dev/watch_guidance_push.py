"""Refresh the Markdown gallery and physics audits as immutable results arrive."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--source',required=True,type=Path)
    p.add_argument('--controls',required=True,type=Path);p.add_argument('--deadline',required=True)
    p.add_argument('--status-name',default='gallery_watch_status.json')
    p.add_argument('--vendor',action='append',default=[]);args=p.parse_args()
    root=args.root.resolve();source=args.source.resolve();deadline=datetime.fromisoformat(args.deadline).timestamp()
    if Path(args.status_name).name!=args.status_name:raise ValueError('Status name must be a filename')
    status_path=root/args.status_name
    if status_path.exists():raise FileExistsError(status_path)
    env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',MPLCONFIGDIR=str(root/'.mplconfig'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    status=dict(state='running',source=str(source),deadline=args.deadline,refreshes=0);last=None;failures=0
    def save():
        status['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=status_path.with_suffix('.pending');tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(status_path)
    def run(module,extra):
        subprocess.run([sys.executable,'-m','scripts.dev.'+module,*extra],cwd=source,env=env,
            check=True,timeout=min(1800,max(1,deadline-time.time())))
    save()
    while time.time()<deadline:
        manifest=root/'gallery_manifest.json'
        data=json.loads(manifest.read_text());markers=[]
        for group in data['groups']:
            for entry in group['variants']:
                path=Path(entry['run']);update=entry.get('updates');seed=entry.get('seed',8311)
                marker=path/f'seed_{seed}_iterates'/f'through_{update:04d}.json' if update is not None else path/'results.json'
                markers.append((str(marker),marker.stat().st_mtime_ns if marker.exists() else None))
        signature=hashlib.sha256(manifest.read_bytes()+json.dumps(markers).encode()).hexdigest()
        try:
            run('plot_guidance_optimization',['--root',str(root)])
            if signature!=last:
                run('render_guidance_push',['--manifest',str(manifest)])
                command=['--root',str(root),'--source',str(source),'--controls',str(args.controls.resolve())]
                for vendor in args.vendor:command+=['--vendor',str(Path(vendor).resolve())]
                run('audit_guidance_push',command)
                run('render_guidance_push',['--manifest',str(manifest)])
                last=signature;status['refreshes']+=1;status.pop('last_error',None)
            failures=0
        except Exception as error:
            failures+=1;status['last_error']=repr(error)
            if failures>=3:status['state']='failed';save();raise
        save();time.sleep(min(30,max(0,deadline-time.time())))
    status['state']='completed';save()


if __name__=='__main__':main()
