"""Bounded drag qualification commands; intentionally no production launcher."""
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback
import time
from collection.common import digest,write_once
from .. import runtime


def wait_idle(uuid,timeout=30):
    """Wait for driver utilization sampling to decay after our prior child exits.

    Foreign processes, ambiguous devices and all other guard errors still fail
    immediately. This is preflight waiting, never an automatic scene retry.
    """
    from ..collect import gpu_status
    deadline=time.monotonic()+timeout
    while True:
        try:return gpu_status(uuid)
        except RuntimeError as error:
            if str(error)!='GPU idle/memory guard failed' or time.monotonic()>=deadline:raise
            time.sleep(1)


def freeze(output,*,consumer=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sources=runtime.sources()
    if consumer:
        for p in [*(runtime.ROOT/'data').glob('*.py'),runtime.ROOT/'stats/normalization.json']:
            sources[str(p.relative_to(runtime.ROOT))]=digest(p)
    for name,expected in sources.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(runtime.ROOT/name,target)
        if digest(target)!=expected:raise ValueError('Source changed while freezing')
    runtime.verify_sources(sources)
    write_once(output/'SNAPSHOT.json',dict(sources=sources,production_approved=False))
    return dict(snapshot=str(output.resolve()),sources=len(sources))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    frozen=sub.add_parser('freeze');frozen.add_argument('--output',required=True);frozen.add_argument('--consumer',action='store_true')
    prepared=sub.add_parser('prepare')
    prepared.add_argument('--output',required=True);prepared.add_argument('--assets',required=True)
    prepared.add_argument('--name',required=True);prepared.add_argument('--count',type=int,default=12)
    prepared.add_argument('--start',type=int,default=0);prepared.add_argument('--split',default='train')
    prepared.add_argument('--evaluation',default='train')
    prepared.add_argument('--material-layout',choices=('homogeneous','mixed'),default='homogeneous')
    prepared.add_argument('--object-counts',nargs='+',type=int)
    prepared.add_argument('--n-grid',type=int,choices=(100,256),default=100)
    prepared.add_argument('--event-counts',nargs='+',type=int,default=[1,2,3])
    prepared.add_argument('--speed-range',nargs=2,type=float,default=[.05,.25],metavar=('MIN','MAX'))
    run=sub.add_parser('run');run.add_argument('--root',required=True);run.add_argument('--output',required=True)
    run.add_argument('--labels',nargs='+',required=True);run.add_argument('--uuid',required=True)
    run.add_argument('--serial',action='store_true')
    run.add_argument('--reference-graph',action='store_true',help='Diagnostic: original 256-thread particles and unfused walls')
    args=parser.parse_args()
    if args.command=='freeze':result=freeze(args.output,consumer=args.consumer)
    elif args.command=='prepare':
        from .prepare import prepare
        result=prepare(args.output,args.assets,name=args.name,count=args.count,start=args.start,
                       split=args.split,evaluation=args.evaluation,material_layout=args.material_layout,
                       object_counts=args.object_counts,event_counts=args.event_counts,speed_range=args.speed_range,n_grid=args.n_grid)
    else:
        from collection.lease import gpu_lease
        from ..collect import gpu_status
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.uuid:
            raise ValueError('Select exactly the explicit GPU UUID before importing the simulator')
        output=Path(args.output)
        if output.exists():raise FileExistsError(output)
        with gpu_lease(args.uuid):
            try:
                wait_idle(args.uuid)
                import warp as wp
                wp.config.kernel_cache_dir=str(output.parent/'warp-cache')
                from .simulate import simulate
                result=simulate(args.root,args.labels,output,serial=args.serial,uuid=args.uuid,reference_graph=args.reference_graph)
                write_once(output/'COMPLETE.json',result)
            except Exception:
                output.mkdir(parents=True,exist_ok=True)
                write_once(output/'FAILURE.json',dict(error=traceback.format_exc(),retained=False))
                raise
    print(json.dumps({k:v for k,v in result.items() if k!='results'},sort_keys=True),flush=True)


if __name__=='__main__':main()
