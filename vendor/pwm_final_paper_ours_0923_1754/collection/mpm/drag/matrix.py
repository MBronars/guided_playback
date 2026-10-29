"""Finite, immutable-plan serial/grouped qualification runner. Not collection."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collection.common import digest,write_once
from .. import runtime


def run(root,output,uuid,*,serial=True,budget_seconds=21600,reference_graph=False):
    root=Path(root).resolve(strict=True);output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'PLAN.json').read_text());runtime.verify_sources(plan['sources'])
    buckets=defaultdict(list)
    for case in plan['cases']:
        manifest=json.loads((root/'cases'/case['label']/'MANIFEST.json').read_text())
        buckets[json.dumps(manifest['clock'],sort_keys=True)].append(case['label'])
    groups=[labels[i:i+4] for labels in buckets.values() for i in range(0,len(labels),4)]
    request=dict(plan_sha256=digest(root/'PLAN.json'),root=str(root),groups=groups,
                 modes=['grouped','serial'] if serial else ['grouped'],uuid=uuid,budget_seconds=budget_seconds,
                 reference_graph=reference_graph,scaling_approved=False)
    write_once(output/'REQUEST.json',request);results=[];start=time.monotonic()
    for mode in request['modes']:
        for index,labels in enumerate(groups):
            if time.monotonic()-start+3600>budget_seconds:
                write_once(output/'STOP.json',dict(reason='deadline_reserve',completed=len(results)));return
            destination=output/mode/f'group-{index:03d}';destination.parent.mkdir(parents=True,exist_ok=True)
            argv=[sys.executable,'-u','-m','collection.mpm.drag.cli','run','--root',str(root),
                  '--output',str(destination),'--labels',*labels,'--uuid',uuid]
            if mode=='serial':argv.append('--serial')
            elif reference_graph:argv.append('--reference-graph')
            with (destination.parent/f'group-{index:03d}.log').open('x') as log:
                child=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,cwd=runtime.ROOT.parent,timeout=3700)
            row=dict(mode=mode,index=index,labels=labels,exit_code=child.returncode)
            if child.returncode:
                write_once(output/'FAILURE.json',row);raise RuntimeError('Qualification child failed: '+str(destination))
            result=json.loads((destination/'COMPLETE.json').read_text())
            row.update(passed=result['passed'],retained=result.get('retained',0),seconds=result.get('seconds'),
                       complete_sha256=digest(destination/'COMPLETE.json'))
            write_once(output/'receipts'/f'{mode}-{index:03d}.json',row);results.append(row)
            print(json.dumps(row),flush=True)
    write_once(output/'COMPLETE.json',dict(groups=results,seconds=time.monotonic()-start,
        raw_integrity_all_pass=all(r['passed'] for r in results),scaling_approved=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--uuid',required=True)
    p.add_argument('--grouped-only',action='store_true');p.add_argument('--budget-seconds',type=int,default=21600)
    p.add_argument('--reference-graph',action='store_true')
    a=p.parse_args()
    run(a.root,a.output,a.uuid,serial=not a.grouped_only,budget_seconds=a.budget_seconds,reference_graph=a.reference_graph)
