"""Bounded public-mesh prefetch for an existing frozen shape plan, no selection.

Eight network threads; content-addressed destinations, validated bytes, atomic
write-once publication. Never changes the plan, receipts or qualification policy.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import uuid

import requests
from .assets import digest,save


def fetch(row,root,limit):
    target=root/'meshes'/(row['sha256']+'.glb')
    if target.exists():
        if digest(target)!=row['sha256']:raise ValueError('Existing download changed')
        return dict(uid=row['uid'],status='already_cached')
    temporary=root/'prefetch'/(row['sha256']+'.part-'+uuid.uuid4().hex)
    url='https://huggingface.co/datasets/allenai/objaverse/resolve/main/'+row['object_path']
    began=time.monotonic();total=0
    try:
        with requests.get(url,stream=True,timeout=(15,45)) as response:
            response.raise_for_status()
            if int(response.headers.get('Content-Length',0))>limit:raise ValueError('mesh_byte_limit')
            with temporary.open('xb') as stream:
                for chunk in response.iter_content(1024**2):
                    total+=len(chunk)
                    if total>limit:raise ValueError('mesh_byte_limit')
                    if time.monotonic()-began>180:raise TimeoutError('Prefetch deadline')
                    stream.write(chunk)
                stream.flush();os.fsync(stream.fileno())
        if digest(temporary)!=row['sha256']:raise ValueError('Catalog hash mismatch')
        try:os.link(temporary,target)
        except FileExistsError:
            if digest(target)!=row['sha256']:raise ValueError('Concurrent content mismatch')
        # Keep the temporary hardlink as provenance; no existing data is deleted.
        return dict(uid=row['uid'],status='cached',bytes=total,seconds=time.monotonic()-began)
    except (requests.RequestException,ValueError,TimeoutError) as error:
        return dict(uid=row['uid'],status='prefetch_failed',error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-began)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=8);args=parser.parse_args()
    if not 1<=args.threads<=8:raise ValueError('At most eight network threads')
    root=args.root.resolve(strict=True);plan_hash=digest(root/'PLAN.json');plan=json.loads((root/'PLAN.json').read_text())
    if plan['target']!=10000:raise ValueError('Expected frozen 10k shape plan')
    (root/'prefetch').mkdir(exist_ok=False);(root/'meshes').mkdir(exist_ok=True)
    seen=set();rows=[]
    for row in plan['candidates']:
        if row['origin']=='new' and row['sha256'] not in seen:
            seen.add(row['sha256']);rows.append(row)
    save(root/'prefetch/PLAN.json',dict(parent_plan_sha256=plan_hash,source_sha256=digest(__file__),
         maximum_threads=args.threads,maximum_candidates=len(rows),effect='Download cache only; no qualification/selection changes'))
    counts={};processed=0
    with (root/'prefetch/RESULTS.jsonl').open('x') as log,ThreadPoolExecutor(max_workers=args.threads) as pool:
        for offset in range(0,len(rows),32):
            if (root/'MANIFEST.json').exists() or list(root.glob('FAILURE-*.json')):break
            if digest(root/'PLAN.json')!=plan_hash:raise ValueError('Shape plan changed')
            for result in pool.map(lambda row:fetch(row,root,plan['policy']['maximum_mesh_bytes']),rows[offset:offset+32]):
                counts[result['status']]=counts.get(result['status'],0)+1;processed+=1
                log.write(json.dumps(result)+'\n');log.flush()
            print(json.dumps(dict(prefetched_candidates=processed,counts=counts)),flush=True)
    save(root/'prefetch/COMPLETE.json',dict(processed=processed,counts=counts,parent_plan_sha256=plan_hash))


if __name__=='__main__':main()
