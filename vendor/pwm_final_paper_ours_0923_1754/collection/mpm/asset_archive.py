"""Archive selected/protected source bytes without rewriting original receipts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from .assets import digest,save


def copy_verified(source,target,expected):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if digest(target)!=expected:raise ValueError('Existing archive differs: '+str(target))
        return
    temporary=target.with_name(target.name+'.copy-'+uuid.uuid4().hex)
    shutil.copy2(source,temporary)
    if digest(temporary)!=expected:raise ValueError('Source/archive copy mismatch: '+str(source))
    try:os.link(temporary,target)
    except FileExistsError:
        if digest(target)!=expected:raise ValueError('Concurrent archive differs')
    # Retain the same-inode temporary link; never clobber an existing archive.


def archive(root):
    plan=json.loads((root/'PLAN.json').read_text());manifest=json.loads((root/'MANIFEST.json').read_text())
    if digest(root/'PLAN.json')!=manifest['plan_sha256']:raise ValueError('Plan changed')
    if (root/'SOURCE_ARCHIVE.json').exists():raise FileExistsError('Archive already recorded')
    uids=[r['uid'] for r in manifest['objects']]+manifest['protected_holdout_uids']
    entries=[]
    for uid in uids:
        value=json.loads((root/'objects'/uid/'RESULT.json').read_text())
        path=Path('meshes')/(value['mesh_sha256']+'.glb')
        entries.append(dict(uid=uid,sha256=value['mesh_sha256'],file=str(path),original_path=value['mesh_path']))
    def copy(entry):copy_verified(Path(entry['original_path']),root/entry['file'],entry['sha256'])
    with ThreadPoolExecutor(max_workers=2) as pool:
        for count,_ in enumerate(pool.map(copy,entries),1):
            if count%500==0:print(json.dumps(dict(archived_sources=count,total=len(entries))),flush=True)
    inputs=[]
    for source,h in plan['inputs_sha256'].items():
        path=Path('inputs')/Path(source).name;copy_verified(Path(source),root/path,h)
        inputs.append(dict(original_path=source,file=str(path),sha256=h))
    save(root/'SOURCE_ARCHIVE.json',dict(status='complete',sources=entries,inputs=inputs,
         manifest_sha256=digest(root/'MANIFEST.json'),archive_code_sha256=digest(__file__),
         policy='Byte-exact independent copies/new cache members; original source files and receipts unchanged',
         accepted_for_training=False))
    print(json.dumps(dict(status='complete',archived_sources=len(entries))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--wait-seconds',type=int,default=0);args=parser.parse_args()
    if args.wait_seconds:
        deadline=time.monotonic()+args.wait_seconds
        while not (args.root/'MANIFEST.json').exists():
            if list(args.root.glob('FAILURE-*.json')):raise RuntimeError('Preparation failed')
            if time.monotonic()>deadline:raise TimeoutError('Archive wait exhausted')
            time.sleep(10)
    archive(args.root.resolve(strict=True))
