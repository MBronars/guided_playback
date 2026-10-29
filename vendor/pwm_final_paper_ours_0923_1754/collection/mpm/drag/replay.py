"""Rebind exact prepared drag inputs to new sources for regression diagnostics.

Never regenerate geometry or overwrite history. This is not a scale approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from collection.common import canonical,digest,write_once
from .. import runtime
from ..contracts import identity
from .protocol import SCHEMAS


def rebind(root,output,labels):
    root=Path(root).resolve(strict=True);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'PLAN.json').read_text());entries={c['label']:c for c in plan['cases']}
    if not labels or len(labels)!=len(set(labels)) or any(label not in entries for label in labels):
        raise ValueError('Explicit unique prepared case labels required')
    if plan['schema'] not in SCHEMAS:raise ValueError('Only versioned scheduled drag replay supported')
    sources=runtime.sources();cases=[]
    for label in labels:
        source=root/'cases'/label;entry=entries[label]
        if digest(source/'initial.npz')!=entry['initial_sha256'] or digest(source/'MANIFEST.json')!=entry['manifest_sha256']:
            raise ValueError('Prepared replay source changed')
        manifest=json.loads((source/'MANIFEST.json').read_text())
        for name,sha in manifest['source_sha256'].items():
            if digest(name)!=sha:raise ValueError('Original snapshot changed: '+name)
        parent=manifest['episode_id'];folder=output/'cases'/label;folder.mkdir(parents=True)
        shutil.copy2(source/'initial.npz',folder/'initial.npz')
        manifest['source_sha256']={str(runtime.ROOT/p):h for p,h in sources.items()}
        manifest['replay_parent']=dict(episode_id=parent,manifest_sha256=entry['manifest_sha256'],
            initial_sha256=entry['initial_sha256'],source_root=str(root),kind='exact-input-new-source-diagnostic')
        manifest['case']['config_sha256']=hashlib.sha256(canonical(dict(
            parent_manifest_sha256=entry['manifest_sha256'],initial_sha256=entry['initial_sha256'],sources=sources)).encode()).hexdigest()
        manifest['episode_id']=identity(manifest['case'])[0]
        write_once(folder/'MANIFEST.json',manifest)
        cases.append(dict(label=label,initial_sha256=digest(folder/'initial.npz'),manifest_sha256=digest(folder/'MANIFEST.json')))
    runtime.verify_sources(sources)
    write_once(output/'PLAN.json',dict(plan,cases=cases,sources=sources,qualification=False,scaling_approved=False,
        replay_parent_plan_sha256=digest(root/'PLAN.json'),replay_parent_root=str(root)))
    return dict(prepared=len(cases),root=str(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--labels',nargs='+',required=True)
    a=p.parse_args();print(json.dumps(rebind(a.root,a.output,a.labels)))
