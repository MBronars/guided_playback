"""Final shape membership/accounting audit, independent of bank selection loop."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from .assets import digest,save


def audit(root,out):
    if out.exists():raise FileExistsError(out)
    plan=json.loads((root/'PLAN.json').read_text());manifest=json.loads((root/'MANIFEST.json').read_text())
    if manifest['plan_sha256']!=digest(root/'PLAN.json'):raise ValueError('Plan hash changed')
    if digest(root/'prepare_shapes.py')!=plan['code_sha256']:raise ValueError('Executed preparation source changed')
    if any(digest(p)!=h for p,h in plan['inputs_sha256'].items()):raise ValueError('Input catalog or historical pools changed')
    selected={r['uid']:r for r in manifest['objects']}
    if len(selected)!=10000 or len(manifest['objects'])!=10000:raise ValueError('Need exactly 10k unique selected UIDs')
    protected={r['uid'] for r in plan['protected']}
    if set(selected)&protected:raise ValueError('Protected UID leak')
    mesh_hashes={};geometry_hashes={};sources={};accounting=[];origins=Counter();reasons=Counter()
    heldout_geometry=set();heldout_mesh=set()
    for candidate in plan['protected']:
        path=root/'objects'/candidate['uid']/'RESULT.json';value=json.loads(path.read_text())
        if value['candidate']!=candidate or value['status']!='protected':raise ValueError('Protected receipt invalid')
        heldout_geometry.add(value['quality']['geometry_sha256']);heldout_mesh.add(value['mesh_sha256'])
        sources[value['mesh_path']]=value['mesh_sha256']
    if heldout_geometry!=set(manifest['protected_geometry_sha256']):raise ValueError('Protected fingerprint set changed')
    if protected!=set(manifest['protected_holdout_uids']):raise ValueError('Protected UID set changed')
    rejected={r['uid']:r['reason'] for r in manifest['rejected']}
    for candidate in plan['candidates']:
        uid=candidate['uid'];path=root/'objects'/uid/'RESULT.json'
        if not path.exists():continue
        value=json.loads(path.read_text())
        if value['candidate']!=candidate:raise ValueError('Candidate receipt mismatch')
        if uid in selected:
            row=selected[uid]
            if value['status']!='qualified' or not value['quality']['eligible']:raise ValueError('Unqualified selected shape')
            if digest(path)!=row['receipt_sha256'] or digest(root/row['file'])!=row['sha256']:raise ValueError('Selected data changed')
            mesh=value['mesh_sha256'];shape=value['quality']['geometry_sha256']
            if mesh!=row['mesh_sha256'] or shape!=row['geometry_sha256']:raise ValueError('Membership fingerprint mismatch')
            if mesh in heldout_mesh or shape in heldout_geometry:raise ValueError('Heldout geometry alias')
            if mesh in mesh_hashes or shape in geometry_hashes:raise ValueError('Duplicate selected shape')
            mesh_hashes[mesh]=uid;geometry_hashes[shape]=uid;sources[value['mesh_path']]=mesh
            origins[row['origin']]+=1;outcome='selected'
        elif uid in rejected:
            outcome='excluded';reasons[str(rejected[uid])]+=1
        else:
            if value['status']!='qualified':raise ValueError('Unaccounted failed candidate')
            outcome='qualified_beyond_capacity'
        accounting.append(dict(uid=uid,outcome=outcome,receipt_sha256=digest(path)))
    counts=Counter(r['outcome'] for r in accounting)
    if counts['selected']!=10000 or counts['excluded']!=len(rejected):raise ValueError('Candidate accounting incomplete')
    # All selected source meshes and every protected heldout source are rehashed.
    def check(entry):
        path,expected=entry
        if digest(path)!=expected:raise ValueError('Source mesh changed: '+path)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(check,sources.items()))
    save(out,dict(status='complete',selected_training_shapes=10000,origins=dict(origins),
        protected_holdout_shapes=len(protected),uid_overlap=0,mesh_hash_overlap=0,normalized_vertex_geometry_overlap=0,
        current_source_meshes_rehashed=len(sources),accounting_counts=dict(counts),accounting=accounting,
        exclusion_reasons=dict(reasons),manifest_sha256=digest(root/'MANIFEST.json'),review_code_sha256=digest(__file__),
        qualification_scope=manifest['policy']['limitations'],accepted_for_training=False))
    print(json.dumps(dict(status='complete',shapes=10000,origins=dict(origins),accounting=dict(counts))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--wait-seconds',type=int,default=0)
    args=parser.parse_args()
    if args.wait_seconds:
        deadline=time.monotonic()+args.wait_seconds
        while True:
            if list(args.root.glob('FAILURE-*.json')):raise RuntimeError('Preparation failed')
            status=args.root/'STATUS.json'
            if status.exists() and json.loads(status.read_text())['phase']=='complete':break
            if time.monotonic()>deadline:raise TimeoutError('Final audit wait exhausted')
            time.sleep(10)
    audit(args.root,args.out)
