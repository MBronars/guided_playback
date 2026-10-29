"""Deterministic CPU-only mesh acquisition and geometry qualification.

This is a dimensionless SHAPE bank, not a physical MPM initialization bank.
No material, particle mass/volume, physical size, scene recipe or trajectory is
chosen here. Historical UID cohorts are preserved; new geometry is training-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from numba import njit

VERSION = 'mpm-shapes-10k-v1'
POLICY = dict(version=VERSION, particles=1024, oversample=20, nn_factor=2.5,
              significant_component_fraction=.03, minimum_largest_component_fraction=.95,
              minimum_axis_extent=.001, maximum_faces=2000000, maximum_mesh_bytes=256*1024**2,
              maximum_protected_local_mesh_bytes=512*1024**2,
              maximum_protected_faces=10000000,
              units='dimensionless; source mesh AABB maximum full extent becomes 1',
              geometry_fingerprint='sorted unique AABB-normalized vertices rounded to 1e-6; ignores appearance',
              limitations='Not invariant to arbitrary rotations/remeshing; not a solid-volume, self-intersection or simulation-stability certificate')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path=Path(path)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True);stream.flush();os.fsync(stream.fileno())


def seed_for(uid):
    return int.from_bytes(hashlib.sha256((VERSION+':'+uid).encode()).digest()[:8],'little')


@njit(cache=True)
def fps(points,count):
    """Fixed start and deterministic first-index tie breaking; no CUDA required."""
    n=len(points); selected=np.empty(count,np.int32); distances=np.full(n,np.inf)
    current=0
    for i in range(count):
        selected[i]=current
        for j in range(n):
            d=0.
            for k in range(3): d+=(points[j,k]-points[current,k])**2
            if d<distances[j]: distances[j]=d
        current=np.argmax(distances)
    return selected


def geometry(vertices,faces,uid,*,fingerprint_only=False):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    vertices=np.asarray(vertices,dtype=np.float64);faces=np.asarray(faces,dtype=np.int64)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all(): raise ValueError('nonfinite_or_invalid_vertices')
    face_limit=POLICY['maximum_protected_faces'] if fingerprint_only else POLICY['maximum_faces']
    if faces.ndim!=2 or faces.shape[1]!=3 or not 1<=len(faces)<=face_limit: raise ValueError('invalid_or_excessive_faces')
    if faces.min()<0 or faces.max()>=len(vertices): raise ValueError('face_index_out_of_bounds')
    # Unreferenced vertices must not change scale or create fake shape diversity.
    used=np.unique(faces); lo=vertices[used].min(0); hi=vertices[used].max(0); extent=hi-lo
    if not np.isfinite(extent).all() or extent.max()<=0: raise ValueError('zero_extent')
    center=(lo+hi)/2; scale=extent.max(); canonical=(vertices-center)/scale
    quantized=np.unique(np.rint(canonical[used]*1e6).astype('<i8'),axis=0)
    fingerprint=hashlib.sha256(quantized.tobytes()).hexdigest()
    info=dict(geometry_sha256=fingerprint,source_aabb_min=lo.tolist(),source_aabb_max=hi.tolist(),
              source_center=center.tolist(),source_max_extent=float(scale),axis_extents_unit=(extent/scale).tolist(),
              vertices=len(used),faces=len(faces),seed_uint64=seed_for(uid))
    if fingerprint_only:
        return None,dict(info,eligible=False,reasons=['protected_role_not_training_candidate'])
    triangles=canonical[faces]
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]); area=np.linalg.norm(cross,axis=1)/2
    if not np.isfinite(area).all() or area.sum()<=0: return None,dict(info,eligible=False,reasons=['zero_or_nonfinite_surface_area'])
    rng=np.random.default_rng(seed_for(uid)); n=POLICY['particles']*POLICY['oversample']
    pick=rng.choice(len(faces),size=n,p=area/area.sum())
    uv=rng.random((n,2)); reflected=uv.sum(1)>1;uv[reflected]=1-uv[reflected]
    tri=triangles[pick]; cloud=tri[:,0]+uv[:,0,None]*(tri[:,1]-tri[:,0])+uv[:,1,None]*(tri[:,2]-tri[:,0])
    ix=fps(cloud,POLICY['particles']); points=cloud[ix].astype(np.float32)
    tree=cKDTree(points); nearest=tree.query(points,k=2)[0][:,1]; radius=POLICY['nn_factor']*float(np.median(nearest))
    pairs=tree.query_pairs(radius,output_type='ndarray')
    adjacency=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(len(points),len(points)))
    _,labels=connected_components(adjacency,directed=False); sizes=np.sort(np.bincount(labels))[::-1]
    significant=int((sizes>=np.ceil(POLICY['significant_component_fraction']*len(points))).sum())
    reasons=[]
    if not np.isfinite(points).all() or nearest.min()<=0: reasons.append('duplicate_or_nonfinite_particles')
    if significant!=1: reasons.append('multiple_or_no_significant_components')
    if sizes[0]/len(points)<POLICY['minimum_largest_component_fraction']: reasons.append('excess_disconnected_particles')
    if (extent/scale).min()<POLICY['minimum_axis_extent']: reasons.append('near_planar_or_linear_geometry')
    info.update(surface_area_unit=float(area.sum()),zero_area_faces=int((area==0).sum()),
                component_sizes=sizes.tolist(),significant_components=significant,median_nn_unit=float(np.median(nearest)),
                eligible=not reasons,reasons=reasons,point_sha256=hashlib.sha256(points.tobytes()).hexdigest())
    return dict(points_unit=points,source_face_index=pick[ix],barycentric_uv=uv[ix].astype(np.float32),
                source_vertex_center=center,source_vertex_scale=np.array(scale)),info


def plan(root,legacy,catalog,index,additional):
    if not 5274<=additional<=20000: raise ValueError('Bounded additional candidate count must be 5274..20000')
    root.mkdir(parents=True,exist_ok=False)
    pools={role:json.loads((legacy/'configs'/name).read_text()) for role,name in
           [('train','objaverse_uid_train.json'),('holdout','objaverse_uid_holdout.json')]}
    known=json.loads((legacy/'configs/objaverse_valid_uid_list.json').read_text())
    bare=lambda uid:uid.rsplit('/',1)[-1]
    train={bare(u) for u in pools['train']};heldout={bare(u) for u in pools['holdout']}
    if train&heldout:raise ValueError('Historical pools overlap')
    with gzip.open(index,'rt') as stream: paths=json.load(stream)
    with catalog.open() as stream: metadata=list(csv.DictReader(stream))
    by_uid={row['file_identifier'].rsplit('/',1)[-1]:row for row in metadata}
    meshroot=legacy/'generated_data/mpm_data/meshes/raw/hf-objaverse-v1'
    def record(uid,role,origin):
        row=by_uid[uid];path=paths[uid]
        if not path.startswith('glbs/') or not path.endswith('/'+uid+'.glb') or '..' in Path(path).parts:raise ValueError('Unsafe asset path')
        return dict(uid=uid,asset_uid=path[len('glbs/'):-4],role=role,origin=origin,sha256=row['sha256'],
                    object_path=path,local_path=str(meshroot/path),file_identifier=row['file_identifier'])
    protected=[record(uid,'holdout','historical') for uid in sorted(heldout)]
    original=[record(uid,'train','historical') for uid in sorted(train)]
    exclude={bare(u) for u in known}; candidates=set(by_uid)&set(paths)-exclude
    ordered=sorted(candidates,key=lambda uid:(hashlib.sha256((VERSION+':candidate:'+uid).encode()).hexdigest(),uid))
    additions=[record(uid,'train','new') for uid in ordered[:additional]]
    sources=[catalog,index,*[legacy/'configs'/name for name in ('objaverse_uid_train.json','objaverse_uid_holdout.json','objaverse_valid_uid_list.json')]]
    save(root/'PLAN.json',dict(policy=POLICY,target=10000,additional_candidates=additional,protected=protected,
         candidates=original+additions,inputs_sha256={str(p):digest(p) for p in sources},
         code_sha256=digest(__file__),selection='Historical train first, then fixed hash order; geometry qualification only, never trajectory outcomes',
         accepted_for_training=False))
    shutil.copy2(__file__,root/'prepare_shapes.py')
    print(json.dumps(dict(root=str(root),historical_train=len(original),protected_holdout=len(protected),additional=len(additions))),flush=True)


def fetch(row,root):
    import requests
    local=Path(row['local_path'])
    if local.is_file():
        if digest(local)!=row['sha256']: raise ValueError('historical_mesh_hash_mismatch')
        return local
    target=root/'meshes'/(row['sha256']+'.glb');target.parent.mkdir(exist_ok=True)
    if target.exists():
        if digest(target)!=row['sha256']: raise ValueError('downloaded_mesh_hash_mismatch')
        return target
    url='https://huggingface.co/datasets/allenai/objaverse/resolve/main/'+row['object_path']
    for retry in range(3):
        temporary=target.with_suffix(f'.part-{os.getpid()}-{retry}')
        try:
            began=time.monotonic();total=0
            with requests.get(url,stream=True,timeout=(15,45)) as response:
                response.raise_for_status()
                if int(response.headers.get('Content-Length',0))>POLICY['maximum_mesh_bytes']:raise ValueError('mesh_byte_limit')
                with temporary.open('xb') as stream:
                    for chunk in response.iter_content(1024**2):
                        total+=len(chunk)
                        if total>POLICY['maximum_mesh_bytes']:raise ValueError('mesh_byte_limit')
                        if time.monotonic()-began>180:raise TimeoutError('download_deadline')
                        stream.write(chunk)
            if digest(temporary)!=row['sha256']:raise ValueError('remote_mesh_hash_mismatch')
            os.replace(temporary,target);return target
        except requests.RequestException:
            if retry==2:raise
            time.sleep(retry+1)
    raise RuntimeError('download_exhausted')


def process(task):
    import trimesh
    row,root=task;root=Path(root);folder=root/'objects'/row['uid']
    receipt=folder/'RESULT.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value['candidate']!=row:raise ValueError('Resume candidate changed')
        if value.get('payload_sha256') and digest(folder/'points.npz')!=value['payload_sha256']:raise ValueError('Resume payload changed')
        return value
    folder.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    value=dict(candidate=row,status='running',accepted_for_training=False)
    try:
        path=fetch(row,root)
        size_limit=POLICY['maximum_protected_local_mesh_bytes'] if row['role']=='holdout' else POLICY['maximum_mesh_bytes']
        if path.stat().st_size>size_limit:raise ValueError('mesh_byte_limit')
        # Scene graph transforms are applied by trimesh's force-mesh conversion;
        # no largest-component extraction, flattening, welding or mesh repair.
        mesh=trimesh.load(path,force='mesh',process=False,skip_materials=True)
        arrays,quality=geometry(mesh.vertices,mesh.faces,row['uid'],fingerprint_only=row['role']=='holdout')
        state='protected' if row['role']=='holdout' else 'qualified' if quality['eligible'] else 'rejected'
        value.update(status=state,quality=quality,mesh_path=str(path),mesh_sha256=digest(path))
        if value['mesh_sha256']!=row['sha256']:raise ValueError('mesh_changed_during_processing')
        if arrays is not None:
            with (folder/'points.npz').open('xb') as stream:np.savez_compressed(stream,**arrays)
            value['payload_sha256']=digest(folder/'points.npz')
    except Exception as error:
        value.update(status='error',error=f'{type(error).__name__}: {error}')
    value['seconds']=time.monotonic()-began;save(receipt,value);return value


def progress(root,phase,processed,accepted,rejected):
    target=root/'STATUS.json';temporary=root/'STATUS.next.json'
    save(temporary,dict(phase=phase,processed=processed,accepted_training=len(accepted),target=10000,
         rejected=rejected,utc=datetime.now(timezone.utc).isoformat(),accepted_for_training=False))
    os.replace(temporary,target)
    print(json.dumps(dict(phase=phase,processed=processed,accepted_training=len(accepted),rejected_count=len(rejected))),flush=True)


def run(root,workers):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only geometry preparation requires CUDA_VISIBLE_DEVICES empty')
    if not 1<=workers<=4:raise ValueError('One to four CPU workers only; reserve resources for folding')
    config=json.loads((root/'PLAN.json').read_text())
    if config['policy']!=POLICY or config['code_sha256']!=digest(__file__):raise ValueError('Frozen policy/code changed')
    if any(digest(p)!=h for p,h in config['inputs_sha256'].items()):raise ValueError('Source catalog/cohorts changed')
    lease=(root/'.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'MANIFEST.json').exists():raise FileExistsError('Bank already finalized; use verify')
    accepted=[];rejected=[];heldout_hashes=set();geometry_seen={};mesh_seen={}
    import multiprocessing
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=context) as executor:
        for i,value in enumerate(executor.map(process,((r,str(root)) for r in config['protected']),chunksize=1)):
            if value['status']=='error' or 'geometry_sha256' not in value.get('quality',{}):raise ValueError('Cannot fingerprint protected heldout geometry: '+value['candidate']['uid'])
            heldout_hashes.add(value['quality']['geometry_sha256']);mesh_seen[value['candidate']['sha256']]='heldout'
            if i%25==0:progress(root,'protecting_holdout',i+1,accepted,rejected)
        # Bounded windows avoid downloading thousands of unneeded candidates
        # after reaching 10k; finish each small window for deterministic resume.
        candidates=config['candidates']
        for start in range(0,len(candidates),32):
            if shutil.disk_usage(root).free<500*1024**3:raise RuntimeError('Less than 500 GiB free')
            for value in executor.map(process,((r,str(root)) for r in candidates[start:start+32]),chunksize=1):
                row=value['candidate'];quality=value.get('quality',{});key=quality.get('geometry_sha256')
                reason=None
                if value['status']!='qualified':reason=value.get('error') or quality['reasons']
                elif row['sha256'] in mesh_seen:reason='duplicate_mesh:'+mesh_seen[row['sha256']]
                elif key in heldout_hashes:reason='heldout_geometry_duplicate'
                elif key in geometry_seen:reason='duplicate_geometry:'+geometry_seen[key]
                if reason:rejected.append(dict(uid=row['uid'],reason=reason))
                elif len(accepted)<config['target']:
                    accepted.append(dict(uid=row['uid'],asset_uid=row['asset_uid'],origin=row['origin'],
                        file=f'objects/{row["uid"]}/points.npz',sha256=value['payload_sha256'],
                        receipt_sha256=digest(root/'objects'/row['uid']/'RESULT.json'),
                        mesh_sha256=row['sha256'],geometry_sha256=key))
                    mesh_seen[row['sha256']]=row['uid'];geometry_seen[key]=row['uid']
            progress(root,'qualifying_train',min(start+32,len(candidates)),accepted,rejected)
            if len(accepted)==config['target']:break
    if len(accepted)!=config['target']:raise ValueError(f'Only {len(accepted)} qualified unique shapes in fixed candidate list')
    save(root/'MANIFEST.json',dict(schema=VERSION,bank_role='train',objects=accepted,rejected=rejected,
         protected_holdout_uids=[r['uid'] for r in config['protected']],protected_geometry_sha256=sorted(heldout_hashes),
         plan_sha256=digest(root/'PLAN.json'),policy=POLICY,accepted_for_training=False,
         scope='Qualified dimensionless geometry only; physical initialization and collection require separate review'))
    verify(root)
    progress(root,'complete',start+32,accepted,rejected)


def verify(root):
    manifest=json.loads((root/'MANIFEST.json').read_text());config=json.loads((root/'PLAN.json').read_text())
    if digest(root/'PLAN.json')!=manifest['plan_sha256']:raise ValueError('Plan changed')
    objects=manifest['objects'];holdout=set(manifest['protected_holdout_uids']);protected=set(manifest['protected_geometry_sha256'])
    if len(objects)!=10000 or any(len({r[key] for r in objects})!=10000 for key in ('uid','mesh_sha256','geometry_sha256')):raise ValueError('10,000 unique shapes required')
    if holdout&{r['uid'] for r in objects} or protected&{r['geometry_sha256'] for r in objects}:raise ValueError('Heldout leak')
    for row in objects:
        if digest(root/row['file'])!=row['sha256']:raise ValueError('Shape payload changed')
        receipt=root/'objects'/row['uid']/'RESULT.json'
        if digest(receipt)!=row['receipt_sha256']:raise ValueError('Receipt changed')
        with np.load(root/row['file'],allow_pickle=False) as arrays:
            points=arrays['points_unit']
            if points.shape!=(1024,3) or points.dtype!=np.float32 or not np.isfinite(points).all() or np.abs(points).max()>.50001:raise ValueError('Invalid canonical particles')
    print(json.dumps(dict(status='verified',training_shapes=len(objects),protected_holdout=len(holdout),gpu_used=False)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('plan','run','verify'))
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--legacy',type=Path)
    parser.add_argument('--catalog',type=Path);parser.add_argument('--index',type=Path)
    parser.add_argument('--additional',type=int,default=8000);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.command=='plan':plan(args.root.resolve(),args.legacy,args.catalog,args.index,args.additional)
    elif args.command=='run':
        try:run(args.root.resolve(strict=True),args.workers)
        except BaseException as error:
            failure=args.root/('FAILURE-'+str(time.time_ns())+'.json')
            save(failure,dict(status='failed',error=f'{type(error).__name__}: {error}',utc=datetime.now(timezone.utc).isoformat()))
            raise
    else:verify(args.root.resolve(strict=True))


if __name__=='__main__':main()
