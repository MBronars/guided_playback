"""CPU-only independent particle/provenance audit and actual-point contact sheets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageDraw

from .assets import digest,save


def review(root,out,*,partial=False,source_samples=32,origin=None):
    out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'PLAN.json').read_text())
    if partial:
        objects=[]
        for candidate in plan['candidates']:
            if origin is not None and candidate['origin']!=origin:continue
            receipt=root/'objects'/candidate['uid']/'RESULT.json'
            if not receipt.exists():continue
            value=json.loads(receipt.read_text())
            if value['status']=='qualified':
                objects.append(dict(uid=candidate['uid'],origin=candidate['origin'],file=f'objects/{candidate["uid"]}/points.npz',sha256=value['payload_sha256'],receipt_sha256=digest(receipt)))
            if len(objects)==100:break
    else:
        if origin is not None:raise ValueError('Origin filter only allowed for partial previews')
        objects=json.loads((root/'MANIFEST.json').read_text())['objects']
    if not objects:raise ValueError('No qualified shapes yet')
    holdout={c['uid'] for c in plan['protected']};unique=set();origins=Counter();statistics=[]
    source_indices=set(np.linspace(0,len(objects)-1,min(source_samples,len(objects)),dtype=int).tolist())
    source_checks=[];sheets=[];columns=10;rows=10;cell_w=190;cell_h=138
    for offset in range(0,len(objects),columns*rows):
        canvas=Image.new('RGB',(columns*cell_w,rows*cell_h+55),'white');draw=ImageDraw.Draw(canvas)
        draw.text((12,10),f'MPM actual 1024-point shapes | {offset+1}-{min(offset+100,len(objects))} | XY and XZ projections',fill='black')
        draw.text((12,28),'Unit full AABB extent = 1. No simulation, material assignment, or mesh repair. Partial preview is not final membership.' if partial else 'Qualified bank membership. Unit full AABB extent = 1. No simulation, material assignment, or mesh repair.',fill='black')
        for j,row in enumerate(objects[offset:offset+100]):
            if row['uid'] in unique or row['uid'] in holdout:raise ValueError('Repeated/heldout UID')
            unique.add(row['uid']);origins[row['origin']]+=1
            if digest(root/row['file'])!=row['sha256']:raise ValueError('Particle file changed')
            receipt=root/'objects'/row['uid']/'RESULT.json'
            if digest(receipt)!=row['receipt_sha256']:raise ValueError('Receipt changed')
            source=json.loads(receipt.read_text())
            if source['status']!='qualified' or not source['quality']['eligible']:raise ValueError('Unqualified bank entry')
            with np.load(root/row['file'],allow_pickle=False) as arrays:
                points=arrays['points_unit'];uv=arrays['barycentric_uv'];faces=arrays['source_face_index']
                center=arrays['source_vertex_center'];scale=arrays['source_vertex_scale']
            if points.shape!=(1024,3) or points.dtype!=np.float32 or not np.isfinite(points).all() or np.max(np.abs(points))>.50001:raise ValueError('Invalid point contract')
            if faces.shape!=(1024,) or faces.dtype.kind not in 'iu' or faces.min()<0 or faces.max()>=source['quality']['faces']:raise ValueError('Invalid source face lineage')
            if uv.shape!=(1024,2) or not np.isfinite(uv).all() or uv.min()<0 or (uv.sum(1)>1.000001).any():raise ValueError('Invalid surface barycentric lineage')
            if hashlib.sha256(points.tobytes()).hexdigest()!=source['quality']['point_sha256']:raise ValueError('Canonical point hash mismatch')
            if offset+j in source_indices:
                import trimesh
                mesh_path=Path(source['mesh_path'])
                if digest(mesh_path)!=source['mesh_sha256']:raise ValueError('Source mesh changed')
                mesh=trimesh.load(mesh_path,force='mesh',process=False,skip_materials=True)
                triangles=(np.asarray(mesh.vertices)[np.asarray(mesh.faces)[faces]]-center)/scale
                reconstructed=triangles[:,0]+uv[:,0,None]*(triangles[:,1]-triangles[:,0])+uv[:,1,None]*(triangles[:,2]-triangles[:,0])
                np.testing.assert_allclose(reconstructed,points,rtol=0,atol=1e-7)
                source_checks.append(dict(uid=row['uid'],maximum_absolute_error_unit=float(np.abs(reconstructed-points).max()),mesh_sha256=source['mesh_sha256']))
            x=(j%columns)*cell_w;y=(j//columns)*cell_h+55
            draw.text((x+5,y+3),f'{offset+j+1:05d} {row["uid"][:12]}',fill='black')
            draw.text((x+5,y+17),row['origin'],fill='#555555')
            for k,axes in enumerate(((0,1),(0,2))):
                pixels=np.rint((points[:,axes]+.5)*78).astype(int)
                for a,b in pixels:draw.point((x+7+k*90+int(a),y+116-int(b)),fill='#246b97')
            draw.text((x+35,y+121),'XY',fill='#555555');draw.text((x+126,y+121),'XZ',fill='#555555')
            statistics.append(dict(uid=row['uid'],axis_extents=np.ptp(points,axis=0).tolist(),
                largest_component_fraction=source['quality']['component_sizes'][0]/1024,origin=row['origin']))
        filename=f'shapes-{offset//100:03d}.png';canvas.save(out/filename);sheets.append(filename)
    with (out/'GALLERY.md').open('x') as stream:
        stream.write('# MPM actual-point shape gallery\n\n'+('Partial preview; not final membership.\n\n' if partial else 'Final qualified shape bank; geometry only, not simulation approval.\n\n'))
        for name in sheets:stream.write(f'![{name}]({name})\n\n')
    save(out/'RESULT.json',dict(status='complete',partial=partial,objects=len(objects),origins=dict(origins),
         sheets=sheets,statistics=statistics,source_reconstruction=source_checks,review_code_sha256=digest(__file__),accepted_for_training=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--partial',action='store_true')
    parser.add_argument('--source-samples',type=int,default=32)
    parser.add_argument('--origin',choices=('historical','new'))
    parser.add_argument('--wait-seconds',type=int,default=0)
    args=parser.parse_args()
    if args.wait_seconds:
        deadline=time.monotonic()+args.wait_seconds
        while True:
            if list(args.root.glob('FAILURE-*.json')):raise RuntimeError('Shape preparation failed; inspect failure receipt')
            status=args.root/'STATUS.json'
            if status.exists() and json.loads(status.read_text())['phase']=='complete':break
            if time.monotonic()>=deadline:raise TimeoutError('Shape preparation review wait exhausted')
            time.sleep(10)
    review(args.root,args.out,partial=args.partial,source_samples=args.source_samples,origin=args.origin)
