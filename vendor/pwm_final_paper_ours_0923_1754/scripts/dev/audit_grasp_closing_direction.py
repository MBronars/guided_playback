"""Measure geometric compression under finger closing; this is not force closure."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from guidance.collision_distance import ConvexCollisionDistance


def probe(query, R, T, axes, epsilons=(.00005, .0001)):
    gaps = query.query(R, T)[0][1:]
    world_axes = axes @ R[1].T
    derivatives = []
    for epsilon in epsilons:
        pair = []
        for i in range(2):
            plus, minus = T.copy(), T.copy()
            plus[2+i] += epsilon * world_axes[i]
            minus[2+i] -= epsilon * world_axes[i]
            pair.append((query.query(R, plus)[0][1+i]
                         - query.query(R, minus)[0][1+i]) / (2*epsilon))
        derivatives.append(pair)
    return gaps, np.array(derivatives)


def analytic_gate(vendors):
    def body(boxes):
        vertices, planes = [], []
        for center, half in boxes:
            center, half = np.array(center), np.broadcast_to(half, (3,))
            vertices.extend(center + np.array([[x,y,z] for x in (-1,1)
                for y in (-1,1) for z in (-1,1)]) * half)
            normal = np.concatenate([np.eye(3), -np.eye(3)])
            planes.append(np.column_stack([normal, -normal@center-np.tile(half,2)]))
        return dict(vertices=np.array(vertices), planes=planes)
    small=body([([0,0,0],.01)])
    cube=body([([0,0,0],.1)])
    walls=body([([-.12,0,0],[.02,.1,.1]),([.12,0,0],[.02,.1,.1])])
    R=np.tile(np.eye(3),(4,1,1));axes=np.array([[1.,0,0],[-1.,0,0]])
    results=[]
    for name,obj,positions,expected in (
        ('outside squeeze',cube,[[.13,0,0],[-.13,0,0]],1.),
        ('inside squeeze requires opening',walls,[[.07,0,0],[-.07,0,0]],-1.),
        ('tangential closing',cube,[[.03,0,.13],[-.03,0,.13]],0.),
    ):
        query=ConvexCollisionDistance(dict(bodies={0:obj,1:small,2:small,3:small}),vendor_paths=vendors)
        T=np.array([[0.,0,0],[0,0,1],*positions])
        gap, derivative=probe(query,R,T,axes)
        np.testing.assert_allclose(gap,.02,atol=1e-7)
        np.testing.assert_allclose(derivative,expected,atol=1e-5)
        results.append(dict(case=name,gaps_m=gap.tolist(),derivative=derivative.tolist()))
    return dict(status='passed',results=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();root=args.root.resolve();out=args.out.resolve()
    out.mkdir(exist_ok=False)
    vendors=[root/'vendor_fcl',root.parent/'overnight_2026-09-14/vendor']
    gate=analytic_gate(vendors);(out/'analytic_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    scenes={s['name']:s for s in json.loads((root/'scenes_v2.json').read_text())['scenes']}
    controls=json.loads((root/'mujoco_controls_matched_v2_summary.json').read_text())['results']
    work=[]
    for c in controls:
        work.append(dict(name=c['scene']+'_recorded',scene=c['scene'],kind='recorded',
            source=str(Path(c['paths']['recorded'])/'execution.npz')))
    # Freeze all completed region plans at invocation, plus the two 100-update
    # physical examples. Later plans do not silently enter this diagnostic set.
    for path in sorted(root.glob('*/seed_8311.npz')):
        run=path.parent
        if 'region_mesh_physical24_r025_h36' not in run.name and not (
            'physical100_h36' in run.name and '_al' not in run.name):continue
        if not (run/'results.json').exists():continue
        protocol=json.loads((run/'protocol.json').read_text())['arguments']
        scene=next(s for s in scenes.values() if Path(s['directory']).resolve()==Path(protocol['input']).resolve().parent)
        work.append(dict(name=run.name,scene=scene['name'],kind='generated',source=str(path)))
    rows=[]
    for item in work:
        scene=scenes[item['scene']];directory=Path(scene['directory'])
        b=torch.load(directory/'input.pt',weights_only=False,map_location='cpu')
        g=torch.load(directory/'collision_geometry.pt',weights_only=False,map_location='cpu')
        c=torch.load(directory/'calibration.pt',weights_only=False,map_location='cpu')['calibration']
        c={k:np.asarray(v,dtype=np.float64) for k,v in c.items()}
        source=Path(item['source']);z=np.load(source)
        if item['kind']=='recorded':R,T=z['desired_R'],z['desired_T']
        else:
            # Campaign exports include all context frames, then future frames.
            assert int(z['context_len'])==b.context_len
            R,T=z['R'][0,b.context_len-1:],z['T'][0,b.context_len-1:]
            anchor=b.body_poses[0,b.context_len-1].numpy()
            np.testing.assert_allclose(R[0],anchor[:,:3,:3],atol=1e-6)
            np.testing.assert_allclose(T[0],anchor[:,:3,3],atol=1e-6)
        R,T=R.astype(np.float64),T.astype(np.float64)
        query=ConvexCollisionDistance(g,vendor_paths=vendors)
        data=[probe(query,rr,tt,c['axis']) for rr,tt in zip(R,T)]
        gaps=np.stack([v[0] for v in data]);derivatives=np.stack([v[1] for v in data])
        near=gaps<=.002;both=near.all(-1)
        stable=np.abs(derivatives[:,0]-derivatives[:,1])<=.05
        squeezing=stable & (derivatives[:,1]>.1)
        opening=stable & (derivatives[:,1]<-.1)
        tangential=stable & (np.abs(derivatives[:,1])<=.1)
        local=np.einsum('fji,fkj->fki',R[:,1],T[:,2:]-T[:,1,None])
        q=c['q_anchor']+((local-c['finger_anchor'])*c['axis']).sum(-1)
        command=np.clip(q-.006*both[:,None],c['limits'][:,0],c['limits'][:,1])
        delta=[]
        for i,(rr,tt) in enumerate(zip(R,T)):
            closed=tt.copy();closed[2:]+=(command[i]-q[i])[:,None]*(c['axis']@rr[1].T)
            delta.append(query.query(rr,closed)[0][1:]-gaps[i])
        np.savez_compressed(out/(item['name']+'.npz'),R=R,T=T,gaps=gaps,derivatives=derivatives,
            near=near,both=both,stable=stable,squeezing=squeezing,opening=opening,tangential=tangential,
            q=q,command=command,command_gap_change=np.array(delta))
        row=dict(**item,task=scene['task'],group=scene['group'],frames=len(R),
            both_near_frames=int(both.sum()),near_finger_frames=int(near.sum()),
            near_closing_compresses=int((near&squeezing).sum()),near_opening_compresses=int((near&opening).sum()),
            near_tangential=int((near&tangential).sum()),near_unstable=int((near&~stable).sum()),
            both_closing_compresses_frames=int((both&squeezing.all(-1)).sum()),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            scene_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [directory/'input.pt',directory/'collision_geometry.pt',directory/'calibration.pt']})
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(results=rows,analytic_gate=gate['status'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        definition='Positive d(gap)/d(q) means decreasing q (closing) locally decreases the signed mesh gap. Two finite-difference scales 0.05/0.1mm; stable if difference<=0.05; directional threshold0.1; near if gap<=2mm.',
        limitations='Local proximity geometry is not force closure or physical success. Feature switches can invalidate a local derivative. Fixed 6mm closing is an exploratory controller assumption; no active experiment or task criterion changed.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
