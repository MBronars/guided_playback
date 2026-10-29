"""Exploratory point-contact gravity-support test for saved trajectory geometry.

This tests a linearized friction-cone model at FCL's nearest witnesses. It omits
contact patches, acceleration and table support; optional torsional friction is
explicitly declared. Qualify it
on recorded controls before interpreting generated-plan failures. It is not a
task acceptance test and does not alter planning costs.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from guidance.collision_distance import ConvexCollisionDistance
from utils.reproducibility import file_sha256


def gravity_support(points, outward_normals, com, *, mass, friction=1., normal_cap=200., radius=.05, rays=16, torsion=0.):
    """Minimum normalized force/torque imbalance under unilateral point friction.

Torque is divided by the declared characteristic length. The polyhedral cone
is inscribed in the circular Coulomb cone. Positive normal forces point inward.
"""
    from scipy.optimize import linprog
    if (not all(math.isfinite(v) for v in (mass,normal_cap,radius,friction,torsion))
            or min(mass,normal_cap,radius)<=0 or min(friction,torsion)<0 or rays<4):
        raise ValueError('Invalid support-test parameters')
    points=np.asarray(points).reshape(-1,3);normals=np.asarray(outward_normals).reshape(-1,3)
    if points.shape!=normals.shape or not np.isfinite([*points.ravel(),*normals.ravel(),*com]).all():raise ValueError('Invalid contact geometry')
    if len(points)==0:return dict(residual=1.,feasible=False,normal_force_N=[])
    columns=[]
    for point,normal in zip(points,normals):
        norm=np.linalg.norm(normal)
        if norm<1e-8:raise ValueError('Undefined contact normal')
        inward=-normal/norm
        helper=np.eye(3)[np.abs(inward).argmin()];u=np.cross(inward,helper);u/=np.linalg.norm(u);v=np.cross(inward,u)
        # The 4-D soft-finger cone couples tangential force and torsional moment
        # through one ellipsoid; independent box bounds would overestimate it.
        for latitude in (np.linspace(-np.pi/2,np.pi/2,7) if torsion else [0.]):
            for angle in np.arange(rays)*2*np.pi/rays:
                force=inward+friction*np.cos(latitude)*(np.cos(angle)*u+np.sin(angle)*v)
                torque=np.cross(point-com,force)+torsion*np.sin(latitude)*inward
                columns.append(np.r_[force,torque/radius])
    W=np.stack(columns,axis=1);b=np.array([0,0,1,0,0,0.]);n=W.shape[1]
    rays_per_contact=n//len(points)
    caps=np.zeros((len(points),n+1))
    for j in range(len(points)):caps[j,j*rays_per_contact:(j+1)*rays_per_contact]=1
    A=np.vstack([np.c_[W,-np.ones(6)],np.c_[-W,-np.ones(6)],caps])
    rhs=np.r_[b,-b,np.full(len(points),normal_cap/(mass*9.81))]
    cost=np.r_[np.zeros(n),1.]
    solution=linprog(cost,A_ub=A,b_ub=rhs,bounds=(0,None),method='highs')
    if not solution.success:raise RuntimeError(solution.message)
    residual=float(solution.x[-1]);feasible=residual<=1e-6
    forces=solution.x[:-1]
    if feasible:
        # Among feasible solutions, report the least total normal force.
        optimal=linprog(np.ones(n),A_eq=W,b_eq=b,A_ub=caps[:,:-1],
            b_ub=rhs[-len(points):],bounds=(0,None),method='highs')
        if optimal.success:forces=optimal.x
    return dict(residual=residual,feasible=feasible,
        normal_force_N=(forces.reshape(len(points),rays_per_contact).sum(-1)*mass*9.81).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('trajectory','geometry','replay-protocol','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--vendor',action='append',default=[])
    p.add_argument('--contact-mm',type=float,default=2.)
    p.add_argument('--friction',type=float,default=1.)
    p.add_argument('--torsion',type=float,default=0.,help='Torsional moment/normal force, in metres; zero is point contact')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    if not math.isfinite(args.contact_mm) or args.contact_mm<=0:raise ValueError('Contact tolerance must be positive and finite')
    geometry=torch.load(args.geometry,map_location='cpu',weights_only=False)
    collision=ConvexCollisionDistance(geometry,vendor_paths=args.vendor)
    receipt=json.loads(args.replay_protocol.read_text())
    # Actual replay fields are preserved in the receipt; do not infer a new mass.
    mass=receipt['body_mass_kg'][0];com_local=np.asarray(receipt['body_com_local'][0])
    data=np.load(args.trajectory)
    if 'context_len' in data:
        C=int(data['context_len']);R=data['R'][0,C-1:];T=data['T'][0,C-1:]
    else:R=data['desired_R'];T=data['desired_T']
    rows=[]
    for frame,(r,t) in enumerate(zip(R,T)):
        distance,witness=collision.query(r,t);points=[];normals=[];contacts=[]
        for k in (1,2):
            d=distance[k]
            if abs(d)>1e-8 and d<=args.contact_mm/1000:
                points.append(witness[k,0]);normals.append((witness[k,1]-witness[k,0])/d);contacts.append(k+1)
        support=gravity_support(points,normals,r[0]@com_local+t[0],mass=mass,friction=args.friction,torsion=args.torsion)
        bottom=(np.asarray(geometry['bodies'][0]['vertices'])@r[0].T+t[0])[:,2].min()
        rows.append(dict(frame=frame,time=frame*.05,contact_bodies=contacts,
            finger_gaps_mm=(distance[1:]*1000).tolist(),bottom_above_table_mm=float(bottom-geometry['metadata']['support_z'])*1000,**support))
    elevated=[v for v in rows if v['bottom_above_table_mm']>10]
    report=dict(status='completed',hypothesis='Two nearest finger witnesses can quasistatically support gravity',
        limitations=__doc__,parameters=dict(mass_kg=mass,friction=args.friction,normal_cap_per_contact_N=200.,
            torque_length_m=.05,cone_longitudes=16,cone_latitudes=7 if args.torsion else 1,
            torsional_friction_m=args.torsion,contact_tolerance_mm=args.contact_mm),
        hashes={str(f.resolve()):file_sha256(f) for f in (args.trajectory,args.geometry,args.replay_protocol)},
        elevated_frames=len(elevated),elevated_feasible_frames=sum(v['feasible'] for v in elevated),frames=rows)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('frames','hashes','limitations')}))


if __name__=='__main__':main()
