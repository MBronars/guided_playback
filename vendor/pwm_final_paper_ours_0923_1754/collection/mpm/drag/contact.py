"""Read-only reconstruction of per-object P2G mass overlap at saved frames.

The 27-node quadratic weights follow kernels.p2g_apic_with_stress. This observes
the actual shared-grid coupling mechanism, not exact mesh contact or impulses.
Float32 weights / float64 CPU reductions need not be bitwise CUDA atomic sums.
Nothing here is called by simulation or changes trajectories/retention.
"""
import itertools
import numpy as np

POLICY='saved-p2g-mass-overlap-v1'
OFFSETS=np.array(list(itertools.product(range(3),repeat=3)),dtype=np.int64)


def grid_mass(x,mass,n_grid,grid_lim_m):
    x=np.asarray(x,dtype=np.float32);mass=np.asarray(mass,dtype=float)
    if (x.ndim!=2 or x.shape[1]!=3 or mass.shape!=(len(x),)
            or not np.isfinite(x).all() or not np.isfinite(mass).all() or np.any(mass<=0)
            or type(n_grid) is not int or n_grid<3 or not np.isfinite(grid_lim_m) or grid_lim_m<=0):
        raise ValueError('Finite positive particle masses and valid grid required')
    gp=x*np.float32(n_grid/grid_lim_m)
    base=np.trunc(gp-np.float32(.5)).astype(np.int64);fx=gp-base.astype(np.float32)
    weights=np.stack((.5*(1.5-fx)**2,.75-(fx-1.)**2,.5*(fx-.5)**2),axis=2)
    coords=base[:,None,:]+OFFSETS[None,:,:]
    if np.any(coords<0) or np.any(coords>=n_grid):raise ValueError('Particle stencil outside grid')
    w=weights[:,0,OFFSETS[:,0]]*weights[:,1,OFFSETS[:,1]]*weights[:,2,OFFSETS[:,2]]
    if np.any(w < -1e-7):raise ValueError('Invalid quadratic interpolation support')
    keys=(coords[:,:,0]*n_grid+coords[:,:,1])*n_grid+coords[:,:,2]
    unique,inverse=np.unique(keys,return_inverse=True)
    values=np.bincount(inverse.ravel(),weights=(w*mass[:,None]).ravel(),minlength=len(unique))
    keep=values>0
    return unique[keep],values[keep]


def overlap(a,b):
    _,i,j=np.intersect1d(a[0],b[0],assume_unique=True,return_indices=True)
    return float(np.minimum(a[1][i],b[1][j]).sum())


def measure(manifest,arrays):
    layout=manifest.get('geometry_provenance',{}).get('drag_interaction',{})
    requested=layout.get('requested_pairs',[])
    x=arrays['x'];ids=arrays['object_ids'];first=manifest['action']['events'][0]['start']
    mass=arrays['particle_volume'].astype(float)*arrays['density'][ids]
    objects=sorted({p[k] for p in requested for k in ('a','b')})
    curves=[[] for _ in requested]
    for frame in range(len(x)):
        grids={oid:grid_mass(x[frame,ids==oid],mass[ids==oid],int(manifest['n_grid']),manifest['grid_lim_m']) for oid in objects}
        for i,p in enumerate(requested):
            left,right=p['a'],p['b']
            curves[i].append(overlap(grids[left],grids[right])/min(mass[ids==left].sum(),mass[ids==right].sum()))
    pairs=[]
    for p,curve in zip(requested,curves):
        curve=np.asarray(curve);com=x[:,ids==p['b']].astype(float).mean(1)
        # 0.01% of the smaller object's mass excludes tiny floating-point tails.
        shared=np.flatnonzero((curve>=1e-4)&(np.arange(len(curve))>=first))
        post=float(np.linalg.norm(com[shared[0]:]-com[shared[0]],axis=-1).max()) if len(shared) else 0.
        before=float(np.linalg.norm(com[:first+1]-com[0],axis=-1).max())
        pairs.append(dict(**p,maximum_shared_mass_fraction=float(curve.max()),
            initial_shared_mass_fraction=float(curve[0]),shared_frames_after_drive=len(shared),
            first_shared_frame=int(shared[0]) if len(shared) else None,
            target_post_coupling_displacement_m=post,target_pre_drive_displacement_m=before,
            transfer_candidate=bool(len(shared)>=2 and post>=.005 and before<=.001),
            shared_mass_fraction_by_frame=curve.tolist()))
    return dict(policy=POLICY,mode=layout.get('mode','legacy_separated'),
        exact_mesh_contact_or_impulse=False,observation='saved-frame reconstruction; not every native step',
        minimum_shared_mass_fraction=1e-4,minimum_target_translation_m=.005,
        maximum_pre_drive_translation_m=.001,pairs=pairs,
        any_transfer_candidate=any(p['transfer_candidate'] for p in pairs))
