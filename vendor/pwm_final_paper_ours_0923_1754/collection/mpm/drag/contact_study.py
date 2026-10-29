"""Diagnostic spatial/contact comparison. NEVER emits training exports or gates.

Analytic blocks use known volume quadrature and fixed +X contact normal. The
geometry-gated variant is an axis-aligned benchmark oracle, NOT a general mesh
contact implementation. Constitutive kernels/transfer weights are unchanged.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
import warp as wp
from collection.common import digest,write_once
from .. import physics,kernels,runtime
from ..contracts import Clock
from ..collect import save_npz
from ..clock import wave_speeds
from .grouped import CommandGroup


class World:
    """Guarded diagnostic without claiming production native-wall observations."""
    def __init__(self,backend,clock):
        self.backend=backend;self.solver=backend.solver;self.clock=clock
        self.config=backend.config;self.device='cuda:0';self.counter=wp.zeros(1,dtype=int,device=self.device)
    @property
    def steps(self):return self.backend.steps
    @property
    def failed(self):return self.backend.failed
    def _particles(self):pass
    def _unchanged(self):pass


@wp.kernel
def clear_bounds(bounds:wp.array(dtype=float)):
    bounds[0]=-1.e10;bounds[1]=1.e10


@wp.kernel
def face_bound(state:physics.MPMStateStruct,half_cell:float,bounds:wp.array(dtype=float),body:int):
    p=wp.tid();f=state.particle_F_trial[p]
    radius=half_cell*(wp.abs(f[0,0])+wp.abs(f[0,1])+wp.abs(f[0,2]))
    if body==0:wp.atomic_max(bounds,0,state.particle_x[p][0]+radius)
    else:wp.atomic_min(bounds,1,state.particle_x[p][0]-radius)


@wp.kernel
def planar_contact(a:physics.MPMStateStruct,b:physics.MPMStateStruct,
                   bounds:wp.array(dtype=float),gated:int,dt:float,impulse:wp.array(dtype=float)):
    i,j,k=wp.tid();ma=a.grid_m[i,j,k];mb=b.grid_m[i,j,k]
    if ma>1.e-12 and mb>1.e-12:
        va=a.grid_v_out[i,j,k];vb=b.grid_v_out[i,j,k]
        approaching=va[0]-vb[0]
        # No tensile impulse and no tangential friction in this diagnostic.
        # Geometry gate is a planar cell-domain oracle; normal is known +X.
        gap=bounds[1]-bounds[0]
        if approaching>0.0 and (gated==0 or gap<=dt*approaching):
            amount=approaching*ma*mb/(ma+mb)
            a.grid_v_out[i,j,k]=va-wp.vec3(amount/ma,0.,0.)
            b.grid_v_out[i,j,k]=vb+wp.vec3(amount/mb,0.,0.)
            wp.atomic_add(impulse,0,amount)


class SplitGroup(CommandGroup):
    """Two independent object grids with one momentum-conserving normal solve."""
    def __init__(self,worlds,*,geometry_gate,half_cell):
        if len(worlds)!=2 or len({w.config['n_grid'] for w in worlds})!=1:
            raise ValueError('Exactly two equally resolved bodies in planar prototype')
        if any(w.solver.grid_postprocess for w in worlds):raise ValueError('Planar benchmark has no boundaries')
        super().__init__(worlds,max_command_speed=.6,fused=False)
        self.bounds=wp.zeros(2,dtype=float,device=self.device)
        self.impulse=wp.zeros(1,dtype=float,device=self.device)
        with wp.ScopedCapture(device=self.device,force_module_load=True) as post:
            wp.launch(clear_bounds,dim=1,inputs=[self.bounds],device=self.device)
            for i,w in enumerate(worlds):
                b=w.backend;s=w.solver;state=s.mpm_state;model=s.mpm_model
                wp.launch(face_bound,dim=b.n,inputs=[state,half_cell,self.bounds,i],device=self.device)
                cmd,mask=self.command_buffers[i]
                wp.launch(physics._command,dim=b.n,inputs=[state.particle_v,cmd,mask],device=self.device)
                shape=(model.n_grid,)*3
                wp.launch(kernels.zero_grid,dim=shape,inputs=[state,model],device=self.device)
                wp.launch(kernels.compute_stress_from_F_trial,dim=b.n,inputs=[state,model,self.clock.substep_dt_s],device=self.device,block_dim=64)
                wp.launch(kernels.p2g_apic_with_stress,dim=b.n,inputs=[state,model,self.clock.substep_dt_s],device=self.device,block_dim=64)
                wp.launch(kernels.grid_normalization_and_gravity,dim=shape,inputs=[state,model,self.clock.substep_dt_s],device=self.device)
            wp.launch(planar_contact,dim=(worlds[0].config['n_grid'],)*3,
                inputs=[worlds[0].solver.mpm_state,worlds[1].solver.mpm_state,self.bounds,int(geometry_gate),self.clock.substep_dt_s,self.impulse],device=self.device)
            for i,w in enumerate(worlds):
                b=w.backend;s=w.solver
                wp.launch(kernels.g2p,dim=b.n,inputs=[s.mpm_state,s.mpm_model,self.clock.substep_dt_s],device=self.device,block_dim=64)
                wp.launch(physics._accumulate,dim=b.n,inputs=[b.master,s.mpm_state.particle_x,s.mpm_state.particle_v,self.clock.substep_dt_s],device=self.device)
                self.guard(w,i)
        self.post=post.graph


def block_fixture(case,n_grid,*,points_axis=16,E=1e5,padded=False):
    if case not in ('near_miss','push_withdraw','free_impact'):raise ValueError('Unknown analytic case')
    side=.1;spacing=side/points_axis
    axis=(np.arange(points_axis)+.5)*spacing-side/2
    cube=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)
    gap=.02 if case=='near_miss' else .04
    x=np.concatenate([cube+[.25-(side+gap)/2,.25,.25],cube+[.25+(side+gap)/2,.25,.25]]).astype(np.float32)
    ids=np.repeat(np.arange(2,dtype=np.int32),len(cube));velocity=np.zeros_like(x)
    if case=='near_miss':velocity[ids==0,1]=.15
    elif case=='free_impact':velocity[ids==0,0]=.2
    static=dict(object_ids=ids,particle_volume=np.full(len(x),spacing**3,np.float32),
        E=np.full(2,E,np.float32),nu=np.full(2,.2,np.float32),density=np.full(2,1000.,np.float32),
        mat_type=np.zeros(2,np.int32),yield_stress=np.zeros(2,np.float32))
    config=dict(n_grid=n_grid,grid_lim_m=.5,floor_m=.01,floor_contact=False,friction=0.,floor_tolerance_m=0.,
        observation_emergency_depth_m=.01,gravity_m_s2=[0.,0.,0.],dt_s=5e-5,precision='float64-master',softening=.1)
    if padded:
        # Add empty cells, preserving dx, grid origin and every particle state.
        # Successful rebounds otherwise leave the original free-space fixture.
        config.update(n_grid=2*n_grid,grid_lim_m=1.)
    courant=config['dt_s']*(float(wave_speeds(static['E'],static['nu'],static['density']).max())+8)/(config['grid_lim_m']/config['n_grid'])
    if courant>.4:raise ValueError('Spatial variant exceeds common-clock CFL budget')
    return x,static,config,velocity,spacing,courant


def prescribed(case,t,ids):
    command=np.zeros((len(ids),3),np.float32);mask=np.zeros(len(ids),np.int32)
    if case=='push_withdraw' and t<.6:
        mask[ids==0]=1;command[ids==0,0]=.2 if t<.3 else -.2
    return command,mask


def metrics(x,v,F,static,t,spacing,case,dx):
    ids=static['object_ids'];mass=static['particle_volume'].astype(float)*static['density'][ids]
    com=np.stack([np.average(x[:,ids==i],weights=mass[ids==i],axis=1) for i in range(2)],axis=1)
    cv=np.stack([np.average(v[:,ids==i],weights=mass[ids==i],axis=1) for i in range(2)],axis=1)
    # Cell domains, not point centres; correct initial gap is 20 or 40 mm.
    radius=.5*spacing*np.abs(F[:,: ,0,:]).sum(-1)
    gap=(x[:,ids==1,0]-radius[:,ids==1]).min(1)-(x[:,ids==0,0]+radius[:,ids==0]).max(1)
    target=np.linalg.norm(com[:,1]-com[0,1],axis=1)
    onset=np.flatnonzero(np.linalg.norm(cv[:,1],axis=1)>1e-3)
    momentum=np.sum(v*mass[None,:,None],axis=1)
    kinetic=.5*np.sum(np.sum(v.astype(float)**2,axis=-1)*mass,axis=1)
    J=np.linalg.det(F.astype(float));E=static['E'].astype(float)[ids];nu=static['nu'].astype(float)[ids]
    mu=E/(2*(1+nu));bulk=E/(3*(1-2*nu))
    W=.5*mu*(J**(-2/3)*np.sum(F.astype(float)**2,axis=(-2,-1))-3)+.25*bulk*(J*J-1-2*np.log(J))
    strain=np.sum(W*static['particle_volume'],axis=1)
    withdrawal=t>=.3
    return dict(case=case,dx_m=dx,first_target_motion_time_s=float(t[onset[0]]) if len(onset) else None,
        estimated_surface_gap_at_first_motion_m=float(gap[onset[0]]) if len(onset) else None,
        minimum_estimated_surface_gap_m=float(gap.min()),final_estimated_surface_gap_m=float(gap[-1]),
        maximum_target_displacement_m=float(target.max()),
        minimum_target_vx_after_withdrawal_m_s=float(cv[withdrawal,1,0].min()) if case=='push_withdraw' else None,
        max_linear_momentum_drift_kg_m_s=float(np.linalg.norm(momentum-momentum[0],axis=1).max()),
        externally_commanded=case=='push_withdraw',
        resolved_energy_proxy_max_ratio=float((kinetic+strain).max()/max(kinetic[0],1e-12)) if case!='push_withdraw' else None,
        energy_note='translational kinetic + neo-Hookean stored energy; affine APIC kinetic not included',
        time_s=t.tolist(),gap_m=gap.tolist(),com_m=com.tolist(),com_velocity_m_s=cv.tolist(),
        momentum_kg_m_s=momentum.tolist(),kinetic_J=kinetic.tolist(),strain_J=strain.tolist())


def analytic(output,case,n_grid,method,uuid,shared_pid=None,*,padded=False):
    from collection.lease import gpu_lease
    from .study_device import DeviceGuard
    if method not in ('shared','multifield','geometry_gated'):raise ValueError('Unknown method')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sources=runtime.sources();started=time.monotonic()
    x,static,config,velocity,spacing,courant=block_fixture(case,n_grid,padded=padded)
    clock=Clock(config['dt_s'],100,140 if case=='push_withdraw' else 120)
    save_npz(output/'INITIAL.npz',dict(x=x,v=velocity,**static))
    write_once(output/'REQUEST.json',dict(case=case,method=method,config=config,clock=asdict(clock),
        courant=courant,spacing_m=spacing,initial_sha256=digest(output/'INITIAL.npz'),sources=sources,
        geometry_note='two 0.1 m solid elastic cubes, 16^3 cell-centre material points each; not surface-sampled assets',
        contact_note='fixed +X frictionless unilateral prototype; geometry gate uses deformed material-cell X extents, not a general contact implementation',
        scaling_approved=False,padded=padded,reference_n_grid=n_grid))
    guard=DeviceGuard(uuid,shared_pid)
    with gpu_lease(uuid):
        write_once(output/'DEVICE.json',guard.check(startup=True));wp.config.kernel_cache_dir=str(output.parent/'warp-cache')
        subsets=[np.ones(len(x),bool)] if method=='shared' else [static['object_ids']==i for i in range(2)]
        worlds=[]
        for i,sel in enumerate(subsets):
            values={k:(v.copy() if len(subsets)==1 else v[sel].copy() if k in ('particle_volume','object_ids') else v[i:i+1].copy()) for k,v in static.items()}
            if len(subsets)>1:values['object_ids'].fill(0)
            b=physics.CollectionBackend(x[sel],values,config,velocity[sel],device='cuda:0')
            # Remove the five far-domain collider kernels for free-space fixture.
            b.solver.grid_postprocess=[];b.solver.collider_params=[];b.solver.modify_bc=[]
            worlds.append(World(b,clock))
        group=CommandGroup(worlds,max_command_speed=.6,fused=False) if method=='shared' else SplitGroup(worlds,geometry_gate=method=='geometry_gated',half_cell=spacing/2)
        records={k:[] for k in ('x','v','F_trial')};last=0.
        for frame in range(clock.transitions+1):
            if frame:
                cmd,mask=prescribed(case,(frame-1)*.005,static['object_ids'])
                group.set_commands([cmd[sel] for sel in subsets],[mask[sel] for sel in subsets])
                for _ in range(clock.substeps_per_transition):
                    if time.monotonic()-last>5:
                        guard.check();last=time.monotonic()
                        if time.monotonic()-started>1800:raise TimeoutError('Bounded analytic study')
                    try:
                        group.step()
                    except FloatingPointError as error:
                        # Preserve rejected diagnostic evidence, never accepted
                        # state/checkpoints. Production read() remains fail-closed.
                        rejected={key:np.concatenate([getattr(w.solver.mpm_state,'particle_'+key).numpy() for w in worlds])
                                  for key in ('x','v','F_trial')}
                        save_npz(output/'REJECTED_STATE.npz',dict(**rejected,object_ids=static['object_ids']))
                        partial={key:np.stack(value) for key,value in records.items()}
                        partial.update(time_s=clock.state_time_s[:len(records['x'])],object_ids=static['object_ids'])
                        save_npz(output/'PARTIAL_TRAJECTORY.npz',partial)
                        write_once(output/'FAILURE.json',dict(reason=str(error),native_step=worlds[0].steps,
                            physical_time_s=worlds[0].steps*clock.substep_dt_s,
                            completed_frames=len(records['x']),diagnostics=[w.backend.diagnostics() for w in worlds],
                            position_min_m=rejected['x'].min(0).tolist(),position_max_m=rejected['x'].max(0).tolist(),
                            accepted=False,scaling_approved=False))
                        raise
            values=[w.backend.read() for w in worlds]
            for key in records:records[key].append(np.concatenate([v[key] for v in values]))
            if frame%40==0:print(json.dumps(dict(case=case,method=method,n_grid=n_grid,frame=frame,total=clock.transitions)),flush=True)
        arrays={k:np.stack(v) for k,v in records.items()};arrays.update(time_s=clock.state_time_s,object_ids=static['object_ids'])
        save_npz(output/'TRAJECTORY.npz',arrays)
        result=metrics(arrays['x'],arrays['v'],arrays['F_trial'],static,arrays['time_s'],spacing,case,config['grid_lim_m']/config['n_grid'])
        result.update(method=method,n_grid=config['n_grid'],reference_n_grid=n_grid,padded=padded,seconds=time.monotonic()-started,
            diagnostics=[w.backend.diagnostics() for w in worlds],scaling_approved=False,
            shared_gpu=shared_pid is not None,throughput_comparison_valid=shared_pid is None)
        runtime.verify_sources(sources);write_once(output/'RESULT.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('time_s','gap_m','com_m','com_velocity_m_s','momentum_kg_m_s','kinetic_J','strain_J','diagnostics')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--case',choices=('near_miss','push_withdraw','free_impact'),required=True)
    p.add_argument('--n-grid',type=int,choices=(32,64,96,128),required=True,
                   help='Cells in the reference 0.5 m box; --padded doubles box and cell count at fixed dx')
    p.add_argument('--method',choices=('shared','multifield','geometry_gated'),required=True)
    p.add_argument('--uuid',required=True);p.add_argument('--share-with-pid',type=int)
    p.add_argument('--padded',action='store_true',help='Double free-space domain/cell count at identical dx and initial state')
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.uuid:raise ValueError('Explicit allocated GPU UUID required')
    analytic(a.output,a.case,a.n_grid,a.method,a.uuid,a.share_with_pid,padded=a.padded)


if __name__=='__main__':main()
