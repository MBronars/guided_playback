"""Guarded real-Warp bridge for the existing isolated prepared-scene collector.

No asset/volume policy is selected here. No production files are changed. The
default kernels are the previously reviewed elastic-law candidate; plasticine's
constitutive inconsistency is a separate recorded blocker, not silently patched.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

HERE = Path(__file__).resolve().parent
SOLVER = HERE
from .solver import MPM_Simulator_WARP
from .state import MPMStateStruct
from .contracts import canonical_json, finite_array
from .material_control import POLICY as HENCKY_POLICY, step_with_consistent_material


@wp.kernel
def _command(v: wp.array(dtype=wp.vec3), command: wp.array(dtype=wp.vec3), selected: wp.array(dtype=int)):
    p=wp.tid()
    if selected[p] != 0: v[p]=command[p]


@wp.kernel
def _accumulate(master: wp.array(dtype=wp.vec3d), x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), dt: wp.float64):
    p=wp.tid()
    master[p]=master[p]+dt*wp.vec3d(wp.float64(v[p][0]),wp.float64(v[p][1]),wp.float64(v[p][2]))
    x[p]=wp.vec3(wp.float32(master[p][0]),wp.float32(master[p][1]),wp.float32(master[p][2]))


@wp.kernel
def _guard(state: MPMStateStruct, inv_dx: float, grid: int, floor: float,
           floor_tolerance: float, failures: wp.array(dtype=int), extrema: wp.array(dtype=float)):
    p=wp.tid()
    x=state.particle_x[p];v=state.particle_v[p];f=state.particle_F_trial[p];c=state.particle_C[p]
    if not wp.isfinite(x) or not wp.isfinite(v) or not wp.isfinite(f) or not wp.isfinite(c):
        wp.atomic_max(failures,0,1)
    else:
        for axis in range(3):
            base=wp.int(x[axis]*inv_dx-0.5)
            if x[axis] < 0.0 or base < 0 or base+2 >= grid:
                wp.atomic_max(failures,1,1)
        j=wp.determinant(f)
        if not wp.isfinite(j) or j <= 0.0: wp.atomic_max(failures,2,1)
        if x[2] < floor-floor_tolerance: wp.atomic_max(failures,3,1)
        wp.atomic_min(extrema,0,j)
        wp.atomic_min(extrema,1,x[2])
        wp.atomic_max(extrema,2,wp.length(v))


def array_fingerprint(values):
    digest=hashlib.sha256()
    for name in sorted(values):
        v=np.ascontiguousarray(values[name])
        digest.update(canonical_json([name,v.shape,str(v.dtype)]).encode());digest.update(v.tobytes())
    return digest.hexdigest()


class GuardedWarpBackend:
    def __init__(self, x, static, config, *, device='cpu', initial=None):
        self.device=device
        self.config=json.loads(canonical_json(config))
        self.material_policy=config.get('plasticine_policy','archived-fcr-log-return')
        if self.material_policy not in ('archived-fcr-log-return',HENCKY_POLICY):
            raise ValueError('Unknown explicit plasticine policy')
        if self.material_policy==HENCKY_POLICY and config.get('softening',.1)!=0:
            raise ValueError('Perfect-plastic diagnostic requires explicit zero softening')
        self.static={k:np.asarray(v).copy() for k,v in static.items()}
        n=len(x); ids=finite_array(static['object_ids'],(n,),'object_ids')
        if not np.issubdtype(ids.dtype,np.integer) or not np.array_equal(np.unique(ids),np.arange(len(static['E']))):
            raise ValueError('Contiguous integer object IDs required')
        if not 1<=n<=100000: raise ValueError('Bounded diagnostic particle limit exceeded')
        x=finite_array(x,(n,3),'x').astype(np.float32)
        volume=finite_array(static['particle_volume'],(n,),'volume').astype(np.float32)
        if np.any(volume<=0): raise ValueError('Positive declared reference volumes required')
        k=len(static['E'])
        for key in ('E','nu','density','yield_stress','mat_type'): finite_array(static[key],(k,),key)
        if np.any(static['E']<0) or np.any(static['density']<=0) or np.any(static['nu']<=-1) or np.any(static['nu']>=.5):
            raise ValueError('Invalid elastic/density parameters')
        if not np.isin(static['mat_type'],[0,1,2]).all(): raise ValueError('Unsupported material')
        if np.any(static['yield_stress']<0): raise ValueError('Nonnegative yield required')
        if config['precision'] not in ('float32','float64-master'): raise ValueError('Explicit position precision required')
        if not 0<=config['floor_m']<config['grid_lim_m'] or config['floor_tolerance_m']<0: raise ValueError('Invalid floor')
        if (not isinstance(config['n_grid'],int) or isinstance(config['n_grid'],bool)
                or not 4<=config['n_grid']<=256 or config['dt_s']<=0):
            raise ValueError('Invalid bounded grid/time')
        finite_array(config['gravity_m_s2'],(3,),'gravity')
        if (not isinstance(config['floor_contact'],bool) or config['friction']<0
                or config.get('softening',.1)<0):
            raise ValueError('Invalid floor/material settings')
        self.n=n;self.steps=0;self.failed=False;self.calls_checked=0
        self.source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [Path(__file__),HERE/'material_control.py',*SOLVER.glob('*.py')]}
        self.solver=MPM_Simulator_WARP(n,n_grid=config['n_grid'],grid_lim=config['grid_lim_m'],device=device)
        s,m=self.solver.mpm_state,self.solver.mpm_model
        s.particle_x=wp.array(x,dtype=wp.vec3,device=device)
        s.particle_vol=wp.array(volume,dtype=float,device=device)
        for field in ('E','nu','yield_stress'):
            setattr(m,field,wp.array(np.asarray(static[field],np.float32)[ids],dtype=float,device=device))
        m.material=wp.array(np.asarray([7,5,2],np.int32)[np.asarray(static['mat_type'],np.int32)[ids]],dtype=int,device=device)
        density=np.asarray(static['density'],np.float32)[ids]
        s.particle_density=wp.array(density,dtype=float,device=device)
        s.particle_mass=wp.array(volume*density,dtype=float,device=device)
        m.gravitational_accelaration=wp.vec3(*config['gravity_m_s2'])
        m.rpic_damping=0.;m.grid_v_damping_scale=1.;m.softening=config.get('softening',.1);m.hardening=0;m.xi=0.
        self.solver.finalize_mu_lam_bulk(device=device)
        # Positive finite input metadata can still overflow or underflow when
        # converted to the solver's float32 arrays. Check the installed values.
        for key in ('E','nu','mu','lam','bulk','yield_stress'):
            finite_array(getattr(m,key).numpy(),(n,),'installed_'+key)
        for key in ('density','mass','vol'):
            values=finite_array(getattr(s,'particle_'+key).numpy(),(n,),'installed_'+key)
            if np.any(values<=0): raise ValueError('Nonpositive installed '+key)
        state={'v':np.zeros((n,3),np.float32), 'F_trial':np.tile(np.eye(3,dtype=np.float32),(n,1,1)),
               'C':np.zeros((n,3,3),np.float32)}
        if initial is not None:
            if set(initial)!=set(state): raise ValueError('Restart needs v/F_trial/C exactly; use checkpoint restore for plastic history')
            state={key:finite_array(initial[key],value.shape,key).astype(np.float32) for key,value in state.items()}
            if np.any(np.asarray(static['mat_type'])!=0): raise ValueError('Fresh/restart elastic state only; plastic history requires full checkpoint')
        for key,dtype in [('v',wp.vec3),('F_trial',wp.mat33),('C',wp.mat33)]:
            setattr(s,'particle_'+key,wp.array(state[key],dtype=dtype,device=device))
        if config['floor_contact']:
            self.solver.add_surface_collider([0,0,config['floor_m']],[0,0,1],surface='separate',friction=config['friction'])
        self.master=wp.array(x.astype(np.float64),dtype=wp.vec3d,device=device)
        self.failure_flags=wp.zeros(4,dtype=int,device=device)
        self.extrema=wp.array([float('inf'),float('inf'),0.],dtype=float,device=device)
        self._cached_command=None;self._cached_selected=None
        self._check()

    def _check(self):
        wp.launch(_guard,dim=self.n,inputs=[self.solver.mpm_state,self.solver.mpm_model.inv_dx,
                  self.config['n_grid'],self.config['floor_m'],self.config['floor_tolerance_m'],
                  self.failure_flags,self.extrema],device=self.device)
        flags=self.failure_flags.numpy()
        self.calls_checked+=1
        if flags.any():
            self.failed=True
            names=['nonfinite_state','unsafe_transfer_stencil','nonpositive_deformation','floor_penetration']
            raise FloatingPointError(','.join(name for name,flag in zip(names,flags) if flag))

    def read(self):
        if self.failed: raise RuntimeError('Failed simulator cannot yield accepted state')
        return {key:getattr(self.solver.mpm_state,'particle_'+key).numpy().copy() for key in ('x','v','F_trial','C')}

    def step(self,dt,command,selected):
        if self.failed: raise RuntimeError('Failed simulator cannot continue')
        if dt!=self.config['dt_s']: raise ValueError('Step differs from declared fixed clock')
        selected=finite_array(selected,(self.n,),'selected')
        if not np.isin(selected,[0,1]).all(): raise ValueError('Selection must be binary')
        selected=selected.astype(np.int32)
        self._check()  # Also catches externally corrupted state before any transfer.
        if command is not None:
            command=finite_array(command,(self.n,3),'command').astype(np.float32)
            if np.any(command[selected==0]!=0): raise ValueError('Command outside selected particles')
            if self._cached_command is None or not np.array_equal(command,self._cached_command):
                self._cached_command=command.copy();self._wp_command=wp.array(command,dtype=wp.vec3,device=self.device)
            if self._cached_selected is None or not np.array_equal(selected,self._cached_selected):
                self._cached_selected=selected.copy();self._wp_selected=wp.array(selected,dtype=int,device=self.device)
            wp.launch(_command,dim=self.n,inputs=[self.solver.mpm_state.particle_v,self._wp_command,self._wp_selected],device=self.device)
        self.solver.time=self.steps*dt
        if self.material_policy==HENCKY_POLICY:
            step_with_consistent_material(self.solver,dt,self.device)
        else:
            self.solver.p2g2p(self.steps,dt,device=self.device)
        if self.config['precision']=='float64-master':
            wp.launch(_accumulate,dim=self.n,inputs=[self.master,self.solver.mpm_state.particle_x,
                      self.solver.mpm_state.particle_v,dt],device=self.device)
        self.steps+=1
        self.solver.time=self.steps*dt
        self._check()

    def diagnostics(self):
        values=self.extrema.numpy()
        return dict(substeps=self.steps, guard_calls=self.calls_checked, failed=self.failed,
                    minimum_det_F_trial=float(values[0]), minimum_z_m=float(values[1]), maximum_speed_m_s=float(values[2]),
                    config=self.config, static_arrays_sha256=array_fingerprint(self.static),source_sha256=self.source_sha256)

    def checkpoint(self):
        if self.failed: raise RuntimeError('Cannot checkpoint failed state')
        arrays=self.read()
        for key in ('F','Jp'):
            arrays[key]=getattr(self.solver.mpm_state,'particle_'+key).numpy().copy()
        for key in ('mu','lam','bulk','yield_stress'):
            arrays['model_'+key]=getattr(self.solver.mpm_model,key).numpy().copy()
        arrays['master']=self.master.numpy().copy()
        return dict(config=self.config.copy(), static_sha256=array_fingerprint(self.static),
                    steps=self.steps, arrays=arrays, schema='mpm-bounded-warp-checkpoint-v1',
                    source_sha256=self.source_sha256,guard_calls=self.calls_checked,extrema=self.extrema.numpy().copy())

    def restore(self,checkpoint):
        if (checkpoint['schema']!='mpm-bounded-warp-checkpoint-v1' or checkpoint['config']!=self.config
                or checkpoint['static_sha256']!=array_fingerprint(self.static)
                or checkpoint['source_sha256']!=self.source_sha256):
            raise ValueError('Checkpoint config/static mismatch')
        if self.steps or self.failed: raise ValueError('Restore into fresh backend only')
        arrays=checkpoint['arrays']
        for key,dtype in [('x',wp.vec3),('v',wp.vec3),('F_trial',wp.mat33),('C',wp.mat33),('F',wp.mat33),('Jp',float)]:
            target=getattr(self.solver.mpm_state,'particle_'+key)
            finite_array(arrays[key],target.numpy().shape,key)
            setattr(self.solver.mpm_state,'particle_'+key,wp.array(arrays[key],dtype=dtype,device=self.device))
        for key in ('mu','lam','bulk','yield_stress'):
            finite_array(arrays['model_'+key],(self.n,),key)
            setattr(self.solver.mpm_model,key,wp.array(arrays['model_'+key],dtype=float,device=self.device))
        finite_array(arrays['master'],(self.n,3),'master')
        self.master=wp.array(arrays['master'],dtype=wp.vec3d,device=self.device)
        if not isinstance(checkpoint['steps'],int) or checkpoint['steps']<0: raise ValueError('Invalid checkpoint clock')
        self.calls_checked=checkpoint['guard_calls']
        self.extrema=wp.array(finite_array(checkpoint['extrema'],(3,),'extrema'),dtype=float,device=self.device)
        self.steps=checkpoint['steps'];self.solver.time=self.steps*self.config['dt_s'];self._check()


def cube_fixture(*, material=0,E=1e5,nu=.2,gravity=(0.,0.,0.),precision='float64-master'):
    """Known-volume 5 cm solid cube: 8³ cell-centre quadrature, not a source asset."""
    axis=(np.arange(8)+.5)*(.05/8)-.025
    x=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)+[.2,.2,.2]
    static=dict(object_ids=np.zeros(len(x),np.int32),particle_volume=np.full(len(x),(.05/8)**3,np.float32),
                E=np.array([E],np.float32),nu=np.array([nu],np.float32),density=np.array([1000.],np.float32),
                mat_type=np.array([material],np.int32),yield_stress=np.array([1e4 if material==1 else 0.],np.float32))
    config=dict(n_grid=32,grid_lim_m=.5,floor_m=.05,floor_contact=False,friction=.3,floor_tolerance_m=0.,
                gravity_m_s2=list(gravity),dt_s=1e-4,precision=precision,softening=.1)
    return x.astype(np.float32),static,config


class ObservationMixin:
    def _check(self):
        wp.launch(_guard, dim=self.n, inputs=[self.solver.mpm_state, self.solver.mpm_model.inv_dx,
            self.config['n_grid'], self.config['floor_m'], self.config['floor_tolerance_m'],
            self.failure_flags, self.extrema], device=self.device)
        flags = self.failure_flags.numpy()
        self.calls_checked += 1
        depth = self.config['floor_m']-float(self.extrema.numpy()[1])
        if flags[3] and not hasattr(self, 'first_floor_alert_step'):
            self.first_floor_alert_step = self.steps
        reasons = [name for name, flag in zip(('nonfinite_state', 'unsafe_transfer_stencil',
                   'nonpositive_deformation'), flags[:3]) if flag]
        if depth > self.config['observation_emergency_depth_m']:
            reasons.append('diagnostic_emergency_floor_depth')
        if reasons:
            self.failed = True
            raise FloatingPointError(','.join(reasons))

    def diagnostics(self):
        return dict(super().diagnostics(), first_2mm_floor_alert_step=getattr(self, 'first_floor_alert_step', None),
            observer_only=True, original_floor_alert_m=self.config['floor_tolerance_m'],
            floor_alert_is_not_acceptance=True, accepted_for_training=False)


class ObservedGrid(ObservationMixin, GuardedWarpBackend):
    pass


SPEED_BUDGET = 8.
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CollectionBackend(ObservedGrid):
    """Fresh initial velocity and original wall planes; no plastic restart shortcuts."""
    def __init__(self, x, static, config, velocity, *, device='cpu'):
        velocity=finite_array(velocity, x.shape, 'fresh initial velocity').astype(np.float32)
        if np.linalg.norm(velocity,axis=-1).max() > SPEED_BUDGET: raise ValueError('Initial speed budget')
        super().__init__(x,static,config,device=device)
        if self.steps: raise AssertionError('Fresh initialization only')
        self.solver.mpm_state.particle_v=wp.array(velocity,dtype=wp.vec3,device=device)
        high=config['grid_lim_m']-.03
        for point,normal in [([0,0,high],[0,0,-1]),([.03,0,0],[1,0,0]),
            ([high,0,0],[-1,0,0]),([0,.03,0],[0,1,0]),([0,high,0],[0,-1,0])]:
            self.solver.add_surface_collider(point,normal,surface='separate',friction=0.)
        self._check()
        self.source_sha256[str(Path(__file__).resolve())]=sha(__file__)

    def _check(self):
        super()._check()
        if float(self.extrema.numpy()[2]) > SPEED_BUDGET:
            self.failed=True
            raise FloatingPointError('declared_advective_clock_budget_exceeded')


def install_original_walls_in_padded_grid(backend):
    if backend.steps or backend.config['n_grid'] not in (100,256) or backend.config['grid_lim_m']!=100/64:
        raise ValueError('Fresh, explicitly padded 100- or 256-cell grid required')
    planes=[[0,0,1.47],[.03,0,0],[1.47,0,0],[0,.03,0],[0,1.47,0]]
    if len(backend.solver.collider_params)!=6:raise ValueError('Original six planes required')
    for param,point in zip(backend.solver.collider_params[1:],planes):param.point=wp.vec3(*point)
