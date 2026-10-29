"""Guard-preserving grouped stepping, extracted from the measured prototypes.

No timestep/material/particle changes. Optional 64-thread launches and ordered
wall fusion must pass frozen full-trajectory qualification before collection.
"""
import ast
import inspect
import json
import os
from pathlib import Path
import textwrap
import time
import types
import numpy as np
import warp as wp
from collection.common import digest, write_once
from . import runtime, physics, observer, walls
from .state import MPMStateStruct, MPMModelStruct, Dirichlet_collider
from .walls import FIELDS, validate_evidence
from .journal import FrameJournal, verify_journal
from .recording import atomic_raw
from .contracts import identity

def no_timer_sync(solver, block=256):
    """Only the profiling timers change; every physics/guard kernel is retained."""
    import inspect
    import textwrap
    from collection.mpm import solver as module
    tree = ast.parse(textwrap.dedent(inspect.getsource(type(solver).p2g2p)))
    changed = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'ScopedTimer':
            for kw in node.keywords:
                if kw.arg == 'synchronize':
                    assert isinstance(kw.value, ast.Constant) and kw.value.value is True
                    kw.value = ast.Constant(value=False); changed += 1
    if changed != 5: raise ValueError('Unexpected profiling timer topology')
    if block not in (64,256): raise ValueError('Unsupported particle launch size')
    if block == 64:
        targets = {'compute_stress_from_F_trial','p2g_apic_with_stress','g2p'}
        tuned = 0
        for node in ast.walk(tree):
            kernel = (node.args[0] if isinstance(node, ast.Call) and node.args else
                      next((k.value for k in node.keywords if k.arg=='kernel'), None)
                      if isinstance(node, ast.Call) else None)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'launch'
                    and isinstance(kernel, ast.Name) and kernel.id in targets):
                if any(k.arg == 'block_dim' for k in node.keywords): raise ValueError('Unexpected launch block override')
                node.keywords.append(ast.keyword(arg='block_dim', value=ast.Constant(value=64))); tuned += 1
        if tuned != 3: raise ValueError('Unexpected particle launch topology')
    ast.fix_missing_locations(tree)
    namespace = dict(vars(module))
    exec(compile(tree, str(Path(__file__).resolve()) + ':timer-only-transform', 'exec'), namespace)
    solver.p2g2p = types.MethodType(namespace['p2g2p'], solver)


@wp.struct
class Bundle:
    colliders:wp.array(dtype=Dirichlet_collider)
    step:wp.array(dtype=int)
    first_grid:wp.array(dtype=int)
    covered:wp.array(dtype=int)


@wp.kernel
def ordered_walls(time:float,dt:float,state:MPMStateStruct,model:MPMModelStruct,bundle:Bundle):
    i,j,k=wp.tid()
    for index in range(6):
        param=bundle.colliders[index]
        # Same observation point: after previous colliders, before this collider.
        if index>0:
            wall=index-1
            if i==0 and j==0 and k==0:bundle.covered[wall]+=1
            offset=wp.vec3(float(i)*model.dx,float(j)*model.dx,float(k)*model.dx)-param.point
            if state.grid_m[i,j,k]>0.0 and wp.dot(offset,param.normal)<0.0:
                if wp.dot(state.grid_v_out[i,j,k],param.normal)<0.0:
                    wp.atomic_min(bundle.first_grid,wall,bundle.step[0])
        if time>=param.start_time and time<param.end_time:
            offset=wp.vec3(float(i)*model.dx-param.point[0],float(j)*model.dx-param.point[1],float(k)*model.dx-param.point[2])
            n=wp.vec3(param.normal[0],param.normal[1],param.normal[2])
            if wp.dot(offset,n)<0.0:
                v=state.grid_v_out[i,j,k]
                normal_component=wp.dot(v,n)
                v=v-wp.min(normal_component,0.0)*n
                if normal_component<0.0 and wp.length(v)>1e-20:
                    v=wp.max(0.0,wp.length(v)+normal_component*param.friction)*wp.normalize(v)
                state.grid_v_out[i,j,k]=v


def install_fused(w):
    w._unchanged();s=w.solver
    if len(s.grid_postprocess)!=11 or len(w.original)!=6 or any(s.modify_bc):raise ValueError('Unreviewed collider topology')
    if any(p.surface_type!=2 or p.start_time>0 or p.end_time<=w.clock.state_time_s[-1] for p in w.original):
        raise ValueError('Only static separate colliders supported')
    first=s.grid_postprocess[0]
    for index,kernel in enumerate(s.grid_postprocess):
        if index%2:
            if kernel is not observer.observe_grid:raise ValueError('Unknown observer')
        elif (kernel.func.__code__ is not first.func.__code__ or kernel.func.__closure__ is not None
              or kernel.key!=first.key or kernel.module is not first.module):
            raise ValueError('Different collider function or captured closure')
    bundle=Bundle();bundle.colliders=wp.array(list(w.original),dtype=Dirichlet_collider,device=w.device)
    bundle.step=w.counter;bundle.first_grid=w.first_grid;bundle.covered=w.covered
    s.grid_postprocess=[ordered_walls];s.collider_params=[bundle];s.modify_bc=[None]
    w.frozen_topology=(tuple(s.grid_postprocess),tuple(s.collider_params),tuple(s.modify_bc))
    w._fused_bundle=bundle
    w.source_sha256[str(Path(__file__).resolve())]=digest(__file__)
    w.wall_metadata=walls.metadata(digest(__file__))
    w._unchanged();return w


@wp.kernel
def pack(flags:wp.array(dtype=int),extrema:wp.array(dtype=float),out:wp.array2d(dtype=float),world:int):
    i=wp.tid()
    if i<4:out[world,i]=float(flags[i])
    else:out[world,i]=extrema[i-4]


def check_values(backend,value):
    flags=value[:4];backend.calls_checked+=1
    if flags[3] and not hasattr(backend,'first_floor_alert_step'):backend.first_floor_alert_step=backend.steps
    reasons=[name for name,flag in zip(('nonfinite_state','unsafe_transfer_stencil','nonpositive_deformation'),flags[:3]) if flag]
    if backend.config['floor_m']-float(value[5])>backend.config['observation_emergency_depth_m']:
        reasons.append('diagnostic_emergency_floor_depth')
    if float(value[6])>physics.SPEED_BUDGET:reasons.append('declared_advective_clock_budget_exceeded')
    if reasons:
        backend.failed=True
        raise FloatingPointError(','.join(reasons))


class Group:
    def __init__(self,worlds, *, block=256, fused=False):
        self.worlds=worlds
        if not worlds or len(worlds)>32:raise ValueError('Empty batch or more than 32 worlds')
        self.clock=worlds[0].clock;self.device='cuda:0'
        if any(w.clock!=self.clock or w.steps or w.device!=self.device for w in worlds):
            raise ValueError('Require fresh worlds on one device and identical physical clocks')
        pointers=[]
        for w in worlds:
            s=w.solver
            pointers.extend([s.mpm_state.particle_x.ptr,s.mpm_state.grid_m.ptr,w.backend.master.ptr])
            if s.pre_p2g_operations or s.particle_velocity_modifiers or any(m is not None for m in s.modify_bc):
                raise ValueError('Require static passive configuration')
            if w.backend.material_policy!='archived-fcr-log-return' or w.config['precision']!='float64-master':
                raise ValueError('Unsupported material/precision policy')
            for p in s.collider_params:
                if hasattr(p,'start_time') and not(p.start_time<=0 and p.end_time>=self.clock.state_time_s[-1]):
                    raise ValueError('Time-dependent collider')
            if fused: install_fused(w)
            no_timer_sync(s, block)
        if len(set(pointers))!=len(pointers):raise ValueError('World buffers alias')
        self.status=wp.empty((len(worlds),7),dtype=float,device=self.device)
        # Pre-transfer checks happen before any world's physics is advanced.
        with wp.ScopedCapture(device=self.device,force_module_load=True) as pre:
            for i,w in enumerate(worlds):
                w._particles()
                wp.launch(observer.next_step,dim=1,inputs=[w.counter],device=self.device)
                self.guard(w,i)
        self.pre=pre.graph
        with wp.ScopedCapture(device=self.device,force_module_load=True) as post:
            for i,w in enumerate(worlds):
                b=w.backend;s=w.solver
                s.p2g2p(0,self.clock.substep_dt_s,device=self.device)
                wp.launch(physics._accumulate,dim=b.n,inputs=[b.master,s.mpm_state.particle_x,
                    s.mpm_state.particle_v,self.clock.substep_dt_s],device=self.device)
                self.guard(w,i)
                w._particles()
        self.post=post.graph
        for w in worlds:w.solver.time=0.

    def guard(self,w,i):
        b=w.backend
        wp.launch(physics._guard,dim=b.n,inputs=[w.solver.mpm_state,w.solver.mpm_model.inv_dx,
            b.config['n_grid'],b.config['floor_m'],b.config['floor_tolerance_m'],b.failure_flags,b.extrema],device=self.device)
        wp.launch(pack,dim=7,inputs=[b.failure_flags,b.extrema,self.status,i],device=self.device)

    def check(self):
        values=self.status.numpy()
        failure=None
        for w,v in zip(self.worlds,values):
            try:check_values(w.backend,v)
            except FloatingPointError as e:failure=e
        if failure:raise failure

    def step(self):
        for w in self.worlds:
            if w.failed or w.steps>=self.clock.transitions*self.clock.substeps_per_transition:
                raise ValueError('Failed world or physical clock exhausted')
            w._unchanged()
        wp.capture_launch(self.pre);self.check()
        wp.capture_launch(self.post)
        for w in self.worlds:
            w.backend.steps+=1;w.solver.time=w.steps*self.clock.substep_dt_s
        self.check()


def load(root, case):
    folder=Path(root)/'cases'/case['label']
    if digest(folder/'initial.npz')!=case['initial_sha256'] or digest(folder/'MANIFEST.json')!=case['manifest_sha256']:
        raise ValueError('Prepared input changed')
    manifest=json.loads((folder/'MANIFEST.json').read_text())
    with np.load(folder/'initial.npz',allow_pickle=False) as f:
        initial={k:f[k].copy() for k in f.files}
    world=runtime.backend(initial,manifest,'cuda:0')
    world.source_sha256[str(Path(__file__).resolve())]=digest(__file__)
    return manifest,initial,world


def simulate(root, cases, outputs, *, block=256, fused=False, serial=False, timeout=1200, uuid=None):
    """Exact prepared inputs; durable per-case outputs compatible with review."""
    from .export import validate_wall_free
    if len(cases)!=len(outputs) or not cases or len(cases)>32 or (serial and len(cases)!=1):
        raise ValueError('Explicit nonempty bounded case/output mapping required')
    started=time.monotonic()
    loaded=[load(root,c) for c in cases]; worlds=[r[2] for r in loaded]
    group=None if serial else Group(worlds,block=block,fused=fused)
    initial_guards=[w.backend.calls_checked for w in worlds]
    records=[{k:[] for k in ('x','v','F_trial','C',*FIELDS)} for _ in cases]
    journals=[]
    for c,out,(m,a,w) in zip(cases,outputs,loaded):
        out=Path(out);out.mkdir(parents=True,exist_ok=False)
        for key in ('x','v'):np.testing.assert_array_equal(w.read()[key],a[key])
        m['source_sha256'].update(w.source_sha256)
        if not serial:
            m['group_execution']=dict(version='grouped-native-guards-v1',worlds=len(cases),block=block,
                fused_walls=fused,source_sha256=digest(__file__),guards_per_substep=2)
        journals.append(FrameJournal(out/'frames',initial_sha256=c['initial_sha256'],
                                     manifest_sha256=c['manifest_sha256'],clock=w.clock))
    clock=worlds[0].clock; last_check=0.; zero=np.zeros(worlds[0].backend.n,np.int32)
    stepping=time.monotonic()
    for frame in range(clock.transitions+1):
        if frame:
            for _ in range(clock.substeps_per_transition):
                now=time.monotonic()
                if now-started>timeout:raise TimeoutError('Bounded grouped simulation deadline')
                if uuid and now-last_check>5:
                    from .collect import gpu_status
                    gpu_status(uuid,allow_own=True,memory_limit_mib=80000);last_check=now
                if serial:worlds[0].step(clock.substep_dt_s,None,zero)
                else:group.step()
        for w,record,journal in zip(worlds,records,journals):
            value=dict(w.read(),**w.wall_snapshot());journal.append(w.steps,value)
            for k in record:record[k].append(value[k].copy())
    simulation_seconds=time.monotonic()-stepping; rows=[]
    for c,out,(m,a,w),record,guards in zip(cases,outputs,loaded,records,initial_guards):
        out=Path(out)
        if w.backend.calls_checked!=guards+2*w.steps:raise ValueError('Native guard coverage changed')
        arrays={k:np.stack(v).astype(np.int64 if k in FIELDS else np.float32) for k,v in record.items()}
        arrays.update({k:a[k] for k in runtime.STATIC})
        arrays.update(state_time_s=clock.state_time_s,selected=np.zeros(w.backend.n,np.int32),
            command_velocity=np.zeros((w.backend.n,3),np.float32),transition_active=np.zeros(clock.transitions,np.int32),
            active_substeps=np.zeros(clock.transitions,np.int32))
        m.update(seed_uint64=identity(m['case'])[1],command_semantics='passive; no velocity overrides',
                 wall_observation=w.wall_metadata)
        validate_evidence(m,arrays)
        for path,sha in m['source_sha256'].items():
            if digest(path)!=sha:raise ValueError('Simulation source changed')
        raw=out/'raw.h5';atomic_raw(raw,m,arrays)
        journal=verify_journal(out/'frames',initial_sha256=c['initial_sha256'],
            manifest_sha256=c['manifest_sha256'],complete=True,raw_path=raw)
        quality=validate_wall_free(raw)
        row=dict(label=c['label'],kind=c['kind'],status='complete',initial_sha256=c['initial_sha256'],
            raw_path=str(raw.resolve()),raw_sha256=digest(raw),journal=journal,quality=quality,
            native_substeps=w.steps,guard_calls=w.backend.calls_checked,
            group_simulation_seconds=simulation_seconds,accepted_for_training=False)
        write_once(out/'RESULT.json',row);rows.append(row)
    return dict(worlds=len(cases),block=block,fused=fused,serial=serial,
                seconds=time.monotonic()-started,simulation_seconds=simulation_seconds,rows=rows)


def grid_test(root, case):
    """Independent same-grid oracle including empty nodes, corners and friction."""
    _,_,w=load(root,case);s=w.solver;shape=s.mpm_state.grid_m.shape
    rng=np.random.default_rng(42);mass=(rng.random(shape)<.01).astype(np.float32)
    velocity=rng.uniform(-2,2,(*shape,3)).astype(np.float32);velocity[::7,::7,::7]=0
    wp.copy(s.mpm_state.grid_m,wp.array(mass,dtype=float,device=w.device))
    wp.copy(s.mpm_state.grid_v_out,wp.array(velocity,dtype=wp.vec3,device=w.device));w.counter.fill_(17)
    for kernel,param in zip(s.grid_postprocess,s.collider_params):
        wp.launch(kernel,dim=shape,inputs=[0.,w.clock.substep_dt_s,s.mpm_state,s.mpm_model,param],device=w.device)
    expected=s.mpm_state.grid_v_out.numpy();first=w.first_grid.numpy();covered=w.covered.numpy()
    wp.copy(s.mpm_state.grid_v_out,wp.array(velocity,dtype=wp.vec3,device=w.device))
    w.first_grid.fill_(observer.NEVER);w.covered.zero_();install_fused(w)
    wp.launch(ordered_walls,dim=shape,inputs=[0.,w.clock.substep_dt_s,s.mpm_state,s.mpm_model,w._fused_bundle],device=w.device)
    np.testing.assert_array_equal(s.mpm_state.grid_v_out.numpy(),expected)
    np.testing.assert_array_equal(w.first_grid.numpy(),first);np.testing.assert_array_equal(w.covered.numpy(),covered)
    w.original[0].friction+=.1
    try:w._unchanged()
    except ValueError:pass
    else:raise AssertionError('Collider mutation escaped checking')
    return dict(grid_cells=int(np.prod(shape)),velocity_and_events_bit_exact=True,mutation_rejected=True)


def corruption_test(root, cases, *, block=256, fused=False):
    worlds=[load(root,c)[2] for c in cases[:2]]
    g=Group(worlds,block=block,fused=fused); original=worlds[1].read()
    bad=worlds[0].read()['x'];bad[0,0]=np.nan
    wp.copy(worlds[0].solver.mpm_state.particle_x,wp.array(bad,dtype=wp.vec3,device='cuda:0'))
    try:g.step()
    except FloatingPointError as e:
        if 'nonfinite_state' not in str(e):raise
    else:raise AssertionError('Injected NaN was not stopped')
    for k,v in original.items():np.testing.assert_array_equal(worlds[1].read()[k],v)
    assert all(w.steps==0 for w in worlds)
    return dict(nonfinite_stopped=True,healthy_world_unchanged=True,physics_steps=0)
