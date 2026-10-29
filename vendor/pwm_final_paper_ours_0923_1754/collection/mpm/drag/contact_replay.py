"""Spatial refinement from an exact real drag settled checkpoint and commands.

Diagnostic NPZ output only. This does not qualify a changed grid for collection.
The same physical box/floor, material state, weights, clock and commands are
used; only spatial resolution changes. No new warmup or patch selection.
"""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import warp as wp
from collection.common import digest,write_once
from collection.lease import gpu_lease
from .. import runtime,physics
from ..clock import wave_speeds
from ..collect import save_npz
from ..contracts import Clock
from ..export import POLICY
from ..validate import read_raw
from .study_device import DeviceGuard
from .contact_study import World
from .grouped import CommandGroup

SUPPORTED_GRIDS=(100,160,192,256)


def run(raw,output,n_grid,uuid,shared_pid=None,*,no_commands=False):
    from scipy.spatial import cKDTree
    raw=Path(raw).resolve(strict=True);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    m,a=read_raw(raw,POLICY)
    cp_path=raw.parent/'settled-full-state.npz';cp_receipt=json.loads((raw.parent/'SETTLED_CHECKPOINT.json').read_text())
    if digest(cp_path)!=cp_receipt['arrays_sha256']:raise ValueError('Saved full state changed')
    static={k:a[k] for k in runtime.STATIC};config=dict(m['solver_config'],n_grid=n_grid)
    clock=Clock(**m['clock']);sources=runtime.sources();started=time.monotonic()
    if config['grid_lim_m']!=100/64 or config['dt_s']!=5e-5 or n_grid not in SUPPORTED_GRIDS:
        raise ValueError('Unsupported spatial replay')
    courant=clock.substep_dt_s*(float(wave_speeds(static['E'],static['nu'],static['density']).max())+8)/(config['grid_lim_m']/n_grid)
    if courant>.4:raise ValueError('Refinement exceeds same-clock stability budget')
    write_once(output/'REQUEST.json',dict(raw_path=str(raw),raw_sha256=digest(raw),
        checkpoint_sha256=digest(cp_path),n_grid=n_grid,config=config,clock=asdict_clock(clock),
        courant=courant,sources=sources,scaling_approved=False,
        held_constant=['positions','master positions','velocity','F/F_trial','C','plastic history','material coefficients','particle weights','selected patch','dt','physical boundary planes']+([] if no_commands else ['commands']),
        changed=['n_grid / dx']+(['commands disabled for residual-motion control'] if no_commands else []),
        commands_enabled=not no_commands,caveat='Diagnostic, no production native-wall certificate'))
    guard=DeviceGuard(uuid,shared_pid)
    with gpu_lease(uuid):
        write_once(output/'DEVICE.json',guard.check(startup=True));wp.config.kernel_cache_dir=str(output.parent/'warp-cache')
        b=physics.CollectionBackend(a['x'][0],static,config,a['v'][0],device='cuda:0')
        checkpoint=b.checkpoint()
        with np.load(cp_path) as f:physical={k:f[k].copy() for k in f.files}
        if set(physical)!=set(checkpoint['arrays']):raise ValueError('Physical state closure differs')
        b.restore(dict(checkpoint,arrays=physical,steps=0))
        for key,value in physical.items():np.testing.assert_array_equal(value,b.checkpoint()['arrays'][key])
        if len(b.solver.collider_params)!=6:raise ValueError('Unexpected boundary topology')
        for param,point in zip(b.solver.collider_params[1:],[[0,0,1.47],[.03,0,0],[1.47,0,0],[0,.03,0],[0,1.47,0]]):
            param.point=wp.vec3(*point)
        world=World(b,clock);group=CommandGroup([world],max_command_speed=.6,fused=False)
        records={k:[] for k in ('x','v','F_trial')};last=0.
        for frame in range(clock.transitions+1):
            if frame:
                command=a['command_velocity'][frame-1];selection=a['selected']*a['transition_active'][frame-1]
                if no_commands:command=np.zeros_like(command);selection=np.zeros_like(selection)
                group.set_commands([command],[selection])
                for _ in range(clock.substeps_per_transition):
                    if time.monotonic()-last>5:
                        guard.check();last=time.monotonic()
                        if time.monotonic()-started>2400:raise TimeoutError('Spatial replay time budget')
                    group.step()
            state=b.read()
            for key in records:records[key].append(state[key])
            if frame%16==0:print(json.dumps(dict(case=raw.parent.name,n_grid=n_grid,frame=frame,total=clock.transitions)),flush=True)
        arrays={k:np.stack(v) for k,v in records.items()};arrays.update(time_s=clock.state_time_s,object_ids=a['object_ids'],selected=a['selected'])
        save_npz(output/'TRAJECTORY.npz',arrays)
        pairs=[]
        for pair in m['geometry_provenance']['drag_interaction']['requested_pairs']:
            left=arrays['x'][:,a['object_ids']==pair['a']];right=arrays['x'][:,a['object_ids']==pair['b']]
            gap=[float(cKDTree(x).query(y)[0].min()) for x,y in zip(left,right)]
            com=right.astype(float).mean(1);motion=np.linalg.norm(com-com[0],axis=1)
            onset=np.flatnonzero(motion>.001)
            pairs.append(dict(**pair,gap_m=gap,target_com_m=com.tolist(),
                target_excursion_m=float(motion.max()),gap_at_1mm_target_motion_m=gap[onset[0]] if len(onset) else None))
        delta=arrays['x'].astype(float)-a['x'].astype(float)
        result=dict(case=raw.parent.name,n_grid=n_grid,dx_m=config['grid_lim_m']/n_grid,
            seconds=time.monotonic()-started,minimum_z_m=float(arrays['x'][...,2].min()),
            position_rms_vs_original_m=float(np.sqrt(np.mean(np.sum(delta**2,axis=-1)))),
            pairs=pairs,initial_state_exact=True,commands_exact=not no_commands,commands_enabled=not no_commands,diagnostics=b.diagnostics(),
            scaling_approved=False,only_shared_grid_refinement=True,
            shared_gpu=shared_pid is not None,throughput_comparison_valid=shared_pid is None)
        runtime.verify_sources(sources);write_once(output/'RESULT.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','diagnostics')}),flush=True)


def asdict_clock(c):return dict(substep_dt_s=c.substep_dt_s,substeps_per_transition=c.substeps_per_transition,transitions=c.transitions)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    p.add_argument('--n-grid',type=int,choices=SUPPORTED_GRIDS,required=True);p.add_argument('--uuid',required=True)
    p.add_argument('--share-with-pid',type=int)
    p.add_argument('--no-commands',action='store_true',help='Residual-motion control, not a commanded replay')
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=args.uuid:raise ValueError('Explicit allocated UUID required')
    run(args.raw,args.output,args.n_grid,args.uuid,args.share_with_pid,no_commands=args.no_commands)
