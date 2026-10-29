"""Exact-state diagnostic of patch size and gentler commands; never exports.

No material, timestep, grid, boundary or position corrections. The conservative
command arm deliberately reduces commanded path; report that tradeoff explicitly.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
from collection.common import digest, write_once
from .protocol import patch_mask, declared_schedule

VARIANTS = ('baseline', 'patch20', 'gentle', 'patch20_gentle')


def commands(manifest, arrays, variant):
    if variant not in VARIANTS: raise ValueError('Unknown actuation ablation')
    action = manifest['action']; transitions = len(arrays['transition_active'])
    original, active = declared_schedule(arrays['selected'], action,
        schema=manifest['action_schema'], transitions=transitions)
    np.testing.assert_array_equal(original, arrays['command_velocity'])
    np.testing.assert_array_equal(active, arrays['transition_active'])
    selected = arrays['selected'].copy()
    if variant.startswith('patch20'):
        selected = patch_mask(arrays['x'][0], arrays['object_ids'], action['object_id'], .2)
    # Every selected point has the same prescribed velocity in this protocol.
    velocity = original[:, np.flatnonzero(arrays['selected'])[0]].copy()
    if 'gentle' in variant:
        for event in action['events']:
            start, moving = event['start'], event['moving']
            # Frame-rate ramp, including a smooth approach to the active hold.
            k = np.arange(moving)
            ramp = np.minimum(1., np.minimum((k + 1) / 3., (moving - k) / 3.))
            velocity[start:start+moving] *= (min(1., .25 / event['speed']) * ramp)[:, None]
    command = velocity[:, None, :] * selected[None, :, None]
    return command.astype(np.float32), selected, active.copy()


def run(raw, output, uuid):
    import warp as wp
    from collection.lease import gpu_lease
    from .. import physics, runtime
    from ..collect import save_npz
    from ..contracts import Clock
    from ..collapse import analyze as collapse
    from ..export import POLICY
    from ..validate import read_raw
    from ..walls import FIELDS, validate_evidence
    from .fragmentation import analyze
    from .simulate import DragWorld
    from .grouped import CommandGroup
    from .study_device import DeviceGuard
    raw = Path(raw).resolve(strict=True); out = Path(output); out.mkdir(parents=True, exist_ok=False)
    m, a = read_raw(raw, POLICY); clock = Clock(**m['clock'])
    if m['n_grid'] != 256: raise ValueError('This ablation requires native-g256 input')
    cp = raw.parent/'settled-full-state.npz'
    receipt = json.loads((raw.parent/'SETTLED_CHECKPOINT.json').read_text())
    if digest(cp) != receipt['arrays_sha256']: raise ValueError('Checkpoint changed')
    with np.load(cp, allow_pickle=False) as f: physical = {k:f[k].copy() for k in f.files}
    for key in ('x', 'v', 'F_trial', 'C'): np.testing.assert_array_equal(physical[key], a[key][0])
    sources = runtime.sources(); started = time.monotonic()
    plans = [commands(m, a, variant) for variant in VARIANTS]
    write_once(out/'REQUEST.json', dict(raw=str(raw), raw_sha256=digest(raw), checkpoint_sha256=digest(cp),
        variants=VARIANTS, sources=sources, clock=asdict(clock), n_grid=256,
        held_constant=['complete settled physical state','materials','native timestep','grid','walls','event timing','directions'],
        changed=['patch20: 20% instead of 10% selected particles','gentle: cap .25m/s and 3-transition ramps; shorter command path'],
        source_action=m['action'], scaling_approved=False))
    guard = DeviceGuard(uuid)
    with gpu_lease(uuid):
        write_once(out/'DEVICE.json', guard.check(startup=True))
        wp.config.kernel_cache_dir = str(out.parent/'warp-cache')
        worlds = []
        for _ in VARIANTS:
            b = physics.CollectionBackend(a['x'][0], {k:a[k] for k in runtime.STATIC}, m['solver_config'], a['v'][0], device='cuda:0')
            physics.install_original_walls_in_padded_grid(b)
            checkpoint = b.checkpoint()
            if set(physical) != set(checkpoint['arrays']): raise ValueError('Incomplete checkpoint closure')
            b.restore(dict(checkpoint, arrays=physical, steps=0))
            for key,value in physical.items(): np.testing.assert_array_equal(value,b.checkpoint()['arrays'][key])
            worlds.append(DragWorld(b, clock))
        group = CommandGroup(worlds, block=64, fused=True, max_command_speed=.6)
        records = [{k:[] for k in ('x','v','F_trial',*FIELDS)} for _ in worlds]
        initial_guards = [w.backend.calls_checked for w in worlds]; last_check = 0.
        for frame in range(clock.transitions+1):
            if frame:
                group.set_commands([p[0][frame-1] for p in plans], [p[1]*p[2][frame-1] for p in plans])
                for _ in range(clock.substeps_per_transition):
                    if time.monotonic()-started > 1500: raise TimeoutError('Bounded actuation comparison')
                    if time.monotonic()-last_check > 10:
                        guard.check(); last_check = time.monotonic()
                    group.step()
            for w, record in zip(worlds, records):
                state = dict(w.read(), **w.wall_snapshot())
                for key in record: record[key].append(state[key].copy())
            if frame % 12 == 0:
                print(json.dumps(dict(case=raw.parent.name,frame=frame,total=clock.transitions,seconds=time.monotonic()-started)), flush=True)
        results=[]
        for variant, w, record, plan, initial_guard in zip(VARIANTS,worlds,records,plans,initial_guards):
            folder=out/variant; folder.mkdir()
            values={k:np.stack(v) for k,v in record.items()}
            values.update(object_ids=a['object_ids'],selected=plan[1],command_velocity=plan[0],transition_active=plan[2],time_s=clock.state_time_s)
            if w.backend.calls_checked-initial_guard != 2*w.steps: raise ValueError('Missing native guards')
            wall_manifest=dict(m,wall_observation=w.wall_metadata,source_sha256=w.source_sha256)
            walls=validate_evidence(wall_manifest,values)
            save_npz(folder/'TRAJECTORY.npz',values)
            frag=analyze(values['x'],a['object_ids'],dx_m=m['solver_config']['grid_lim_m']/256,selected=plan[1])
            col=collapse(values['x'],a['object_ids'],values['F_trial'])
            oid=m['action']['object_id'];body=a['object_ids']==oid
            com=values['x'][:,body].mean(1); original_patch=a['selected'].astype(bool)
            bulk_z=values['x'][:,body & ~original_patch,2].mean(1)
            path=float(np.linalg.norm(plan[0][:,np.flatnonzero(plan[1])[0]],axis=-1).sum()*clock.substep_dt_s*clock.substeps_per_transition)
            result=dict(variant=variant,fragmentation=frag,collapse=col,walls=walls,
                exact_initial_full_state=True,commanded_path_m=path,selected_particles=int(plan[1].sum()),
                object_com_max_lift_m=float(np.max(com[:,2]-com[0,2])),
                original_unselected_com_max_lift_m=float(np.max(bulk_z-bulk_z[0])),
                object_com_excursion_m=float(np.max(np.linalg.norm(com-com[0],axis=1))),
                trajectory_sha256=digest(folder/'TRAJECTORY.npz'),native_guard_calls=w.backend.calls_checked-initial_guard,
                scaling_approved=False)
            write_once(folder/'RESULT.json',result); results.append(result)
        runtime.verify_sources(sources)
        write_once(out/'COMPLETE.json',dict(case=raw.parent.name,seconds=time.monotonic()-started,results=results,scaling_approved=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True);parser.add_argument('--uuid',required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.uuid: raise ValueError('Explicit GPU UUID required')
    run(args.raw,args.output,args.uuid)
