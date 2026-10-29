"""Diagnostic dt/dt2/dt4 comparison from IDENTICAL full settled physical states.

Not a production collector. Changing the clock for this experiment is explicit;
all plastic history, deformation, APIC state, momentum and material points are
copied bit-for-bit. Patch membership is fixed before the three branches split.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
from collection.common import digest,write_once
from collection.lease import gpu_lease
from .cli import wait_idle

POSITION_RMS_M=.001
MAXIMUM_POINT_M=.005


def run(root,labels,output,uuid,*,divisors=(1,2,4),settled_source=None,passive_control=False,
        reference_graph=False,serial_reference=False):
    divisors=tuple(divisors)
    if (not (1 if serial_reference else 2)<=len(divisors)<=6
            or any(type(d) is not int or d not in (1,2,4,8,16,32) for d in divisors)
            or any(b!=2*a for a,b in zip(divisors,divisors[1:]))):
        raise ValueError('Use two to six consecutive power-of-two timestep divisors, at most 32')
    import warp as wp
    from .. import runtime,physics
    from ..contracts import Clock,SettlingMonitor
    from ..collect import save_npz,gpu_status
    from ..collapse import analyze,CollapsePolicy
    from ..walls import FIELDS,validate_evidence
    from .protocol import SETTLING,patch_mask,declared_schedule,SEQUENCE_SCHEMAS
    from .simulate import DragWorld
    from .grouped import CommandGroup
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=json.loads((root/'PLAN.json').read_text());entries={c['label']:c for c in plan['cases']}
    if not 1<=len(labels)<=4 or len(labels)!=len(set(labels)):raise ValueError('Bounded unique control labels required')
    sources=runtime.sources();started=time.monotonic();last_check=0.
    max_speed=.6 if plan['schema'] in SEQUENCE_SCHEMAS else .3
    parent=None
    if settled_source is not None:
        settled_source=Path(settled_source).resolve(strict=True)
        parent_request=json.loads((settled_source/'REQUEST.json').read_text())
        parent=json.loads((settled_source/'RESULT.json').read_text())
        if Path(parent_request['root']).resolve()!=root.resolve() or not set(labels)<=set(parent_request['labels']):
            raise ValueError('Saved settled states belong to different prepared inputs')
        if parent_request.get('plan_sha256',digest(root/'PLAN.json'))!=digest(root/'PLAN.json'):
            raise ValueError('Prepared inputs changed since saved settling')
        for label in labels:
            name=label+'-settled-full-state.npz'
            if parent.get('files',{}).get(name)!=digest(settled_source/name):raise ValueError('Unverified full settled checkpoint')
    write_once(output/'REQUEST.json',dict(root=str(root),labels=labels,divisors=divisors,source_sha256=sources,
        plan_sha256=digest(root/'PLAN.json'),passive_control=passive_control,reference_graph=reference_graph,
        serial_reference=serial_reference,
        settled_source=None if settled_source is None else str(settled_source),
        settled_source_result_sha256=None if parent is None else digest(settled_source/'RESULT.json'),
        thresholds=dict(position_vector_rms_m=POSITION_RMS_M,selected_vector_rms_m=POSITION_RMS_M,
                        maximum_particle_difference_m=MAXIMUM_POINT_M),scaling_approved=False))
    def tick():
        nonlocal last_check
        if time.monotonic()-started>3600:raise TimeoutError('Convergence control deadline')
        if time.monotonic()-last_check>5:
            gpu_status(uuid,allow_own=True,memory_limit_mib=80000);last_check=time.monotonic()
    with gpu_lease(uuid):
        wait_idle(uuid);wp.config.kernel_cache_dir=str(output.parent/'convergence-cache')
        manifests=[];initials=[];worlds=[]
        for label in labels:
            folder=root/'cases'/label;entry=entries[label]
            if digest(folder/'initial.npz')!=entry['initial_sha256'] or digest(folder/'MANIFEST.json')!=entry['manifest_sha256']:
                raise ValueError('Prepared control changed')
            m=json.loads((folder/'MANIFEST.json').read_text())
            with np.load(folder/'initial.npz') as f:a={k:f[k].copy() for k in f.files}
            b=physics.CollectionBackend(a['x'],{k:a[k] for k in runtime.STATIC},m['solver_config'],a['v'],device='cuda:0')
            physics.install_original_walls_in_padded_grid(b)
            c=Clock(**m['clock']);worlds.append(DragWorld(b,Clock(c.substep_dt_s,c.substeps_per_transition,96)))
            manifests.append(m);initials.append(a)
        if len({w.clock for w in worlds})!=1:raise ValueError('Control worlds need identical base clocks')
        if len({m['clock']['transitions'] for m in manifests})!=1:raise ValueError('Control recording lengths must match')
        transitions=manifests[0]['clock']['transitions']
        if parent is None:
            group=CommandGroup(worlds,block=256 if reference_graph else 64,fused=not reference_graph,max_command_speed=max_speed)
            monitors=[SettlingMonitor(a['object_ids'],SETTLING) for a in initials]
            previous=[w.read()['x'] for w in worlds]
            for frame in range(96):
                for _ in range(worlds[0].clock.substeps_per_transition):tick();group.step()
                for i,(w,monitor) in enumerate(zip(worlds,monitors)):
                    value=w.read();monitor.observe(previous[i],value['x'],value['v']);previous[i]=value['x']
            write_once(output/'SETTLING.json',dict(rows=[dict(passed=m.consecutive>=8,history=m.history) for m in monitors]))
            if not all(m.consecutive>=8 for m in monitors):
                write_once(output/'RESULT.json',dict(passed=False,reason='settling',scaling_approved=False));return
        else:
            for label,w in zip(labels,worlds):
                cp=w.backend.checkpoint()
                with np.load(settled_source/(label+'-settled-full-state.npz')) as f:
                    arrays={k:f[k].copy() for k in f.files}
                if set(arrays)!=set(cp['arrays']):raise ValueError('Incomplete physical checkpoint')
                w.backend.restore(dict(cp,arrays=arrays,steps=0))
                for key,value in arrays.items():np.testing.assert_array_equal(value,w.backend.checkpoint()['arrays'][key])
            write_once(output/'SETTLING.json',dict(restored_all_physical_arrays_exactly=True,
                parent=str(settled_source),parent_result_sha256=digest(settled_source/'RESULT.json')))
        checkpoints=[w.backend.checkpoint() for w in worlds];masks=[];commands=[];gates=[]
        for label,m,a,cp in zip(labels,manifests,initials,checkpoints):
            action=m['action'];mask=patch_mask(cp['arrays']['x'],a['object_ids'],action['object_id'],action['fraction'])
            cmd,gate=declared_schedule(mask,action,schema=m['action_schema'],transitions=transitions)
            if passive_control:cmd.fill(0);gate.fill(0)
            masks.append(mask);commands.append(cmd);gates.append(gate)
            save_npz(output/(label+'-settled-full-state.npz'),cp['arrays'])
        all_values={};checks=[]
        executions=[(d,False) for d in divisors]+([(divisors[0],True)] if serial_reference else [])
        for divisor,serial in executions:
            branches=[]
            for m,a,cp in zip(manifests,initials,checkpoints):
                c=Clock(**m['clock']);clock=Clock(c.substep_dt_s/divisor,c.substeps_per_transition*divisor,transitions)
                config=dict(cp['config'],dt_s=clock.substep_dt_s)
                b=physics.CollectionBackend(a['x'],{k:a[k] for k in runtime.STATIC},config,a['v'],device='cuda:0')
                # A new observational epoch and dt only, not a state reset.
                b.restore(dict(cp,config=config,steps=0))
                for key,value in cp['arrays'].items():np.testing.assert_array_equal(value,b.checkpoint()['arrays'][key])
                physics.install_original_walls_in_padded_grid(b);branches.append(DragWorld(b,clock))
            stepping=None if serial else CommandGroup(branches,block=256 if reference_graph else 64,fused=not reference_graph,max_command_speed=max_speed)
            records=[{k:[] for k in ('x','v','F_trial','C',*FIELDS)} for _ in branches]
            guards=[w.backend.calls_checked for w in branches]
            for frame in range(transitions+1):
                if frame:
                    cmd=[c[frame-1] for c in commands];sel=[mask*gate[frame-1] for mask,gate in zip(masks,gates)]
                    if not serial:stepping.set_commands(cmd,sel)
                    for _ in range(branches[0].clock.substeps_per_transition):
                        tick()
                        if serial:
                            for w,c,s in zip(branches,cmd,sel):w.step(w.clock.substep_dt_s,c,s)
                        else:stepping.step()
                for w,record in zip(branches,records):
                    value=dict(w.read(),**w.wall_snapshot())
                    for key in record:record[key].append(value[key].copy())
            values=[]
            for i,(label,w,record,a) in enumerate(zip(labels,branches,records,initials)):
                if w.backend.calls_checked-guards[i]!=2*w.steps:raise ValueError('Incomplete native guards')
                arrays={k:np.stack(v) for k,v in record.items()};values.append(arrays)
                evidence=validate_evidence(dict(task='drag',action_schema=plan['schema'],clock=asdict(w.clock),
                    wall_observation=w.wall_metadata,source_sha256={str(runtime.ROOT/p):s for p,s in sources.items()}),arrays)
                quality=analyze(arrays['x'],a['object_ids'],arrays['F_trial'],CollapsePolicy())
                checks.append(dict(label=label,divisor=divisor,serial=serial,wall_free=evidence['wall_free'],
                    collapse_pass=quality['eligible'],floor_pass=float(arrays['x'][...,2].min())>=.088))
                suffix='-serial' if serial else ''
                save_npz(output/f'{label}-dtdiv{divisor}{suffix}.npz',dict(arrays,selected=masks[i],
                    command_velocity=commands[i],transition_active=gates[i],state_time_s=w.clock.state_time_s))
            all_values[('serial',divisor) if serial else divisor]=values
            print(json.dumps(dict(completed_dt_divisor=divisor,serial=serial,labels=labels)),flush=True)
        comparisons=[]
        for lo,hi in zip(divisors,divisors[1:]):
            for i,label in enumerate(labels):
                delta=all_values[lo][i]['x'].astype(float)-all_values[hi][i]['x'].astype(float)
                rms=float(np.sqrt(np.mean(np.sum(delta**2,axis=-1))))
                selected_rms=float(np.sqrt(np.mean(np.sum(delta[:,masks[i]!=0]**2,axis=-1))))
                maximum=float(np.linalg.norm(delta,axis=-1).max())
                comparisons.append(dict(label=label,divisors=[lo,hi],position_rms_m=rms,selected_rms_m=selected_rms,
                    maximum_particle_difference_m=maximum,passed=max(rms,selected_rms)<=POSITION_RMS_M and maximum<=MAXIMUM_POINT_M))
        execution_comparisons=[]
        if serial_reference:
            from .review import compare_events
            for i,label in enumerate(labels):
                a=all_values[divisors[0]][i];b=all_values[('serial',divisors[0])][i]
                np.testing.assert_array_equal(a['x'][0],b['x'][0])
                dx=a['x'].astype(float)-b['x'].astype(float)
                rms=float(np.sqrt(np.mean(np.sum(dx**2,axis=-1))))
                events=[compare_events(a[k],b[k]) for k in ('wall_first_grid_step','wall_first_particle_step')]
                execution_comparisons.append(dict(label=label,position_rms_m=rms,events=events,
                    passed=rms<=1e-4 and all(e['contact_presence_equal'] and e['maximum_step_difference']<=2 for e in events)))
        runtime.verify_sources(sources)
        report=dict(passed=all(r['passed'] for r in comparisons+execution_comparisons) and all(
            r['wall_free'] and r['collapse_pass'] and r['floor_pass'] for r in checks),comparisons=comparisons,
            checks=checks,seconds=time.monotonic()-started,scaling_approved=False,
            diagnostic='timestep-control' if len(divisors)>1 else 'same-state-execution-parity',
            execution_comparisons=execution_comparisons,
            files={p.name:digest(p) for p in output.glob('*.npz')})
        write_once(output/'RESULT.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--output',required=True);p.add_argument('--uuid',required=True)
    p.add_argument('--divisors',nargs='+',type=int,default=[1,2,4])
    p.add_argument('--settled-source');p.add_argument('--passive-control',action='store_true')
    p.add_argument('--reference-graph',action='store_true');p.add_argument('--serial-reference',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.uuid:raise ValueError('Select exact allocated GPU UUID')
    run(a.root,a.labels,a.output,a.uuid,divisors=a.divisors,settled_source=a.settled_source,
        passive_control=a.passive_control,reference_graph=a.reference_graph,serial_reference=a.serial_reference)
