"""Split-aware prepared drag inputs. No simulation or production approval."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from collection.common import canonical, digest, write_once
from .. import runtime
from ..collect import save_npz
from ..recipes import load_bank, scene
from ..clock import choose_clock
from ..contracts import identity
from ..collapse import CollapsePolicy
from ..protocol import cohort
from .protocol import CURRENT_SEQUENCE_VERSION as VERSION, FAMILIES, SETTLING, sequence_action
from .interactions import POLICY as INTERACTION_POLICY, MIXTURE, mode_for, place
from .fragmentation import FragmentationPolicy


def prepare(root, assets, *, name, count=12, split='train', evaluation='train', start=0,
            material_layout='homogeneous',object_counts=None,event_counts=(1,2,3),speed_range=(.05,.25),n_grid=100):
    if type(n_grid) is not int or n_grid not in (100,256):raise ValueError('Explicit supported drag grid: 100 or 256')
    if not name or type(count) is not int or not 1<=count<=108 or type(start) is not int or start<0:
        raise ValueError('Explicit run namespace and 1..108 bounded attempts required')
    if material_layout not in ('homogeneous','mixed'):raise ValueError('Unknown material layout')
    if (not event_counts or any(type(n) is not int or n not in (1,2,3) for n in event_counts)
            or tuple(sorted(set(event_counts)))!=tuple(event_counts)):
        raise ValueError('Event counts must be an ordered subset of 1,2,3')
    if len(speed_range)!=2 or not np.isfinite(speed_range).all() or not .05<=speed_range[0]<speed_range[1]<=.25:
        raise ValueError('Continuous gentle speed range must lie within .05.. .25 m/s')
    role=cohort(split,evaluation);assets=Path(assets).resolve(strict=True)
    allowed=(5,6,7) if evaluation.startswith('count_') else (1,2,3,4)
    counts=tuple(allowed if object_counts is None else object_counts)
    if not counts or any(type(n) is not int or n not in allowed for n in counts) or tuple(sorted(set(counts)))!=counts:
        raise ValueError('Object counts must be an ordered nonempty subset of the declared split/count regime')
    metadata,objects=load_bank(assets/role,assets/'pool',role=role,split_manifest=assets/'SPLITS.json')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    sources=runtime.sources();cases=[];used=set()
    # A namespace-specific cyclic permutation covers the entire bank without
    # relying on a coupon-collector draw. Rejections remain visible afterwards.
    order=sorted(range(len(objects)),key=lambda i:hashlib.sha256(
        f'{VERSION}:{split}:{evaluation}:{name}:{metadata["objects"][i]["asset_uid"]}'.encode()).hexdigest())
    for index in range(start,start+count):
        n=counts[index%len(counts)];material=(index//len(counts))%2
        phase=(index//(2*len(counts)))%3
        event_count=event_counts[(index//(6*len(counts)))%len(event_counts)]
        family=FAMILIES[(index//(6*len(counts)*len(event_counts)))%3]
        namespace=f'{VERSION}:{split}:{evaluation}:{name}:{index}'
        seed=int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:4],'little')
        rng=np.random.default_rng(seed)
        cursor=(index//len(counts))*sum(counts)+sum(counts[:index%len(counts)])
        picks=[order[(cursor+j)%len(order)] for j in range(n)];used.update(picks)
        spec=dict(index=index,count=n,scenario='drop' if n==1 else 'random',seed=seed,
                  recipe_version='fresh-shapes-v1',material_offset=material,stiffness_phase=phase,
                  asset_indices=picks,split=split,evaluation=evaluation)
        initial,geometry=scene(spec,metadata,objects)
        action=sequence_action(rng,object_id=(index//len(counts))%n,event_count=event_count,
                               first_family=family,speed_range=speed_range,schema=VERSION)
        interaction=place(initial,geometry,action,rng,mode_for(index,counts))
        initial['v'].fill(0)
        # Sand is excluded from BOTH driven objects and distractors. The full
        # default count/type/E/event-count/first-family crossing is 216 cases.
        initial['mat_type'].fill(material);initial['E'].fill((1e4,1e5,1e6)[phase])
        initial['yield_stress'].fill(10000 if material==1 else 0)
        if material_layout=='mixed':
            initial['mat_type'][:]=(material+np.arange(n))%2
            initial['E'][:]=np.array([1e4,1e5,1e6],np.float32)[(phase+np.arange(n))%3]
            initial['yield_stress'][:]=np.where(initial['mat_type']==1,10000,0)
        from scipy.spatial import cKDTree
        placement_reasons=[];pairs=[]
        if np.any(initial['x']<.06) or np.any(initial['x']>1.44):
            placement_reasons.append('drag_initial_wall_margin')
        for left in range(n):
            for right in range(left+1,n):
                distance=cKDTree(initial['x'][initial['object_ids']==left]).query(
                    initial['x'][initial['object_ids']==right])[0].min()
                pairs.append(dict(a=left,b=right,gap_m=float(distance),required_m=.03))
                if distance<.03:placement_reasons.append(f'drag_initial_spacing:{left}:{right}')
        if placement_reasons:raise ValueError(placement_reasons)
        geometry.update(recipe_version='drag-fresh-shapes-v1',placement_reasons=placement_reasons,
                        pair_gaps=pairs,requested_pairs=interaction['requested_pairs'],drag_interaction=interaction,
                        material_policy=material_layout+' types/E; nu=.2 rho=1000 plasticine yield=10000',
                        interaction_semantics='Shared-grid MPM; one driven patch; targeted contact is not guaranteed',
                        drag_layout='floor-clearance 15mm, zero initial velocity; '+interaction['mode'])
        clock,wave=choose_clock(initial,dx_m=(100/64)/n_grid)
        clock=replace(clock,transitions=16+32*event_count)
        config=dict(n_grid=n_grid,grid_lim_m=100/64,floor_m=.1,floor_contact=True,friction=.3,
                    floor_tolerance_m=.002,observation_emergency_depth_m=.06,gravity_m_s2=[0.,0.,-9.8],
                    dt_s=clock.substep_dt_s,precision='float64-master',softening=.1,
                    plasticine_policy='archived-fcr-log-return')
        folder=root/'cases'/f'drag-{index:06d}';folder.mkdir(parents=True)
        save_npz(folder/'initial.npz',initial)
        case=dict(release_id=VERSION,case_index=index,seed_namespace=namespace,
                  asset_ids=geometry['asset_ids'],task='drag',config_sha256=hashlib.sha256(canonical(dict(
                      initial_sha256=digest(folder/'initial.npz'),solver=config,action=action,
                      settling_policy=asdict(SETTLING),settling_transitions=96,
                      fragmentation_policy=asdict(FragmentationPolicy()),sources=sources)).encode()).hexdigest())
        manifest=dict(task='drag',case=case,episode_id=identity(case)[0],n_objects=n,clock=asdict(clock),
            action_schema=VERSION,action=action,scenario=action['events'][0]['family'],requested_first_family=family,settling_policy=asdict(SETTLING),
            drag_material_policy='elastic-plasticine-only-v1',wall_selection_policy='strict-wall-free-v1',
            settling_mode='group-quiet-tail-v1',settling_transitions=96,drag_object_ids=[action['object_id']],
            command_semantics='prescribed pre-P2G velocity; not achieved displacement',
            floor_m=.1,grid_lim_m=100/64,n_grid=n_grid,solver_config=config,volume_policy=metadata['volume_policy'],
            units=dict(position='m',velocity='m/s',time='s',volume='m^3',density='kg/m^3',E='Pa'),
            population=VERSION,split=split,evaluation=evaluation,object_cohort=role,
            split_manifest_sha256=digest(assets/'SPLITS.json'),bank_manifest_sha256=digest(assets/role/'MANIFEST.json'),
            collapse_policy=asdict(CollapsePolicy()),material_layout=material_layout,
            fragmentation_policy=asdict(FragmentationPolicy()),
            material_names=[('elastic','plasticine','sand')[int(v)] for v in initial['mat_type']],
            geometry_provenance=geometry,wave_budget=wave,
            source_sha256={str(runtime.ROOT/p):h for p,h in sources.items()},accepted_for_training=False)
        write_once(folder/'MANIFEST.json',manifest)
        cases.append(dict(label=folder.name,initial_sha256=digest(folder/'initial.npz'),manifest_sha256=digest(folder/'MANIFEST.json')))
    runtime.verify_sources(sources)
    write_once(root/'PLAN.json',dict(schema=VERSION,cases=cases,sources=sources,assets=str(assets),
        bank_sha256=digest(assets/'COMPLETE.json'),split=split,evaluation=evaluation,namespace=name,material_layout=material_layout,
        object_counts=list(counts),event_counts=list(event_counts),speed_range_m_s=list(speed_range),n_grid=n_grid,
        drag_material_policy='elastic-plasticine-only-v1',wall_selection_policy='strict-wall-free-v1',
        interaction_policy=INTERACTION_POLICY,multi_object_layout_mixture=MIXTURE,
        fragmentation_policy=asdict(FragmentationPolicy()),
        shape_sequence_policy='namespace-cyclic-bank-v1',
        planned_unique_shapes=len(used),bank_shapes=len(objects),start=start,count=count,
        qualification=False,scaling_approved=False))
    return dict(prepared=len(cases),root=str(root))
