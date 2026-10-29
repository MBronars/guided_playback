"""Explicit scheduled-drag adapter; never infer commands from future positions."""
from dataclasses import replace
import numpy as np
import torch
from data.unified_dataset import UnifiedMPMDataset

SCHEMA='mpm-drag-command-schedule-v1'
SEQUENCE_SCHEMA='mpm-drag-command-sequence-v3'
SEQUENCE_SCHEMAS={'mpm-drag-command-sequence-v2',SEQUENCE_SCHEMA}
GENTLE_SEQUENCE_SCHEMA='mpm-drag-command-sequence-v4'
SEQUENCE_SCHEMAS.add(GENTLE_SEQUENCE_SCHEMA)


def commands(sample):
    positions=np.asarray(sample['particle_pos.pyd']);b,t,n,_=positions.shape
    velocity=np.asarray(sample['grab_velocity.pyd'])
    selected=np.asarray(sample['is_grabbed.pyd']);active=np.asarray(sample['drag_active.pyd'])
    rows=sample.get('provenance.json')
    if not isinstance(rows,list) or len(rows)!=b:raise ValueError('Drag requires per-row provenance')
    schemas={row.get('action_schema') for row in rows}
    if len(schemas)!=1 or not schemas<={SCHEMA,*SEQUENCE_SCHEMAS} or any(row.get('task')!='drag' for row in rows):
        raise ValueError('Unsupported drag command schema')
    sequence=next(iter(schemas)) in SEQUENCE_SCHEMAS
    times=np.asarray([row['state_time_s'] for row in rows],dtype=np.float64)
    if (t not in ((49,81,113) if sequence else (49,)) or velocity.shape!=(b,t-1,n,3) or velocity.dtype!=np.float32
            or selected.shape!=(b,n) or active.shape!=(b,t)
            or times.shape!=(b,t) or not np.isfinite(velocity).all() or not np.isfinite(times).all()
            or not np.all(np.diff(times,axis=1)>0) or not np.isin(selected,[0,1]).all()
            or not np.isin(active,[0,1]).all() or active[:,-1].any()
            or not selected.any(axis=1).all() or not active.any(axis=1).all()):
        raise ValueError('Malformed scheduled drag arrays/clock')
    gentle=schemas=={GENTLE_SEQUENCE_SCHEMA}
    if not np.allclose(np.diff(times,axis=1),.0417,rtol=0,atol=1e-12) or np.linalg.norm(velocity,axis=-1).max()>(.250001 if gentle else .600001 if sequence else .300001):
        raise ValueError('Scheduled drag clock/speed outside the versioned contract')
    if sequence:
        materials=np.asarray(sample['material_properties.pyd'])
        if (materials.shape!=(b,n,5) or not np.isin(materials[...,3],[0,1]).all()
                or any(row.get('drag_material_policy')!='elastic-plasticine-only-v1'
                       or row.get('wall_selection_policy')!='strict-wall-free-v1' for row in rows)):
            raise ValueError('Sequence dragging requires sand-free materials and strict wall filtering')
        for row,g in zip(rows,active):
            if gentle and row.get('fragmentation_policy',{}).get('version')!='saved-point-fragmentation-review-v1':
                raise ValueError('Gentle drag requires fragmentation-filter provenance')
            events=row.get('action',{}).get('events',[])
            starts=np.count_nonzero(g & ~np.r_[False,g[:-1]])
            if len(events)!=(t-17)//32 or starts!=len(events):
                raise ValueError('Sequence event count/recording mismatch')
    gate=active[:,:-1,None].astype(bool)&selected[:,None,:].astype(bool)
    if np.any(velocity[~gate]):raise ValueError('Velocity outside declared active patch')
    delta=(velocity.astype(np.float64)*np.diff(times,axis=1)[:,:,None,None]).astype(np.float32)
    return torch.from_numpy(delta),torch.from_numpy(gate),torch.from_numpy(selected.astype(bool))


class UnifiedMPMDragDataset(UnifiedMPMDataset):
    """Opt-in pilot adapter. Same MPM coordinates, windows and frozen stats."""
    def __init__(self,*args,**kwargs):
        if kwargs.get('blocklist_paths'):
            raise ValueError('Drag exports must already be filtered; row-sidecar exclusions not supported')
        super().__init__(*args,**kwargs)

    def _validate_command_input(self,sample):
        commands(sample)

    def _process_sample(self,sample):
        delta,gate,selected=commands(sample)
        batch=super()._process_sample(sample)
        c=batch.context_len;f=batch.num_future_frames
        action=torch.stack([delta[i,s+c-1:s+c-1+f] for i,s in enumerate(self._window_starts)])
        future_gate=torch.stack([gate[i,s+c-1:s+c-1+f] for i,s in enumerate(self._window_starts)])
        context_gate=torch.stack([gate[i,s:s+c-1] for i,s in enumerate(self._window_starts)])
        return replace(batch,actions=action,action_mask=selected,action_active=future_gate,
                       context_action_active=context_gate,domain='mpm')
