"""Pure, versioned drag commands and object-local selection (SI units)."""
from dataclasses import asdict
import numpy as np
from ..contracts import SettlingPolicy, finite_array
from .interactions import POLICY as INTERACTION_POLICY, MIXTURE

VERSION = 'mpm-drag-command-schedule-v1'
LEGACY_SEQUENCE_VERSION = 'mpm-drag-command-sequence-v2'
SEQUENCE_VERSION = 'mpm-drag-command-sequence-v3'
GENTLE_SEQUENCE_VERSION = 'mpm-drag-command-sequence-v4'
CURRENT_SEQUENCE_VERSION = GENTLE_SEQUENCE_VERSION
SEQUENCE_SCHEMAS = (LEGACY_SEQUENCE_VERSION, SEQUENCE_VERSION, GENTLE_SEQUENCE_VERSION)
SCHEMAS = (VERSION, *SEQUENCE_SCHEMAS)
FRAME_DT = .0417
VARIATION = dict(schema=CURRENT_SEQUENCE_VERSION,materials=['elastic','plasticine'],
    event_counts=[1,2,3],speed_range_m_s=[.05,.25],maximum_event_command_path_m=.25,
    recording_transitions=[48,80,112],observation_interval_s=FRAME_DT,patch_fraction=.1,
    patch_reselection=False,wall_policy='strict-wall-free-v1',
    reverse_semantics='single leg opposite previous horizontal stage; sampled reference if none',
    interaction_policy=INTERACTION_POLICY,multi_object_layout_mixture=MIXTURE,
    targeted_command_path_goal_m=.14,all_lift_targeted_policy='last event becomes pull',
    crowded_two_object_policy='close pair; no chain claim',
    command_profile='symmetric-three-transition-ramp-v1',speed_sampling='continuous uniform, no clipping',
    fragmentation_policy='saved-point-fragmentation-review-v1')
FAMILIES = ('pull_release', 'lift_hold_release', 'reverse_release')
SETTLING = SettlingPolicy(8, 120, .005, .015, .0006)


def patch_mask(x, object_ids, object_id, fraction=.1):
    x=finite_array(x,name='selection positions')
    ids=np.asarray(object_ids)
    if (ids.ndim!=1 or x.shape!=(len(ids),3) or ids.dtype.kind not in 'iu'
            or type(object_id) is not int or not .02<=fraction<=.2):
        raise ValueError('Invalid object-local patch request')
    candidates=np.flatnonzero(ids==object_id)
    if not len(candidates):raise ValueError('Selected object missing')
    # A nearest-neighbour spatial patch around the topmost material point;
    # surface samples do not contain topology, so connectivity is not claimed.
    # stable material indices break ties, never select a neighbouring object.
    anchor=candidates[np.argmax(x[candidates,2])]
    distance=np.sum((x[candidates].astype(float)-x[anchor])**2,axis=1)
    order=np.lexsort((candidates,distance))
    selected=np.zeros(len(ids),np.int32)
    selected[candidates[order[:max(1,int(np.ceil(len(candidates)*fraction)))]]]=1
    return selected


def schedule(selected, family, *, speed=.15, angle=0., start=4, transitions=48):
    selected=finite_array(selected,name='selected')
    if (selected.ndim!=1 or not np.isin(selected,[0,1]).all() or not selected.any()
            or family not in FAMILIES or not np.isfinite([speed,angle]).all()
            or not .05<=speed<=.3 or type(start) is not int or not 0<=start<=8
            or transitions!=48):raise ValueError('Outside declared drag schedule')
    command=np.zeros((transitions,len(selected),3),np.float32)
    active=np.zeros(transitions,np.int32)
    direction=np.array([np.cos(angle),np.sin(angle),0.],np.float32)
    if family=='lift_hold_release':direction=np.array([0.,0.,1.],np.float32)
    active[start:start+24]=1
    command[start:start+12]=selected[None,:,None]*direction*speed
    if family=='pull_release':command[start+12:start+24]=selected[None,:,None]*direction*speed
    elif family=='reverse_release':command[start+12:start+24]=-selected[None,:,None]*direction*speed
    # lift family has twelve active, exactly zero-velocity hold transitions.
    return command,active


def check_schedule(command, selected, active, *, transitions=48, schema=VERSION):
    if schema not in SCHEMAS or (schema==VERSION and transitions!=48) or (
            schema in SEQUENCE_SCHEMAS and transitions not in (48,80,112)):
        raise ValueError('Unknown drag schema/recording length')
    selected=finite_array(selected,name='selected');active=finite_array(active,(transitions,),'active')
    command=finite_array(command,(transitions,len(selected),3),'command')
    if (selected.ndim!=1 or not np.isin(selected,[0,1]).all() or not np.isin(active,[0,1]).all()
            or not selected.any() or not active.any() or active[-1]
            or np.any(command[:,selected==0]) or np.any(command[active==0])
            or np.linalg.norm(command,axis=-1).max()>(.250001 if schema==GENTLE_SEQUENCE_VERSION else .300001 if schema==VERSION else .600001)):
        raise ValueError('Invalid selection/activity/velocity/release contract')
    return dict(schema=schema,velocity_units='m/s',application='before every native P2G',
                activity_semantics='zero velocity with active=1 is hold; active=0 is release',
                command_is_achieved_motion=False,settling_policy=asdict(SETTLING))


def sequence_action(rng, *, object_id, event_count, first_family, speed_range=(.05,.6),schema=SEQUENCE_VERSION):
    """One fixed material patch, 1--3 separated episodes, no robot regrasp claim.

    Speeds are continuous. Integer durations preserve the observation clock.
    Each stage has one moving direction. Reverse opposes the previous horizontal
    stage (ignoring intervening lifts), or a sampled reference if none exists.
    The 25 cm command-path bound applies per event.
    """
    if (schema not in SEQUENCE_SCHEMAS or type(event_count) is not int or not 1<=event_count<=3 or first_family not in FAMILIES
            or len(speed_range)!=2 or not np.isfinite(speed_range).all()
            or not .05<=speed_range[0]<speed_range[1]<=(.25 if schema==GENTLE_SEQUENCE_VERSION else .6)):
        raise ValueError('Invalid sequence sampling range')
    events=[];last_horizontal_angle=None
    for i in range(event_count):
        family=first_family if i==0 else FAMILIES[int(rng.integers(3))]
        speed=float(rng.uniform(*speed_range))
        max_moving=min(20,int(np.floor(.25/(speed*FRAME_DT))))
        moving=int(rng.integers(6,max_moving+1))
        hold=int(rng.integers(4,9)) if family=='lift_hold_release' else 0
        # Keep every active interval inside a 24-transition budget in its slot.
        if moving+hold>24:moving=24-hold
        angle=float(rng.uniform(-np.pi,np.pi))
        if family=='reverse_release' and last_horizontal_angle is not None:
            angle=last_horizontal_angle
        if family!='lift_hold_release':
            last_horizontal_angle=angle+(np.pi if family=='reverse_release' else 0.)
            last_horizontal_angle=float(np.arctan2(np.sin(last_horizontal_angle),np.cos(last_horizontal_angle)))
        events.append(dict(family=family,speed=speed,angle=angle,
                           start=4+32*i+int(rng.integers(3)),moving=moving,hold=hold))
    return dict(family='sequence',object_id=object_id,fraction=.1,events=events,
                patch_semantics='same material indices across releases; no physical regrasp')


def declared_schedule(selected, action, *, schema, transitions):
    """Reconstruct the exact command arrays from the versioned manifest."""
    if schema==VERSION:
        if transitions!=48:raise ValueError('Invalid legacy drag length')
        return schedule(selected,action['family'],speed=action['speed'],angle=action['angle'],start=action['start'])
    if schema not in SEQUENCE_SCHEMAS:raise ValueError('Unknown drag action schema')
    events=action.get('events')
    if (action.get('family')!='sequence' or not isinstance(events,list) or not 1<=len(events)<=3
            or transitions!=16+32*len(events) or action.get('fraction')!=.1
            or action.get('patch_semantics')!='same material indices across releases; no physical regrasp'):
        raise ValueError('Invalid sand-free sequence declaration')
    selected=finite_array(selected,name='selected')
    if selected.ndim!=1 or not np.isin(selected,[0,1]).all() or not selected.any():
        raise ValueError('Invalid sequence patch')
    cmd=np.zeros((transitions,len(selected),3),np.float32);active=np.zeros(transitions,np.int32)
    last_horizontal=None
    for i,event in enumerate(events):
        if set(event)!={'family','speed','angle','start','moving','hold'}:
            raise ValueError('Invalid sequence event fields')
        family,speed,angle,start,moving,hold=(event[k] for k in ('family','speed','angle','start','moving','hold'))
        if (family not in FAMILIES or not np.isfinite([speed,angle]).all() or not .05<=speed<=(.25 if schema==GENTLE_SEQUENCE_VERSION else .6)
                or any(type(v) is not int for v in (start,moving,hold))
                or not 4+32*i<=start<=6+32*i or not 6<=moving<=20 or moving+hold>24
                or speed*moving*FRAME_DT>.25+1e-12
                or (schema==LEGACY_SEQUENCE_VERSION and family=='reverse_release' and (moving<8 or moving%2))
                or (family=='lift_hold_release' and not 4<=hold<=8)
                or (family!='lift_hold_release' and hold!=0)):
            raise ValueError('Sequence event outside declared timing/speed/path contract')
        direction=np.array([np.cos(angle),np.sin(angle),0.],np.float32)
        if family=='lift_hold_release':direction=np.array([0.,0.,1.],np.float32)
        elif schema in (SEQUENCE_VERSION,GENTLE_SEQUENCE_VERSION):
            if family=='reverse_release':
                if last_horizontal is not None and not np.allclose(direction,last_horizontal,rtol=0,atol=1e-6):
                    raise ValueError('Reverse reference must match previous horizontal stage')
                direction=-direction
            last_horizontal=direction.copy()
        active[start:start+moving+hold]=1
        cmd[start:start+moving]=selected[None,:,None]*direction*speed
        if schema==GENTLE_SEQUENCE_VERSION:
            steps=np.arange(moving)
            ramp=np.minimum(1.,np.minimum((steps+1)/3.,(moving-steps)/3.))
            cmd[start:start+moving]*=ramp[:,None,None]
        if schema==LEGACY_SEQUENCE_VERSION and family=='reverse_release':
            cmd[start+moving//2:start+moving]*=-1
    check_schedule(cmd,selected,active,transitions=transitions,schema=schema)
    return cmd,active
