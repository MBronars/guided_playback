"""Versioned multi-event dragging. Synthetic tests are not physics approval."""
from dataclasses import asdict
import numpy as np
import pytest
from collection.mpm.drag.protocol import (SEQUENCE_VERSION, LEGACY_SEQUENCE_VERSION, FAMILIES, sequence_action,
    declared_schedule, check_schedule)
from collection.mpm.contracts import Clock
from collection.mpm.recording import atomic_raw
from collection.mpm.validate import validate_raw, _batch_payload
from collection.mpm.tests.test_mpm_drag import fixture


def sequence_fixture(events=3):
    m,a=fixture();t=17+32*events
    action=sequence_action(np.random.default_rng(38),object_id=0,event_count=events,
                           first_family='lift_hold_release',speed_range=(.3,.6))
    m.update(action_schema=SEQUENCE_VERSION,action=action,clock=asdict(Clock(5e-5,834,t-1)),
             drag_material_policy='elastic-plasticine-only-v1',wall_selection_policy='strict-wall-free-v1')
    for key in ('x','v','F_trial','C'):
        a[key]=np.repeat(a[key][:1],t,axis=0)
    a['state_time_s']=Clock(**m['clock']).state_time_s
    a['command_velocity'],a['transition_active']=declared_schedule(a['selected'],action,
        schema=SEQUENCE_VERSION,transitions=t-1)
    a['active_substeps']=a['transition_active']*834
    steps=np.arange(t,dtype=np.int64)*834
    a.update(wall_simulation_steps=steps,wall_covered_substeps=np.repeat(steps[:,None],5,axis=1),
             wall_first_grid_step=np.full((t,5),-1,np.int64),wall_first_particle_step=np.full((t,5),-1,np.int64))
    return m,a


@pytest.mark.parametrize('events',[1,2,3])
@pytest.mark.parametrize('family',FAMILIES)
def test_sequence_timing_paths_and_holds(events,family):
    for seed in range(30):
        action=sequence_action(np.random.default_rng(seed),object_id=0,event_count=events,first_family=family)
        cmd,gate=declared_schedule(np.array([1,0],np.int32),action,schema=SEQUENCE_VERSION,transitions=16+32*events)
        check_schedule(cmd,np.array([1,0]),gate,transitions=len(gate),schema=SEQUENCE_VERSION)
        starts=np.flatnonzero(np.diff(np.r_[0,gate,0])==1);ends=np.flatnonzero(np.diff(np.r_[0,gate,0])==-1)
        assert len(starts)==events and starts[0]>=4 and len(gate)-ends[-1]>=12
        assert np.all(starts[1:]-ends[:-1]>=6) and not cmd[:,1].any()
        previous_horizontal=None
        for e in action['events']:
            s=e['start'];moving=e['moving']
            np.testing.assert_array_equal(cmd[s:s+moving,0],np.broadcast_to(cmd[s,0],(moving,3)))
            assert np.linalg.norm(cmd[s:s+moving,0],axis=-1).sum()*.0417<=.250001
            if e['family']=='lift_hold_release':
                assert gate[s+moving:s+moving+e['hold']].all()
                assert not cmd[s+moving:s+moving+e['hold']].any()
            if e['family']!='lift_hold_release':
                direction=cmd[s,0]/e['speed']
                if e['family']=='reverse_release' and previous_horizontal is not None:
                    np.testing.assert_allclose(direction,-previous_horizontal,atol=2e-7)
                previous_horizontal=direction


def explicit_action(families):
    return dict(family='sequence',fraction=.1,object_id=0,
        patch_semantics='same material indices across releases; no physical regrasp',
        events=[dict(family=f,speed=.2,angle=0.,start=4+32*i,moving=8,
                     hold=4 if f=='lift_hold_release' else 0) for i,f in enumerate(families)])


@pytest.mark.parametrize('families',[
    ['pull_release','reverse_release','lift_hold_release'],
    ['pull_release','lift_hold_release','reverse_release'],
    ['reverse_release'],
])
def test_single_leg_reverse_and_lift_preserve_reference(families):
    action=explicit_action(families)
    cmd,_=declared_schedule(np.array([1,0]),action,schema=SEQUENCE_VERSION,transitions=16+32*len(families))
    for e in action['events']:
        expected=[-.2,0,0] if e['family']=='reverse_release' else [0,0,.2] if e['family']=='lift_hold_release' else [.2,0,0]
        np.testing.assert_allclose(cmd[e['start']:e['start']+e['moving'],0],np.tile(expected,(8,1)))


def test_reverse_rejects_unrelated_reference():
    action=explicit_action(['pull_release','reverse_release'])
    action['events'][1]['angle']=1.
    with pytest.raises(ValueError,match='previous horizontal'):
        declared_schedule(np.array([1]),action,schema=SEQUENCE_VERSION,transitions=80)


def test_consecutive_reverse_events_alternate_between_stages_only():
    action=explicit_action(['pull_release','reverse_release','reverse_release'])
    action['events'][2]['angle']=np.pi
    cmd,_=declared_schedule(np.array([1]),action,schema=SEQUENCE_VERSION,transitions=112)
    for i,sign in enumerate([1,-1,1]):
        np.testing.assert_allclose(cmd[4+32*i:12+32*i,0],np.tile([sign*.2,0,0],(8,1)),atol=1e-7)


def test_legacy_v2_keeps_out_and_back_and_odd_length_rejection():
    action=explicit_action(['reverse_release'])
    cmd,_=declared_schedule(np.array([1]),action,schema=LEGACY_SEQUENCE_VERSION,transitions=48)
    np.testing.assert_allclose(cmd[4:8,0],np.tile([.2,0,0],(4,1)))
    np.testing.assert_array_equal(cmd[8:12],-cmd[4:8])
    action['events'][0]['moving']=7
    with pytest.raises(ValueError):
        declared_schedule(np.array([1]),action,schema=LEGACY_SEQUENCE_VERSION,transitions=48)
    cmd,_=declared_schedule(np.array([1]),action,schema=SEQUENCE_VERSION,transitions=48)
    np.testing.assert_allclose(cmd[4:11,0],np.tile([-.2,0,0],(7,1)))


def test_legacy_v2_raw_export_and_consumer_still_supported(tmp_path):
    m,a=sequence_fixture(1)
    m.update(action_schema=LEGACY_SEQUENCE_VERSION,action=explicit_action(['reverse_release']))
    a['command_velocity'],a['transition_active']=declared_schedule(a['selected'],m['action'],
        schema=LEGACY_SEQUENCE_VERSION,transitions=48)
    a['active_substeps']=a['transition_active']*834
    p=tmp_path/'legacy-v2.h5';atomic_raw(p,m,a)
    result=validate_raw(p);assert result['integrity_pass'],result
    from data.mpm_drag import commands
    payload={k+'.pyd':v for k,v in _batch_payload([(m,a)]).items()}
    payload['provenance.json']=[dict(m,state_time_s=a['state_time_s'].tolist())]
    delta,_,_=commands(payload)
    np.testing.assert_allclose(delta.numpy()[0],a['command_velocity']*.0417,atol=2e-9)


@pytest.mark.parametrize('events',[1,2,3])
def test_sequence_raw_and_consumer(events,tmp_path):
    m,a=sequence_fixture(events);p=tmp_path/'raw.h5';atomic_raw(p,m,a)
    result=validate_raw(p);assert result['integrity_pass'],result
    from data.mpm_drag import commands
    payload={k+'.pyd':v for k,v in _batch_payload([(m,a)]).items()}
    payload['provenance.json']=[dict(m,state_time_s=a['state_time_s'].tolist())]
    delta,gate,_=commands(payload)
    np.testing.assert_array_equal(gate.numpy()[0],a['transition_active'][:,None].astype(bool)&a['selected'][None,:].astype(bool))
    np.testing.assert_allclose(delta.numpy()[0],a['command_velocity']*.0417,atol=2e-9)
    payload['material_properties.pyd'][0,-1,3]=2
    with pytest.raises(ValueError,match='sand-free'):commands(payload)


@pytest.mark.parametrize('damage',['sand','policy','wall','duration','overlap','speed','long_path','events','command'])
def test_sequence_rejects_corruption(damage,tmp_path):
    m,a=sequence_fixture()
    if damage=='sand':a['mat_type'][0]=2
    elif damage=='policy':m['wall_selection_policy']='sand-only-guarded-clearance-v2'
    elif damage=='wall':a['wall_first_grid_step'][1:,0]=1
    elif damage=='duration':m['action']['events'][0]['moving']=99
    elif damage=='overlap':m['action']['events'][1]['start']=4
    elif damage=='speed':m['action']['events'][0]['speed']=.7
    elif damage=='long_path':m['action']['events'][0].update(family='pull_release',moving=20,hold=0,speed=.6)
    elif damage=='events':m['action']['events'].pop()
    else:a['command_velocity'][4]*=0
    if damage=='command':a['command_velocity'][10]=.1
    p=tmp_path/'raw.h5';atomic_raw(p,m,a)
    result=validate_raw(p);assert not result['integrity_pass'],result


def test_legacy_speed_limit_is_not_relaxed():
    m,a=sequence_fixture(1)
    with pytest.raises(ValueError):check_schedule(a['command_velocity'],a['selected'],a['transition_active'])
