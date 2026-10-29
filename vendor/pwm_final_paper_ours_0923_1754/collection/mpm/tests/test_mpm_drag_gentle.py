"""V4 production commands; historical v2/v3 semantics remain independently tested."""
from dataclasses import asdict
import numpy as np
import pytest
from collection.mpm.drag.protocol import (GENTLE_SEQUENCE_VERSION as VERSION,SEQUENCE_VERSION,
    sequence_action,declared_schedule,FAMILIES)
from collection.mpm.drag.fragmentation import FragmentationPolicy


def gentle_fixture(events=3):
    from collection.mpm.tests.test_mpm_drag_sequence import sequence_fixture
    m,a=sequence_fixture(events)
    action=sequence_action(np.random.default_rng(51),object_id=0,event_count=events,
        first_family='lift_hold_release',speed_range=(.05,.25),schema=VERSION)
    m.update(action=action,action_schema=VERSION,fragmentation_policy=asdict(FragmentationPolicy()))
    a['command_velocity'],a['transition_active']=declared_schedule(a['selected'],action,schema=VERSION,transitions=len(a['x'])-1)
    a['active_substeps']=834*a['transition_active']
    return m,a


@pytest.mark.parametrize('events',[1,2,3])
@pytest.mark.parametrize('family',FAMILIES)
def test_gentle_preserves_gate_direction_and_continuous_speed(events,family):
    action=sequence_action(np.random.default_rng(6),object_id=0,event_count=events,
        first_family=family,speed_range=(.05,.25),schema=VERSION)
    selected=np.array([1,0],np.int32)
    cmd,gate=declared_schedule(selected,action,schema=VERSION,transitions=16+32*events)
    legacy,old_gate=declared_schedule(selected,action,schema=SEQUENCE_VERSION,transitions=16+32*events)
    np.testing.assert_array_equal(gate,old_gate)
    assert not cmd[gate==0].any() and not cmd[:,1].any()
    for event in action['events']:
        start,moving=event['start'],event['moving'];k=np.arange(moving)
        ramp=np.minimum(1,np.minimum((k+1)/3,(moving-k)/3))
        expected=legacy[start:start+moving]*ramp[:,None,None]
        np.testing.assert_allclose(cmd[start:start+moving],expected,rtol=1e-7,atol=1e-9)
        assert np.linalg.norm(cmd[start:start+moving,0],axis=-1).max()<=.250001


@pytest.mark.parametrize('damage',[None,'ramp','fragmentation','speed'])
def test_v4_raw_roundtrip_and_fail_closed(tmp_path,damage):
    from collection.mpm.recording import atomic_raw
    from collection.mpm.validate import validate_raw
    m,a=gentle_fixture()
    if damage=='ramp':a['command_velocity'][m['action']['events'][0]['start']]*=2
    elif damage=='fragmentation':del m['fragmentation_policy']
    elif damage=='speed':m['action']['events'][0]['speed']=.3
    path=tmp_path/'raw.h5';atomic_raw(path,m,a);result=validate_raw(path)
    assert result['integrity_pass']==(damage is None),result


def test_v4_rejects_aggressive_sampler_range():
    with pytest.raises(ValueError):
        sequence_action(np.random.default_rng(1),object_id=0,event_count=1,first_family='pull_release',schema=VERSION)
