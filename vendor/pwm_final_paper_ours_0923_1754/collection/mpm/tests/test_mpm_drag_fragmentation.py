"""Synthetic contract tests; passing does not calibrate fracture thresholds."""
import numpy as np
import pytest
from collection.mpm.drag.fragmentation import analyze, FragmentationPolicy
from collection.mpm.drag.actuation_ablation import commands, VARIANTS
from collection.mpm.drag.protocol import patch_mask, declared_schedule, SEQUENCE_VERSION


def fixture():
    p=np.stack(np.meshgrid(np.arange(10),np.arange(4),np.arange(4),indexing='ij'),-1).reshape(-1,3)*.005
    return np.repeat(p[None],6,axis=0),np.zeros(len(p),np.int32)


def test_rigid_motion_and_moderate_stretch_are_not_fragmentation():
    x,ids=fixture();theta=.7
    rot=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    x[1:]=x[1:]@rot.T*1.4+[.3,.2,.1]
    assert not analyze(x,ids,dx_m=.006)['quarantine']


def test_persistent_split_and_transient_split():
    x,ids=fixture();x[2:,:32,2]+=.2
    report=analyze(x,ids,dx_m=.006)
    assert report['quarantine']
    assert report['objects'][0]['flagged_states']==[4,5]
    x[3:]=x[0]
    assert not analyze(x,ids,dx_m=.006)['quarantine']


def test_initial_disconnected_parts_and_other_objects_are_not_new_splits():
    x,ids=fixture();x[:,:32,2]+=.2
    assert not analyze(x,ids,dx_m=.006)['quarantine']
    x,ids=fixture();joined=np.concatenate([x,x+[.001,.001,.001]],axis=1)
    both=np.concatenate([ids,ids+1]);joined[2:,:32,2]+=.2
    # A nearby different object must not reconnect the first object's fragment.
    assert analyze(joined,both,dx_m=.006)['quarantine']


def test_changing_outliers_cannot_accumulate_persistence():
    x,ids=fixture()
    for k in range(1,len(x)):x[k,(k-1)*16:k*16,2]+=.2
    assert not analyze(x,ids,dx_m=.006)['quarantine']


@pytest.mark.parametrize('kwargs',[dict(dx_m=0),dict(dx_m=float('nan')),dict(dx_m=.006,selected=[1])])
def test_bad_inputs(kwargs):
    x,ids=fixture()
    with pytest.raises(ValueError):analyze(x,ids,**kwargs)


@pytest.mark.parametrize('kwargs',[dict(consecutive_states=1),dict(minimum_particles=True),dict(minimum_fraction=.8)])
def test_bad_policy(kwargs):
    with pytest.raises(ValueError):FragmentationPolicy(**kwargs)


@pytest.mark.parametrize('variant',VARIANTS)
def test_actuation_variants_preserve_selection_hold_release_direction(variant):
    x,ids=fixture();selected=patch_mask(x[0],ids,0,.1)
    action=dict(family='sequence',object_id=0,fraction=.1,patch_semantics='same material indices across releases; no physical regrasp',
        events=[dict(family='lift_hold_release',speed=.55,angle=0.,start=4,moving=8,hold=4)])
    m=dict(action=action,action_schema=SEQUENCE_VERSION)
    original,active=declared_schedule(selected,action,schema=SEQUENCE_VERSION,transitions=48)
    a=dict(x=x,object_ids=ids,selected=selected,command_velocity=original,transition_active=active)
    cmd,sel,gate=commands(m,a,variant)
    np.testing.assert_array_equal(gate,active)
    assert not cmd[gate==0].any() and not cmd[:,sel==0].any()
    assert not cmd[12:16].any() and gate[12:16].all()
    assert not cmd[:,:,:2].any() and np.all(cmd[:,:,2]>=0)
    assert sel.sum()==(32 if variant.startswith('patch20') else 16)
    if 'gentle' in variant:
        assert np.linalg.norm(cmd,axis=-1).max()<=.250001
        v=cmd[4:12,np.flatnonzero(sel)[0],2]
        assert v[0]<v[1]<v[2] and v[-1]<v[-2]<v[-3]
    elif variant=='baseline':np.testing.assert_array_equal(cmd,original)


@pytest.mark.parametrize('split',[False,True])
def test_versioned_raw_filter_rejects_detachment_only_when_present(tmp_path,split):
    from dataclasses import asdict
    from collection.mpm.tests.test_mpm_drag import fixture as raw_fixture
    from collection.mpm.recording import atomic_raw
    from collection.mpm.validate import validate_raw
    m,a=raw_fixture();m['fragmentation_policy']=asdict(FragmentationPolicy())
    if split:a['x'][20:,:4,2]+=.35
    path=tmp_path/'raw.h5';atomic_raw(path,m,a);result=validate_raw(path)
    assert 'fragmentation' in result['metrics'],result
    assert result['metrics']['fragmentation']['quarantine']==split
    if split:
        assert result['status']=='rejected'
        assert 'object_0:persistent_new_spatial_split' in result['reasons']
    else:assert result['integrity_pass'],result


def test_raw_filter_policy_cannot_be_loosened(tmp_path):
    from dataclasses import asdict
    from collection.mpm.tests.test_mpm_drag import fixture as raw_fixture
    from collection.mpm.recording import atomic_raw
    from collection.mpm.validate import validate_raw
    m,a=raw_fixture();m['fragmentation_policy']=asdict(FragmentationPolicy())
    m['fragmentation_policy']['minimum_fraction']=.4
    path=tmp_path/'raw.h5';atomic_raw(path,m,a);result=validate_raw(path)
    assert not result['integrity_pass']
    assert any('pinned policy' in r for r in result['reasons'])


def test_settling_split_is_screened_even_if_it_rejoins(tmp_path):
    from dataclasses import asdict
    from collection.mpm.tests.test_mpm_drag import fixture as raw_fixture
    from collection.mpm.recording import atomic_raw
    from collection.mpm.validate import validate_raw
    m,a=raw_fixture();m['fragmentation_policy']=asdict(FragmentationPolicy())
    a['settle_x'][20:40,:4,2]+=.35
    path=tmp_path/'raw.h5';atomic_raw(path,m,a);result=validate_raw(path)
    assert result['status']=='rejected',result
    assert 'settle_object_0:persistent_new_spatial_split' in result['reasons']
