"""Synthetic drag contracts, not numerical qualification."""
from dataclasses import asdict
import numpy as np
import pytest
from collection.mpm.drag.protocol import VERSION,SETTLING,FAMILIES,patch_mask,schedule,check_schedule
from collection.mpm.contracts import Clock,identity
from collection.mpm.collapse import CollapsePolicy
from collection.mpm.walls import FIELDS,metadata
from collection.mpm.recording import atomic_raw
from collection.mpm.validate import validate_raw


def fixture():
    n=20;clock=Clock(5e-5,834,48)
    x=np.random.default_rng(3).uniform(.4,.5,(n,3)).astype(np.float32)
    ids=np.zeros(n,np.int32);selected=patch_mask(x,ids,0)
    command,active=schedule(selected,'lift_hold_release')
    case=dict(release_id='synthetic-drag-only',case_index=0,seed_namespace='unit-test',
              asset_ids=['synthetic'],config_sha256='b'*64,task='drag')
    m=dict(case=case,episode_id=identity(case)[0],task='drag',n_objects=1,clock=asdict(clock),
           floor_m=.1,grid_lim_m=100/64,n_grid=100,volume_policy='synthetic',
           source_sha256={'test':'a'*64},units=dict(position='m',velocity='m/s',time='s',
           volume='m^3',density='kg/m^3',E='Pa'),population='synthetic',split='diagnostic',
           action_schema=VERSION,settling_policy=asdict(SETTLING),settling_mode='group-quiet-tail-v1',
           solver_config=dict(dt_s=clock.substep_dt_s),
           settling_transitions=96,drag_object_ids=[0],collapse_policy=asdict(CollapsePolicy()),
           wall_observation=metadata('a'*64),action=dict(family='lift_hold_release',object_id=0,
           fraction=.1,speed=.15,angle=0.,start=4))
    a=dict(x=np.tile(x,(49,1,1)),v=np.zeros((49,n,3),np.float32),
           F_trial=np.tile(np.eye(3,dtype=np.float32),(49,n,1,1)),C=np.zeros((49,n,3,3),np.float32),
           object_ids=ids,particle_volume=np.full(n,1e-6,np.float32),E=np.array([1e4],np.float32),
           nu=np.array([.2],np.float32),density=np.array([1000],np.float32),mat_type=np.array([0],np.int32),
           yield_stress=np.array([0],np.float32),state_time_s=clock.state_time_s,selected=selected,
           command_velocity=command,transition_active=active,active_substeps=active*834,
           settle_x=np.tile(x,(97,1,1)),settle_v=np.zeros((97,n,3),np.float32),
           settle_F_trial=np.tile(np.eye(3,dtype=np.float32),(97,n,1,1)))
    for prefix,states in (('',49),('settle_',97)):
        steps=np.arange(states,dtype=np.int64)*834
        a.update({prefix+'wall_simulation_steps':steps,prefix+'wall_covered_substeps':np.repeat(steps[:,None],5,axis=1),
                  prefix+'wall_first_grid_step':np.full((states,5),-1,np.int64),
                  prefix+'wall_first_particle_step':np.full((states,5),-1,np.int64)})
    return m,a


@pytest.mark.parametrize('family',FAMILIES)
def test_schedule(family):
    selected=np.array([0,1,0,1],np.int32);command,active=schedule(selected,family)
    check_schedule(command,selected,active)
    assert not command[:,selected==0].any()
    assert active.sum()==24 and not active[-1]
    if family=='lift_hold_release':
        assert active[16:28].all() and not command[16:28].any()
        assert not active[28:].any() and not command[28:].any()


def test_patch_object_scope_and_stability():
    x=np.array([[.5,.5,.6],[.5,.5,.59],[.5,.5,.601]],np.float32)
    mask=patch_mask(x,np.array([0,0,1]),0,.2)
    np.testing.assert_array_equal(mask,[1,0,0])
    with pytest.raises(ValueError):patch_mask(x,np.array([0,0,1]),2)


def test_raw_scheduled_contract(tmp_path):
    m,a=fixture();path=tmp_path/'raw.h5';atomic_raw(path,m,a)
    result=validate_raw(path)
    assert result['integrity_pass'],result
    assert result['metrics']['settling_transitions']==96


@pytest.mark.parametrize('corruption',['command','gate','mask','settle_wall','settle_missing_wall',
    'settle_collapse','settle_deformation','settle_velocity','schema','settle_length','recipe','clock','solver_clock'])
def test_raw_rejects_corruption(tmp_path,corruption):
    m,a=fixture()
    if corruption=='command':a['command_velocity'][4]*=2
    elif corruption=='gate':a['transition_active'][20]=0;a['active_substeps'][20]=0
    elif corruption=='mask':a['selected'][:]=1
    elif corruption=='settle_wall':a['settle_wall_first_grid_step'][1:,0]=1
    elif corruption=='settle_missing_wall':del a['settle_wall_covered_substeps']
    elif corruption=='settle_collapse':a['settle_x'][30]=a['settle_x'][30].mean(0)
    elif corruption=='settle_deformation':a['settle_F_trial'][30]*=.001
    elif corruption=='settle_velocity':a['settle_v'][-8:]=.1
    elif corruption=='schema':m['action_schema']='unknown'
    elif corruption=='settle_length':a['settle_x']=a['settle_x'][:-1];a['settle_v']=a['settle_v'][:-1]
    elif corruption=='recipe':m['action']['speed']=.2
    elif corruption=='clock':
        m['clock']['substep_dt_s']=1/24/834;m['solver_config']['dt_s']=m['clock']['substep_dt_s']
        a['state_time_s']=Clock(**m['clock']).state_time_s
    elif corruption=='solver_clock':m['solver_config']['dt_s']/=2
    path=tmp_path/'raw.h5';atomic_raw(path,m,a)
    result=validate_raw(path)
    assert not result['integrity_pass'],(corruption,result)
