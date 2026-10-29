"""Analytic prototype invariants; do not qualify general object contact."""
import numpy as np
import pytest
from collection.mpm.drag.contact_study import block_fixture,prescribed,metrics


def test_spatial_arms_have_identical_particles_weights_materials_and_time():
    base=block_fixture('push_withdraw',32)
    for n in (64,96,128):
        trial=block_fixture('push_withdraw',n)
        np.testing.assert_array_equal(base[0],trial[0]);np.testing.assert_array_equal(base[3],trial[3])
        for key in base[1]:np.testing.assert_array_equal(base[1][key],trial[1][key])
        assert {k:v for k,v in base[2].items() if k!='n_grid'}=={k:v for k,v in trial[2].items() if k!='n_grid'}
        assert trial[-1]<.4
    assert base[1]['particle_volume'].sum()==pytest.approx(2*.1**3)


def test_action_has_single_direction_stages_and_release():
    ids=np.array([0,0,1,1])
    for t,v in [(0.,.2),(.299,.2),(.3,-.2),(.599,-.2),(.6,0.)]:
        cmd,mask=prescribed('push_withdraw',t,ids)
        assert np.all(cmd[ids==0,0]==np.float32(v));assert not cmd[ids==1].any()
        assert mask.sum()==(2 if t<.6 else 0)


def test_padding_adds_only_empty_grid_cells():
    for n in (32,64,96,128):
        base=block_fixture('push_withdraw',n);pad=block_fixture('push_withdraw',n,padded=True)
        np.testing.assert_array_equal(base[0],pad[0]);np.testing.assert_array_equal(base[3],pad[3])
        for key in base[1]:np.testing.assert_array_equal(base[1][key],pad[1][key])
        assert pad[2]['n_grid']==2*base[2]['n_grid']
        assert pad[2]['grid_lim_m']/pad[2]['n_grid']==base[2]['grid_lim_m']/base[2]['n_grid']
        assert pad[-1]==base[-1]


@pytest.mark.parametrize('grid,admitted',[(192,True),(256,True),(257,False)])
def test_diagnostic_grid_bound_before_allocating(monkeypatch,grid,admitted):
    from collection.mpm import physics
    class AllocationReached(Exception):pass
    def sentinel(*args,**kwargs):raise AllocationReached()
    monkeypatch.setattr(physics,'MPM_Simulator_WARP',sentinel)
    x,s,c,_,_,_=block_fixture('near_miss',32);c=dict(c,n_grid=grid)
    with pytest.raises(AllocationReached if admitted else ValueError):
        physics.GuardedWarpBackend(x,s,c,device='cpu')


def test_material_cell_gap_and_undeformed_energy():
    x,s,c,v,h,_=block_fixture('free_impact',32)
    r=metrics(np.stack([x,x]),np.stack([v,v]),np.tile(np.eye(3),(2,len(x),1,1)),s,np.array([0.,.005]),h,'free_impact',.5/32)
    assert r['gap_m'][0]==pytest.approx(.04,abs=1e-7)
    assert np.max(np.abs(r['strain_J']))<1e-8
    assert r['max_linear_momentum_drift_kg_m_s']==0


def test_energy_proxy_uses_fixture_material_coefficients():
    x,s,c,v,h,_=block_fixture('free_impact',32)
    f=np.tile(1.01*np.eye(3),(2,len(x),1,1));args=(np.stack([x,x]),np.stack([v,v]),f)
    first=metrics(*args,s,np.array([0.,.005]),h,'free_impact',.5/32)
    second=metrics(*args,dict(s,E=2*s['E']),np.array([0.,.005]),h,'free_impact',.5/32)
    np.testing.assert_allclose(second['strain_J'],2*np.array(first['strain_J']))


@pytest.mark.parametrize('gated,gap,va,vb,expected_a,expected_b',[
    (0,.03,.2,0.,.05,.05),(1,.03,.2,0.,.2,0.),(1,0.,.2,0.,.05,.05),
    (0,0.,-.2,0.,-.2,0.),(1,0.,-.2,0.,-.2,0.)])
def test_planar_contact_conserves_momentum_and_never_attracts(tmp_path,monkeypatch,gated,gap,va,vb,expected_a,expected_b):
    import warp as wp
    from collection.mpm.state import MPMStateStruct
    from collection.mpm.drag.contact_study import planar_contact
    monkeypatch.setattr(wp.config,'kernel_cache_dir',str(tmp_path/'warp'))
    states=[]
    for mass,velocity in ((1.,va),(3.,vb)):
        s=MPMStateStruct();s.grid_m=wp.full((1,1,1),mass,dtype=float,device='cpu')
        s.grid_v_out=wp.array(np.array([[[[velocity,.07,0.]]]],np.float32),dtype=wp.vec3,device='cpu');states.append(s)
    bounds=wp.array([0.,gap],dtype=float,device='cpu');impulse=wp.zeros(1,dtype=float,device='cpu')
    wp.launch(planar_contact,dim=(1,1,1),inputs=[*states,bounds,gated,5e-5,impulse],device='cpu')
    a,b=[s.grid_v_out.numpy()[0,0,0] for s in states]
    assert a[0]==pytest.approx(expected_a,abs=1e-7);assert b[0]==pytest.approx(expected_b,abs=1e-7)
    np.testing.assert_allclose(a+3*b,[va+3*vb,.28,0.],atol=1e-7)
    assert a[1]==b[1]==np.float32(.07)
    assert float(a[0]**2+3*b[0]**2)<=va**2+3*vb**2+1e-7
