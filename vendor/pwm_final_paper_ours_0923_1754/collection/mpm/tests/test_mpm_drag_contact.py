import numpy as np
import pytest
from collection.mpm.drag.contact import grid_mass,overlap,measure


def test_quadratic_grid_mass_partition_and_symmetry():
    rng=np.random.default_rng(1);x=rng.uniform(.2,1.2,size=(1024,3));mass=rng.uniform(.1,2,len(x))
    grid=grid_mass(x,mass,100,100/64)
    assert grid[1].sum()==pytest.approx(mass.sum(),rel=2e-7)
    assert overlap(grid,grid)==pytest.approx(grid[1].sum())
    b=grid_mass(x+.01,mass,100,100/64)
    assert overlap(grid,b)==overlap(b,grid)


def test_reconstruction_matches_actual_p2g_cpu_kernel(tmp_path,monkeypatch):
    import warp as wp
    from collection.mpm.solver import MPM_Simulator_WARP
    from collection.mpm.kernels import p2g_apic_with_stress
    monkeypatch.setattr(wp.config,'kernel_cache_dir',str(tmp_path/'warp-cache'))
    rng=np.random.default_rng(4);x=rng.uniform(.1,.4,size=(127,3)).astype(np.float32)
    mass=rng.uniform(.01,.1,len(x)).astype(np.float32)
    sim=MPM_Simulator_WARP(len(x),n_grid=32,grid_lim=.5,device='cpu')
    sim.mpm_state.particle_x=wp.array(x,dtype=wp.vec3,device='cpu')
    sim.mpm_state.particle_mass=wp.array(mass,dtype=float,device='cpu')
    wp.launch(p2g_apic_with_stress,dim=len(x),inputs=[sim.mpm_state,sim.mpm_model,0.],device='cpu')
    nodes,values=grid_mass(x,mass,32,.5)
    expected=np.zeros(32**3);expected[nodes]=values
    np.testing.assert_allclose(sim.mpm_state.grid_m.numpy().ravel(),expected,atol=1e-8,rtol=2e-6)


def test_20mm_particle_cutoff_can_miss_real_p2g_support_overlap():
    a=grid_mass([[.5,.5,.5]],[1.],100,100/64)
    close=grid_mass([[.525,.5,.5]],[1.],100,100/64)
    far=grid_mass([[.6,.5,.5]],[1.],100,100/64)
    assert overlap(a,close)>1e-4
    assert overlap(a,far)==0


@pytest.mark.parametrize('point',[[-1.,.5,.5],[float('nan'),.5,.5],[2.,.5,.5]])
def test_bad_particle_stencil_is_not_silently_clipped(point):
    with pytest.raises(ValueError):grid_mass([point],[1.],100,100/64)


@pytest.mark.parametrize('moving,pre_motion,expected',[(False,False,False),(True,False,True),(True,True,False)])
def test_transfer_needs_motion_after_coupling_and_quiet_baseline(moving,pre_motion,expected):
    x=np.full((20,2,3),.5);x[:,1,0]+=.025
    if moving:x[8:,:,1]+=np.linspace(0,.03,12)[:,None]
    if pre_motion:x[1:,:,2]+=.003
    m=dict(n_grid=100,grid_lim_m=100/64,action=dict(events=[dict(start=4)]),
           geometry_provenance=dict(drag_interaction=dict(mode='pair',requested_pairs=[dict(a=0,b=1)])))
    a=dict(x=x,object_ids=np.array([0,1]),particle_volume=np.ones(2),density=np.ones(2))
    assert measure(m,a)['any_transfer_candidate'] is expected
