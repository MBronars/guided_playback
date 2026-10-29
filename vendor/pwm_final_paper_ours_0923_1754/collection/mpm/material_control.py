"""Opt-in Hencky perfect-plastic control, NOT a replacement dataset policy.

Plasticine's original log-strain yield criterion is paired with its matching
Hencky elastic energy. Y means ||dev(tau)||_F in Pa, not uniaxial yield stress.
Damage/softening are intentionally unsupported by this diagnostic variant.
Equations and limitations are documented in MATERIAL_CANDIDATE.md.
"""
import warp as wp
from .state import MPMModelStruct, MPMStateStruct
from .kernels import (sand_return_mapping, kirchoff_stress_neoHookean,
                       kirchoff_stress_drucker_prager, zero_grid,
                       p2g_apic_with_stress, grid_normalization_and_gravity, g2p)

POLICY='hencky-perfect-plastic-diagnostic-v1'


@wp.func
def project_hencky(f: wp.mat33, mu: float, yield_norm: float):
    u=wp.mat33(0.0);v=wp.mat33(0.0);sig=wp.vec3(0.0)
    wp.svd3(f,u,sig,v)
    eps=wp.vec3(wp.log(sig[0]),wp.log(sig[1]),wp.log(sig[2]))
    mean=(eps[0]+eps[1]+eps[2])/3.0
    hat=eps-wp.vec3(mean);length=wp.length(hat)
    result=f
    if mu > 0.0 and 2.0*mu*length > yield_norm:
        eps=wp.vec3(mean)+(yield_norm/(2.0*mu*length))*hat
        result=u*wp.diag(wp.vec3(wp.exp(eps[0]),wp.exp(eps[1]),wp.exp(eps[2])))*wp.transpose(v)
    return result


@wp.func
def stress_hencky(u: wp.mat33, sig: wp.vec3, mu: float, lam: float):
    eps=wp.vec3(wp.log(sig[0]),wp.log(sig[1]),wp.log(sig[2]))
    tau=2.0*mu*eps+lam*(eps[0]+eps[1]+eps[2])*wp.vec3(1.0)
    return u*wp.diag(tau)*wp.transpose(u)


@wp.kernel
def compute_consistent_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    p=wp.tid()
    if state.particle_selection[p]==0:
        mat=model.material[p]
        f=state.particle_F_trial[p]
        if mat==5:
            f=project_hencky(f,model.mu[p],model.yield_stress[p])
        elif mat==2:
            f=sand_return_mapping(f,state,model,p)
        state.particle_F[p]=f
        u=wp.mat33(0.0);v=wp.mat33(0.0);sig=wp.vec3(0.0)
        wp.svd3(f,u,sig,v)
        stress=wp.mat33(0.0)
        if mat==5:
            stress=stress_hencky(u,sig,model.mu[p],model.lam[p])
        elif mat==2:
            stress=kirchoff_stress_drucker_prager(f,u,v,sig,model.mu[p],model.lam[p])
        elif mat==7:
            stress=kirchoff_stress_neoHookean(p,f,u,v,wp.determinant(f),sig,model.mu[p],model.lam[p])
        state.particle_stress[p]=(stress+wp.transpose(stress))/2.0


def step_with_consistent_material(solver, dt, device):
    """Same original transfer kernels/order; only the material kernel changes.

    This explicitly bounded bridge supports only the backend's fixed floor and
    pre-P2G commands. Unsupported solver features fail rather than being dropped.
    Tests compare this path bit-for-bit with the original elastic/sand paths.
    """
    if (solver.pre_p2g_operations or solver.particle_velocity_modifiers
            or any(f is not None for f in solver.modify_bc)
            or solver.mpm_model.grid_v_damping_scale!=1):
        raise ValueError('Unsupported operation in bounded material-control bridge')
    s,m=solver.mpm_state,solver.mpm_model
    grid=(m.grid_dim_x,m.grid_dim_y,m.grid_dim_z)
    wp.launch(zero_grid,dim=grid,inputs=[s,m],device=device)
    wp.launch(compute_consistent_stress,dim=solver.n_particles,inputs=[s,m,dt],device=device)
    wp.launch(p2g_apic_with_stress,dim=solver.n_particles,inputs=[s,m,dt],device=device)
    wp.launch(grid_normalization_and_gravity,dim=grid,inputs=[s,m,dt],device=device)
    for kernel,parameters in zip(solver.grid_postprocess,solver.collider_params):
        wp.launch(kernel,dim=grid,inputs=[solver.time,dt,s,m,parameters],device=device)
    wp.launch(g2p,dim=solver.n_particles,inputs=[s,m,dt],device=device)
    solver.time+=dt
