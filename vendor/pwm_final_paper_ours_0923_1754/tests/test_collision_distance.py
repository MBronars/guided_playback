"""Analytic geometry and finite-difference checks for the optional FCL backend."""
import itertools

import pytest
import torch

pytest.importorskip('fcl')
pytest.importorskip('scipy')

from guidance.collision_distance import ConvexCollisionDistance
from guidance.mesh_physical import MeshPickPhysicalObjective, MeshPlacePhysicalObjective, MeshPickValidator, MeshPlaceValidator
from guidance.physical_constraints import PhysicalConstraints
from guidance.place_objective import PlaceValidator
from tests.test_physical_guidance import resting_released_scene


def cubes():
    vertices=torch.tensor(list(itertools.product([-.5,.5],repeat=3)),dtype=torch.float64)
    normals=torch.cat([torch.eye(3),-torch.eye(3)]).double()
    planes=torch.cat([normals,torch.full((6,1),-.5,dtype=torch.float64)],dim=-1)
    body=dict(vertices=vertices,planes=[planes])
    return ConvexCollisionDistance(dict(bodies={0:body,1:body}),other_bodies=(1,))


@pytest.mark.parametrize('separation,expected',[(1.4,.4),(.7,-.3)])
def test_convex_signed_distance_and_translation_gradient(separation,expected):
    query=cubes();R=torch.eye(3,dtype=torch.float64).repeat(2,1,1)
    t=torch.tensor([[0.,0,0],[separation,.11,.17]],dtype=torch.float64,requires_grad=True)
    distance=query(R,t,torch.arange(2))
    assert float(distance.detach()[0])==pytest.approx(expected,abs=1e-7)
    gradient,=torch.autograd.grad(distance.sum(),t)
    torch.testing.assert_close(gradient,torch.tensor([[-1.,0,0],[1.,0,0]],dtype=torch.float64),atol=1e-6,rtol=0)
    translated=query(R,t.detach()+torch.tensor([.3,-.7,.2]),torch.arange(2))
    torch.testing.assert_close(translated,distance.detach(),atol=1e-7,rtol=0)


@pytest.mark.parametrize('separation',[1.5,.7])
def test_convex_rotation_derivative_matches_finite_difference(separation):
    query=cubes();angle=torch.tensor(.23,dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[0.,0,0],[separation,.13,.09]],dtype=torch.float64)
    def evaluate(theta):
        zero=theta*0;one=zero+1
        rz=torch.stack([theta.cos(),-theta.sin(),zero,theta.sin(),theta.cos(),zero,zero,zero,one]).reshape(3,3)
        return query(torch.stack([torch.eye(3,dtype=torch.float64),rz]),t,torch.arange(2)).sum()
    value=evaluate(angle);gradient,=torch.autograd.grad(value,angle)
    eps=1e-5
    finite=(evaluate(angle.detach()+eps)-evaluate(angle.detach()-eps))/(2*eps)
    torch.testing.assert_close(gradient,finite,atol=1e-5,rtol=1e-3)


def test_collision_queries_handle_leading_dimensions_and_body_reordering():
    query=cubes()
    R=torch.eye(3,dtype=torch.float64).expand(1,3,2,3,3).clone()
    t=torch.zeros(1,3,2,3,dtype=torch.float64);t[:,:,1,0]=torch.tensor([1.5,1.2,.8])
    original=query(R,t,torch.arange(2))
    reordered=query(R.flip(-3),t.flip(-2),torch.tensor([1,0]))
    assert original.shape==(1,3,1)
    torch.testing.assert_close(original,reordered,rtol=0,atol=1e-8)


@pytest.mark.parametrize('task',['pick','place'])
def test_mesh_objective_constraints_match_independent_mesh_screen(task):
    b,g,p,grasp=resting_released_scene()
    target=p['t'][:,-1,0].clone()
    cls=MeshPlacePhysicalObjective if task=='place' else MeshPickPhysicalObjective
    extra=dict(target_R=torch.eye(3)[None]) if task=='place' else {}
    objective=cls(b,target,g,grasp['calibration'],**extra)
    validator=(MeshPlaceValidator if task=='place' else MeshPickValidator)(b,target,g,
        grasp['calibration'],collision=objective.collision,**extra)
    # Translation and particle positions stay consistent in this perturbation.
    p['t'][:,-2,0,2]+=.01
    p['positions'][:,-2,b.particle_body[0]==0,2]+=.01
    p['t'].requires_grad_()
    loss,diag=objective(p,b)
    grad,=torch.autograd.grad(loss,p['t'])
    assert torch.isfinite(grad).all() and grad.norm()>0
    actual=PhysicalConstraints(objective,task=task)(p)
    expected=validator(p)['residuals_m']
    for key,value in actual.items():
        assert float(value.detach())==pytest.approx(expected[key],abs=2e-7)
    metrics=objective.measure(p)
    assert 'mesh_penetration_max_mm' in metrics
    assert 'mesh_distance' in metrics['traces'] and 'gaps' in metrics['traces']


def test_sparse_surface_samples_cannot_falsely_certify_release_with_mesh_screen():
    b,g,p,grasp=resting_released_scene()
    # Keep only box-corner coverage for the object, duplicating those samples in
    # the existing slots. Every point remains on its actual collision surface.
    obj=(b.particle_body[0]==0).nonzero(as_tuple=True)[0]
    assert len(obj)==16
    b.positions[:,:,obj[8:]]=b.positions[:,:,obj[:8]].clone()
    b.body_templates[0,0,8:16]=b.body_templates[0,0,:8].clone()
    p['positions'][:,:,obj[8:]]=p['positions'][:,:,obj[:8]].clone()
    # Fingers are only 2 mm from the box faces, although the sampled corners are
    # much farther apart. The requested release clearance is 5 mm.
    p['t'][:,:,2,0]+=.01;p['t'][:,:,3,0]-=.01
    p['positions'][:,:,b.particle_body[0]==2,0]+=.01
    p['positions'][:,:,b.particle_body[0]==3,0]-=.01
    target=p['t'][:,-1,0].clone()
    assert PlaceValidator(b,target,g,grasp['calibration'])(p)['accepted']
    objective=MeshPlacePhysicalObjective(b,target,g,grasp['calibration'])
    check=MeshPlaceValidator(b,target,g,grasp['calibration'],collision=objective.collision)(p)
    assert not check['accepted']
    assert check['residuals_m']['release']==pytest.approx(.003,abs=1e-7)
    assert objective.terms(objective.features(p))['release']==pytest.approx(.003,abs=1e-7)
