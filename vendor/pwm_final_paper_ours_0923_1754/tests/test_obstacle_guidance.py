"""Full-body obstacle distances, pose derivatives and between-frame collisions."""
import copy

import pytest
import torch

pytest.importorskip('fcl')
pytest.importorskip('scipy')

from guidance.obstacles import (box_geometry, validate_obstacles, interpolate_poses,
    StaticObstacleDistance, ObstacleObjective, ObstacleValidator)
from tests.test_physical_guidance import resting_released_scene
from guidance.mesh_physical import MeshPickPhysicalObjective, MeshPickValidator


def spec(centre=(0., 0., 0.), size=(1., 1., 1.)):
    return dict(version=1, units='metres', boxes=[dict(name='block', centre=list(centre), size=list(size))])


@pytest.mark.parametrize('separation,expected', [(1.4, .4), (.7, -.3)])
def test_static_box_signed_distance_full_body_and_translation_derivative(separation, expected):
    distance=StaticObstacleDistance(dict(bodies={7:box_geometry([1.,1.,1.])}),spec())
    R=torch.eye(3,dtype=torch.float64).expand(1,1,1,3,3)
    t=torch.tensor([[[[separation,.11,.17]]]],dtype=torch.float64,requires_grad=True)
    value=distance(R,t,torch.tensor([7]))
    assert value.shape==(1,1,1,1)
    assert float(value.detach())==pytest.approx(expected,abs=1e-7)
    grad,=torch.autograd.grad(value.sum(),t)
    torch.testing.assert_close(grad,torch.tensor([[[[1.,0,0]]]],dtype=torch.float64),atol=1e-6,rtol=0)


def test_dense_probe_detects_obstacle_crossing_with_clear_endpoints():
    distance=StaticObstacleDistance(dict(bodies={0:box_geometry([.2,.2,.2])}),spec(size=(.2,.2,.2)))
    R=torch.eye(3,dtype=torch.float64).expand(1,2,1,3,3)
    t=torch.tensor([[[[-.5,.013,.017]],[[.5,.013,.017]]]],dtype=torch.float64)
    assert distance(R,t,torch.tensor([0])).min()>.29
    assert distance(R,t,torch.tensor([0]),substeps=4).min()<-.17


def test_rotation_sweep_detects_collision_and_retains_endpoint_bits():
    # Long thin bar swings through an obstacle; origin and both end poses clear.
    from utils.pose_ops import so3_exp
    distance=StaticObstacleDistance(dict(bodies={0:box_geometry([1.,.04,.04])}),
        spec(centre=(.4,0.,0.),size=(.08,.08,.08)))
    angles=torch.tensor([-.8,.8],dtype=torch.float64,requires_grad=True)
    rv=torch.stack([angles*0,angles*0,angles],-1)
    R=so3_exp(rv)[None,:,None];t=torch.zeros(1,2,1,3,dtype=torch.float64)
    dense_R,dense_t=interpolate_poses(R,t,4)
    assert torch.equal(dense_R[:,::4],R) and torch.equal(dense_t[:,::4],t)
    assert distance(R,t,torch.tensor([0])).min()>.1
    value=distance(R,t,torch.tensor([0]),substeps=4).min()
    assert value<-.05
    grad,=torch.autograd.grad(value,angles)
    assert torch.isfinite(grad).all()


def test_dense_rotation_and_translation_gradients_match_finite_differences():
    from utils.pose_ops import so3_exp
    distance=StaticObstacleDistance(dict(bodies={0:box_geometry([.8,.3,.2])}),
        spec(centre=(1.1,.21,.17),size=(.2,.2,.2)))
    x=torch.tensor([.12,.31,.05],dtype=torch.float64,requires_grad=True)
    def evaluate(v):
        rv=torch.stack([v[:2]*0,v[:2]*0,v[:2]],-1)
        R=so3_exp(rv)[None,:,None]
        t=torch.stack([torch.zeros_like(v),torch.stack([v[2],v[2]*0,v[2]*0])])[None,:,None]
        return distance(R,t,torch.tensor([0]),substeps=3).sum()
    grad,=torch.autograd.grad(evaluate(x),x)
    eps=1e-5
    finite=torch.stack([(evaluate(x.detach()+torch.eye(3,dtype=x.dtype)[i]*eps)-
                         evaluate(x.detach()-torch.eye(3,dtype=x.dtype)[i]*eps))/(2*eps) for i in range(3)])
    torch.testing.assert_close(grad,finite,atol=2e-5,rtol=1e-3)


def test_obstacle_query_covers_every_body_and_respects_body_reordering():
    distance=StaticObstacleDistance(dict(bodies={2:box_geometry([.1,.1,.1]),9:box_geometry([.1,.1,.1])}),
        spec(centre=(0.,0.,0.),size=(.1,.1,.1)))
    R=torch.eye(3,dtype=torch.float64).expand(1,1,2,3,3)
    t=torch.tensor([[[[.6,0,0],[.02,.01,.007]]]],dtype=torch.float64)
    value=distance(R,t,torch.tensor([2,9]))
    assert value[0,0,0,0]>.49 and value[0,0,1,0]<0
    torch.testing.assert_close(value,distance(R.flip(2),t.flip(2),torch.tensor([9,2])),rtol=0,atol=1e-8)


def test_zero_obstacle_weight_preserves_loss_but_audit_still_rejects_collision():
    b,g,p,grasp=resting_released_scene();target=p['t'][:,-1,0].clone()
    centre=b.body_poses[0,b.context_len-1,1,:3,3].tolist()
    distance=StaticObstacleDistance(g,spec(centre=centre,size=(.1,.1,.1)))
    base=MeshPickPhysicalObjective(b,target,g,grasp['calibration'])
    wrapped=ObstacleObjective(base,distance,weight=0.)
    torch.testing.assert_close(base(p,b)[0],wrapped(p,b)[0],rtol=0,atol=0)
    validator=MeshPickValidator(b,target,g,grasp['calibration'],collision=base.collision)
    result=ObstacleValidator(validator,b,distance)(p)
    assert result['residuals_m']['obstacle']>0
    assert not result['accepted'] and not result['interaction_accepted']
    assert len(result['obstacle_clearance_by_body_mm'])==4


def test_physical_ablation_keeps_independent_obstacle_cost_and_gradient():
    b,g,p,grasp=resting_released_scene();target=p['t'][:,-1,0].clone()
    centre=b.body_poses[0,b.context_len-1,1,:3,3]+torch.tensor([.03,.011,.017])
    distance=StaticObstacleDistance(g,spec(centre=centre.tolist(),size=(.1,.1,.1)))
    base=MeshPickPhysicalObjective(b,target,g,grasp['calibration'],physical_weight=0.)
    wrapped=ObstacleObjective(base,distance,weight=5.)
    p['t'].requires_grad_();loss,diag=wrapped(p,b)
    assert base(p,b)[0]==0 and loss>0 and diag['obstacle_cost_mm']>0
    grad,=torch.autograd.grad(loss,p['t'])
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad[:,:,1:])>0


@pytest.mark.parametrize('bad',['units','negative_size','reflection','nan','duplicate'])
def test_invalid_obstacle_spec_is_rejected(bad):
    s=spec()
    if bad=='units':s['units']='centimetres'
    if bad=='negative_size':s['boxes'][0]['size'][0]=-1
    if bad=='reflection':s['boxes'][0]['rotation']=[[-1,0,0],[0,1,0],[0,0,1]]
    if bad=='nan':s['boxes'][0]['centre'][0]=float('nan')
    if bad=='duplicate':s['boxes'].append(copy.deepcopy(s['boxes'][0]))
    with pytest.raises(ValueError):validate_obstacles(s)


def test_box_rotation_is_world_geometry_not_an_axis_aligned_envelope():
    s=spec(size=(1.,.1,.1));s['boxes'][0]['rotation']=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    query=StaticObstacleDistance(dict(bodies={0:box_geometry([.1,.1,.1])}),s)
    R=torch.eye(3,dtype=torch.float64).expand(1,1,1,3,3)
    t=torch.tensor([[[[.3,0.,0.]]]],dtype=torch.float64)
    assert float(query(R,t,torch.tensor([0])))==pytest.approx(.2,abs=1e-7)
