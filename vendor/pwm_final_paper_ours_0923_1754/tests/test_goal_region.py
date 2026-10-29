"""Region goals express a different task while preserving physical constraints."""
import pytest
import torch

from guidance.goal_region import PositionGoalRegion,PositionRegionValidator
from guidance.physical_constraints import PhysicalConstraints
from guidance.physical_objective import PickPhysicalObjective
from guidance.place_objective import PlacePhysicalObjective
from guidance.pick_validation import PickValidator
from tests.test_physical_guidance import resting_released_scene


def test_minimum_height_and_workspace_region_has_correct_distance_and_gradient():
    region=PositionGoalRegion([-.1,-.1,.1],[.1,.1,float('inf')])
    positions=torch.tensor([[0.,0.,.3],[.13,0.,.06],[0.,0.,.1]],requires_grad=True)
    distance=region.distance(positions)
    torch.testing.assert_close(distance,torch.tensor([0.,.05,0.]))
    violation=region.signed_violation(positions)
    torch.testing.assert_close(violation,torch.tensor([-.1,.04,0.]))
    gradient,=torch.autograd.grad(distance.sum(),positions)
    torch.testing.assert_close(gradient,torch.tensor([[0.,0.,0.],[.6,0.,-.8],[0.,0.,0.]]))
    assert torch.isfinite(gradient).all()


def test_region_physical_objective_and_screen_distinguish_point_target():
    b,g,p,grasp=resting_released_scene()
    # All positions are inside the region, far from the unrelated nominal point.
    target=p['t'][:,-1,0].clone()+torch.tensor([1.,0.,0.])
    center=p['t'][0,-1,0].clone()
    lower=center-torch.tensor([.1,.1,.01]);upper=center+torch.tensor([.1,.1,.5])
    region=PositionGoalRegion(lower,upper)
    objective=PickPhysicalObjective(b,target,g,grasp['calibration'],goal_frames=6,goal_region=region)
    assert objective.terms(objective.features(p))['goal']==0
    _,diag=objective(p,b)
    assert diag['goal_cost_mm']==0 and diag['target_distance_mm']==pytest.approx(1000.,abs=.001)
    base=PickValidator(b,target,g,grasp['calibration'])
    region_validator=PositionRegionValidator(base,region)
    before,after=base(p),region_validator(p)
    assert before['residuals_m']['goal']>.9 and after['residuals_m']['goal']<0
    assert before['interaction_accepted']==after['interaction_accepted']
    for key in before['residuals_m']:
        if key!='goal':assert before['residuals_m'][key]==after['residuals_m'][key]
    residuals=PhysicalConstraints(objective)(p)
    assert float(residuals['goal'])==pytest.approx(after['residuals_m']['goal'])
    # A single earlier held frame outside the workspace invalidates the region.
    p['t'][:,-3,0,0]=upper[0]+.03
    p['t'].requires_grad_()
    goal=objective.terms(objective.features(p))['goal']
    assert float(goal.detach())==pytest.approx(.005,abs=1e-7)
    assert region_validator(p)['residuals_m']['goal']==pytest.approx(.03,abs=1e-7)
    gradient,=torch.autograd.grad(goal,p['t'])
    assert gradient[0,-3,0,0]>0 and torch.isfinite(gradient).all()


@pytest.mark.parametrize('lower,upper',[
    ([0,0],[1,1,1]),([0,0,0],[1,1]),
    ([0,0,float('nan')],[1,1,1]),([0,0,0],[1,1,float('nan')]),
    ([0,0,0],[1,1,0]),([0,0,2],[1,1,1]),
    ([0,0,float('inf')],[1,1,float('inf')]),
    ([float('-inf')]*3,[float('inf')]*3)])
def test_invalid_or_empty_region_is_rejected(lower,upper):
    with pytest.raises(ValueError):PositionGoalRegion(lower,upper)


def test_region_bounds_are_fixed_task_inputs_and_require_complete_hold():
    lower=torch.tensor([-.1,-.1,.1],requires_grad=True)
    region=PositionGoalRegion(lower,[.1,.1,float('inf')])
    assert not region.lower.requires_grad
    with torch.no_grad():lower.add_(1)
    assert float(region.lower[0])==pytest.approx(-.1)
    validator=PositionRegionValidator(lambda _:None,region,hold_frames=6)
    with pytest.raises(ValueError,match='cover'):
        validator(dict(t=torch.zeros(1,5,4,3),rigid_idx=torch.arange(4)))


def test_placing_cannot_silently_override_a_region_with_its_point_goal():
    b,g,p,grasp=resting_released_scene()
    with pytest.raises(ValueError,match='picking only'):
        PlacePhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'],
            goal_region=PositionGoalRegion([-.1,-.1,.1],[.1,.1,float('inf')]))
