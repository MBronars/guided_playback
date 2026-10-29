"""Peak penalties and joint constraints do not depend on grasp proposals."""
import torch
import pytest

from guidance.constraints import ConstrainedGraspObjective
from guidance.physical_constraints import PhysicalConstraints
from guidance.physical_objective import PickPhysicalObjective
from guidance.physical_objective import PhysicalTermSubset
from guidance.place_objective import PlacePhysicalObjective, PlaceValidator
from guidance.pick_validation import PickValidator
from tests.test_grasp_guidance import calibrated_scene


@pytest.mark.parametrize('cls', [PickPhysicalObjective, PlacePhysicalObjective])
@pytest.mark.parametrize('reduction', ['mean', 'max'])
def test_physical_subsets_keep_goal_gradients_and_do_not_query_disabled_geometry(cls, reduction):
    b,g,p,grasp=resting_released_scene()
    p['t'].requires_grad_()
    target=p['t'][:,-1,0].detach()+torch.tensor([.08,.03,.01])
    extra=dict(target_R=torch.eye(3)[None]) if cls is PlacePhysicalObjective else {}
    base=cls(b,target,g,grasp['calibration'],goal_frames=6,goal_reduction=reduction,**extra)
    full,_=base(p,b)
    subset,_=PhysicalTermSubset(base,set(base.weights)-{'rotation'})(p,b)
    torch.testing.assert_close(full,subset,atol=0,rtol=0)
    base.physical_weight=0.
    reference,_=base(p,b)
    expected,=torch.autograd.grad(reference,p['t'],retain_graph=True)
    base.physical_weight=1.
    def forbidden(*_):raise AssertionError('Goal-only evaluation queried geometry')
    base.features=forbidden
    actual,_=PhysicalTermSubset(base,[])(p,b)
    gradient,=torch.autograd.grad(actual,p['t'])
    torch.testing.assert_close(actual,reference,atol=1e-7,rtol=0)
    torch.testing.assert_close(gradient,expected,atol=1e-7,rtol=0)


def test_physical_subset_removes_only_declared_terms_and_rejects_typos():
    b,g,p,grasp=resting_released_scene()
    p['t'][:,2,2,0]+=.08
    p['t'].requires_grad_()
    base=PickPhysicalObjective(b,p['t'][:,-1,0].detach()+.1,g,grasp['calibration'],goal_frames=6)
    full,_=base(p,b)
    terms=base.terms(base.features(p))
    reduced,_=PhysicalTermSubset(base,set(base.weights)-{'joint','joint_speed'})(p,b)
    removed=base.weights['joint']*terms['joint']+base.weights['joint_speed']*terms['joint_speed']
    torch.testing.assert_close(full-reduced,removed)
    grad,=torch.autograd.grad(full-reduced-removed,p['t'])
    torch.testing.assert_close(grad,torch.zeros_like(grad),atol=1e-6,rtol=0)
    with pytest.raises(ValueError,match='Unknown physical terms'):
        PhysicalTermSubset(base,['joint_speeed'])


@pytest.mark.parametrize('cls', [PickPhysicalObjective, PlacePhysicalObjective])
def test_goal_weight_changes_only_position_cost_and_its_gradient(cls):
    b,g,p,grasp=resting_released_scene()
    target=p['t'][:,-1,0].clone()+torch.tensor([.08,0,0])
    p['t'].requires_grad_()
    base=cls(b,target,g,grasp['calibration'])
    weighted=cls(b,target,g,grasp['calibration'],goal_weight=3.)
    a,da=base(p,b);c,dc=weighted(p,b)
    term=base.terms(base.features(p))['goal']
    torch.testing.assert_close(c-a,2*term)
    actual,=torch.autograd.grad(c-a,p['t'],retain_graph=True)
    expected,=torch.autograd.grad(2*term,p['t'])
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
    assert da==dc  # Raw measurements retain their physical units.
    assert base.weights==weighted.weights


@pytest.mark.parametrize('value', [-1.,float('nan'),float('inf')])
def test_goal_weights_reject_invalid_values(value):
    b,g,p,grasp=resting_released_scene()
    with pytest.raises(ValueError,match='Goal weight'):
        PickPhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'],goal_weight=value)


@pytest.mark.parametrize('cls', [PickPhysicalObjective, PlacePhysicalObjective])
def test_physical_ablation_preserves_position_rotation_goals_and_raw_diagnostics(cls):
    b,g,p,grasp=resting_released_scene()
    target=p['t'][:,-1,0].clone()+torch.tensor([.08,0,0])
    p['t'][:,2,2,0]+=.08  # Impossible finger excursion supplies a physical penalty.
    p['R'][:,:,0]=torch.diag(torch.tensor([-1.,-1.,1.]))
    p['t'].requires_grad_()
    extra=dict(target_R=torch.eye(3)[None]) if cls is PlacePhysicalObjective else {}
    base=cls(b,target,g,grasp['calibration'],goal_frames=6,**extra)
    ablated=cls(b,target,g,grasp['calibration'],goal_frames=6,physical_weight=0.,**extra)
    a,da=base(p,b);z,dz=ablated(p,b)
    expected=.08+(torch.pi*.04 if cls is PlacePhysicalObjective else 0.)
    assert float(z.detach())==pytest.approx(expected,abs=1e-7)
    assert a>z and da==dz
    grad,=torch.autograd.grad(z,p['t'])
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad[:,:,1:])==0
    assert torch.all(grad[:,-6:,0,0]<0)
    terms=base.terms(base.features(p))
    # Preserve the precise default evaluation order and scalar value.
    torch.testing.assert_close(a,terms['goal']+sum(base.weights[k]*terms[k] for k in base.weights),atol=0,rtol=0)


@pytest.mark.parametrize('value', [-1.,float('nan'),float('inf')])
def test_physical_scale_rejects_invalid_values(value):
    b,g,p,grasp=resting_released_scene()
    with pytest.raises(ValueError,match='Physical weight'):
        PickPhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'],physical_weight=value)


def test_physical_cost_penalizes_a_brief_collision_and_excess_joint_speed():
    b,g,p,grasp,_=calibrated_scene()
    objective=PickPhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'])
    f=objective.features(p)
    assert f['joint_violation'].max()==0
    assert f['joint_speed_violation'].max()==0
    f['inside_object'].zero_();f['inside_gripper_hull'].zero_()
    f['inside_object'][:,2,0]=.011
    assert objective.terms(f)['penetration']>.01
    altered={k:v.clone() for k,v in p.items()}
    altered['t'][:,2,2]+=torch.tensor([-.03,0,0])
    altered['t'].requires_grad_()
    loss,_=objective(altered,b)
    grad,=torch.autograd.grad(loss,altered['t'])
    assert torch.isfinite(grad).all() and float(grad.norm())>0
    terms=objective.terms(objective.features(altered))
    assert terms['joint']>.01 and terms['joint_speed']>.015


def test_physical_cost_uses_only_calibration_and_has_no_closure_horizon_requirement():
    b,g,p,grasp,_=calibrated_scene()
    objective=PickPhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'])
    short={k:(v[:,:2] if k!='rigid_idx' else v) for k,v in p.items()}
    loss,diag=objective(short,b)
    assert torch.isfinite(loss)
    assert 'joint_cost_mm' in diag and not any(k.startswith('stage_') for k in diag)


def resting_released_scene():
    b,g,p,grasp,_=calibrated_scene()
    C=b.context_len
    # Stationary object on support; fingers open 10 mm from their observed q.
    p={k:v.clone() for k,v in p.items()}
    p['t'][:]=b.body_poses[:,C-1:C,:,:3,3]
    p['R'][:]=b.body_poses[:,C-1:C,:,:3,:3]
    p['t'][:,:,2,0]-=.01;p['t'][:,:,3,0]+=.01
    for k in range(4):
        rows=b.particle_body[0]==k
        p['positions'][:,:,rows]=b.positions[:,C-1:C,rows]+(p['t'][:,:,k]-b.body_poses[:,C-1:C,k,:3,3])[:,:,None]
    return b,g,p,grasp


def test_placing_accepts_stationary_released_object_and_rejects_hovering():
    b,g,p,grasp=resting_released_scene();target=p['t'][:,-1,0].clone()
    v=PlaceValidator(b,target,g,grasp['calibration'],target_R=torch.eye(3)[None])
    assert v(p)['accepted']
    assert not PickValidator(b,target,g,grasp['calibration'])(p)['accepted']
    bad={k:value.clone() for k,value in p.items()}
    bad['t'][:,:,0,2]+=.04;bad['positions'][:,:,b.particle_body[0]==0,2]+=.04
    result=v(bad)
    assert not result['accepted'] and result['residuals_m']['rest']>.03


def test_placing_cost_allows_release_and_detects_wrong_rotation():
    b,g,p,grasp=resting_released_scene()
    o=PlacePhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'],target_R=torch.eye(3)[None])
    f=o.features(p); terms=o.terms(f)
    assert terms['contact']==0 and terms['release']==0 and terms['rest']<1e-7
    _,diag=o(p,b)
    assert 'rotation_cost_mm' not in diag and 'rotation_cost_rad' in diag
    p['R'][:,:,0]=torch.diag(torch.tensor([-1.,-1.,1.]))
    assert o.terms(o.features(p))['rotation']>3.14


def test_placing_progress_distinguishes_endpoint_from_held_goal_cost():
    b,g,p,grasp=resting_released_scene()
    target=p['t'][:,-1,0].clone()
    p['t'][:,-6:-1,0,0]+=.012
    p['positions'][:,-6:-1,b.particle_body[0]==0,0]+=.012
    objective=PlacePhysicalObjective(b,target,g,grasp['calibration'])
    loss,diag=objective(p,b)
    assert diag['target_distance_mm']==pytest.approx(0.,abs=1e-6)
    assert diag['goal_cost_mm']==pytest.approx(10.,abs=1e-5)
    terms=objective.terms(objective.features(p))
    # Reporting does not change the held-goal objective or its physical costs.
    expected=terms['goal']+sum(objective.weights[k]*terms[k] for k in objective.weights)
    torch.testing.assert_close(loss,expected,rtol=0,atol=0)
    assert objective.measure(p)['target_distance_mm']==diag['target_distance_mm']


def test_pick_hold_objective_reaches_earlier_frames_without_changing_endpoint_metric():
    b,g,p,grasp=resting_released_scene()
    target=p['t'][:,-1,0].clone()
    p['t'][:,-6:-1,0,0]+=.012
    p['t'].requires_grad_()
    terminal=PickPhysicalObjective(b,target,g,grasp['calibration'])
    held=PickPhysicalObjective(b,target,g,grasp['calibration'],goal_frames=6)
    final_term=terminal.terms(terminal.features(p))['goal']
    held_term=held.terms(held.features(p))['goal']
    assert float(final_term.detach())==pytest.approx(0.,abs=1e-7)
    assert float(held_term.detach())==pytest.approx(.010,abs=1e-7)
    gradient,=torch.autograd.grad(held_term,p['t'])
    assert torch.all(gradient[:,-6:-1,0,0]>0)
    assert torch.count_nonzero(gradient[:,:-6])==0
    _,diag=held(p,b)
    assert diag['target_distance_mm']==0 and diag['goal_cost_mm']==pytest.approx(10.,abs=1e-5)
    # The task validator still requires the maximum over six frames, independent
    # of whether the optimizer uses terminal or mean-held goal costs.
    expected=PickValidator(b,target,g,grasp['calibration'])(p)['residuals_m']['goal']
    assert float(PhysicalConstraints(held)(p)['goal'].detach())==pytest.approx(expected,abs=1e-7)


@pytest.mark.parametrize('frames',[0,-1,True,1.5])
def test_pick_goal_interval_rejects_invalid_lengths(frames):
    b,g,p,grasp=resting_released_scene()
    with pytest.raises(ValueError,match='positive integer'):
        PickPhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'],goal_frames=frames)


def test_pick_goal_interval_cannot_silently_shrink_to_short_prediction():
    b,g,p,grasp=resting_released_scene()
    objective=PickPhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'],goal_frames=p['t'].shape[1]+1)
    with pytest.raises(ValueError,match='cover'):
        objective(p,b)


@pytest.mark.parametrize('cls',[PickPhysicalObjective,PlacePhysicalObjective])
def test_worst_frame_position_goal_matches_hold_screen_and_directs_gradient(cls):
    b,g,p,grasp=resting_released_scene();target=p['t'][:,-1,0].clone()
    p['t'][:,-6:,0,0]+=torch.tensor([.010,.031,.009,.008,.007,.006])
    p['t'].requires_grad_()
    mean=cls(b,target,g,grasp['calibration'],goal_frames=6)
    worst=cls(b,target,g,grasp['calibration'],goal_frames=6,goal_reduction='max')
    a=mean.terms(mean.features(p));c=worst.terms(worst.features(p))
    assert float(a['goal'].detach())==pytest.approx(.071/6,abs=1e-7)
    assert float(c['goal'].detach())==pytest.approx(.031,abs=1e-7)
    for key in a:
        if key!='goal':torch.testing.assert_close(a[key],c[key],atol=0,rtol=0)
    grad,=torch.autograd.grad(c['goal'],p['t'])
    assert grad[0,-5,0,0]==1 and torch.count_nonzero(grad)==1
    screen=PhysicalConstraints(worst,task='place' if cls is PlacePhysicalObjective else 'pick')(p)
    torch.testing.assert_close(c['goal']-.020,screen['goal'],atol=1e-7,rtol=0)


@pytest.mark.parametrize('task', ['pick', 'place'])
def test_differentiable_constraints_match_independent_geometry_screen(task):
    b,g,p,grasp=resting_released_scene()
    target=p['t'][:,-1,0].clone()+torch.tensor([.03,0,0])
    cls=PlacePhysicalObjective if task=='place' else PickPhysicalObjective
    extra=dict(target_R=torch.eye(3)[None]) if task=='place' else {}
    objective=cls(b,target,g,grasp['calibration'],**extra)
    validator=(PlaceValidator if task=='place' else PickValidator)(b,target,g,grasp['calibration'],**extra)
    # Introduce a transient penetration, an impossible finger opening and motion
    # during the final hold. All must survive the differentiable reduction.
    p['t'][:,2,2,0]+=.06
    p['t'][:,-2,0,2]+=.01
    p['positions'][:,-2,b.particle_body[0]==0,2]+=.01
    p['t'].requires_grad_();p['positions'].requires_grad_()
    constraints=PhysicalConstraints(objective,task=task)
    actual=constraints(p)
    expected=validator(p)['residuals_m']
    assert actual.keys()==expected.keys()
    for key,value in actual.items():
        assert float(value.detach())==pytest.approx(expected[key],abs=2e-8)
    loss=sum(v.clamp_min(0).square() for v in actual.values())
    grads=torch.autograd.grad(loss,(p['t'],p['positions']))
    assert all(torch.isfinite(v).all() for v in grads)
    # Released placing particles already satisfy proximity constraints; their
    # hinge gradient may be zero. Pose/joint violations must still have a route.
    assert grads[0].norm()>0


def test_physical_augmented_lagrangian_updates_duals_without_grasp_proposals():
    b,g,p,grasp=resting_released_scene()
    target=p['t'][:,-1,0]+torch.tensor([.05,0,0])
    base=PlacePhysicalObjective(b,target,g,grasp['calibration'])
    objective=ConstrainedGraspObjective(base,dual_every=1,
        residual_fn=PhysicalConstraints(base,task='place'))
    first,diag=objective(p,b)
    objective.after_step(diag,0)
    second,updated=objective(p,b)
    assert objective.multipliers['goal']>0 and objective.multipliers['rest']==0
    assert second>first
    assert updated['selection_loss']==diag['selection_loss']
    assert updated['constraint_goal_m']==pytest.approx(.03,abs=1e-7)


def test_released_object_does_not_hide_gripper_below_table():
    b,g,p,grasp=resting_released_scene()
    # Leave speed headroom instead of placing a floating-point comparison
    # exactly at the joint's per-frame motion limit.
    for body,dx in ((2,.001),(3,-.001)):
        p['t'][:,:,body,0]+=dx
        p['positions'][:,:,b.particle_body[0]==body,0]+=dx
    target=p['t'][:,-1,0].clone()
    validator=PlaceValidator(b,target,g,grasp['calibration'])
    assert validator(p)['accepted']
    # Move the whole gripper away from the object and under the table, keeping
    # its articulated opening intact. The object remains correctly released.
    shift=torch.tensor([.5,0,-.3])
    p['t'][:,:,1:]+=shift
    p['positions'][:,:,b.particle_body[0]!=0]+=shift
    result=validator(p)
    assert not result['accepted']
    assert result['residuals_m']['gripper_support']>.1
    assert all(v<=0 for k,v in result['residuals_m'].items() if k!='gripper_support')
    objective=PlacePhysicalObjective(b,target,g,grasp['calibration'])
    p['t'].requires_grad_()
    terms=objective.terms(objective.features(p))
    gradient,=torch.autograd.grad(terms['gripper_support'],p['t'])
    assert torch.isfinite(gradient).all()
    assert gradient[:,:,1:,2].sum()<0  # Descent moves the gripper upward.
    constraints=PhysicalConstraints(objective,task='place')(p)
    assert float(constraints['gripper_support'].detach())==pytest.approx(result['residuals_m']['gripper_support'],abs=1e-7)


def test_gripper_support_uses_collision_mesh_and_body_column_mapping():
    b,g,p,grasp=resting_released_scene()
    objective=PickPhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'])
    original=objective.features(p)['gripper_bottom']
    # Particle rows are deliberately unmodified; support comes from mesh poses.
    p['t'][:,:,2,2]-=.02
    altered=objective.features(p)['gripper_bottom']
    torch.testing.assert_close(altered[:,1:,1],original[:,1:,1]-.02)
    order=torch.tensor([3,1,0,2])
    reordered=dict(p,R=p['R'][:,:,order],t=p['t'][:,:,order],rigid_idx=order)
    torch.testing.assert_close(objective.features(reordered)['gripper_bottom'],altered)


@pytest.mark.parametrize('task',['pick','place'])
def test_tolerance_scaled_constraints_keep_raw_screen_and_dual_units_consistent(task):
    b,g,p,grasp=resting_released_scene()
    target=p['t'][:,-1,0]+torch.tensor([.05,0,0])
    base=(PickPhysicalObjective if task=='pick' else PlacePhysicalObjective)(b,target,g,grasp['calibration'])
    residuals=PhysicalConstraints(base,task=task)
    scales=residuals.tolerance_scales()
    assert scales.keys()==residuals(p).keys()
    assert scales['goal']==.020 and scales['gripper_support']==.003
    wrapped=ConstrainedGraspObjective(base,penalty=.02,dual_every=1,residual_fn=residuals,residual_scales=scales)
    first,diag=wrapped(p,b)
    assert diag['constraint_goal_m']==pytest.approx(.03,abs=1e-7)
    assert diag['constraint_goal_scaled']==pytest.approx(1.5,abs=1e-6)
    raw=residuals(p)
    expected=sum(.01*(v/scales[k]).clamp_min(0).square() for k,v in raw.items())
    assert float(first-base(p,b)[0])==pytest.approx(float(expected),abs=1e-7)
    wrapped.after_step(diag,0)
    assert wrapped.multipliers['goal']==pytest.approx(.03,abs=1e-7)
    second,updated=wrapped(p,b)
    assert second>first and updated['selection_loss']==diag['selection_loss']
    p['t'].requires_grad_()
    gradient,=torch.autograd.grad(wrapped(p,b)[0],p['t'])
    assert torch.isfinite(gradient).all() and gradient.norm()>0


def test_constraint_scaling_rejects_missing_or_invalid_units():
    b,g,p,grasp=resting_released_scene()
    base=PlacePhysicalObjective(b,p['t'][:,-1,0],g,grasp['calibration'])
    residuals=PhysicalConstraints(base,task='place')
    with pytest.raises(ValueError,match='positive'):
        ConstrainedGraspObjective(base,residual_scales={'goal':0})
    wrapped=ConstrainedGraspObjective(base,residual_fn=residuals,residual_scales={'goal':.02})
    with pytest.raises(ValueError,match='exactly'):
        wrapped(p,b)
