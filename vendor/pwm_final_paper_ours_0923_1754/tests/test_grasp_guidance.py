"""Stage guidance must penalize separated holds, half-turns and brief violations."""
import math

import pytest
import torch

from guidance.grasp import StagedPickObjective, rotation_chord
from guidance.pick_validation import PickValidator, selection_key
from guidance.constraints import ConstrainedGraspObjective
from tests.test_pick_objective import scene


def calibrated_scene():
    b,g,p = scene()
    C = b.context_len
    pose = b.body_poses[0,C-1]
    rel = pose[[2,3],:3,3]-pose[1,:3,3]
    axis = torch.tensor([[-1.,0,0],[1.,0,0]])
    q = torch.tensor([.026,.026])
    cal = dict(q_anchor=q, finger_anchor=rel, axis=axis,
        pad_local=torch.tensor([[.004,0,0],[-.004,0,0]]),
        pad_normal_local=-axis, approach_local=torch.tensor([0.,0.,-1.]),
        limits=torch.tensor([[0.,.04,.2],[0.,.04,.2]]), dt=.05)
    proposal = dict(contacts=torch.tensor([[-.02,0,0],[.02,0,0]]), normals=axis,
        hand_R_object=torch.eye(3), hand_t_object=pose[1,:3,3]-pose[0,:3,3],
        q_open=q, q_close=q, approach_distance=.03)
    grasp = dict(calibration=cal, proposals=[proposal])
    o = StagedPickObjective(b,p['t'][:,-1,0],g,grasp,approach_frames=0,close_frame=0,hold_frames=3)
    return b,g,p,grasp,o


def test_rotation_chord_distinguishes_half_turn_and_has_finite_identity_gradient():
    R = torch.eye(3).requires_grad_()
    value = rotation_chord(R,torch.eye(3))
    grad, = torch.autograd.grad(value,R)
    assert value == 0 and torch.isfinite(grad).all()
    assert rotation_chord(torch.diag(torch.tensor([-1.,-1.,1.])),torch.eye(3)) == pytest.approx(2.)


def test_held_pose_has_zero_relative_motion_and_valid_joint_recovery():
    b,g,p,grasp,o = calibrated_scene()
    f = o.features(p)
    torch.testing.assert_close(f['q'],torch.full((1,6,2),.026),atol=1e-7,rtol=0)
    assert f['hold_translation_step'].max() < 1e-7
    assert f['hold_rotation_step'].max() < 1e-7
    assert f['normal_error'].max() < 1e-6
    assert f['joint_violation'].max() == 0


def test_separated_object_does_not_disable_required_contact_cost():
    b,g,p,grasp,o = calibrated_scene()
    base = o.stage_terms(o.features(p))['contact']
    bad = {k:v.clone() for k,v in p.items()}
    bad['t'][:,:,0,0] += .10
    bad['positions'][:,:,b.particle_body[0]==0,0] += .10
    bad['t'].requires_grad_()
    loss, diag = o(bad,b)
    grad, = torch.autograd.grad(loss,bad['t'])
    assert torch.isfinite(grad).all() and grad.norm() > 0
    assert o.stage_terms(o.features(bad))['contact'] > base+.07
    assert diag['constraint_contact_m'] > .08


def test_object_rotation_is_detected_with_unchanged_object_centre():
    b,g,p,grasp,o = calibrated_scene()
    p = {k:v.clone() for k,v in p.items()}
    p['R'][:,3:,0] = torch.diag(torch.tensor([-1.,-1.,1.]))
    f = o.features(p)
    assert f['hold_translation_step'].max() < 1e-7
    assert f['hold_rotation_step'].max() > 1.999
    assert o.measure(p)['stage_relative_rotation_step_deg'] > 30


def test_single_frame_violation_is_not_averaged_out_of_constraint():
    b,g,p,grasp,o = calibrated_scene()
    f = o.features(p)
    f['depth'].zero_()
    f['depth'][:,2] = .011
    assert o.residuals(f)['penetration'] == pytest.approx(.01)
    terms = o.stage_terms(f)
    assert terms['penetration'] > .011


def test_hold_must_fit_after_closure_and_invalid_settings_fail():
    b,g,p,grasp,o = calibrated_scene()
    bad = StagedPickObjective(b,p['t'][:,-1,0],g,grasp,approach_frames=2,close_frame=4,hold_frames=3)
    with pytest.raises(ValueError,match='complete hold'):
        bad.features(p)
    with pytest.raises(ValueError,match='Stage weights'):
        StagedPickObjective(b,p['t'][:,-1,0],g,grasp,contact_weight=math.nan)


def test_validator_acceptance_does_not_require_a_particular_grasp_proposal():
    b,g,p,grasp,o = calibrated_scene()
    validator = PickValidator(b,p['t'][:,-1,0],g,grasp['calibration'],hold_frames=2)
    valid = validator(p)
    assert valid['accepted'] and valid['interaction_accepted']
    assert valid['max_violation_m'] == 0
    # A goal-accurate isolated object must fail even when weighted reward improves.
    bad = {k:v.clone() for k,v in p.items()}
    bad['positions'][:,:,b.particle_body[0]>0,0] += .10
    bad['t'][:,:,1:,0] += .10
    invalid = validator(bad)
    assert not invalid['accepted'] and not invalid['interaction_accepted']
    assert selection_key(10.,valid) < selection_key(0.,invalid)


def test_validator_reports_goal_and_interaction_separately():
    b,g,p,grasp,o = calibrated_scene()
    validator = PickValidator(b,p['t'][:,-1,0]+torch.tensor([.1,0,0]),g,grasp['calibration'],hold_frames=2)
    result=validator(p)
    assert result['interaction_accepted'] and not result['accepted']
    assert result['residuals_m']['goal']>.07
    # Missing a task goal is preferable to a goal-accurate but separated object.
    invalid=dict(accepted=False,interaction_accepted=False,max_violation_m=.01,
        max_interaction_violation_m=.01)
    assert selection_key(.1,result)<selection_key(0.,invalid)


def test_invalid_fallback_ranking_uses_interaction_error_before_goal_error():
    closer_contact=dict(accepted=False,interaction_accepted=False,
        max_violation_m=.2,max_interaction_violation_m=.001)
    phantom_goal=dict(accepted=False,interaction_accepted=False,
        max_violation_m=.03,max_interaction_violation_m=.03)
    assert selection_key(.2,closer_contact)<selection_key(0.,phantom_goal)


def test_dual_updates_raise_violated_constraint_cost_without_changing_selection_score():
    b,g,p,grasp,o = calibrated_scene()
    p={k:v.clone() for k,v in p.items()}
    p['t'][:,:,0,0]+=.04
    p['positions'][:,:,b.particle_body[0]==0,0]+=.04
    p['t'].requires_grad_()
    constrained=ConstrainedGraspObjective(o,dual_every=1)
    first,diag=constrained(p,b)
    grad,=torch.autograd.grad(first,p['t'])
    assert torch.isfinite(grad).all() and grad.norm()>0
    constrained.after_step(diag,0)
    second,after=constrained(p,b)
    assert second>first
    assert constrained.multipliers['contact']>0
    assert after['selection_loss']==diag['selection_loss']


def test_feasible_constraints_do_not_accumulate_positive_dual_weights():
    b,g,p,grasp,o = calibrated_scene()
    constrained=ConstrainedGraspObjective(o,dual_every=1)
    loss,diag=constrained(p,b)
    constrained.after_step(diag,0)
    assert all(v==0 for v in constrained.multipliers.values())


def test_prefix_stage_cost_does_not_demand_the_final_goal_in_an_early_window():
    b,g,p,grasp,o=calibrated_scene()
    full=StagedPickObjective(b,p['t'][:,-1,0]+.1,g,grasp,
        approach_frames=2,close_frame=4,hold_frames=2)
    short={k:v[:,:2] if k!='rigid_idx' else v for k,v in p.items()}
    short['t']=short['t'].clone().requires_grad_()
    f=full.features(short,allow_prefix=True)
    terms=full.stage_terms(f,complete=False)
    assert terms['goal']==0 and terms['contact']==0 and terms['hold']==0
    loss=full.prefix_loss(short,complete=False)
    grad,=torch.autograd.grad(loss,short['t'])
    assert torch.isfinite(loss) and torch.isfinite(grad).all()
