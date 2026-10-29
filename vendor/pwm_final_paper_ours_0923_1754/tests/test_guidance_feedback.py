"""Feedback must carry observed rigid histories without stale future or joint state."""
from dataclasses import replace

import pytest
import torch

from guidance.feedback import rigid_feedback_batch, rigid_planning_batch, reanchor_finger_calibration
from guidance.physical_objective import PickPhysicalObjective
from tests.test_articulated_bodies import gripper
from tests.test_grasp_guidance import calibrated_scene
from utils.pose_native import materialize_pose_native


def test_longer_planning_horizon_preserves_observations_and_discards_all_future():
    batch=gripper(second='prismatic')[0].slice(1)
    C=batch.context_len
    # Observed particles need not be reconstructed or quantized to extend time.
    batch.positions[:,:C,0,0]+=.000003
    poisoned=replace(batch,positions=batch.positions.clone(),body_poses=batch.body_poses.clone(),
        actions=torch.full_like(batch.actions,float('nan')),
        pose_actions=torch.full_like(batch.pose_actions,float('nan')),
        pose_deltas_gt=torch.full_like(batch.pose_deltas_gt,float('nan')))
    poisoned.positions[:,C:]=float('nan');poisoned.body_poses[:,C:]=float('nan')
    before=torch.get_rng_state()
    short=rigid_planning_batch(batch,horizon=16)
    long=rigid_planning_batch(poisoned,horizon=68)
    assert torch.equal(torch.get_rng_state(),before)
    assert torch.equal(long.context,batch.context)
    assert torch.equal(long.body_poses[:,:C],batch.body_poses[:,:C])
    for key in ('positions','body_poses','actions','pose_actions','pose_deltas_gt'):
        a,b=getattr(short,key),getattr(long,key)
        assert torch.equal(a,b[:,:a.shape[1]]) and torch.isfinite(b).all()
    assert long.num_future_frames==68
    assert torch.count_nonzero(long.pose_deltas_gt)==torch.count_nonzero(long.pose_actions)==0
    assert torch.equal(long.future[:,-1],batch.context[:,-1])
    assert long.body_templates is batch.body_templates and long.pose_action_mask is batch.pose_action_mask


def test_feedback_rebuilds_observed_history_and_cannot_copy_recorded_future():
    batch = gripper(second='prismatic')[0].slice(1)
    C = batch.context_len
    observed = batch.body_poses[:, -C:].clone()
    R, t = observed[..., :3, :3], observed[..., :3, 3]
    original_positions, original_poses = batch.positions.clone(), batch.body_poses.clone()
    poisoned_positions, poisoned_poses = batch.positions.clone(), batch.body_poses.clone()
    poisoned_positions[:, C:] = float('nan')
    poisoned_poses[:, C:] = float('nan')
    poisoned = replace(batch, positions=poisoned_positions, body_poses=poisoned_poses,
                       actions=torch.full_like(batch.actions, float('nan')),
                       pose_actions=torch.full_like(batch.pose_actions, float('nan')),
                       pose_deltas_gt=torch.full_like(batch.pose_deltas_gt, float('nan')))
    before = torch.get_rng_state()
    feedback = rigid_feedback_batch(batch, R, t, horizon=7)
    other = rigid_feedback_batch(poisoned, R, t, horizon=7)
    assert torch.equal(torch.get_rng_state(), before)
    for field in ('positions', 'body_poses', 'actions', 'pose_actions', 'pose_deltas_gt'):
        assert torch.equal(getattr(feedback, field), getattr(other, field))
        assert torch.isfinite(getattr(feedback, field)).all()
    torch.testing.assert_close(feedback.context, original_positions[:, -C:], rtol=0, atol=2e-6)
    torch.testing.assert_close(feedback.body_poses[:, :C], observed, rtol=0, atol=0)
    assert feedback.rho is batch.rho and feedback.body_parent is batch.body_parent
    assert feedback.body_templates is batch.body_templates
    assert torch.equal(batch.positions, original_positions) and torch.equal(batch.body_poses, original_poses)
    assert torch.count_nonzero(feedback.actions) == torch.count_nonzero(feedback.pose_deltas_gt) == 0
    normalized = feedback.pose_delta_stats.normalize(feedback.pose_deltas_gt, clip=False)
    particles, *_ = materialize_pose_native(feedback, None, normalized)
    torch.testing.assert_close(particles, feedback.future, rtol=0, atol=2e-6)


def test_reanchoring_preserves_joint_coordinates_without_spurious_first_step_speed():
    batch, geometry, pred, grasp, _ = calibrated_scene()
    C = batch.context_len
    c = grasp['calibration']
    R = batch.body_poses[:, :C, :, :3, :3].clone()
    t = batch.body_poses[:, :C, :, :3, 3].clone()
    opening_change = .007
    for f, body in enumerate((2, 3)):
        t[:, :, body] += (R[:, :, 1] @ (opening_change*c['axis'][f])[..., None])[..., 0]
    observed = rigid_feedback_batch(batch, R, t, horizon=8)
    calibration = reanchor_finger_calibration(observed, c)
    torch.testing.assert_close(calibration['q_anchor'], c['q_anchor']+opening_change, rtol=0, atol=1e-7)
    stationary = dict(positions=observed.future, R=observed.body_poses[:, C:, :, :3, :3],
                      t=observed.body_poses[:, C:, :, :3, 3], rigid_idx=torch.arange(4))
    objective = PickPhysicalObjective(observed, pred['t'][:, -1, 0], geometry, calibration)
    features = objective.features(stationary)
    torch.testing.assert_close(features['joint_q'][:, 0], calibration['q_anchor'][None], rtol=0, atol=1e-7)
    assert float(features['joint_speed_violation'].max()) == 0
    # The old anchor would invent a 7 mm first-step closure/opening.
    stale = PickPhysicalObjective(observed, pred['t'][:, -1, 0], geometry, c).features(stationary)
    assert float((stale['joint_q'][:, 0] - c['q_anchor']).abs().min()) > .0069


@pytest.mark.parametrize('invalid', ['reflection', 'nan', 'short_history', 'zero_horizon'])
def test_feedback_rejects_invalid_observations(invalid):
    batch = gripper(second='prismatic')[0].slice(1)
    R = batch.body_poses[:, :batch.context_len, :, :3, :3].clone()
    t = batch.body_poses[:, :batch.context_len, :, :3, 3].clone()
    horizon = 4
    if invalid == 'reflection': R[0, 0, 0, :, 0] *= -1
    if invalid == 'nan': t[0, 0, 0, 0] = float('nan')
    if invalid == 'short_history': R, t = R[:, :1], t[:, :1]
    if invalid == 'zero_horizon': horizon = 0
    with pytest.raises(ValueError):
        rigid_feedback_batch(batch, R, t, horizon=horizon)


def test_receding_schedule_keeps_deadline_and_complete_final_hold():
    from guidance.receding_feedback import replan_schedule
    schedule=replan_schedule(36,4,8,6)
    assert [v['frame'] for v in schedule]==[4,12,20,28]
    assert [v['horizon'] for v in schedule]==[32,24,16,8]
    assert sum(v['commit'] for v in schedule)==32
    # A four-frame remainder cannot supply a six-frame goal hold. The previous
    # plan executes its remaining twelve frames instead of inventing padding.
    assert replan_schedule(36,16,8,6)==[
        dict(frame=16,horizon=20,commit=8),dict(frame=24,horizon=12,commit=12)]
    for args in [(36,32,8,6),(36,4,0,6),(36,4,8,0),(36,4,True,6)]:
        with pytest.raises(ValueError):replan_schedule(*args)
