"""Prescribed actuator paths must survive delta encoding and window integration."""
from dataclasses import replace

import pytest
import torch

from guidance.objectives import materialize
from guidance.unconditional import sample_rollout
from guidance.waypoints import grasp_waypoints
from tests.test_grasp_guidance import calibrated_scene
from tests.test_unconditional_guidance import setup
from utils.pose_native import JOINT_FREE, JOINT_PRISMATIC, pose_channel_mask
from utils.pose_ops import so3_exp


def articulated_calibrated_scene():
    batch,_,_,grasp,_=calibrated_scene()
    batch=replace(batch,body_parent=torch.tensor([[-1,-1,1,1]]),
        joint_type=torch.tensor([[JOINT_FREE,JOINT_FREE,JOINT_PRISMATIC,JOINT_PRISMATIC]]),
        joint_axis=torch.cat([torch.zeros(2,3),grasp['calibration']['axis']])[None],
        joint_pivot=torch.zeros(1,4,3))
    return batch,grasp


def test_geometric_waypoints_encode_the_whole_actuator_path_without_passive_motion():
    batch,grasp=articulated_calibrated_scene()
    grasp['proposals'][0]['hand_R_object']=so3_exp(torch.tensor([0.,0.,.8]))
    grasp['proposals'][0]['q_close']=torch.tensor([.02,.02])
    target=batch.body_poses[:,batch.context_len-1,0,:3,3]+torch.tensor([0.,0.,.1])
    control=grasp_waypoints(batch,target,grasp,horizon=6,approach_frames=1,close_frame=3,hold_frames=2)
    pred=materialize(batch,torch.zeros_like(batch.future),control['known_pose'])
    expected=control['actuator_poses'][:,1:]
    torch.testing.assert_close(pred['t'][:,:,1:],expected[...,:3,3],atol=1e-6,rtol=0)
    torch.testing.assert_close(pred['R'][:,:,1:],expected[...,:3,:3],atol=1e-6,rtol=0)
    assert not control['known_mask'][:,:,0].any()
    assert torch.count_nonzero(control['known_pose'][:,:,0])==0
    # Reach, close, then lift: the last two frames preserve the full held pose.
    torch.testing.assert_close(expected[:,-1],expected[:,-2],atol=1e-7,rtol=0)
    torch.testing.assert_close(control['finger_q'][:,-1],torch.tensor([[.02,.02]]))
    # Future recordings cannot influence geometric waypoint construction.
    poses=batch.body_poses.clone(); poses[:,batch.context_len:,:,:3,3]+=10
    altered=grasp_waypoints(replace(batch,body_poses=poses),target,grasp,
        horizon=6,approach_frames=1,close_frame=3,hold_frames=2,scope='fingers')
    assert not altered['known_mask'][:,:,1].any()
    torch.testing.assert_close(altered['known_pose'][:,:,2:],control['known_pose'][:,:,2:],atol=0,rtol=0)


def test_inpainting_keeps_exact_actuator_deltas_across_windows_and_replays():
    batch,model,diffusion=setup()
    known=batch.pose_delta_stats.normalize(batch.pose_deltas_gt,clip=False)
    mask=pose_channel_mask(batch,torch.arange(4)).bool()[None,None] & batch.pose_action_mask[:,None,:,None]
    sampling=dict(future_len=4,steps=2)
    with torch.no_grad():
        initial=sample_rollout(model,diffusion,batch,**sampling)
        result=sample_rollout(model,diffusion,batch,noise=initial['noise'],rng_states=initial['rng_states'],
            known_pose=known,known_mask=mask,**sampling)
        repeat=sample_rollout(model,diffusion,batch,noise=initial['noise'],rng_states=initial['rng_states'],
            known_pose=known,known_mask=mask,**sampling)
        empty=sample_rollout(model,diffusion,batch,noise=initial['noise'],rng_states=initial['rng_states'],
            known_pose=known,known_mask=torch.zeros_like(mask),**sampling)
    torch.testing.assert_close(empty['positions'],initial['positions'],atol=0,rtol=0)
    torch.testing.assert_close(repeat['positions'],result['positions'],atol=0,rtol=0)
    torch.testing.assert_close(result['pose_deltas'].masked_select(mask),known.masked_select(mask),atol=0,rtol=0)
    C=batch.context_len
    torch.testing.assert_close(result['T'][:,C:,:3],batch.body_poses[:,C:,:3,:3,3],atol=2e-6,rtol=0)
    torch.testing.assert_close(result['R'][:,C:,:3],batch.body_poses[:,C:,:3,:3,:3],atol=2e-6,rtol=0)
    assert not torch.equal(result['T'][:,C:,3],batch.body_poses[:,C:,3,:3,3])
    torch.testing.assert_close(result['positions'][:,:C],batch.context,atol=0,rtol=0)


def test_inpainting_rejects_passive_bodies_and_inactive_joint_channels():
    batch,model,diffusion=setup()
    known=torch.zeros(1,batch.num_future_frames,4,6)
    with pytest.raises(ValueError,match='actuated bodies'):
        sample_rollout(model,diffusion,batch,future_len=4,steps=2,
            known_pose=known,known_mask=torch.ones_like(known,dtype=torch.bool))
    mask=torch.zeros_like(known,dtype=torch.bool); mask[:,:,1,3]=True
    with pytest.raises(ValueError,match='inactive articulation'):
        sample_rollout(model,diffusion,batch,future_len=4,steps=2,known_pose=known,known_mask=mask)


def test_waypoint_search_pairs_noise_and_leaves_passive_motion_generated():
    from copy import deepcopy
    from guidance.control_search import waypoint_search
    from guidance.pick_validation import PickValidator, selection_key
    batch,grasp=articulated_calibrated_scene()
    _,geometry,_,_,_=calibrated_scene()
    _,model,diffusion=setup()
    alternate=deepcopy(grasp['proposals'][0]); alternate['hand_t_object'][0]+=.01
    grasp['proposals'].append(alternate)
    target=batch.body_poses[:,batch.context_len-1,0,:3,3]+torch.tensor([0.,0.,.05])
    validator=PickValidator(batch,target,geometry,grasp['calibration'],hold_frames=2)
    with torch.no_grad(): initial=sample_rollout(model,diffusion,batch,future_len=4,steps=2)
    candidates=[]; before=torch.get_rng_state()
    result=waypoint_search(model,diffusion,batch,target,grasp,initial=initial,validator=validator,
        proposals=2,phases=((1,2),(2,3)),hold_frames=2,future_len=4,steps=2,
        candidate_callback=lambda i,out,row:candidates.append((out,row)))
    assert torch.equal(torch.get_rng_state(),before)
    assert len(candidates)==4 and result['nfe']==4*initial['nfe']
    best=min((row for _,row in candidates),key=lambda row:selection_key(row['loss'],row['validation']))
    assert result['draw']==best['draw']
    assert not result['control']['known_mask'][:,:,0].any()
    for out,row in candidates:
        for actual,expected in zip(out['noise'],initial['noise']):
            for a,b in zip(actual,expected): torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(out['positions'][:,:batch.context_len],batch.context,atol=0,rtol=0)
        assert torch.count_nonzero(out['pose_deltas'][:,:,0])>0
