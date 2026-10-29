"""Replay commands distinguish achieved geometry from commanded grip closure."""
import pytest
import torch
import numpy as np

from guidance.replay_control import finger_command, native_tracks_to_model


def test_recorded_positive_control_is_unmodified_even_beyond_achieved_limits():
    recorded=torch.tensor([-.002,.05]); achieved=torch.tensor([.02,.03])
    torch.testing.assert_close(finger_command(achieved,mode='recorded',recorded=recorded),recorded,atol=0,rtol=0)


def test_preload_begins_at_declared_closure_and_respects_joint_limits():
    achieved=torch.tensor([.02,.003])
    early=finger_command(achieved,mode='preload',frame=17,close_frame=18)
    closed=finger_command(achieved,mode='preload',frame=18,close_frame=18)
    torch.testing.assert_close(early,achieved,atol=0,rtol=0)
    torch.testing.assert_close(closed,torch.tensor([.014,0.]),atol=1e-8,rtol=0)
    with pytest.raises(ValueError,match='explicit closure'):
        finger_command(achieved,mode='preload')


def test_replay_metric_uses_the_template_origin_not_the_native_link_origin():
    from utils.pose_ops import so3_exp
    native_q=torch.tensor([[[2**-.5,0.,0.,2**-.5],[2**-.5,0.,2**-.5,0.]]],dtype=torch.float64)
    native_R=torch.tensor([[[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],[[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]]]],dtype=torch.float64)
    t=torch.tensor([[[.2,.3,.4],[.4,.5,.6]]],dtype=torch.float64)
    mr=so3_exp(torch.tensor([[0.,.3,0.],[.1,0.,0.]],dtype=torch.float64))
    R=native_R @ mr.transpose(-1,-2)
    mt=torch.tensor([[.03,-.04,.01],[.02,.01,.05]],dtype=torch.float64)
    shift=torch.tensor([-.45,0.,-.10],dtype=torch.float64)
    native_t=(R @ mt[...,None])[...,0]+t-shift
    native=torch.cat([native_t,native_q],dim=-1)
    actual_R,actual_t=native_tracks_to_model(native,mr,mt,shift)
    torch.testing.assert_close(actual_R,R,atol=1e-9,rtol=0)
    torch.testing.assert_close(actual_t,t,atol=1e-9,rtol=0)
def test_latched_preload_does_not_turn_closing_into_opening_when_proximity_flickers():
    from guidance.replay_control import mesh_preload_commands
    q=np.array([[.033,.030],[.0315,.0285],[.0312,.0281],[.0311,.0280]])
    near=np.array([False,True,False,True])
    old,old_gate=mesh_preload_commands(q,near)
    np.testing.assert_array_equal(old,q-.006*near[:,None])
    assert np.all(old[2]-old[1]>.005)
    new,active=mesh_preload_commands(q,near,policy='latched')
    np.testing.assert_array_equal(active,[False,True,True,True])
    assert np.all(new[2]-new[1]<0)
    np.testing.assert_array_equal(q,np.array([[.033,.030],[.0315,.0285],[.0312,.0281],[.0311,.0280]]))


def test_latched_preload_releases_on_opening_and_needs_a_new_closing_stroke():
    from guidance.replay_control import mesh_preload_commands
    q=np.array([[.030,.030],[.029,.029],[.0292,.0292],[.032,.032],[.0319,.0319],[.0295,.0295]])
    near=np.ones(len(q),dtype=bool)
    command,active=mesh_preload_commands(q,near,policy='latched')
    np.testing.assert_array_equal(active,[True,True,True,False,False,True])
    np.testing.assert_array_equal(command[3:5],q[3:5])


def test_latched_preload_uses_aperture_not_incompatible_per_finger_minima():
    from guidance.replay_control import mesh_preload_commands
    q=np.array([[.03,.03],[.02,.04],[.04,.02],[.03,.03]])
    _,active=mesh_preload_commands(q,np.array([True,False,True,False]),policy='latched')
    assert active.all()  # Translating a fixed-width grasp is not an opening stroke.
