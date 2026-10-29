"""Physical counterexamples for picking measurements and collision gradients."""
import itertools

import pytest
import torch

from data.batch import TrajectoryBatch
from data.dataset import NormalizationStats
from data.unified_dataset import _augment_pose_channels, _rigid_sidecar_from_particles
from guidance.pick_objective import PickObjective, component_margin


def cube(half):
    half = torch.tensor(half)
    vertices = torch.tensor(list(itertools.product((-1., 1.), repeat=3))) * half
    normals = torch.cat([torch.eye(3), -torch.eye(3)])
    return vertices, torch.cat([normals, -half.repeat(2)[:, None]], dim=1)


def scene(*, contact=True, penetrating=False, falling=False):
    C, F = 2, 6
    halfs = [(0.02, 0.02, 0.02), (0.04, 0.025, 0.01), (0.004, 0.006, 0.008), (0.004, 0.006, 0.008)]
    gap = 0.026 if contact else 0.10
    if penetrating: gap = 0.019
    centers = torch.tensor([[0., 0., .02], [0., 0., .15], [-gap, 0., .02], [gap, 0., .02]])
    poses = torch.eye(4).expand(1, C+F, 4, 4, 4).clone()
    poses[..., :3, 3] = centers
    motion = torch.cat([torch.zeros(C), torch.linspace(.01, .06, F)])
    if falling:
        poses[:, :, 0, 2, 3] += .10
        motion = -motion
    if contact:
        poses[:, :, :, 2, 3] += motion[None, :, None]
    else:
        poses[:, :, 0, 2, 3] += motion
    pieces = [cube(h) for h in halfs]
    points = [v for v,_ in pieces]
    # Include samples at the finger contact patches. Eight box corners alone are
    # a poor proximity measurement for fingers touching the middle of a face.
    points[0] = torch.cat([points[0], torch.tensor(list(itertools.product(
        (-.02,.02),(-.006,.006),(-.008,.008))))])
    X = torch.cat([v[None,None]+poses[:,:,i,None,:3,3] for i,v in enumerate(points)], dim=2)
    pb = torch.arange(4).repeat_interleave(torch.tensor([len(v) for v in points]))[None]
    sc = _rigid_sidecar_from_particles(X, pb, 4)
    sc['particle_body'] = pb
    stats = NormalizationStats(min_val=torch.full((3,), -.1), max_val=torch.full((3,), .1))
    ps = NormalizationStats(min_val=torch.full((6,), -.1), max_val=torch.full((6,), .1))
    act = pb > 0
    sc = _augment_pose_channels(sc, act, C, ps)
    N = X.shape[2]
    b = TrajectoryBatch(positions=X, actions=torch.zeros(1,F,N,3), action_mask=act,
        object_ids=act.long(), valid_lengths=torch.tensor([N]), E=torch.ones(1,N),
        nu=torch.ones(1,N)*.3, rho=torch.ones(1,N), context_len=C, delta_stats=stats,
        position_stats=stats, **sc)
    geo = dict(bodies={i:dict(vertices=v,planes=[p]) for i,(v,p) in enumerate(pieces)},
               metadata=dict(support_z=0.0))
    pred = dict(positions=b.future, R=b.body_poses[:,C:,:,:3,:3], t=b.body_poses[:,C:,:,:3,3],
                rigid_idx=torch.arange(4))
    return b, geo, pred


def test_signed_component_depth_and_gradient_on_known_box():
    _, planes = cube([.02,.03,.04])
    points = torch.tensor([[.018,.001,.002],[.025,0.,0.]],requires_grad=True)
    signed = component_margin(points,[planes])
    torch.testing.assert_close(signed,torch.tensor([-.002,.005]),atol=1e-8,rtol=0)
    depth = (-signed).clamp_min(0).sum()
    grad, = torch.autograd.grad(depth,points)
    torch.testing.assert_close(grad,torch.tensor([[-1.,0.,0.],[0.,0.,0.]]),atol=0,rtol=0)


def test_coherent_lift_is_close_without_phantom_or_penetration():
    b,g,p = scene()
    o = PickObjective(b,p['t'][:,-1,0],g)
    m = o.measure(p)
    assert m['both_fingers_close_fraction'] == 1.0
    assert m['unsupported_rise_transitions'] == 0
    assert m['gripper_in_object_max_mm'] < 1e-4
    assert m['object_in_gripper_hull_max_mm'] < 1e-4
    assert m['held_hand_frame_step_mm'] < 1e-4


def test_free_rise_is_flagged_but_free_fall_is_not():
    for falling in (False, True):
        b,g,p = scene(contact=False,falling=falling)
        o = PickObjective(b,p['t'][:,-1,0],g)
        m = o.measure(p)
        assert m['unsupported_rise_transitions'] == (0 if falling else 6)
        assert not m['sustained_hover_proxy']


def test_penetration_is_detected_when_point_proximity_alone_could_look_close():
    b,g,p = scene(penetrating=True)
    o = PickObjective(b,p['t'][:,-1,0],g)
    m = o.measure(p)
    assert m['gripper_in_object_max_mm'] == pytest.approx(5.0,abs=1e-4)
    assert m['gripper_in_object_over_1mm_frames'] == 6
    assert o.terms(o.features(p))['penetration'] > .0039


def test_support_intrusion_and_sustained_hover_are_distinguished():
    b,g,p = scene(contact=False)
    o = PickObjective(b,p['t'][:,-1,0],g)
    p = {k:v.clone() for k,v in p.items()}
    p['t'][:,:,0,2] = .08
    p['positions'][:,:,b.particle_body[0]==0,2] = b.body_templates[:,0,:16,2][:,None]+.08
    assert o.measure(p)['sustained_hover_proxy']
    p['t'][:,:,0,2] = .015
    p['positions'][:,:,b.particle_body[0]==0,2] = b.body_templates[:,0,:16,2][:,None]+.015
    m = o.measure(p)
    assert m['support_reference_intrusion_max_mm'] == pytest.approx(5.0,abs=1e-4)
    assert not m['sustained_hover_proxy']
