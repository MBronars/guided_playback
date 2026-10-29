"""Optional scheduling cannot displace primary runs; paired worlds retain provenance."""
import copy
import json

import numpy as np
import pytest

from scripts.dev.guidance_obstacle_study import available_lane, find_fixture, second_barrier_fixture, MINIMUM_SECONDS
from scripts.dev.guidance_study import write, command, run_directory
from scripts.dev.guidance_report_utils import obstacle_evaluation_protocol
from scripts.dev.render_guidance_surfaces import obstacle_surface_points


def test_lane_requires_all_original_receipts_and_remaining_time(tmp_path):
    plan=dict(scenes=[dict(name='pick'),dict(name='place')],lanes={'0':['a'],'1':['b']})
    write(tmp_path/'gpu0.json',dict(state='completed'))
    write(tmp_path/'gpu1.json',dict(state='running'))
    assert available_lane(tmp_path,plan,10000) is None
    for scene in plan['scenes']:
        path=tmp_path/'runs'/scene['name']/'a'
        write(path/'results.json',[{}]);write(path.with_suffix('.status.json'),dict(state='completed'))
    assert available_lane(tmp_path,plan,MINIMUM_SECONDS)==0
    assert available_lane(tmp_path,plan,MINIMUM_SECONDS-1) is None
    assert available_lane(tmp_path,plan,10000,only_gpu=1) is None
    write(tmp_path/'gpu0.json',dict(state='failed'))
    assert available_lane(tmp_path,plan,10000) is None


def test_obstacle_evaluation_keeps_original_optimizer_provenance(tmp_path):
    source=tmp_path/'original.json';box=tmp_path/'box.json'
    protocol=dict(arguments=dict(obstacles=None,iterations=20),sampling=dict(horizon=68),target=[[1,2,3]])
    write(source,protocol)
    write(box,dict(version=1,units='metres',boxes=[dict(name='box',centre=[0,0,0],size=[.1,.2,.3])]))
    before=copy.deepcopy(protocol);modified=obstacle_evaluation_protocol(protocol,source,box)
    assert protocol==before==json.loads(source.read_text())
    assert modified['sampling']==protocol['sampling'] and modified['target']==protocol['target']
    assert modified['evaluation_override']['original_arguments']==protocol['arguments']
    assert modified['evaluation_override']['trajectory_reoptimized'] is False
    assert modified['arguments']['obstacles']==str(box)
    assert modified['obstacle_spec']['boxes'][0]['rotation']==np.eye(3).tolist()
    with pytest.raises(ValueError):obstacle_evaluation_protocol(modified,source,box)


def test_deterministic_fixture_blocks_transport_and_keeps_endpoints_clear():
    pytest.importorskip('fcl')
    from guidance.obstacles import box_geometry, StaticObstacleDistance
    import torch
    geometry=dict(bodies={0:box_geometry([.06,.06,.06])},metadata=dict(support_z=0.))
    R=np.eye(3)[None];T=np.array([[-.2,0.,.04]])
    args=(geometry,R,T,[.2,0.,.04],None)
    spec,receipt=find_fixture(*args)
    assert spec==find_fixture(*args)[0]
    assert receipt['initial_clearance_by_body_mm']['0']>=10
    assert receipt['target_object_clearance_mm']>=10 and receipt['nominal_min_clearance_mm']<=-1
    distance=StaticObstacleDistance(geometry,spec)
    values=distance(torch.eye(3,dtype=torch.float64).expand(1,3,1,3,3),
        torch.tensor([[[[-.2,0.,.04]],[[0.,0.,.04]],[[.2,0.,.04]]]],dtype=torch.float64),torch.tensor([0]))
    assert values[0,0].min()>.01 and values[0,2].min()>.01 and values[0,1].min()<0


def test_point_obstacle_lies_on_rotated_surfaces():
    box=dict(size=[.1,.2,.3],centre=[1,2,3],rotation=[[0,-1,0],[1,0,0],[0,0,1]])
    points=obstacle_surface_points(box);local=(points-np.asarray(box['centre']))@np.asarray(box['rotation'])
    relative=np.abs(local)/(np.asarray(box['size'])/2)
    assert (relative<=1+1e-12).all()
    np.testing.assert_allclose(relative.max(-1),1,atol=1e-12)


def test_second_barrier_preserves_first_and_keeps_start_and_goal_clear():
    pytest.importorskip('fcl')
    from guidance.obstacles import box_geometry
    geometry=dict(bodies={0:box_geometry([.06,.06,.06])},metadata=dict(support_z=0.))
    R=np.eye(3)[None];T=np.array([[-.2,0.,.04]])
    args=(geometry,R,T,[.2,0.,.04],None)
    first,_=find_fixture(*args);original=copy.deepcopy(first)
    two,receipt=second_barrier_fixture(*args,first)
    assert first==original and two['boxes'][0]==first['boxes'][0] and len(two['boxes'])==2
    assert two==second_barrier_fixture(*args,first)[0]
    assert min(receipt['initial_clearance_by_body_mm'].values())>=10
    assert receipt['target_object_clearance_mm']>=10
    assert receipt['between_obstacles_clearance_mm']>=10
    assert receipt['nominal_second_barrier_clearance_mm']<=-1


def test_optional_command_and_reference_use_same_primary_fixture(tmp_path):
    plan=dict(checkpoint='checkpoint',horizon=68,hold_frames=6,goal_reduction='max',iterations=20,
        deterministic=True,vendors=[],reference_root=str(tmp_path/'primary'),
        arms=dict(avoid=dict(terms=None,obstacles=True),reference=dict(source_arm='full')))
    scene=dict(name='pick',directory='scene',task='pick',target=[1,2,3],target_R=None,seed=9515,obstacles='barrier.json')
    cmd=command(plan,scene,'avoid',tmp_path/'out',reference='original.pt')
    assert cmd[cmd.index('--obstacles')+1]=='barrier.json'
    assert cmd[cmd.index('--initial-rollout')+1]=='original.pt'
    assert '--physical-terms' not in cmd
    assert run_directory(tmp_path,plan,scene,'reference')==tmp_path/'primary/runs/pick/full'
