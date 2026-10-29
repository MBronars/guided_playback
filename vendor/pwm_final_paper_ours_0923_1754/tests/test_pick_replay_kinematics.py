"""Joint origins compose before revolute/prismatic motions."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip('scipy')
from scripts.dev.audit_pick_replay_kinematics import forward_kinematics


def test_rotating_parent_carries_child_origin_and_slider_axis():
    robot=ET.fromstring('''<robot><link name="root"/><link name="hand"/><link name="finger"/>
      <joint name="hinge" type="revolute"><parent link="root"/><child link="hand"/>
       <origin xyz="0.1 0 0"/><axis xyz="0 0 1"/></joint>
      <joint name="slide" type="prismatic"><parent link="hand"/><child link="finger"/>
       <origin xyz="0.1 0 0"/><axis xyz="1 0 0"/></joint></robot>''')
    result=forward_kinematics(robot,dict(hinge=np.array([0.,np.pi/2]),slide=np.array([0.,.02])))
    np.testing.assert_allclose(result['finger'][:,:3,3],[[.2,0,0],[.1,.12,0]],atol=1e-12)
    np.testing.assert_allclose(result['finger'][1,:3,:3],[[0,-1,0],[1,0,0],[0,0,1]],atol=1e-12)
