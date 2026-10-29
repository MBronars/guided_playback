"""Independent engine checks for supplied obstacle dimensions and contact masks."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

mujoco=pytest.importorskip('mujoco')
pytest.importorskip('scipy')

from scripts.dev.replay_pickup_mujoco import add_obstacles


@pytest.mark.parametrize('category',[1,2])
def test_static_box_collides_with_object_and_gripper_categories(category):
    root=ET.Element('mujoco')
    ET.SubElement(root,'option',gravity='0 0 0',timestep='.001')
    world=ET.SubElement(root,'worldbody')
    rotation=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]
    spec=dict(version=1,units='metres',boxes=[dict(name='barrier',centre=[.2,0.,0.],
        size=[.2,.1,.3],rotation=rotation)])
    add_obstacles(world,spec)
    body=ET.SubElement(world,'body',pos='.12 0 0')
    ET.SubElement(body,'freejoint')
    ET.SubElement(body,'geom',type='sphere',size='.05',mass='1',contype=str(category),conaffinity='0')
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    np.testing.assert_allclose(model.geom_size[0],[.1,.05,.15],rtol=0,atol=1e-12)
    np.testing.assert_allclose(data.geom_xmat[0].reshape(3,3),rotation,rtol=0,atol=1e-12)
    assert data.ncon==1 and data.contact[0].dist<-.0199
    initial=data.qpos[0]
    for _ in range(20):mujoco.mj_step(model,data)
    assert data.qpos[0]<initial-.001  # Contact pushes the dynamic body out.
    assert np.isfinite(data.qpos).all() and not data.warning.number.any()
