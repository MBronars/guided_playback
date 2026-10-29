"""Corrupted observations preserve rigid and articulated geometric contracts."""
import torch

from guidance.observations import rigid_observation_bias, density_estimation_bias
from tests.test_articulated_bodies import gripper


def test_coherent_bias_preserves_templates_child_relations_and_true_future():
    batch=gripper(second='prismatic')[0].slice(1)
    C=batch.context_len
    before=torch.get_rng_state()
    changed,receipt=rigid_observation_bias(batch,translation_m=.005,rotation_deg=5,seed=57,roots=(0,))
    assert torch.equal(torch.get_rng_state(),before)
    assert not torch.equal(changed.context,batch.context)
    torch.testing.assert_close(changed.positions[:,C:],batch.positions[:,C:],rtol=0,atol=0)
    torch.testing.assert_close(changed.body_poses[:,C:],batch.body_poses[:,C:],rtol=0,atol=0)
    for k in receipt['transforms'][0]['bodies']:
        rows=batch.particle_body[0]==k
        template=batch.body_templates[0,k,batch.body_template_mask[0,k]]
        pose=changed.body_poses[0,:C,k]
        expected=template[None] @ pose[:,:3,:3].transpose(-1,-2)+pose[:,None,:3,3]
        torch.testing.assert_close(changed.positions[0,:C,rows],expected,atol=2e-6,rtol=0)
    for child in (1,2):
        def relative(b):
            pose=b.body_poses[0,:C]
            return (pose[:,0,:3,:3].transpose(-1,-2) @ (pose[:,child,:3,3]-pose[:,0,:3,3])[...,None])[...,0]
        torch.testing.assert_close(relative(changed),relative(batch),atol=2e-6,rtol=0)


def test_zero_bias_is_exact_and_independent_of_future_reference():
    batch=gripper(second='prismatic')[0].slice(1)
    changed,_=rigid_observation_bias(batch,seed=88,roots=(3,))
    torch.testing.assert_close(changed.positions,batch.positions,atol=0,rtol=0)
    torch.testing.assert_close(changed.body_poses,batch.body_poses,atol=0,rtol=0)


def test_density_bias_changes_only_selected_material_inputs():
    batch=gripper(second='prismatic')[0].slice(1)
    changed,receipt=density_estimation_bias(batch,scale=1.25,bodies=(3,))
    object_rows=batch.particle_body==3
    torch.testing.assert_close(changed.rho[object_rows],batch.rho[object_rows]*1.25,rtol=0,atol=0)
    torch.testing.assert_close(changed.rho[~object_rows],batch.rho[~object_rows],rtol=0,atol=0)
    assert changed.positions is batch.positions and changed.body_poses is batch.body_poses
    assert changed.E is batch.E and changed.nu is batch.nu
    unchanged,_=density_estimation_bias(batch,scale=1.,bodies=(3,))
    torch.testing.assert_close(unchanged.rho,batch.rho,rtol=0,atol=0)
