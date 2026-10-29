import numpy as np
import pytest

pytest.importorskip('scipy')

from scripts.dev.audit_grasp_support import gravity_support


def test_opposed_contacts_support_gravity_with_expected_minimum_normal_force():
    result=gravity_support([[-.03,0,0],[.03,0,0]],[[-1,0,0],[1,0,0]],np.zeros(3),mass=.5,friction=.5)
    assert result['feasible'] and result['residual']<1e-8
    np.testing.assert_allclose(result['normal_force_N'],[4.905,4.905],atol=1e-7)


def test_unilateral_or_frictionless_contact_cannot_support_side_grasp():
    assert not gravity_support([[-.03,0,0]],[[-1,0,0]],np.zeros(3),mass=.5)['feasible']
    assert not gravity_support([[-.03,0,0],[.03,0,0]],[[-1,0,0],[1,0,0]],np.zeros(3),mass=.5,friction=0)['feasible']


def test_force_limit_and_moment_balance_matter():
    p=[[-.03,0,0],[.03,0,0]];n=[[-1,0,0],[1,0,0]]
    assert not gravity_support(p,n,np.zeros(3),mass=.5,normal_cap=1)['feasible']
    assert not gravity_support(p,n,np.array([0,.03,0]),mass=.5)['feasible']
    with pytest.raises(ValueError,match='Undefined'):gravity_support([[0,0,0]],[[0,0,0]],np.zeros(3),mass=.5)


def test_soft_finger_torsion_supports_offset_gravity_without_independent_box_cones():
    p=[[-.03,0,0],[.03,0,0]];n=[[-1,0,0],[1,0,0]]
    # Contact torque about the pinch axis is necessary for this COM offset.
    result=gravity_support(p,n,np.array([0,.003,0]),mass=.5,torsion=.005)
    assert result['feasible']
    assert sum(result['normal_force_N'])>4.905
