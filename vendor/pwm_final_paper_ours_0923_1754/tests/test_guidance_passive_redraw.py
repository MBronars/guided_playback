import torch

from scripts.dev.audit_fixed_gripper import passive_redraw


def test_passive_redraw_preserves_actuator_inactive_and_caller_rng():
    noise=[(torch.randn(1,n,8,3),torch.randn(1,n,4,6)) for n in (16,16,4)]
    original=[(a.clone(),z.clone()) for a,z in noise]
    mask=torch.zeros(1,1,4,6,dtype=torch.bool);mask[:,:,0]=True
    state=torch.get_rng_state().clone()
    a=passive_redraw(noise,mask,9301);b=passive_redraw(noise,mask,9301)
    c=passive_redraw(noise,mask,9302)
    assert torch.equal(state,torch.get_rng_state())
    for (ap,az),(bp,bz),(cp,cz),(p,z),(op,oz) in zip(a,b,c,noise,original):
        m=mask.expand_as(z)
        assert torch.equal(ap,p) and torch.equal(az[~m],z[~m])
        assert torch.equal(ap,bp) and torch.equal(az,bz)
        assert not torch.equal(az[m],cz[m]) and not torch.equal(az[m],z[m])
        assert torch.equal(p,op) and torch.equal(z,oz)
