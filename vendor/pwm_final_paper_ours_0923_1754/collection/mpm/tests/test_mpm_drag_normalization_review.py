"""Regression for the production round-trip false positive; not physics changes."""
import pytest
import torch
from data.normalization import domain_normalizers
from collection.mpm.drag.review import normalization_review


@pytest.mark.parametrize('profile',['xy_p99_per_domain','xy_p99_global'])
def test_affine_oracle_sweep(profile):
    stats=domain_normalizers(profile,'mpm')[0]
    rng=torch.Generator().manual_seed(712)
    actions=torch.cat((torch.zeros(1,3),torch.rand(10000,3,generator=rng)*.022-.011,
                       stats.min_val[None],stats.max_val[None]))
    result=normalization_review(stats,actions)
    assert result['maximum_oracle_error_m']<result['maximum_oracle_bound_m']


def test_old_roundtrip_assertion_rejects_valid_float32():
    stats=domain_normalizers('xy_p99_global','mpm')[0]
    actions=torch.linspace(-.010425,.010425,30000)[:,None].expand(-1,3).contiguous()
    back=stats.denormalize(stats.normalize(actions))
    assert not torch.isclose(back,actions,atol=1e-8,rtol=1e-5).all()
    normalization_review(stats,actions)


@pytest.mark.parametrize('method',['normalize','denormalize'])
def test_oracle_rejects_corruption(method,monkeypatch):
    stats=domain_normalizers('xy_p99_global','mpm')[0]
    original=getattr(stats,method)
    monkeypatch.setattr(stats,method,lambda x:original(x)+1e-4)
    with pytest.raises(AssertionError):normalization_review(stats,torch.zeros(10,3))


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_rejected(value):
    stats=domain_normalizers('xy_p99_global','mpm')[0]
    with pytest.raises(ValueError):normalization_review(stats,torch.full((2,3),value))
