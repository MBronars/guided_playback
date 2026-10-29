"""Window branching preserves sampler semantics and creates distinct continuations."""
import torch

from guidance.beam import beam_search
from guidance.unconditional import sample_rollout
from tests.test_unconditional_guidance import setup


class PrefixObjective:
    def __init__(self): self.seen=[]
    def prefix_loss(self,pred,*,complete):
        self.seen.append((pred['t'].shape[1],complete))
        return (pred['t'][:,-1,3,0]-.08).square().mean()


def test_one_branch_one_beam_is_exactly_ordinary_sampling():
    batch,model,diffusion=setup()
    torch.manual_seed(354)
    with torch.no_grad(): ref=sample_rollout(model,diffusion,batch,future_len=4,steps=2)
    rng=torch.get_rng_state()
    torch.manual_seed(354)
    result=beam_search(model,diffusion,batch,PrefixObjective(),width=1,branches=1,future_len=4,steps=2,diversity_body=0)
    torch.testing.assert_close(result['rollout']['positions'],ref['positions'],atol=0,rtol=0)
    assert torch.equal(torch.get_rng_state(),rng)
    assert result['nfe']==ref['nfe'] and result['lineage']==[0,0]


def test_branches_have_fresh_noise_and_selected_path_can_be_replayed():
    batch,model,diffusion=setup()
    objective=PrefixObjective(); candidates=[]; validated=[]
    def validator(pred):
        validated.append(pred['t'].shape[1])
        return dict(accepted=False,max_violation_m=float(pred['t'][:,-1,3].norm()))
    result=beam_search(model,diffusion,batch,objective,width=2,branches=2,future_len=4,steps=2,
        diversity_body=0,validator=validator,candidate_callback=lambda _,out,row:candidates.append(out))
    assert len(candidates)==4 and validated==[6]*4
    assert not result['validation']['accepted']
    assert all((length==6)==complete for length,complete in objective.seen)
    for i,left in enumerate(candidates):
        for right in candidates[i+1:]:
            assert not torch.equal(left['noise'][-1][1],right['noise'][-1][1])
    selected=result['rollout']
    with torch.no_grad():
        replay=sample_rollout(model,diffusion,batch,future_len=4,steps=2,
            noise=selected['noise'],rng_states=selected['rng_states'])
    torch.testing.assert_close(replay['positions'],selected['positions'],atol=0,rtol=0)
    assert result['nfe']>selected['nfe']
