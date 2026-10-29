"""Sampling invariants and contact-proxy limits; not simulation qualification."""
from collections import Counter
import itertools
import numpy as np
import pytest
from collection.mpm.drag.interactions import mode_for, place, measure
from collection.mpm.drag.protocol import sequence_action, declared_schedule, SEQUENCE_VERSION, FAMILIES


@pytest.mark.parametrize('counts',[(1,2,3,4),(2,4),(1,),(3,)])
def test_mixture_is_per_object_count(counts):
    for n in counts:
        modes=[mode_for(i,counts) for i in range(10*len(counts)) if counts[i%len(counts)]==n]
        assert Counter(modes)==(dict(single=10) if n==1 else dict(separated=3,pair=5,crowded=2))


def geometry(n,rng):
    vertices=np.array(list(itertools.product((-1.,1.),repeat=3)))
    clouds=[]
    for _ in range(n):
        cloud=vertices*rng.uniform(.03,.1,size=3)
        angle=rng.uniform(-np.pi,np.pi)
        cloud[:,:2]=cloud[:,:2] @ np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])
        clouds.append(cloud+rng.uniform(.2,1.2,size=3))
    return dict(x=np.concatenate(clouds).astype(np.float32),object_ids=np.repeat(np.arange(n),8),E=np.full(n,1e5)),dict(transforms=[{} for _ in range(n)])


@pytest.mark.parametrize('n,mode',[(1,'single'),*( (n,m) for n in (2,3,4) for m in ('separated','pair','crowded'))])
def test_layout_and_commands_keep_contract(n,mode):
    from scipy.spatial import cKDTree
    for seed in range(30):
        rng=np.random.default_rng(seed);initial,geo=geometry(n,rng)
        old=initial['x'].copy();driven=seed%n
        action=sequence_action(rng,object_id=driven,event_count=1+seed%3,first_family=FAMILIES[seed%3])
        result=place(initial,geo,action,rng,mode)
        assert np.all((initial['x']>=.06)&(initial['x']<=1.44))
        for oid in range(n):
            sel=initial['object_ids']==oid
            # Two float32 endpoint roundings at metre-scale coordinates.
            np.testing.assert_allclose(np.diff(initial['x'][sel],axis=0),np.diff(old[sel],axis=0),atol=2.5e-7)
            assert initial['x'][sel,2].min()==pytest.approx(.115,abs=1e-7)
            for other in range(oid):
                assert cKDTree(initial['x'][sel]).query(initial['x'][initial['object_ids']==other])[0].min()>=.03
        selected=(initial['object_ids']==driven).astype(np.int32)
        command,active=declared_schedule(selected,action,schema=SEQUENCE_VERSION,transitions=16+32*len(action['events']))
        assert len(result['requested_pairs'])==(0 if mode in ('single','separated') else n-1 if mode=='crowded' else 1)
        if mode in ('pair','crowded'):
            event=action['events'][result['aimed_event']]
            target=result['requested_pairs'][0]['b']
            # Initial AABB centres lie along the first actual horizontal command.
            p=[(initial['x'][initial['object_ids']==i].min(0)+initial['x'][initial['object_ids']==i].max(0))/2 for i in (driven,target)]
            delta=(p[1]-p[0])[:2];delta/=np.linalg.norm(delta)
            direction=command[event['start'],selected.astype(bool)][0,:2]/event['speed']
            np.testing.assert_allclose(delta,direction,atol=2e-6)


def test_all_lifts_get_final_pull_only_in_targeted_modes():
    for mode in ('separated','pair','crowded'):
        rng=np.random.default_rng(3);initial,geo=geometry(3,rng)
        action=sequence_action(rng,object_id=0,event_count=1,first_family='lift_hold_release')
        result=place(initial,geo,action,rng,mode)
        assert action['events'][0]['family']==('lift_hold_release' if mode=='separated' else 'pull_release')
        assert result['all_lift_final_event_changed_to_pull']==(mode!='separated')


@pytest.mark.parametrize('near,moving,expected',[(False,True,False),(True,False,False),(True,True,True)])
def test_proxy_requires_proximity_and_later_target_motion(near,moving,expected):
    x=np.zeros((20,2,3));x[:,1,0]=.01 if near else .2
    if moving:x[8:,:,1]=np.linspace(0,.03,12)[:,None]
    m=dict(action=dict(events=[dict(start=4)]),geometry_provenance=dict(drag_interaction=dict(mode='pair',requested_pairs=[dict(a=0,b=1)])))
    r=measure(m,dict(x=x,object_ids=np.array([0,1])))
    assert r['any_transfer_candidate'] is expected
    assert r['exact_contact_measurement'] is False
