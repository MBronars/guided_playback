"""Rigid ceilings use body membership, preserve mixed objects and share scene capacity."""
from copy import deepcopy
import random

import pytest
import torch

from models.unet_std import GraphProvider, particle_rigidity
from tests.test_unet_adaptive_allocation import cfg, precision
from tests.test_unet_std_stratified import make_batch
from tests.test_unet_std_pose_native import make_batch as pose_batch


def test_ceiling_can_be_below_preferred_minimum():
    c = cfg(l1_rule_min=16, l1_rigid_max=8, l1_scene_budget=32)
    assert c.l1_for_scene([0, 0], [256, 256], rigid=[False, True]) == [24, 8]
    assert c.l1_for_scene([0, 0], [256, 256], rigid=[True, True]) == [8, 8]
    assert c.l1_for_object(0, 3, rigid=True) == 3
    assert c.ladder(3) == [3, 1, 1]
    with pytest.raises(AssertionError, match="explicit object rigidity"):
        c.l1_for_object(0, 256)
    with pytest.raises(AssertionError, match="rigidity flag"):
        c.l1_for_scene([0], [256])


def test_random_rigid_allocations_respect_ceiling_minimum_and_order():
    rng = random.Random(1223)
    c = cfg(l1_rule_min=16, l1_rigid_max=8)
    for _ in range(400):
        counts = [rng.randint(1, 500) for _ in range(rng.randint(1, 8))]
        rigid = [rng.choice([True, False]) for _ in counts]
        requests = [c.l1_for_object(0, n, rigid=r) for n, r in zip(counts, rigid)]
        floors = [min(16, n) for n in requests]
        c.l1_scene_budget = rng.randint(sum(floors), max(sum(floors), 160))
        result = c.l1_for_scene([0] * len(counts), counts, rigid=rigid)
        assert sum(result) <= c.l1_scene_budget
        for n, floor, want, got, r in zip(counts, floors, requests, result, rigid):
            assert floor <= got <= want <= min(n, 8 if r else 32)
        order = list(range(len(counts)))
        rng.shuffle(order)
        assert c.l1_for_scene([3] * len(counts), [counts[i] for i in order], rigid=[rigid[i] for i in order]) == [result[i] for i in order]


def test_membership_uses_bodies_including_passive_rigid_and_multiple_links():
    b = pose_batch(deform_n=128, rigid=((1, 32), (1, 64), (2, 128)))
    mask = particle_rigidity(b)
    assert not mask[:, :128].any() and mask[:, 128:].all()
    changed = deepcopy(b)
    changed.action_mask = ~b.action_mask
    changed.mat_type.fill_(0)
    changed.pose_action_mask = ~b.pose_action_mask
    assert torch.equal(particle_rigidity(changed), mask)
    particle_only = pose_batch(deform_n=48, rigid=(), sidecar=False)
    assert not particle_rigidity(particle_only).any()
    particle_only.body_type = torch.zeros((2, 1), dtype=torch.long)
    with pytest.raises(AssertionError, match="both body_type and particle_body"):
        particle_rigidity(particle_only)


def test_provider_caps_rigid_objects_and_preserves_mixed_objects(precision):
    c = cfg(l1_rigid_max=8)
    _, pos, pos0, oid, mid, _ = [v.to(precision[0]) for v in make_batch(counts=(128,128,128), mats=(0,0,0))]
    rigid = oid > 0
    p = GraphProvider(c, pos, pos0, oid, mid, rigid)
    assert p.rigid_objects == [False, True, True]
    assert p.sizes[1] == [32, 8, 8]
    mixed = rigid.clone()
    mixed[:, 256] = False
    q = GraphProvider(c, pos, pos0, oid, mid, mixed)
    assert q.rigid_objects == [False, True, False]
    assert q.sizes[1] == [16, 8, 16]  # Round extras down to the 8-token hierarchy quantum.
    # Shared tensor sizes cannot silently use element 0's classification.
    mixed[1, 128] = False
    with pytest.raises(AssertionError, match="rigidity must match"):
        GraphProvider(c, pos, pos0, oid, mid, mixed)
    with pytest.raises(AssertionError, match="requires particle_rigid"):
        GraphProvider(c, pos, pos0, oid, mid)
