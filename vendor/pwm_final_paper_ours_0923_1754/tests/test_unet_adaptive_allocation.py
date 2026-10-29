"""Capped UNet allocation: budget invariants, real graph operations, and saved semantics."""
from dataclasses import asdict
import itertools
import random

from omegaconf import OmegaConf
import pytest
import torch

from inference import load_model_and_diffusion
from models.factory import parse_model_config
from models.unet_std import GraphProvider, UNet, UNetWrapper, build_unet_config
from tests.test_unet_std_stratified import make_batch, make_cfg, open_gates
from tests.test_unet_std_pose_native import _model_cfg, _noisy_pose_gt, make_batch as pose_batch
from utils.checkpoint import validate_resume_configuration
from utils.config import load_config


def cfg(**changes):
    options = dict(l1_by_mat={}, l1_rule="capped_quarter", l1_rule_min=8,
                   l1_rule_max=32, l1_scene_budget=48, strides=(4, 4, 2),
                   widths=(16, 32, 64, 64), enc_depths=(1, 1, 1, 1),
                   dec_depths=(1, 1, 1), register_from_level=2, register_objs=1,
                   pool_mode="pointnet", unpool_mode="pointnet")
    options.update(changes)
    return make_cfg(**options)


def test_production_examples_and_small_objects():
    c = build_unet_config(parse_model_config(load_config("configs/unet_adaptive.yaml").model)[1])
    assert [c.l1_for_object(0, n, rigid=False) for n in (1, 7, 8, 100, 128, 512, 671, 1000, 4096, 10000)] == [
        1, 7, 8, 96, 128, 128, 128, 256, 512, 512]
    assert c.l1_for_scene([3, 3], [128, 128], rigid=[True, True]) == [128, 128]
    assert c.l1_for_scene([3, 4], [512, 671], rigid=[True, True]) == [128, 128]
    assert c.l1_for_scene([0] * 4, [1024] * 4, rigid=[False] * 4) == [256] * 4
    assert c.l1_for_scene([5, 4, 4], [10000, 1000, 1000], rigid=[False, True, True]) == [512, 128, 128]
    assert c.l1_for_scene([0, 1, 2], [4096] * 3, rigid=[False] * 3) == [336] * 3
    assert c.l1_for_scene([0, 1, 2], [4096] * 3, rigid=[False, False, True]) == [448, 448, 128]
    assert c.l1_for_scene([0] * 8, [4096] * 8, rigid=[False] * 8) == [128] * 8
    with pytest.raises(AssertionError, match="cannot preserve.*minimums"):
        c.l1_for_scene([0] * 9, [4096] * 9, rigid=[False] * 9)
    assert c.ladder(1) == [1, 1, 1]
    assert c.ladder(7) == [7, 1, 1]
    assert c.ladder(336) == [336, 84, 42]


def test_random_scenes_respect_bounds_budget_and_order():
    rng = random.Random(917)
    c = cfg()
    # Independent invariants, including budgets that cannot fill an entire stride.
    for _ in range(800):
        counts = [rng.randint(1, 600) for _ in range(rng.randint(1, 8))]
        floor = [min(8, n if n < 8 else n // 8 * 8) for n in counts]
        c.l1_scene_budget = rng.randint(sum(floor), max(sum(floor), 200))
        result = c.l1_for_scene([0] * len(counts), counts)
        assert sum(result) <= c.l1_scene_budget
        for n, minimum, tokens in zip(counts, floor, result):
            assert minimum <= tokens <= min(n, 32)
            assert tokens % 8 == 0 or tokens == n < 8
            ladder = c.ladder(tokens)
            assert all(1 <= b <= a for a, b in zip([n] + ladder, ladder))
        order = list(range(len(counts)))
        rng.shuffle(order)
        assert c.l1_for_scene([3] * len(counts), [counts[i] for i in order]) == [result[i] for i in order]
    requests = [c.l1_for_object(0, n) for n in range(1, 2000)]
    assert requests == sorted(requests)
    c.l1_scene_budget = 0
    assert c.l1_for_scene([0] * 20, [500] * 20) == [32] * 20


@pytest.mark.parametrize("options", [
    dict(l1_scene_budget=-1), dict(l1_scene_budget=True), dict(l1_scene_budget=12.5),
    dict(l1_rule="pow2_quarter"), dict(l1_rule=""), dict(l1_by_mat={0: 8}),
    dict(l1_default=8), dict(l1_rule_min=7), dict(l1_rule_min=4), dict(l1_rule_max=4),
    dict(l1_rigid_max=-1), dict(l1_rigid_max=True), dict(l1_rigid_max=8.5),
    dict(l1_rigid_max=4), dict(l1_rigid_max=64),
    dict(l1_rigid_max=8, l1_rule="pow2_quarter", l1_scene_budget=0),
])
def test_invalid_policy_is_explicit(options):
    with pytest.raises((AssertionError, ValueError)):
        cfg(**options)


@pytest.mark.parametrize("overrides", [
    ["model.unet_l1_scene_budget=-1"], ["model.unet_l1_scene_budget=true"],
    ["model.unet_l1_scene_budget=1.5"], ["model.unet_l1_by_mat=0:256"],
    ["model.unet_l1_default=128"], ["model.unet_l1_rule=pow2_quarter"],
    ["model.unet_l1_rigid_max=-1"], ["model.unet_l1_rigid_max=true"],
    ["model.unet_l1_rigid_max=4"], ["model.unet_l1_rigid_max=1024"],
])
def test_invalid_recipe_fails_before_stats(monkeypatch, overrides):
    from data import normalization
    monkeypatch.setattr(normalization, "resolve_normalization", lambda *a, **k: pytest.fail("stats access"))
    with pytest.raises(ValueError):
        load_config("configs/unet_adaptive.yaml", overrides)


@pytest.fixture(params=[("cpu", None), ("cuda", None), ("cuda", torch.bfloat16)],
                ids=["cpu-fp32", "cuda-fp32", "cuda-bf16"])
def precision(request):
    device, dtype = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device(device), dtype


def autocast(precision):
    device, dtype = precision
    return torch.autocast(device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None)


@pytest.mark.parametrize("counts", [(1, 7, 33, 150, 256), (32, 100, 127), (160, 200, 300)])
def test_graph_indices_nesting_and_noise_independence(precision, counts):
    c = cfg()
    _, pos, pos0, oid, mid, _ = [v.to(precision[0]) for v in make_batch(counts=counts, mats=[0] * len(counts))]
    p = GraphProvider(c, pos, pos0, oid, mid)
    noisy = GraphProvider(c, pos + torch.randn_like(pos) * 50, pos0, oid, mid)
    assert p.sizes == noisy.sizes
    for ids, other in zip(p.ids, noisy.ids):
        torch.testing.assert_close(ids, other, rtol=0, atol=0)
    for l in range(len(p.sizes)):
        for o, ((offset, count), (raw, n)) in enumerate(zip(p.segs[l], p.segs[0])):
            ids = p.ids[l][:, offset:offset + count]
            assert ((ids >= raw) & (ids < raw + n)).all()
            assert all(row.unique().numel() == count for row in ids)
            if l:
                prev, _ = p.segs[l - 1][o]
                if l > 1:
                    assert torch.equal(ids, p.ids[l - 1][:, prev:prev + count])
                assert torch.equal(p.ids[l - 1].gather(1, p.anchor_slots[l - 1]), p.ids[l])
    for l in range(len(p.interp)):
        for (fo, fn), (co, cn) in zip(p.segs[l], p.segs[l + 1]):
            interp, group = p.interp[l][:, :, fo:fo + fn], p.groups[l][:, :, co:co + cn]
            assert ((interp >= co) & (interp < co + cn)).all()
            assert ((group >= fo) & (group < fo + fn)).all()


@pytest.mark.parametrize("pool,unpool", list(itertools.product(["attn", "pointnet"], repeat=2)))
def test_forward_backward_with_tiny_and_capped_objects(precision, pool, unpool):
    torch.manual_seed(311)
    c = cfg(pool_mode=pool, unpool_mode=unpool)
    model = UNet(c)
    open_gates(model)
    model.to(precision[0])
    inputs = [v.to(precision[0]) for v in make_batch(counts=(1, 7, 33, 150, 256), mats=(0,) * 5)]
    with autocast(precision):
        out = model(*inputs, n_context=2)
        loss = out.float().square().mean()
    assert out.shape == inputs[0].shape[:3] + (3,) and torch.isfinite(out).all()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
    assert model.stem[0].weight.grad.abs().sum() > 0


def wrapper_config(**options):
    values = dict(unet_l1_by_mat="", unet_l1_rule="capped_quarter",
                  unet_l1_rule_min=4, unet_l1_rule_max=16, unet_l1_scene_budget=24,
                  unet_l1_rigid_max=8)
    values.update(options)
    return _model_cfg(**values)


@pytest.mark.parametrize("scene", ["mixed", "rigid"])
def test_native_objective_optimizer_steps(precision, scene):
    from diffusion import TrajectoryFlowMatching
    from utils.trajectory_objective import trajectory_objective
    torch.manual_seed(377)
    model = UNetWrapper(wrapper_config()).to(precision[0])
    batch = pose_batch(deform_n=0 if scene == "rigid" else 128,
                       rigid=((1, 30), (1, 34), (2, 96))).to(precision[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(391)
        with autocast(precision):
            loss, _, _ = trajectory_objective(model, batch, TrajectoryFlowMatching(), noise_context=False)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    for component in ("stem", "enc", "pools", "unpools", "dec", "pose"):
        assert any(component in n and not torch.equal(p, before[n]) for n, p in model.named_parameters()), component


@pytest.mark.parametrize("policy", ["legacy", "count_only", "adaptive"])
@pytest.mark.parametrize("ema", [False, True])
def test_saved_policy_and_outputs_survive_loading(precision, tmp_path, policy, ema):
    options = _model_cfg() if policy == "legacy" else wrapper_config()
    if policy == "count_only":
        options.unet_l1_rigid_max = 0
    torch.manual_seed(409)
    model = UNetWrapper(options)
    open_gates(model)
    model.to(precision[0]).eval()
    batch = pose_batch().to(precision[0])
    noisy, pose = torch.randn_like(batch.future), _noisy_pose_gt(batch) + .1
    times = torch.full(noisy.shape[:2], 500, dtype=torch.long, device=precision[0])
    config = dict(model=dict(asdict(options), architecture="unet"),
                  data=dict(format="unified", stats_location="unified://xy_p99",
                            context_len=batch.context_len, future_len=batch.num_future_frames),
                  train=dict(noise_context=False), denoise_method="flow_matching",
                  flow_matching=dict(num_train_timesteps=1000, shift=1.0,
                                     prediction_type="velocity", inference_steps=4))
    if policy == "legacy":
        config["model"].pop("unet_l1_scene_budget")  # Checkpoint predates this field.
    if policy != "adaptive":
        config["model"].pop("unet_l1_rigid_max")  # Both prior policies predate the rigid ceiling.
    path = tmp_path / "saved.pt"
    torch.save(dict(cfg=config, **{"ema" if ema else "model":
                                 dict(model.named_parameters()) if ema else model.state_dict()}), path)
    loaded, _, _, _ = load_model_and_diffusion(path, precision[0])
    torch.testing.assert_close(model.state_dict(), loaded.state_dict(), rtol=0, atol=0)
    assert loaded.model.cfg.l1_scene_budget == (0 if policy == "legacy" else 24)
    assert loaded.model.cfg.l1_rigid_max == (8 if policy == "adaptive" else 0)
    with torch.no_grad(), autocast(precision):
        expected = model(batch, noisy, times, noisy_pose=pose)
        actual = loaded(batch, noisy, times, noisy_pose=pose)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_allocation_only_changes_no_parameter_shapes_or_initial_values():
    torch.manual_seed(419)
    fixed = UNetWrapper(_model_cfg())  # Largest table count = 16.
    torch.manual_seed(419)
    adaptive = UNetWrapper(wrapper_config())
    torch.testing.assert_close(fixed.state_dict(), adaptive.state_dict(), rtol=0, atol=0)
    assert fixed.model.cfg.level_sizes == adaptive.model.cfg.level_sizes


def test_equal_allocations_preserve_active_network_outputs_exactly():
    torch.manual_seed(421)
    fixed = UNetWrapper(_model_cfg()).eval()
    open_gates(fixed)
    adaptive = UNetWrapper(wrapper_config(unet_l1_scene_budget=32)).eval()
    adaptive.load_state_dict(fixed.state_dict(), strict=True)
    batch = pose_batch(deform_n=64, rigid=((1, 16), (1, 16), (2, 32)))
    noisy, pose = torch.randn_like(batch.future), _noisy_pose_gt(batch) + .1
    times = torch.full(noisy.shape[:2], 500, dtype=torch.long)
    with torch.no_grad():
        expected = fixed(batch, noisy, times, noisy_pose=pose)
        actual = adaptive(batch, noisy, times, noisy_pose=pose)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("override", ["model.unet_l1_scene_budget=512", "model.unet_l1_rule_min=64",
                                      "model.unet_l1_scene_budget=0", "model.unet_l1_rigid_max=0",
                                      "model.unet_l1_rigid_max=64"])
def test_resume_rejects_changed_allocation_semantics(override):
    base = load_config("configs/unet_adaptive.yaml")
    changed = load_config("configs/unet_adaptive.yaml", [override])
    with pytest.raises(ValueError, match="training contract"):
        validate_resume_configuration({"cfg": OmegaConf.to_container(base)}, changed)


def test_legacy_resume_with_absent_new_field():
    base = load_config("configs/unet.yaml")
    state = {"cfg": OmegaConf.to_container(base)}
    state["cfg"]["model"].pop("unet_l1_scene_budget", None)
    validate_resume_configuration(state, base)


def test_water_recipe_needs_no_material_allocation_override(tmp_path):
    from tests.test_water3d import _water_artifact
    config = load_config("configs/unet_adaptive.yaml", ["data.domains=[water]",
        "model.num_mat_types=7", f"normalization.artifact={_water_artifact(tmp_path)}"])
    c = build_unet_config(parse_model_config(config.model)[1])
    assert c.l1_for_scene([6], [4096], rigid=[False]) == [512]
    assert not c.l1_by_mat


def test_water_material_forward_backward(precision):
    torch.manual_seed(433)
    model = UNetWrapper(wrapper_config(num_mat_types=7)).to(precision[0])
    batch = pose_batch(deform_n=96, rigid=(), sidecar=False).to(precision[0])
    batch.mat_type.fill_(6)
    noisy = torch.randn_like(batch.future)
    times = torch.full(noisy.shape[:2], 500, dtype=torch.long, device=precision[0])
    with autocast(precision):
        output = model(batch, noisy, times)
    assert output.shape == noisy.shape and torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert model.model.stem[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
