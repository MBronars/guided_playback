"""Checkpoint durability and exact fixed-input training continuation."""
from pathlib import Path
import random

from accelerate import Accelerator
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from utils.checkpoint import atomic_torch_save, capture_runtime, restore_runtime
from utils.ema import EMA


def test_failed_checkpoint_write_preserves_the_previous_artifact(tmp_path, monkeypatch):
    from train import save_checkpoint as particle_save
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    ema = EMA(model)
    cfg = OmegaConf.create({"log": {}})
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous checkpoint")

    def fail(state, destination):
        if hasattr(destination, "write"):
            destination.write(b"partial checkpoint")
        else:
            Path(destination).write_bytes(b"partial checkpoint")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="interrupted"):
        particle_save(path, model, ema, optimizer, scheduler, 3, cfg, corpus={})
    assert path.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.iterdir()) == [path], "Temporary files leaked after a failed save"


def test_runtime_replays_all_cpu_random_streams(tmp_path):
    accelerator = Accelerator(cpu=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    state = capture_runtime(accelerator, next_step=9, best_val=0.125, loss_ema=0.0)
    expected = (random.random(), np.random.rand(), torch.randn(3))
    path = tmp_path / "runtime.pt"
    atomic_torch_save(state, path)
    # New metadata must not break weights-only inference readers.
    restored = torch.load(path, weights_only=True)
    loop = restore_runtime(restored, accelerator)
    assert loop == dict(next_step=9, best_val=0.125, loss_ema=0.0)
    observed = (random.random(), np.random.rand(), torch.randn(3))
    assert observed[:2] == expected[:2]
    torch.testing.assert_close(observed[2], expected[2], rtol=0, atol=0)


def test_runtime_rejects_world_size_changes():
    accelerator = Accelerator(cpu=True)
    state = capture_runtime(accelerator)
    state["world_size"] = 2
    with pytest.raises(RuntimeError, match="world size"):
        restore_runtime(state, accelerator)


def test_runtime_rejects_bf16_fp32_change_even_without_a_scaler():
    from types import SimpleNamespace
    accelerator = Accelerator(cpu=True)
    state = capture_runtime(accelerator)
    other = "bf16" if accelerator.mixed_precision == "no" else "no"
    fake = SimpleNamespace(num_processes=1, mixed_precision=other, scaler=None)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="precision"):
        restore_runtime(state, fake)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_legacy_runtime_needs_its_saved_configuration_precision():
    accelerator = Accelerator(cpu=True)
    state = capture_runtime(accelerator)
    state["version"] = 1
    state.pop("mixed_precision")
    with pytest.raises(RuntimeError, match="precision"):
        restore_runtime(state, accelerator)
    restore_runtime(state, accelerator, legacy_precision=accelerator.mixed_precision)


@pytest.mark.parametrize("kind", ["shape", "metadata"])
def test_invalid_ema_restore_does_not_partially_replace_model_weights(kind):
    from utils.checkpoint import load_training_state
    model = torch.nn.Linear(1, 1)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    state = {"model": {name: torch.zeros_like(tensor) for name, tensor in before.items()},
             "ema": {name: tensor.clone() for name, tensor in before.items()},
             "ema_meta": dict(version=1, updates=0, decay=0.99, warmup=False), "step": 0}
    if kind == "shape":
        state["ema"]["weight"] = torch.ones(3, 1)
    else:
        state["ema_meta"]["updates"] = -1
    with pytest.raises(RuntimeError, match="EMA"):
        load_training_state(state, model, ema=EMA(model))
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


def test_runtime_cannot_be_taken_mid_accumulation():
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match="accumulation"):
        capture_runtime(SimpleNamespace(sync_gradients=False))


@pytest.mark.parametrize("kind,warmup", [("cosine", 0), ("cosine", 5), ("constant", 5), ("wsd", 0)])
def test_fixed_input_resume_matches_uninterrupted_training_bitwise(tmp_path, kind, warmup):
    from utils.checkpoint import load_training_state, training_checkpoint
    from utils.lr_schedule import make_scheduler
    accelerator = Accelerator(cpu=True)
    cfg = OmegaConf.create({"train": {"steps": 100, "lr": 0.01,
                                     "warmup_steps": warmup, "lr_schedule": kind}})

    def construct():
        model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = make_scheduler(optimizer, cfg.train)
        return model, EMA(model, decay=0.99, warmup=True), optimizer, scheduler

    def update(model, ema, optimizer, scheduler):
        # Exercise Python, NumPy, CPU Torch and dropout RNG, plus Adam moments.
        inputs = torch.randn(4, 3) + random.random()
        target = torch.tensor(np.random.randn(4, 1), dtype=torch.float32)
        optimizer.zero_grad()
        loss = (model(inputs) - target).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update()
        return loss.item()

    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    original = construct()
    for _ in range(3):
        update(*original)
    model, ema, optimizer, scheduler = original
    checkpoint = training_checkpoint(model, ema, optimizer, scheduler, 2, cfg, next_step=3,
                                     runtime=capture_runtime(accelerator, best_val=0.125, loss_ema=0.0))
    path = tmp_path / "resume.pt"
    atomic_torch_save(checkpoint, path)
    expected_losses = [update(*original) for _ in range(4)]
    resumed = construct()
    rm, re, ro, rs = resumed
    loaded = torch.load(path, weights_only=True)
    assert load_training_state(loaded, rm, ro, rs, re) == 3
    assert rs.last_epoch == 3, "Resume must not execute a scheduler tick"
    loop = restore_runtime(loaded["runtime"], accelerator)
    assert loop["loss_ema"] == 0.0 and loop["best_val"] == 0.125
    assert [update(*resumed) for _ in range(4)] == expected_losses
    for expected, observed in ((model.state_dict(), rm.state_dict()), (ema.state_dict(), re.state_dict()),
                               (optimizer.state_dict(), ro.state_dict())):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert scheduler.state_dict() == rs.state_dict()
    assert ema.metadata() == re.metadata()


def test_two_rank_schedule_spans_the_logical_update_budget():
    from utils.lr_schedule import make_scheduler
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scheduler = make_scheduler(optimizer, dict(lr=1, steps=100), num_processes=2)
    for logical_step in range(100):
        optimizer.step()
        for _ in range(2):  # AcceleratedScheduler's default behavior.
            scheduler.step()
        if logical_step == 49:
            assert scheduler.get_last_lr()[0] == pytest.approx(0.505)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.01)


def test_resume_rejects_a_silently_changed_schedule():
    from utils.lr_schedule import make_scheduler, validate_scheduler_state
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    before = make_scheduler(optimizer, dict(lr=1, steps=100))
    after = make_scheduler(optimizer, dict(lr=1, steps=200))
    with pytest.raises(RuntimeError, match="Resume schedule differs"):
        validate_scheduler_state(after, before.state_dict())
