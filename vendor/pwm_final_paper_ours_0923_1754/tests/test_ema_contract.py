"""EMA must survive resume and temporary evaluation without corrupting weights."""
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from train import load_ema_checkpoint, save_checkpoint
from utils.ema import EMA


def setup(warmup=False):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.25)
    return model, EMA(model, decay=0.99, warmup=warmup)


def test_checkpoint_preserves_warmup_ema(tmp_path: Path):
    model, ema = setup(warmup=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    for value in (1.0, 2.0, 3.0):
        model.weight.data.fill_(value)
        ema.update()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, ema, optimizer, scheduler, 3, OmegaConf.create({"log": {}}), corpus={})
    resumed_model, resumed = setup(warmup=True)
    load_ema_checkpoint(str(path), resumed, torch.device("cpu"))
    for candidate, shadow in ((model, ema), (resumed_model, resumed)):
        candidate.weight.data.fill_(4.0)
        shadow.update()
    torch.testing.assert_close(resumed.shadow["weight"], ema.shadow["weight"], rtol=0, atol=0)
    assert resumed.updates == ema.updates


def test_shadow_evaluation_cannot_mutate_ema_storage():
    model, ema = setup()
    expected = ema.shadow["weight"].clone()
    ema.apply_shadow()
    with torch.no_grad():
        model.weight.add_(1)
    torch.testing.assert_close(ema.shadow["weight"], expected, rtol=0, atol=0)
    ema.restore()
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)


def test_nested_shadow_application_cannot_lose_raw_weights():
    model, ema = setup()
    model.weight.data.fill_(2)
    ema.apply_shadow()
    with pytest.raises(RuntimeError, match="already applied"):
        ema.apply_shadow()
    ema.restore()
    assert model.weight.item() == 2.0


def test_updating_applied_shadow_is_an_error():
    _, ema = setup()
    ema.apply_shadow()
    with pytest.raises(RuntimeError, match="applied"):
        ema.update()
    ema.restore()


@pytest.mark.parametrize("bad", [{}, {"weight": torch.zeros(3)},
                                {"weight": torch.zeros(1, 1), "extra": torch.zeros(1)}])
def test_load_validates_all_weights_before_mutation(bad):
    _, ema = setup()
    expected = {k: v.clone() for k, v in ema.shadow.items()}
    with pytest.raises(RuntimeError, match="EMA"):
        ema.load_state_dict(bad)
    assert ema.shadow.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(ema.shadow[name], expected[name], rtol=0, atol=0)


def test_loading_does_not_alias_the_input_checkpoint():
    _, ema = setup()
    state = {"weight": torch.ones(1, 1)}
    ema.load_state_dict(state)
    state["weight"].zero_()
    assert ema.shadow["weight"].item() == 1.0


def test_inference_state_remains_a_plain_parameter_mapping():
    model, ema = setup()
    assert ema.state_dict().keys() == dict(model.named_parameters()).keys()
    assert all(isinstance(value, torch.Tensor) for value in ema.state_dict().values())


def test_context_manager_restores_raw_weights_on_exception():
    model, ema = setup()
    model.weight.data.fill_(2)
    with pytest.raises(ValueError, match="evaluation failed"):
        with ema.average_parameters():
            raise ValueError("evaluation failed")
    assert model.weight.item() == 2.0
    ema.update()


def test_legacy_warmup_limit_is_explicit():
    _, ema = setup(warmup=True)
    with pytest.warns(UserWarning, match="no EMA update count"):
        ema.load_state_dict({"weight": torch.ones(1, 1)})
    assert ema._effective_decay() == ema.decay


def test_invalid_metadata_is_rejected_before_mutation():
    _, ema = setup()
    before = ema.shadow["weight"].clone()
    with pytest.raises(RuntimeError, match="metadata"):
        ema.load_state_dict({"weight": torch.ones(1, 1)},
                            metadata=dict(updates=-1, decay=0.99, warmup=False))
    torch.testing.assert_close(ema.shadow["weight"], before, rtol=0, atol=0)
