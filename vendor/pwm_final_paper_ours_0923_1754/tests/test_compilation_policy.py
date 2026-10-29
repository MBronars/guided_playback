"""Compilation configuration must describe which modules actually compile."""
import pytest
import torch
import torch._dynamo as dynamo

from models.unet_std import Block, UNet, UNetWrapper
from tests.test_unet_std_pose_native import _model_cfg
from utils.compilation import prepare_compilation


@pytest.fixture
def compiler_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.nn.Module, "compile", lambda self, **kwargs: calls.append(self))
    # Restore process-wide policy after testing the same setup used by the trainer.
    monkeypatch.setattr(dynamo.config, "optimize_ddp", dynamo.config.optimize_ddp)
    return calls


def test_compile_disabled_is_a_true_noop(compiler_calls):
    model = torch.nn.Linear(2, 2)
    assert prepare_compilation(model, {"compile": False}) is model
    assert not compiler_calls


def test_regional_compile_rejects_an_unsupported_architecture(compiler_calls):
    with pytest.raises(ValueError, match="UNet"):
        prepare_compilation(torch.nn.Linear(2, 2), {"compile": True, "compile_regional": True})


@pytest.mark.parametrize("level", [0, 1])
def test_external_checkpointing_including_level_zero_stays_eager(compiler_calls, level):
    model = UNetWrapper(_model_cfg(unet_ckpt_from_level=level, unet_ckpt_ingraph=False))
    prepare_compilation(model, {"compile": True, "compile_regional": True})
    network = next(unit for unit in model.modules() if isinstance(unit, UNet))
    forbidden = [block for row in network.enc[level:] for block in row]
    forbidden += [block for row in network.dec[level:] for block in row]
    assert not any(block in compiler_calls for block in forbidden)


def test_ingraph_checkpointing_compiles_coarse_blocks_but_keeps_ragged_l0_eager(compiler_calls):
    model = UNetWrapper(_model_cfg(unet_ckpt_from_level=0, unet_ckpt_ingraph=True))
    keys = list(model.state_dict())
    prepare_compilation(model, {"compile": True, "compile_regional": True})
    network = next(unit for unit in model.modules() if isinstance(unit, UNet))
    assert all(block in compiler_calls for row in network.enc[1:] for block in row)
    assert all(block not in compiler_calls for block in network.enc[0])
    assert all(unit not in compiler_calls for unit in (network.pools[0], network.unpools[0], network.stem, network.head))
    assert network.t_mlp in compiler_calls
    assert any(isinstance(unit, Block) for unit in compiler_calls)
    assert list(model.state_dict()) == keys
