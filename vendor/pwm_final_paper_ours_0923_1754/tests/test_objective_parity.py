"""Objective gradients/draws compared with actual frozen objective and FM sources.

Do not load frozen train.py: it can reimport the *current* objective and produce
vacuous parity. Here both objective functions and both FM classes come from their
own source modules; retained model equivalence is independently tested elsewhere.
"""
import copy

import pytest
import torch

from diffusion import TrajectoryFlowMatching
from tests.test_model_scope import _construct, _reference_module
from tests.test_unet_std_pose_native import make_batch
from utils.trajectory_objective import trajectory_objective

SCENES = {"deformable": dict(rigid=(), sidecar=False), "mixed": {},
          "rigid": dict(deform_n=0, rigid=((1, 24), (2, 16)))}


def _run(architecture, scene, prediction_type, options):
    """The frozen and the current objective from one seed: (loss, metrics, draw, gradients) each."""
    batch = make_batch(**SCENES[scene])
    if scene == "deformable":
        batch.mat_type.zero_()
    candidate = _construct(architecture, False)
    reference_model = copy.deepcopy(candidate)
    reference = _reference_module("utils/trajectory_objective.py")
    reference_fm = _reference_module("diffusion/flow_matching.py").TrajectoryFlowMatching
    # This binding makes isinstance checks select the frozen FM code as well.
    reference.TrajectoryFlowMatching = reference_fm
    results = []
    for function, cls, model in (
        (reference.trajectory_objective, reference_fm, reference_model),
        (trajectory_objective, TrajectoryFlowMatching, candidate),
    ):
        fm = cls(prediction_type=prediction_type)
        torch.manual_seed(198473)
        loss, metrics, draw = function(model, batch, fm, diffusion_forcing=True, **options)
        if function is trajectory_objective:
            # The current objective exposes effective geometry for validation
            # probes. These frozen recipes do not replace the Batch; keep the
            # entire historical draw/loss/gradient comparison exact.
            assert draw.pop("batch") is batch
        assert torch.isfinite(loss) and loss > 0
        loss.backward()
        grads = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                 for name, parameter in model.named_parameters()}
        assert any(value is not None and torch.count_nonzero(value) > 0 for value in grads.values())
        results.append((loss, metrics, draw, grads))
    return batch, results


def _compare(architecture, scene, prediction_type, options):
    _, results = _run(architecture, scene, prediction_type, options)
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


def _compare_rigid_history_exemption(architecture, scene, prediction_type, options):
    """Intended change since the frozen revision (ISSUES.md ISSUE-001): light context noise no
    longer jitters rigid rows, which are template @ pose. Their history is the clean context;
    every other row and every other part of the draw is the frozen objective's."""
    batch, ((_, _, frozen, _), (_, _, current, _)) = _run(architecture, scene, prediction_type, options)
    rigid = batch.body_type.gather(1, batch.particle_body) == 1
    clean = batch.delta_stats.normalize(batch.context_deltas)
    torch.testing.assert_close(current["context"], torch.where(rigid[:, None, :, None], clean, frozen["context"]),
                               rtol=0, atol=0)
    assert not torch.equal(current["context"], frozen["context"])
    same = ("timesteps", "x0", "x_t", "valid_mask", "context_timesteps", "pose_x0", "target", "pose_target")
    torch.testing.assert_close({k: current[k] for k in same}, {k: frozen[k] for k in same}, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("scene", ["deformable", "mixed", "rigid"])
@pytest.mark.parametrize("prediction_type", ["velocity", "sample"])
@pytest.mark.parametrize("context", ["clean", "noise", "light", "clean_clock"])
def test_frozen_objective_loss_draw_and_gradient_parity(architecture, scene, prediction_type, context):
    options = {
        "clean": {}, "noise": dict(noise_context=True),
        "light": dict(light_context_noise_std=0.025),
        "clean_clock": dict(clean_ctx_timesteps=True),
    }[context]
    options = dict(options, clip_deltas=False)
    if context == "light" and scene != "deformable":
        _compare_rigid_history_exemption(architecture, scene, prediction_type, options)
    else:
        _compare(architecture, scene, prediction_type, options)


@pytest.mark.parametrize("architecture", ["transformer", "unet"])
@pytest.mark.parametrize("prediction_type", ["velocity", "sample"])
@pytest.mark.parametrize("auxiliary", ["normalized", "mesh", "physics"])
def test_retained_auxiliary_objective_parity(architecture, prediction_type, auxiliary):
    options = {
        "normalized": dict(aux_loss_weights={"pos_cum": 0.1, "accel": 0.2,
            "com": 0.3, "spread": 0.4, "spread_axes": 0.5}),
        "mesh": dict(lambda_mesh=0.2, mesh_pair_K=4),
        "physics": dict(aux_loss_weights={"strain_inc": 0.1, "strain_cum": 0.2, "floor": 0.3}),
    }[auxiliary]
    _compare(architecture, "deformable", prediction_type,
             dict(options, noise_context=True, clip_deltas=True))
