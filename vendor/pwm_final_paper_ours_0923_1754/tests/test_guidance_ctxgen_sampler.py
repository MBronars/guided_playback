"""The differentiable sampler's context-generating path (guidance.sampler, context_predicted) equals
the production sampler bitwise on the future rows for the same noise, returns the whole window,
and carries a gradient from an objective on the integrated window to both pose-noise tensors."""
import torch

from diffusion.flow_matching import TrajectoryFlowMatching
from guidance import objectives
from guidance.sampler import draw_noise, sample_window
from inference import run_diffusion_inference_df
from models.unet_std import UNetWrapper
from tests.test_unet_std_pose_native import _model_cfg, make_batch


def _ctxgen_model():
    torch.manual_seed(29417)
    model = UNetWrapper(_model_cfg()).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.025)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._context_predicted = True          # what models.factory sets for train.pose_noise_context
    return model


def test_ctxgen_sampler_matches_production_on_future_rows():
    batch = make_batch(deform_n=0)           # entirely rigid: the path the port covers
    model, diffusion = _ctxgen_model(), TrajectoryFlowMatching(prediction_type="sample")
    C1 = batch.context_len - 1
    for schedule, kw in (("full_sequence", {}), ("stream", dict(stream_keep=1))):
        torch.manual_seed(1234)
        ref = run_diffusion_inference_df(model, batch, diffusion, num_inference_steps=4, schedule=schedule, **kw)
        torch.manual_seed(1234)
        with torch.no_grad():
            noise = draw_noise(model, batch)
            assert len(noise) == 4
            ours = sample_window(model, diffusion, batch, noise, steps=4, schedule=schedule, **kw)
        assert ours[1].shape[1] == C1 + batch.num_future_frames
        torch.testing.assert_close(ours[0][:, C1:], ref[0], rtol=0, atol=0)
        torch.testing.assert_close(ours[1][:, C1:], ref[1], rtol=0, atol=0)
        assert torch.isfinite(ours[1]).all()


def test_ctxgen_objective_gradient_reaches_both_pose_noises():
    batch = make_batch(deform_n=0)
    model, diffusion = _ctxgen_model(), TrajectoryFlowMatching(prediction_type="sample")
    torch.manual_seed(7)
    noise = tuple(n.detach().requires_grad_(True) for n in draw_noise(model, batch))
    x0p, x0pose = sample_window(model, diffusion, batch, noise, steps=3, schedule="full_sequence")
    pred = objectives.materialize(batch, x0p, x0pose)
    assert pred["positions"].shape[1] == batch.context_len - 1 + batch.num_future_frames
    body = int(pred["rigid_idx"][0])
    loss, _ = objectives.body_target(body, pred["t"][:, 0, 0].detach() + 0.05)(pred, batch)
    loss.backward()
    assert noise[1].grad is not None and torch.isfinite(noise[1].grad).all() and noise[1].grad.norm() > 0
    assert noise[3].grad is not None and torch.isfinite(noise[3].grad).all() and noise[3].grad.norm() > 0
    # the existing box term is finite and differentiable on the same window
    box = objectives.box_clearance([(pred["t"][0, -1, 0].detach().tolist(), [0.05, 0.05, 0.05])])
    val, diag = box(pred, batch)
    assert torch.isfinite(val) and "box_worst_mm" in diag
