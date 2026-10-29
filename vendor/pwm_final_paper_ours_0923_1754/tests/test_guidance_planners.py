"""Planners run end to end on a tiny pose-native UNet, account their NFE, and the
receding-horizon driver commits the model's own output as context while the
actuated bodies follow the executed plan."""

import torch

from diffusion.flow_matching import TrajectoryFlowMatching
from guidance import objectives, planners
from guidance.horizon import receding_horizon
from guidance.plan import actuated_bodies, body_rows, integrate, plan_from_batch
from guidance.planners import window_nfe
from models.unet_std import UNetWrapper
from tests.test_unet_std_pose_native import _model_cfg, make_batch


def _model():
    torch.manual_seed(29417)
    model = UNetWrapper(_model_cfg()).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.normal_(0, 0.025)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _objective(batch):
    C, passive = batch.context_len, 3
    target = batch.body_poses[:, C - 1, passive, :3, 3] + torch.tensor([0.03, 0.0, 0.0])
    return objectives.weighted((1.0, objectives.body_target(passive, target)),
                               (100.0, objectives.phantom_motion(passive, contact_radius=0.02)))


def test_every_planner_runs_and_accounts_nfe():
    batch, model, diffusion = make_batch(), _model(), TrajectoryFlowMatching()
    objective = _objective(batch)
    per_window = window_nfe(diffusion, batch, 2, "pyramid", 1.0)
    assert per_window >= 1
    gen = torch.Generator().manual_seed(0)
    cases = {
        "best_of_k": (dict(k=2), 2 * per_window),
        "cem": (dict(iterations=1, population=2, elites=1, waypoints=2), 2 * per_window),
        "noise_opt": (dict(restarts=1, iterations=2), 2 * per_window),
        "actuator_opt": (dict(restarts=1, iterations=2), 2 * per_window),
        "joint_opt": (dict(restarts=1, iterations=2), 2 * per_window),
    }
    for name, (kwargs, expected_nfe) in cases.items():
        result = planners.PLANNERS[name](model, diffusion, batch, objective, steps=2, generator=gen, **kwargs)
        assert result.deltas.shape == plan_from_batch(batch).shape, name
        assert torch.isfinite(result.deltas).all() and torch.isfinite(torch.tensor(result.loss)), name
        assert result.nfe == expected_nfe, (name, result.nfe, expected_nfe)
        assert result.loss <= min(t["loss"] for t in result.trace) + 1e-9, name
        assert "target_distance_mm" in result.diagnostics and "phantom_mm" in result.diagnostics, name
        assert result.pred["positions"].shape == batch.future.shape, name


def test_receding_horizon_commits_model_output_and_planned_actuator():
    batch, model, diffusion = make_batch(), _model(), TrajectoryFlowMatching()
    C, N = batch.context_len, batch.num_particles
    H = batch.num_frames - C                                   # 4 future frames in the test layout
    objective = _objective(batch)
    gen = torch.Generator().manual_seed(1)
    out = receding_horizon(model, diffusion, batch, planners.best_of_k, objective, horizon=H, stride=1,
                           future_len=2, steps=2, generator=gen, planner_kwargs=dict(k=1))
    assert out["positions"].shape == (batch.batch_size, C + H, N, 3)
    assert out["deltas"].shape == (batch.batch_size, H, actuated_bodies(batch).numel(), 6)
    assert len(out["windows"]) == 3 and out["nfe"] == sum(w.nfe for w in out["windows"])
    # context frames are untouched ground truth
    torch.testing.assert_close(out["positions"][:, :C], batch.positions[:, :C])
    # actuated particles equal template @ (anchor pose composed with the executed deltas)
    aidx = actuated_bodies(batch)
    anchor = batch.body_poses[:, C - 1, aidx]
    R, t = integrate(anchor[:, :, :3, :3].float(), anchor[:, :, :3, 3].float(), out["deltas"].float())
    for j, body in enumerate(aidx.tolist()):
        n_t = int(batch.body_template_mask[0, body].sum())
        tmpl = batch.body_templates[:, body, :n_t].float()
        pts = torch.einsum("bfij,bpj->bfpi", R[:, :, j], tmpl) + t[:, :, j, None]
        rows = body_rows(batch, body)
        got = torch.stack([out["positions"][e, C:, rows[e]] for e in range(batch.batch_size)])
        torch.testing.assert_close(got, pts, rtol=1e-4, atol=1e-5)
        # and the carried pose track for that body is the plan's
        col = int((out["rigid_idx"] == body).nonzero()[0])
        torch.testing.assert_close(out["T"][:, C:, col], t[:, :, j], rtol=1e-4, atol=1e-5)
    # a single full window with the GT plan and one sample is the plain sampler path
    single = receding_horizon(model, diffusion, batch, planners.best_of_k, objective, horizon=H, stride=H,
                              future_len=H, steps=2, generator=torch.Generator().manual_seed(2), planner_kwargs=dict(k=1))
    assert len(single["windows"]) == 1 and single["positions"].shape == out["positions"].shape
